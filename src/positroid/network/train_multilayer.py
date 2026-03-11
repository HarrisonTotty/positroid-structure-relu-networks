"""Multi-layer ReLU network training for binary classification.

Supports arbitrary-depth hidden layers with sigmoid output and BCE loss.
Two parameter modes:
- unconstrained: He-init weights, zero biases
- tp_exponential: W_i = exp(outer(a_i, b_i)) per hidden layer, output unconstrained
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from positroid.network.relu_network import ReluLayer, ReluNetwork
from positroid.network.train import (
    SGD,
    Adam,
    _raw_to_sorted,
    binary_cross_entropy,
    sigmoid,
)

# ── Config / History ──


@dataclass
class MultiLayerTrainConfig:
    """Training configuration for multi-layer networks."""

    layer_dims: list[int] = field(default_factory=lambda: [8, 6])
    learning_rate: float = 0.01
    epochs: int = 200
    batch_size: int = 32
    optimizer: str = "adam"
    param_mode: str = "unconstrained"
    seed: int = 42
    beta1: float = 0.9
    beta2: float = 0.999
    adam_eps: float = 1e-8


@dataclass
class MultiLayerTrainHistory:
    """Training history for multi-layer networks."""

    losses: list[float] = field(default_factory=list)
    accuracies: list[float] = field(default_factory=list)


# ── General Forward / Backward ──


def multilayer_forward(
    x: np.ndarray,
    weights: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Forward pass through L layers: ReLU on hidden, sigmoid on output.

    Args:
        x: Input data, shape (batch, input_dim).
        weights: List of (W_i, b_i) pairs for each layer.

    Returns:
        (y_pred, pre_acts, post_acts) where:
        - y_pred: sigmoid output, shape (batch, 1)
        - pre_acts: pre-activation at each layer (before ReLU/sigmoid)
        - post_acts: post-activation at each layer (after ReLU; input for last)
    """
    pre_acts: list[np.ndarray] = []
    post_acts: list[np.ndarray] = []
    z = x
    n_layers = len(weights)
    for i, (w, b) in enumerate(weights):
        pre = z @ w.T + b  # (batch, out_dim)
        pre_acts.append(pre)
        z = np.maximum(pre, 0) if i < n_layers - 1 else sigmoid(pre)
        post_acts.append(z)
    return post_acts[-1], pre_acts, post_acts


def multilayer_backward(
    x: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pre_acts: list[np.ndarray],
    post_acts: list[np.ndarray],
    weights: list[tuple[np.ndarray, np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Backward pass for multi-layer BCE + sigmoid network.

    Returns list of (dW_i, db_i) for each layer.
    """
    batch = x.shape[0]
    y_t = y_true.reshape(-1, 1)
    n_layers = len(weights)

    grads: list[tuple[np.ndarray, np.ndarray]] = [
        (np.zeros_like(w), np.zeros_like(b)) for w, b in weights
    ]

    # Output gradient: d(BCE+sigmoid)/d(z_out) = y_pred - y_true
    dz = y_pred - y_t  # (batch, output_dim)

    for i in range(n_layers - 1, -1, -1):
        layer_input = x if i == 0 else post_acts[i - 1]

        dw = dz.T @ layer_input / batch
        db = np.mean(dz, axis=0)
        grads[i] = (dw, db)

        if i > 0:
            # Propagate through this layer's weight
            w_i = weights[i][0]
            dz_prev = dz @ w_i  # (batch, prev_dim)
            # ReLU derivative at layer i-1
            dz = dz_prev * (pre_acts[i - 1] > 0).astype(float)

    return grads


# ── Parameter Classes ──


class MultiLayerUnconstrainedParams:
    """Standard unconstrained parameters for multi-layer network."""

    def __init__(
        self,
        input_dim: int,
        layer_dims: list[int],
        rng: np.random.Generator,
    ) -> None:
        dims = [input_dim] + layer_dims + [1]
        self._weights: list[np.ndarray] = []
        self._biases: list[np.ndarray] = []
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / dims[i])
            self._weights.append(rng.normal(0, scale, (dims[i + 1], dims[i])))
            self._biases.append(np.zeros(dims[i + 1]))

    def weights(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return list of (W_i, b_i) pairs."""
        return list(zip(self._weights, self._biases, strict=True))

    def param_list(self) -> list[np.ndarray]:
        """Flat list of parameter arrays: [W0, b0, W1, b1, ...]."""
        result: list[np.ndarray] = []
        for w, b in zip(self._weights, self._biases, strict=True):
            result.append(w)
            result.append(b)
        return result

    def to_relu_network(self) -> ReluNetwork:
        layers = [
            ReluLayer(w.copy(), b.copy()) for w, b in zip(self._weights, self._biases, strict=True)
        ]
        return ReluNetwork(layers)

    def compute_grads(
        self,
        x: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        pre_acts: list[np.ndarray],
        post_acts: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Compute gradients for all parameters."""
        layer_grads = multilayer_backward(
            x,
            y_true,
            y_pred,
            pre_acts,
            post_acts,
            self.weights(),
        )
        result: list[np.ndarray] = []
        for dw, db in layer_grads:
            result.append(dw)
            result.append(db)
        return result


class MultiLayerTPExponentialParams:
    """TP-constrained hidden layers via exponential kernel, unconstrained output.

    Each hidden layer i: W_i = exp(outer(a_i, b_i)) with strictly increasing a, b.
    Output layer: standard unconstrained weights.
    """

    def __init__(
        self,
        input_dim: int,
        layer_dims: list[int],
        rng: np.random.Generator,
    ) -> None:
        dims = [input_dim] + layer_dims + [1]
        n_hidden = len(layer_dims)

        # TP params for each hidden layer
        self._a_raws: list[np.ndarray] = []
        self._b_raws: list[np.ndarray] = []
        self._biases: list[np.ndarray] = []

        for i in range(n_hidden):
            out_dim = dims[i + 1]
            in_dim = dims[i]
            a_raw = np.concatenate(
                [
                    [rng.uniform(0.5, 1.0)],
                    rng.uniform(0.2, 0.5, size=out_dim - 1),
                ]
            )
            b_raw = np.concatenate(
                [
                    [rng.uniform(0.5, 1.0)],
                    rng.uniform(0.2, 0.5, size=in_dim - 1),
                ]
            )
            self._a_raws.append(a_raw)
            self._b_raws.append(b_raw)
            self._biases.append(np.zeros(out_dim))

        # Output layer: unconstrained
        out_in = dims[-2]
        scale_out = np.sqrt(2.0 / out_in)
        self._w_out = rng.normal(0, scale_out, (1, out_in))
        self._b_out = np.zeros(1)

    def _sorted_params(self, layer_idx: int) -> tuple[np.ndarray, np.ndarray]:
        return (
            _raw_to_sorted(self._a_raws[layer_idx]),
            _raw_to_sorted(self._b_raws[layer_idx]),
        )

    def _weight_matrix(self, layer_idx: int) -> np.ndarray:
        a, b = self._sorted_params(layer_idx)
        return np.exp(np.outer(a, b))

    def weights(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return list of (W_i, b_i) pairs."""
        result: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(len(self._a_raws)):
            result.append((self._weight_matrix(i), self._biases[i]))
        result.append((self._w_out, self._b_out))
        return result

    def param_list(self) -> list[np.ndarray]:
        """Flat list: [a0_raw, b0_raw, bias0, a1_raw, b1_raw, bias1, ..., w_out, b_out]."""
        result: list[np.ndarray] = []
        for i in range(len(self._a_raws)):
            result.append(self._a_raws[i])
            result.append(self._b_raws[i])
            result.append(self._biases[i])
        result.append(self._w_out)
        result.append(self._b_out)
        return result

    def to_relu_network(self) -> ReluNetwork:
        layers: list[ReluLayer] = []
        for i in range(len(self._a_raws)):
            layers.append(
                ReluLayer(
                    self._weight_matrix(i).copy(),
                    self._biases[i].copy(),
                )
            )
        layers.append(ReluLayer(self._w_out.copy(), self._b_out.copy()))
        return ReluNetwork(layers)

    def compute_grads(
        self,
        x: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        pre_acts: list[np.ndarray],
        post_acts: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Compute gradients w.r.t. all raw params.

        For each hidden layer, chain dW through exp kernel + cumulative softplus.
        Output layer gradients are direct.
        """
        all_weights = self.weights()
        layer_grads = multilayer_backward(
            x,
            y_true,
            y_pred,
            pre_acts,
            post_acts,
            all_weights,
        )

        n_hidden = len(self._a_raws)
        result: list[np.ndarray] = []

        for i in range(n_hidden):
            dw_i, db_i = layer_grads[i]
            a, b = self._sorted_params(i)
            w_i = np.exp(np.outer(a, b))

            # Chain through exponential kernel
            dl_da = np.sum(dw_i * w_i * b[np.newaxis, :], axis=1)
            dl_db = np.sum(dw_i * w_i * a[:, np.newaxis], axis=0)

            # Chain through cumulative softplus
            dl_da_cumsum = np.cumsum(dl_da[::-1])[::-1]
            sig_a = sigmoid(self._a_raws[i])
            da_raw = dl_da_cumsum.copy()
            da_raw[1:] *= sig_a[1:]

            dl_db_cumsum = np.cumsum(dl_db[::-1])[::-1]
            sig_b = sigmoid(self._b_raws[i])
            db_raw = dl_db_cumsum.copy()
            db_raw[1:] *= sig_b[1:]

            result.append(da_raw)
            result.append(db_raw)
            result.append(db_i)

        # Output layer: direct gradients
        dw_out, db_out = layer_grads[-1]
        result.append(dw_out)
        result.append(db_out)

        return result


# ── Training Loop ──


def train_multilayer(
    x: np.ndarray,
    y: np.ndarray,
    config: MultiLayerTrainConfig,
) -> tuple[ReluNetwork, MultiLayerTrainHistory]:
    """Train a multi-layer ReLU network for binary classification.

    Args:
        x: Input data, shape (n_samples, input_dim).
        y: Labels, shape (n_samples,), values in {0, 1}.
        config: Training configuration.

    Returns:
        (trained_network, history)
    """
    rng = np.random.default_rng(config.seed)
    input_dim = x.shape[1]

    params: MultiLayerUnconstrainedParams | MultiLayerTPExponentialParams
    if config.param_mode == "tp_exponential":
        params = MultiLayerTPExponentialParams(input_dim, config.layer_dims, rng)
    else:
        params = MultiLayerUnconstrainedParams(input_dim, config.layer_dims, rng)

    if config.optimizer == "adam":
        opt: SGD | Adam = Adam(
            params.param_list(),
            config.learning_rate,
            config.beta1,
            config.beta2,
            config.adam_eps,
        )
    else:
        opt = SGD(params.param_list(), config.learning_rate)

    history = MultiLayerTrainHistory()
    n_samples = x.shape[0]

    for _epoch in range(config.epochs):
        perm = rng.permutation(n_samples)
        x_shuf, y_shuf = x[perm], y[perm]

        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, config.batch_size):
            x_batch = x_shuf[start : start + config.batch_size]
            y_batch = y_shuf[start : start + config.batch_size]

            all_weights = params.weights()
            y_pred, pre_acts, post_acts = multilayer_forward(x_batch, all_weights)

            loss = binary_cross_entropy(y_pred, y_batch)
            epoch_loss += loss
            n_batches += 1

            grads = params.compute_grads(
                x_batch,
                y_batch,
                y_pred,
                pre_acts,
                post_acts,
            )
            opt.step(grads)

        history.losses.append(epoch_loss / n_batches)

        # Full-dataset accuracy
        all_weights = params.weights()
        full_pred, _, _ = multilayer_forward(x, all_weights)
        preds = (full_pred.ravel() > 0.5).astype(float)
        accuracy = float(np.mean(preds == y))
        history.accuracies.append(accuracy)

    return params.to_relu_network(), history
