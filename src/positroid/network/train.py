"""Numpy-only training for small ReLU networks.

Supports single hidden layer + linear output for binary classification
(sigmoid output, BCE loss). Three parameter modes:
- Unconstrained: standard W1, b1, W2, b2
- TP-constrained (exponential): W1[i,j] = exp(a_i * b_j)
- TP-constrained (Cauchy): W1[i,j] = 1/(a_i + b_j)
Both TP modes enforce strictly increasing a, b via cumulative softplus.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from positroid.network.relu_network import ReluLayer, ReluNetwork

# ── Activation / Loss ──


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    neg = ~pos
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def binary_cross_entropy(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Mean binary cross-entropy loss."""
    p = np.clip(y_pred.ravel(), eps, 1 - eps)
    t = y_true.ravel()
    return -float(np.mean(t * np.log(p) + (1 - t) * np.log(1 - p)))


# ── Forward / Backward ──


def forward_pass(
    x: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Forward pass: input → hidden (ReLU) → output (sigmoid).

    Returns (y_pred, z1_pre, z1_post, z2_pre).
    """
    z1_pre = x @ w1.T + b1  # (batch, H)
    z1_post = np.maximum(z1_pre, 0)  # (batch, H)
    z2_pre = z1_post @ w2.T + b2  # (batch, 1)
    y_pred = sigmoid(z2_pre)  # (batch, 1)
    return y_pred, z1_pre, z1_post, z2_pre


def backward_pass(
    x: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    z1_pre: np.ndarray,
    z1_post: np.ndarray,
    w2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Backward pass for BCE + sigmoid (simplifies to y_pred - y_true).

    Returns (dW1, db1, dW2, db2).
    """
    batch = x.shape[0]
    y_t = y_true.reshape(-1, 1)

    # Output gradient: d(BCE)/d(z2_pre) = y_pred - y_true
    dz2 = y_pred - y_t  # (batch, 1)

    dw2 = dz2.T @ z1_post / batch  # (1, H)
    db2 = np.mean(dz2, axis=0)  # (1,)

    # Hidden gradient
    dz1_post = dz2 @ w2  # (batch, H)
    dz1_pre = dz1_post * (z1_pre > 0).astype(float)  # ReLU derivative

    dw1 = dz1_pre.T @ x / batch  # (H, 2)
    db1 = np.mean(dz1_pre, axis=0)  # (H,)

    return dw1, db1, dw2, db2


# ── Softplus utilities for TP reparameterization ──


def _softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus: log(1 + exp(x))."""
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20))))


def _raw_to_sorted(raw: np.ndarray) -> np.ndarray:
    """Convert unconstrained raw params to strictly increasing sequence.

    a[0] = raw[0], a[i] = a[i-1] + softplus(raw[i]) for i > 0.
    """
    result = np.empty_like(raw)
    result[0] = raw[0]
    for i in range(1, len(raw)):
        result[i] = result[i - 1] + _softplus(raw[i])
    return result


# ── Parameter Classes ──


@dataclass
class TrainConfig:
    """Training configuration."""

    hidden_dim: int = 10
    learning_rate: float = 0.01
    epochs: int = 200
    batch_size: int = 32
    optimizer: str = "adam"
    tp_constrained: bool = False
    tp_kernel: str = "exponential"
    seed: int = 42
    beta1: float = 0.9
    beta2: float = 0.999
    adam_eps: float = 1e-8


@dataclass
class TrainHistory:
    """Training history."""

    losses: list[float] = field(default_factory=list)
    accuracies: list[float] = field(default_factory=list)
    snapshots: dict[int, ReluNetwork] = field(default_factory=dict)


class UnconstrainedParams:
    """Standard unconstrained parameters: W1, b1, W2, b2."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        rng: np.random.Generator,
    ) -> None:
        scale1 = np.sqrt(2.0 / input_dim)
        self.w1 = rng.normal(0, scale1, (hidden_dim, input_dim))
        self.b1 = np.zeros(hidden_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.w2 = rng.normal(0, scale2, (1, hidden_dim))
        self.b2 = np.zeros(1)

    def weights(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (W1, b1, W2, b2)."""
        return self.w1, self.b1, self.w2, self.b2

    def param_list(self) -> list[np.ndarray]:
        """Flat list of parameter arrays for optimizer."""
        return [self.w1, self.b1, self.w2, self.b2]

    def to_relu_network(self) -> ReluNetwork:
        return ReluNetwork(
            [
                ReluLayer(self.w1.copy(), self.b1.copy()),
                ReluLayer(self.w2.copy(), self.b2.copy()),
            ]
        )

    def compute_grads(
        self,
        x: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        z1_pre: np.ndarray,
        z1_post: np.ndarray,
    ) -> list[np.ndarray]:
        """Compute gradients for all parameters."""
        dw1, db1, dw2, db2 = backward_pass(
            x,
            y_true,
            y_pred,
            z1_pre,
            z1_post,
            self.w2,
        )
        return [dw1, db1, dw2, db2]


class TPConstrainedParams:
    """TP-constrained W1 via exponential kernel reparameterization.

    W1[i,j] = exp(a[i] * b[j]) where a, b are strictly increasing.
    Free parameters: a_raw, b_raw (unconstrained), plus b1, W2, b2.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        rng: np.random.Generator,
    ) -> None:
        # Initialize so that sorted params land in [0.5, 2.5] range
        self.a_raw = np.concatenate(
            [
                [rng.uniform(0.5, 1.0)],
                rng.uniform(0.2, 0.5, size=hidden_dim - 1),
            ]
        )
        self.b_raw = np.concatenate(
            [
                [rng.uniform(0.5, 1.0)],
                rng.uniform(0.2, 0.5, size=input_dim - 1),
            ]
        )
        self.b1 = np.zeros(hidden_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.w2 = rng.normal(0, scale2, (1, hidden_dim))
        self.b2 = np.zeros(1)

    def sorted_params(self) -> tuple[np.ndarray, np.ndarray]:
        """Convert raw params to strictly increasing sequences."""
        return _raw_to_sorted(self.a_raw), _raw_to_sorted(self.b_raw)

    def weight_matrix(self) -> np.ndarray:
        """Compute W1 = exp(outer(a, b))."""
        a, b = self.sorted_params()
        return np.exp(np.outer(a, b))

    def weights(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (W1, b1, W2, b2)."""
        return self.weight_matrix(), self.b1, self.w2, self.b2

    def param_list(self) -> list[np.ndarray]:
        """Flat list of parameter arrays for optimizer."""
        return [self.a_raw, self.b_raw, self.b1, self.w2, self.b2]

    def to_relu_network(self) -> ReluNetwork:
        w1 = self.weight_matrix()
        return ReluNetwork(
            [
                ReluLayer(w1.copy(), self.b1.copy()),
                ReluLayer(self.w2.copy(), self.b2.copy()),
            ]
        )

    def compute_grads(
        self,
        x: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        z1_pre: np.ndarray,
        z1_post: np.ndarray,
    ) -> list[np.ndarray]:
        """Compute gradients w.r.t. a_raw, b_raw, b1, W2, b2.

        Chain rule through the exponential kernel:
          dL/da[i] = sum_j dL/dW1[i,j] * b[j] * W1[i,j]
          dL/db[j] = sum_i dL/dW1[i,j] * a[i] * W1[i,j]
        Then chain through cumulative softplus for da_raw, db_raw.
        """
        a, b = self.sorted_params()
        w1 = np.exp(np.outer(a, b))

        # Standard backward pass to get dL/dW1
        dw1, db1, dw2, db2 = backward_pass(
            x,
            y_true,
            y_pred,
            z1_pre,
            z1_post,
            self.w2,
        )

        # Chain through exponential kernel
        # dL/da[i] = sum_j dL/dW1[i,j] * b[j] * W1[i,j]
        dl_da = np.sum(dw1 * w1 * b[np.newaxis, :], axis=1)  # (H,)
        # dL/db[j] = sum_i dL/dW1[i,j] * a[i] * W1[i,j]
        dl_db = np.sum(dw1 * w1 * a[:, np.newaxis], axis=0)  # (input_dim,)

        # Chain through cumulative softplus: da/da_raw
        # da_raw[0]: da[i]/da_raw[0] = 1 for all i
        # da_raw[k] (k>0): da[i]/da_raw[k] = sigmoid(a_raw[k]) for i >= k
        # So dL/da_raw[k] = sigmoid(a_raw[k]) * sum_{i>=k} dL/da[i]
        # This is a reverse cumsum scaled by sigmoid
        dl_da_cumsum = np.cumsum(dl_da[::-1])[::-1]  # reverse cumsum
        sig_a = sigmoid(self.a_raw)
        da_raw = dl_da_cumsum.copy()
        da_raw[1:] *= sig_a[1:]
        # da_raw[0] doesn't get sigmoid scaling (direct derivative = 1)

        dl_db_cumsum = np.cumsum(dl_db[::-1])[::-1]
        sig_b = sigmoid(self.b_raw)
        db_raw = dl_db_cumsum.copy()
        db_raw[1:] *= sig_b[1:]

        return [da_raw, db_raw, db1, dw2, db2]


class CauchyConstrainedParams:
    """TP-constrained W1 via Cauchy kernel reparameterization.

    W1[i,j] = 1/(a[i] + b[j]) where a, b are strictly increasing and positive.
    Free parameters: a_raw, b_raw (unconstrained), plus b1, W2, b2.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        rng: np.random.Generator,
    ) -> None:
        # Initialize so that sorted params land in positive range [1, ~4]
        self.a_raw = np.concatenate(
            [
                [rng.uniform(1.0, 2.0)],
                rng.uniform(0.3, 0.8, size=hidden_dim - 1),
            ]
        )
        self.b_raw = np.concatenate(
            [
                [rng.uniform(1.0, 2.0)],
                rng.uniform(0.3, 0.8, size=input_dim - 1),
            ]
        )
        self.b1 = np.zeros(hidden_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.w2 = rng.normal(0, scale2, (1, hidden_dim))
        self.b2 = np.zeros(1)

    def sorted_params(self) -> tuple[np.ndarray, np.ndarray]:
        """Convert raw params to strictly increasing sequences."""
        return _raw_to_sorted(self.a_raw), _raw_to_sorted(self.b_raw)

    def weight_matrix(self) -> np.ndarray:
        """Compute W1 = 1/(outer_sum(a, b))."""
        a, b = self.sorted_params()
        w1: np.ndarray = 1.0 / (a[:, np.newaxis] + b[np.newaxis, :])
        return w1

    def weights(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (W1, b1, W2, b2)."""
        return self.weight_matrix(), self.b1, self.w2, self.b2

    def param_list(self) -> list[np.ndarray]:
        """Flat list of parameter arrays for optimizer."""
        return [self.a_raw, self.b_raw, self.b1, self.w2, self.b2]

    def to_relu_network(self) -> ReluNetwork:
        w1 = self.weight_matrix()
        return ReluNetwork(
            [
                ReluLayer(w1.copy(), self.b1.copy()),
                ReluLayer(self.w2.copy(), self.b2.copy()),
            ]
        )

    def compute_grads(
        self,
        x: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        z1_pre: np.ndarray,
        z1_post: np.ndarray,
    ) -> list[np.ndarray]:
        """Compute gradients w.r.t. a_raw, b_raw, b1, W2, b2.

        Chain rule through the Cauchy kernel:
          dW1/da_i = -W1[i,j]^2,  dW1/db_j = -W1[i,j]^2
          dL/da[i] = sum_j dL/dW1[i,j] * (-W1[i,j]^2)
          dL/db[j] = sum_i dL/dW1[i,j] * (-W1[i,j]^2)
        Then chain through cumulative softplus for da_raw, db_raw.
        """
        a, b = self.sorted_params()
        w1 = 1.0 / (a[:, np.newaxis] + b[np.newaxis, :])

        dw1, db1, dw2, db2 = backward_pass(
            x,
            y_true,
            y_pred,
            z1_pre,
            z1_post,
            self.w2,
        )

        # Chain through Cauchy kernel: dW1/da_i = dW1/db_j = -W1^2
        w1_sq = w1 * w1
        dl_da = np.sum(dw1 * (-w1_sq), axis=1)  # (H,)
        dl_db = np.sum(dw1 * (-w1_sq), axis=0)  # (input_dim,)

        # Chain through cumulative softplus (same as exponential kernel)
        dl_da_cumsum = np.cumsum(dl_da[::-1])[::-1]
        sig_a = sigmoid(self.a_raw)
        da_raw = dl_da_cumsum.copy()
        da_raw[1:] *= sig_a[1:]

        dl_db_cumsum = np.cumsum(dl_db[::-1])[::-1]
        sig_b = sigmoid(self.b_raw)
        db_raw = dl_db_cumsum.copy()
        db_raw[1:] *= sig_b[1:]

        return [da_raw, db_raw, db1, dw2, db2]


# ── Optimizers ──


class SGD:
    """Stochastic gradient descent."""

    def __init__(self, params: list[np.ndarray], lr: float) -> None:
        self.params = params
        self.lr = lr

    def step(self, grads: list[np.ndarray]) -> None:
        for p, g in zip(self.params, grads, strict=True):
            p -= self.lr * g


class Adam:
    """Adam optimizer."""

    def __init__(
        self,
        params: list[np.ndarray],
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads: list[np.ndarray]) -> None:
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads, strict=True)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g * g
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ── Training Loop ──


def train(
    x: np.ndarray,
    y: np.ndarray,
    config: TrainConfig,
    snapshot_epochs: list[int] | None = None,
) -> tuple[ReluNetwork, TrainHistory]:
    """Train a single-hidden-layer ReLU network for binary classification.

    Args:
        x: Input data, shape (n_samples, input_dim).
        y: Labels, shape (n_samples,), values in {0, 1}.
        config: Training configuration.
        snapshot_epochs: Save network snapshots at these epochs.

    Returns:
        (trained_network, history)
    """
    rng = np.random.default_rng(config.seed)
    input_dim = x.shape[1]

    # Initialize parameters
    if config.tp_constrained:
        tp_params: TPConstrainedParams | CauchyConstrainedParams
        if config.tp_kernel == "cauchy":
            tp_params = CauchyConstrainedParams(input_dim, config.hidden_dim, rng)
        else:
            tp_params = TPConstrainedParams(input_dim, config.hidden_dim, rng)
        params: UnconstrainedParams | TPConstrainedParams | CauchyConstrainedParams = tp_params
    else:
        params = UnconstrainedParams(input_dim, config.hidden_dim, rng)

    # Create optimizer
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

    history = TrainHistory()
    n_samples = x.shape[0]

    for epoch in range(config.epochs):
        perm = rng.permutation(n_samples)
        x_shuf, y_shuf = x[perm], y[perm]

        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, config.batch_size):
            x_batch = x_shuf[start : start + config.batch_size]
            y_batch = y_shuf[start : start + config.batch_size]

            w1, b1, w2, b2 = params.weights()
            y_pred, z1_pre, z1_post, _ = forward_pass(x_batch, w1, b1, w2, b2)

            loss = binary_cross_entropy(y_pred, y_batch)
            epoch_loss += loss
            n_batches += 1

            grads = params.compute_grads(x_batch, y_batch, y_pred, z1_pre, z1_post)
            opt.step(grads)

        history.losses.append(epoch_loss / n_batches)

        # Compute accuracy on full dataset
        w1, b1, w2, b2 = params.weights()
        full_pred, _, _, _ = forward_pass(x, w1, b1, w2, b2)
        preds = (full_pred.ravel() > 0.5).astype(float)
        accuracy = float(np.mean(preds == y))
        history.accuracies.append(accuracy)

        if snapshot_epochs is not None and epoch in snapshot_epochs:
            history.snapshots[epoch] = params.to_relu_network()

    return params.to_relu_network(), history
