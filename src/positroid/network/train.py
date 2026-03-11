"""Numpy-only training for small ReLU networks.

Supports single hidden layer + linear output for binary classification
(sigmoid output, BCE loss). Eight parameter modes:
- Unconstrained: standard W1, b1, W2, b2
- TP-constrained (exponential): W1[i,j] = exp(a_i * b_j)
- TP-constrained (Cauchy): W1[i,j] = 1/(a_i + b_j)
- Sinusoidal (non-TP): W1[i,j] = 2 + sin(a_i * b_j)
- Quadratic distance (non-TP): W1[i,j] = (a_i - b_j)^2 + 1
- Permuted exponential (non-TP): W1[i,j] = exp(a_i * b_perm[j])
- Negated bidiagonal (non-TP): W1 = B @ exp(outer(a,b))
- Fixed convergent bias-only (non-TP): frozen W1, train only biases
All kernel modes enforce strictly increasing a, b via cumulative softplus.
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
    param_mode: str | None = None
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


class SinusoidalConstrainedParams:
    """Non-TP W1 via sinusoidal kernel reparameterization.

    W1[i,j] = 2 + sin(a[i] * b[j]) where a, b are strictly increasing.
    Not totally positive: 2x2 minors can be negative (e.g. a=[1,2], b=[1,2]).
    Free parameters: a_raw, b_raw (unconstrained), plus b1, W2, b2.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        rng: np.random.Generator,
    ) -> None:
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
        """Compute W1 = 2 + sin(outer(a, b))."""
        a, b = self.sorted_params()
        return 2.0 + np.sin(np.outer(a, b))

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

        Chain rule through the sinusoidal kernel:
          dW1/da_i = b_j * cos(a_i * b_j)
          dW1/db_j = a_i * cos(a_i * b_j)
        Then chain through cumulative softplus for da_raw, db_raw.
        """
        a, b = self.sorted_params()
        ab = np.outer(a, b)
        cos_ab = np.cos(ab)

        dw1, db1, dw2, db2 = backward_pass(
            x,
            y_true,
            y_pred,
            z1_pre,
            z1_post,
            self.w2,
        )

        # Chain through sinusoidal kernel
        dl_da = np.sum(dw1 * cos_ab * b[np.newaxis, :], axis=1)  # (H,)
        dl_db = np.sum(dw1 * cos_ab * a[:, np.newaxis], axis=0)  # (input_dim,)

        # Chain through cumulative softplus
        dl_da_cumsum = np.cumsum(dl_da[::-1])[::-1]
        sig_a = sigmoid(self.a_raw)
        da_raw = dl_da_cumsum.copy()
        da_raw[1:] *= sig_a[1:]

        dl_db_cumsum = np.cumsum(dl_db[::-1])[::-1]
        sig_b = sigmoid(self.b_raw)
        db_raw = dl_db_cumsum.copy()
        db_raw[1:] *= sig_b[1:]

        return [da_raw, db_raw, db1, dw2, db2]


class QuadraticDistanceConstrainedParams:
    """Non-TP W1 via quadratic distance kernel reparameterization.

    W1[i,j] = (a[i] - b[j])^2 + 1 where a, b are strictly increasing.
    Not totally positive: 2x2 minors can be negative (e.g. a=[1,2], b=[1,2]).
    Free parameters: a_raw, b_raw (unconstrained), plus b1, W2, b2.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        rng: np.random.Generator,
    ) -> None:
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
        """Compute W1 = (a_i - b_j)^2 + 1."""
        a, b = self.sorted_params()
        diff = a[:, np.newaxis] - b[np.newaxis, :]
        w1: np.ndarray = diff * diff + 1.0
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

        Chain rule through the quadratic distance kernel:
          dW1/da_i = 2 * (a_i - b_j)
          dW1/db_j = -2 * (a_i - b_j)
        Then chain through cumulative softplus for da_raw, db_raw.
        """
        a, b = self.sorted_params()
        diff = a[:, np.newaxis] - b[np.newaxis, :]  # (H, input_dim)

        dw1, db1, dw2, db2 = backward_pass(
            x,
            y_true,
            y_pred,
            z1_pre,
            z1_post,
            self.w2,
        )

        # Chain through quadratic distance kernel
        dl_da = np.sum(dw1 * 2.0 * diff, axis=1)  # (H,)
        dl_db = np.sum(dw1 * (-2.0 * diff), axis=0)  # (input_dim,)

        # Chain through cumulative softplus
        dl_da_cumsum = np.cumsum(dl_da[::-1])[::-1]
        sig_a = sigmoid(self.a_raw)
        da_raw = dl_da_cumsum.copy()
        da_raw[1:] *= sig_a[1:]

        dl_db_cumsum = np.cumsum(dl_db[::-1])[::-1]
        sig_b = sigmoid(self.b_raw)
        db_raw = dl_db_cumsum.copy()
        db_raw[1:] *= sig_b[1:]

        return [da_raw, db_raw, db1, dw2, db2]


class PermutedExponentialConstrainedParams:
    """Non-TP W1 via exponential kernel with reversed column permutation.

    W1[i,j] = exp(a[i] * b[perm[j]]) where perm reverses the columns.
    Preserves normal-convergence geometry (rows converge to common direction
    as a_i grows), but breaks TP by disrupting cyclic minor ordering.
    Free parameters: a_raw, b_raw (unconstrained), plus b1, W2, b2.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        rng: np.random.Generator,
    ) -> None:
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
        # Column reversal permutation
        self._perm = np.arange(input_dim)[::-1].copy()
        self._inv_perm = np.empty_like(self._perm)
        self._inv_perm[self._perm] = np.arange(input_dim)
        self.b1 = np.zeros(hidden_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.w2 = rng.normal(0, scale2, (1, hidden_dim))
        self.b2 = np.zeros(1)

    def sorted_params(self) -> tuple[np.ndarray, np.ndarray]:
        """Convert raw params to strictly increasing sequences."""
        return _raw_to_sorted(self.a_raw), _raw_to_sorted(self.b_raw)

    def weight_matrix(self) -> np.ndarray:
        """Compute W1[i,j] = exp(a[i] * b[perm[j]])."""
        a, b = self.sorted_params()
        b_perm = b[self._perm]
        return np.exp(np.outer(a, b_perm))

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

        Chain rule through the permuted exponential kernel:
          W1[i,j] = exp(a[i] * b_perm[j])
          dL/da[i] = sum_j dL/dW1[i,j] * b_perm[j] * W1[i,j]
          dL/db_perm[j] = sum_i dL/dW1[i,j] * a[i] * W1[i,j]
          dL/db = dL/db_perm[inv_perm]
        Then chain through cumulative softplus for da_raw, db_raw.
        """
        a, b = self.sorted_params()
        b_perm = b[self._perm]
        w1 = np.exp(np.outer(a, b_perm))

        dw1, db1, dw2, db2 = backward_pass(
            x,
            y_true,
            y_pred,
            z1_pre,
            z1_post,
            self.w2,
        )

        # Chain through permuted exponential kernel
        dl_da = np.sum(dw1 * w1 * b_perm[np.newaxis, :], axis=1)  # (H,)
        dl_db_perm = np.sum(dw1 * w1 * a[:, np.newaxis], axis=0)  # (input_dim,)
        # Unpermute: db[k] = db_perm[inv_perm[k]]
        dl_db = dl_db_perm[self._inv_perm]

        # Chain through cumulative softplus
        dl_da_cumsum = np.cumsum(dl_da[::-1])[::-1]
        sig_a = sigmoid(self.a_raw)
        da_raw = dl_da_cumsum.copy()
        da_raw[1:] *= sig_a[1:]

        dl_db_cumsum = np.cumsum(dl_db[::-1])[::-1]
        sig_b = sigmoid(self.b_raw)
        db_raw = dl_db_cumsum.copy()
        db_raw[1:] *= sig_b[1:]

        return [da_raw, db_raw, db1, dw2, db2]


class NegatedBidiagonalConstrainedParams:
    """Non-TP W1 via Loewner-Whitney with negated bidiagonal factor.

    W1 = B @ E where E = exp(outer(a, b)) is TP, and B is an H×H lower
    bidiagonal matrix with B[i,i]=1 and B[i+1,i] = (-1)^(i+1) * BIDIAG_SCALE.
    The alternating-sign subdiagonal breaks total positivity while
    preserving the exponential kernel's convergence geometry (for large a_i,
    E[i,:] >> E[i-1,:], so W1[i,:] ≈ E[i,:] asymptotically).
    Free parameters: a_raw, b_raw (unconstrained), plus b1, W2, b2.
    """

    BIDIAG_SCALE = 1.5

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        rng: np.random.Generator,
    ) -> None:
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
        # Build fixed bidiagonal matrix B and its transpose
        self._B = np.eye(hidden_dim)
        for i in range(hidden_dim - 1):
            self._B[i + 1, i] = (-1) ** (i + 1) * self.BIDIAG_SCALE
        self._BT = self._B.T.copy()
        self.b1 = np.zeros(hidden_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.w2 = rng.normal(0, scale2, (1, hidden_dim))
        self.b2 = np.zeros(1)

    def sorted_params(self) -> tuple[np.ndarray, np.ndarray]:
        """Convert raw params to strictly increasing sequences."""
        return _raw_to_sorted(self.a_raw), _raw_to_sorted(self.b_raw)

    def weight_matrix(self) -> np.ndarray:
        """Compute W1 = B @ exp(outer(a, b))."""
        a, b = self.sorted_params()
        exp_ab = np.exp(np.outer(a, b))
        w1: np.ndarray = self._B @ exp_ab
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

        Chain rule through the bidiagonal-exponential product:
          W1 = B @ E, so dL/dE = B^T @ dL/dW1
          dL/da[i] = sum_j dL/dE[i,j] * b[j] * E[i,j]
          dL/db[j] = sum_i dL/dE[i,j] * a[i] * E[i,j]
        Then chain through cumulative softplus for da_raw, db_raw.
        """
        a, b = self.sorted_params()
        exp_ab = np.exp(np.outer(a, b))

        dw1, db1, dw2, db2 = backward_pass(
            x,
            y_true,
            y_pred,
            z1_pre,
            z1_post,
            self.w2,
        )

        # Chain through B: dL/dE = B^T @ dL/dW1
        d_exp = self._BT @ dw1  # (H, input_dim)

        # Chain through exponential kernel
        dl_da = np.sum(d_exp * exp_ab * b[np.newaxis, :], axis=1)  # (H,)
        dl_db = np.sum(d_exp * exp_ab * a[:, np.newaxis], axis=0)  # (input_dim,)

        # Chain through cumulative softplus
        dl_da_cumsum = np.cumsum(dl_da[::-1])[::-1]
        sig_a = sigmoid(self.a_raw)
        da_raw = dl_da_cumsum.copy()
        da_raw[1:] *= sig_a[1:]

        dl_db_cumsum = np.cumsum(dl_db[::-1])[::-1]
        sig_b = sigmoid(self.b_raw)
        db_raw = dl_db_cumsum.copy()
        db_raw[1:] *= sig_b[1:]

        return [da_raw, db_raw, db1, dw2, db2]


class FixedConvergentBiasOnlyParams:
    """Non-TP W1 with normal convergence, bias-only training.

    Generates a fixed W1 using the permuted exponential kernel (not TP)
    and freezes it. Only biases and output-layer weights are trained.
    This isolates the question: does gradient descent on biases alone
    (with a convergent-geometry weight matrix) always produce positroids?

    param_list() returns [b1, w2, b2] — only 3 parameter arrays.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        rng: np.random.Generator,
    ) -> None:
        # Generate a_raw, b_raw just to build the fixed W1
        a_raw = np.concatenate(
            [
                [rng.uniform(0.5, 1.0)],
                rng.uniform(0.2, 0.5, size=hidden_dim - 1),
            ]
        )
        b_raw = np.concatenate(
            [
                [rng.uniform(0.5, 1.0)],
                rng.uniform(0.2, 0.5, size=input_dim - 1),
            ]
        )
        a = _raw_to_sorted(a_raw)
        b = _raw_to_sorted(b_raw)
        # Reversed-column exponential: not TP, but has normal convergence
        perm = np.arange(input_dim)[::-1]
        b_perm = b[perm]
        self._w1_fixed = np.exp(np.outer(a, b_perm))
        self.b1 = np.zeros(hidden_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.w2 = rng.normal(0, scale2, (1, hidden_dim))
        self.b2 = np.zeros(1)

    def weight_matrix(self) -> np.ndarray:
        """Return the frozen W1."""
        return self._w1_fixed

    def weights(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (W1, b1, W2, b2)."""
        return self._w1_fixed, self.b1, self.w2, self.b2

    def param_list(self) -> list[np.ndarray]:
        """Only biases and output-layer weights are trainable."""
        return [self.b1, self.w2, self.b2]

    def to_relu_network(self) -> ReluNetwork:
        return ReluNetwork(
            [
                ReluLayer(self._w1_fixed.copy(), self.b1.copy()),
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
        """Compute gradients for b1, W2, b2 only (W1 is frozen)."""
        _, db1, dw2, db2 = backward_pass(
            x,
            y_true,
            y_pred,
            z1_pre,
            z1_post,
            self.w2,
        )
        return [db1, dw2, db2]


class LoewnerWhitneyConstrainedParams:
    """TP-constrained W1 via Loewner-Whitney bidiagonal factorization.

    Constructs W1 as a product of elementary bidiagonal matrices:
    1. Positive diagonal (d params)
    2. Upper bidiagonal operations in wiring diagram order (d(d-1)/2 params)
    3. Lower bidiagonal operations in Neville order (d*(2H-d-1)/2 params)

    Total: H*d parameters. All positive via exp(raw) reparameterization.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        rng: np.random.Generator,
    ) -> None:
        self._d = input_dim
        self._H = hidden_dim
        # Diagonal: d params
        self._diag_raw = rng.uniform(-0.5, 0.5, size=input_dim)
        # Upper bidiagonal: d(d-1)/2 params
        n_upper = input_dim * (input_dim - 1) // 2
        self._upper_raw = rng.uniform(-1, 0.5, size=n_upper)
        # Lower bidiagonal: d*(2H-d-1)/2 params
        n_lower = sum(hidden_dim - j - 1 for j in range(input_dim))
        self._lower_raw = rng.uniform(-1, 0.5, size=n_lower)
        self.b1 = np.zeros(hidden_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.w2 = rng.normal(0, scale2, (1, hidden_dim))
        self.b2 = np.zeros(1)

    def weight_matrix(self) -> np.ndarray:
        """Build H×d TP matrix via bidiagonal factorization."""
        from positroid.linalg.totally_positive import tp_from_loewner_whitney

        return tp_from_loewner_whitney(
            np.exp(self._diag_raw),
            np.exp(self._upper_raw),
            np.exp(self._lower_raw),
            self._H,
            self._d,
        )

    def weights(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (W1, b1, W2, b2)."""
        return self.weight_matrix(), self.b1, self.w2, self.b2

    def param_list(self) -> list[np.ndarray]:
        """Flat list of parameter arrays for optimizer."""
        return [self._diag_raw, self._upper_raw, self._lower_raw, self.b1, self.w2, self.b2]

    def to_relu_network(self) -> ReluNetwork:
        w1 = self.weight_matrix()
        return ReluNetwork(
            [
                ReluLayer(w1.copy(), self.b1.copy()),
                ReluLayer(self.w2.copy(), self.b2.copy()),
            ]
        )

    def _build_ops_and_prev(
        self,
    ) -> tuple[np.ndarray, list[tuple[str, int, int]], list[np.ndarray]]:
        """Build W1 while recording each operation and A[row-1] before it.

        Returns:
            W1: The weight matrix
            ops: List of (stage, row_modified, row_source_or_col) for each op.
                 stage='U' for upper bidiag, 'L' for lower bidiag.
            a_prev_list: A[source_row] before each operation.
        """
        diag = np.exp(self._diag_raw)
        upper = np.exp(self._upper_raw)
        lower = np.exp(self._lower_raw)
        h, d = self._H, self._d

        a = np.zeros((h, d))
        for i in range(d):
            a[i, i] = diag[i]

        ops: list[tuple[str, int, int]] = []
        a_prev_list: list[np.ndarray] = []

        # Upper bidiagonal
        u_idx = 0
        for level in range(1, d):
            for k in range(d - level, d):
                a_prev_list.append(a[k].copy())
                a[k - 1, :] += upper[u_idx] * a[k, :]
                ops.append(("U", k - 1, k))
                u_idx += 1

        # Lower bidiagonal
        l_idx = 0
        for j in range(d - 1, -1, -1):
            for i in range(j + 1, h):
                a_prev_list.append(a[i - 1].copy())
                a[i, :] += lower[l_idx] * a[i - 1, :]
                ops.append(("L", i, i - 1))
                l_idx += 1

        return a, ops, a_prev_list

    def compute_grads(
        self,
        x: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        z1_pre: np.ndarray,
        z1_post: np.ndarray,
    ) -> list[np.ndarray]:
        """Compute gradients w.r.t. diag_raw, upper_raw, lower_raw, b1, W2, b2.

        Three-stage chain rule:
        1. Standard backward_pass → dL/dW1
        2. Reverse bidiagonal ops to get d_diag, d_upper, d_lower
        3. Chain through exp: d_raw = d_param * param
        """
        diag = np.exp(self._diag_raw)
        upper = np.exp(self._upper_raw)
        lower = np.exp(self._lower_raw)
        d = self._d

        _, ops, a_prev_list = self._build_ops_and_prev()

        # Stage 1: standard backward pass
        dw1, db1, dw2, db2 = backward_pass(
            x,
            y_true,
            y_pred,
            z1_pre,
            z1_post,
            self.w2,
        )

        # Stage 2: reverse bidiagonal ops
        da = dw1.copy()
        d_upper = np.zeros_like(upper)
        d_lower = np.zeros_like(lower)

        # Build forward index mapping for each op
        all_params: list[tuple[str, int]] = []
        ui = 0
        li = 0
        for stage, _, _ in ops:
            if stage == "U":
                all_params.append(("U", ui))
                ui += 1
            else:
                all_params.append(("L", li))
                li += 1

        # Backward: reverse order of all ops
        for k in range(len(ops) - 1, -1, -1):
            _, row_mod, row_src = ops[k]
            a_prev = a_prev_list[k]
            param_type, param_idx = all_params[k]
            if param_type == "U":
                d_upper[param_idx] = np.dot(da[row_mod], a_prev)
                da[row_src] += upper[param_idx] * da[row_mod]
            else:
                d_lower[param_idx] = np.dot(da[row_mod], a_prev)
                da[row_src] += lower[param_idx] * da[row_mod]

        # da now holds gradients w.r.t. the diagonal matrix
        d_diag = np.array([da[i, i] for i in range(d)])

        # Stage 3: chain through exp
        d_diag_raw = d_diag * diag
        d_upper_raw = d_upper * upper
        d_lower_raw = d_lower * lower

        return [d_diag_raw, d_upper_raw, d_lower_raw, db1, dw2, db2]


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

    # Resolve param_mode: explicit param_mode takes priority, else derive
    # from legacy tp_constrained/tp_kernel fields.
    mode = config.param_mode
    if mode is None:
        mode = f"tp_{config.tp_kernel}" if config.tp_constrained else "unconstrained"

    # Initialize parameters
    params: (
        UnconstrainedParams
        | TPConstrainedParams
        | CauchyConstrainedParams
        | SinusoidalConstrainedParams
        | QuadraticDistanceConstrainedParams
        | PermutedExponentialConstrainedParams
        | NegatedBidiagonalConstrainedParams
        | FixedConvergentBiasOnlyParams
        | LoewnerWhitneyConstrainedParams
    )
    if mode == "tp_exponential":
        params = TPConstrainedParams(input_dim, config.hidden_dim, rng)
    elif mode == "tp_cauchy":
        params = CauchyConstrainedParams(input_dim, config.hidden_dim, rng)
    elif mode == "sinusoidal":
        params = SinusoidalConstrainedParams(input_dim, config.hidden_dim, rng)
    elif mode == "quadratic_distance":
        params = QuadraticDistanceConstrainedParams(input_dim, config.hidden_dim, rng)
    elif mode == "permuted_exponential":
        params = PermutedExponentialConstrainedParams(input_dim, config.hidden_dim, rng)
    elif mode == "negated_bidiagonal":
        params = NegatedBidiagonalConstrainedParams(input_dim, config.hidden_dim, rng)
    elif mode == "fixed_convergent_bias_only":
        params = FixedConvergentBiasOnlyParams(input_dim, config.hidden_dim, rng)
    elif mode == "tp_loewner_whitney":
        params = LoewnerWhitneyConstrainedParams(input_dim, config.hidden_dim, rng)
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


# ── Multiclass Training ──


def _softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax. z: (batch, C) -> (batch, C)."""
    z_shifted = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    result: np.ndarray = exp_z / exp_z.sum(axis=1, keepdims=True)
    return result


def _cross_entropy(y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-12) -> float:
    """Mean cross-entropy. y_pred: (batch, C) probs, y_true: (batch,) int labels."""
    batch = y_pred.shape[0]
    log_probs = np.log(np.clip(y_pred, eps, 1.0))
    return -float(np.sum(log_probs[np.arange(batch), y_true.astype(int)]) / batch)


def forward_pass_multiclass(
    x: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Forward pass: input -> hidden (ReLU) -> C-class output (softmax).

    Args:
        x: (batch, d)
        w1: (H, d), b1: (H,), w2: (C, H), b2: (C,)

    Returns (probs, z1_pre, z1_post, z2_pre).
    """
    z1_pre = x @ w1.T + b1  # (batch, H)
    z1_post = np.maximum(z1_pre, 0)  # (batch, H)
    z2_pre = z1_post @ w2.T + b2  # (batch, C)
    probs = _softmax(z2_pre)  # (batch, C)
    return probs, z1_pre, z1_post, z2_pre


def backward_pass_multiclass(
    x: np.ndarray,
    y_true: np.ndarray,
    probs: np.ndarray,
    z1_pre: np.ndarray,
    z1_post: np.ndarray,
    w2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Backward pass for softmax + cross-entropy.

    Matches the binary backward_pass convention: 1/batch applied at each
    leaf gradient, NOT in the upstream dz2.

    Returns (dW1, db1, dW2, db2).
    """
    batch = x.shape[0]
    num_classes = probs.shape[1]

    # d(CE)/d(z2_pre) = probs - one_hot(y_true)  (no 1/batch here)
    one_hot = np.zeros((batch, num_classes))
    one_hot[np.arange(batch), y_true.astype(int)] = 1.0
    dz2 = probs - one_hot  # (batch, C)

    dw2 = dz2.T @ z1_post / batch  # (C, H)
    db2 = np.mean(dz2, axis=0)  # (C,)

    # Hidden gradient
    dz1_post = dz2 @ w2  # (batch, H)
    dz1_pre = dz1_post * (z1_pre > 0).astype(float)

    dw1 = dz1_pre.T @ x / batch  # (H, d)
    db1 = np.mean(dz1_pre, axis=0)  # (H,)

    return dw1, db1, dw2, db2


def train_multiclass(
    x: np.ndarray,
    y: np.ndarray,
    config: TrainConfig,
    num_classes: int = 10,
) -> tuple[ReluNetwork, TrainHistory]:
    """Train a single-hidden-layer ReLU network for multiclass classification.

    Architecture: x -> W1@x+b1 -> ReLU -> W2@h+b2 -> softmax -> probs
    W2 shape: (C, H), b2 shape: (C,)

    Args:
        x: Input data, shape (n_samples, input_dim).
        y: Labels, shape (n_samples,), integer class indices in [0, C).
        config: Training configuration.
        num_classes: Number of output classes.

    Returns:
        (trained_network, history)
    """
    rng = np.random.default_rng(config.seed)
    input_dim = x.shape[1]
    h = config.hidden_dim

    # Initialize parameters (unconstrained only for multiclass baseline)
    scale1 = np.sqrt(2.0 / input_dim)
    w1 = rng.normal(0, scale1, (h, input_dim))
    b1 = np.zeros(h)
    scale2 = np.sqrt(2.0 / h)
    w2 = rng.normal(0, scale2, (num_classes, h))
    b2 = np.zeros(num_classes)

    param_arrays = [w1, b1, w2, b2]

    opt: SGD | Adam
    if config.optimizer == "adam":
        opt = Adam(
            param_arrays,
            config.learning_rate,
            config.beta1,
            config.beta2,
            config.adam_eps,
        )
    else:
        opt = SGD(param_arrays, config.learning_rate)

    history = TrainHistory()
    n_samples = x.shape[0]

    for _epoch in range(config.epochs):
        perm = rng.permutation(n_samples)
        x_shuf, y_shuf = x[perm], y[perm]

        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, config.batch_size):
            x_batch = x_shuf[start : start + config.batch_size]
            y_batch = y_shuf[start : start + config.batch_size]

            probs, z1_pre, z1_post, _ = forward_pass_multiclass(x_batch, w1, b1, w2, b2)

            loss = _cross_entropy(probs, y_batch)
            epoch_loss += loss
            n_batches += 1

            dw1, db1_g, dw2, db2_g = backward_pass_multiclass(
                x_batch, y_batch, probs, z1_pre, z1_post, w2
            )
            opt.step([dw1, db1_g, dw2, db2_g])

        history.losses.append(epoch_loss / n_batches)

        # Full-dataset accuracy
        probs_full, _, _, _ = forward_pass_multiclass(x, w1, b1, w2, b2)
        preds = np.argmax(probs_full, axis=1)
        accuracy = float(np.mean(preds == y.astype(int)))
        history.accuracies.append(accuracy)

    net = ReluNetwork(
        [
            ReluLayer(w1.copy(), b1.copy()),
            ReluLayer(w2.copy(), b2.copy()),
        ]
    )
    return net, history
