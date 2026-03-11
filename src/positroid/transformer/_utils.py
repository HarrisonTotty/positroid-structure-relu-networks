"""Shared numerical utilities for transformer components."""

from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = x.max(axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / e.sum(axis=axis, keepdims=True)  # type: ignore[no-any-return]


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    neg = ~pos
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def cross_entropy(y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-12) -> float:
    """Mean cross-entropy. y_pred: (batch, C) probs, y_true: (batch,) int labels."""
    batch = y_pred.shape[0]
    log_probs = np.log(np.clip(y_pred, eps, 1.0))
    return -float(np.sum(log_probs[np.arange(batch), y_true.astype(int)]) / batch)


def batch_det(mat: np.ndarray, k: int) -> np.ndarray:
    """Batch determinant for (batch, k, k) matrices."""
    if k == 1:
        return mat[:, 0, 0].copy()
    if k == 2:
        return mat[:, 0, 0] * mat[:, 1, 1] - mat[:, 0, 1] * mat[:, 1, 0]  # type: ignore[no-any-return]
    if k == 3:
        return (  # type: ignore[no-any-return]
            mat[:, 0, 0] * (mat[:, 1, 1] * mat[:, 2, 2] - mat[:, 1, 2] * mat[:, 2, 1])
            - mat[:, 0, 1] * (mat[:, 1, 0] * mat[:, 2, 2] - mat[:, 1, 2] * mat[:, 2, 0])
            + mat[:, 0, 2] * (mat[:, 1, 0] * mat[:, 2, 1] - mat[:, 1, 1] * mat[:, 2, 0])
        )
    return np.linalg.det(mat)  # type: ignore[no-any-return]


def batch_det_grad(mat: np.ndarray, d_dets: np.ndarray, k: int) -> np.ndarray:
    """Gradient of det w.r.t. mat: d_mat[b] = d_dets[b] * cofactor(mat[b])."""
    if k == 1:
        return d_dets[:, None, None] * np.ones_like(mat)  # type: ignore[no-any-return]
    if k == 2:
        d_mat = np.empty_like(mat)
        d_mat[:, 0, 0] = d_dets * mat[:, 1, 1]
        d_mat[:, 0, 1] = d_dets * (-mat[:, 1, 0])
        d_mat[:, 1, 0] = d_dets * (-mat[:, 0, 1])
        d_mat[:, 1, 1] = d_dets * mat[:, 0, 0]
        return d_mat
    if k == 3:
        d_mat = np.empty_like(mat)
        d_mat[:, 0, 0] = d_dets * (mat[:, 1, 1] * mat[:, 2, 2] - mat[:, 1, 2] * mat[:, 2, 1])
        d_mat[:, 0, 1] = d_dets * (-(mat[:, 1, 0] * mat[:, 2, 2] - mat[:, 1, 2] * mat[:, 2, 0]))
        d_mat[:, 0, 2] = d_dets * (mat[:, 1, 0] * mat[:, 2, 1] - mat[:, 1, 1] * mat[:, 2, 0])
        d_mat[:, 1, 0] = d_dets * (-(mat[:, 0, 1] * mat[:, 2, 2] - mat[:, 0, 2] * mat[:, 2, 1]))
        d_mat[:, 1, 1] = d_dets * (mat[:, 0, 0] * mat[:, 2, 2] - mat[:, 0, 2] * mat[:, 2, 0])
        d_mat[:, 1, 2] = d_dets * (-(mat[:, 0, 0] * mat[:, 2, 1] - mat[:, 0, 1] * mat[:, 2, 0]))
        d_mat[:, 2, 0] = d_dets * (mat[:, 0, 1] * mat[:, 1, 2] - mat[:, 0, 2] * mat[:, 1, 1])
        d_mat[:, 2, 1] = d_dets * (-(mat[:, 0, 0] * mat[:, 1, 2] - mat[:, 0, 2] * mat[:, 1, 0]))
        d_mat[:, 2, 2] = d_dets * (mat[:, 0, 0] * mat[:, 1, 1] - mat[:, 0, 1] * mat[:, 1, 0])
        return d_mat
    # General k: compute cofactor matrix directly (works for singular matrices)
    cof = np.empty_like(mat)
    rows_all = list(range(k))
    cols_all = list(range(k))
    for i in range(k):
        minor_rows = rows_all[:i] + rows_all[i + 1 :]
        for j in range(k):
            minor_cols = cols_all[:j] + cols_all[j + 1 :]
            sub = mat[:, minor_rows, :][:, :, minor_cols]  # (batch, k-1, k-1)
            sign = (-1) ** (i + j)
            cof[:, i, j] = sign * np.linalg.det(sub)
    return d_dets[:, None, None] * cof  # type: ignore[no-any-return]


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
