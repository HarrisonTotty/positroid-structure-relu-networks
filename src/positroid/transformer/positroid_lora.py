"""Positroid LoRA: low-rank adaptation via boundary measurement.

Proposal D from the Cognihedron at Scale brainstorming document.

Standard LoRA: ΔW = B @ A where B ∈ R^{d_out × r}, A ∈ R^{r × d_in}.
Positroid LoRA: B = boundary_measurement(exp(t), r, d_out)^T.

The boundary measurement constrains B to lie in Gr+(r, d_out), providing:
  1. Structured sparsity from the plabic graph factorization
  2. Better conditioning (no rank-deficient B)
  3. Natural initialization (all weights = 1 → uniform positroid)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from positroid.positroid_cell.boundary_map import (
    boundary_measurement_backward,
    boundary_measurement_matrix,
)


class PositroidLoRA:
    """Low-rank adapter with boundary measurement parameterization.

    ΔW = B(t)^T @ A  where:
      B(t) = boundary_measurement(exp(t), r, d_out) ∈ R^{r × d_out}
      A ∈ R^{r × d_in}  (standard learnable matrix)

    Parameters:
        face_raw: r*(d_out - r) raw face weights
        A: (r, d_in) down-projection
        alpha: scalar scaling factor
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        rank: int,
        alpha: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank
        self.alpha = alpha
        self.n_face = rank * (d_out - rank)

        rng = np.random.default_rng(seed)
        # Initialize face weights near zero → B ≈ [I_r | 0]
        self.face_raw = rng.standard_normal(self.n_face) * 0.01
        # A initialized small so ΔW starts near zero
        self.A = rng.standard_normal((rank, d_in)) * (0.01 / rank**0.5)

    def params(self) -> list[np.ndarray]:
        return [self.face_raw, self.A]

    def param_count(self) -> int:
        return self.n_face + self.rank * self.d_in

    def get_delta_w(self) -> np.ndarray:
        """Compute the low-rank weight update ΔW = alpha * B^T @ A.

        Returns:
            delta_W: (d_out, d_in)
        """
        weights = np.exp(self.face_raw)
        B = boundary_measurement_matrix(weights, self.rank, self.d_out)  # (r, d_out)
        result: np.ndarray = self.alpha * B.T @ self.A  # (d_out, d_in)
        return result

    def forward(
        self, X: np.ndarray, W_base: np.ndarray | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Apply LoRA-adapted linear transformation.

        Args:
            X: (batch, d_in) input.
            W_base: (d_out, d_in) frozen base weight. If None, only ΔW is applied.

        Returns:
            output: (batch, d_out)
            cache: intermediates for backward.
        """
        weights = np.exp(self.face_raw)
        B = boundary_measurement_matrix(weights, self.rank, self.d_out)  # (r, d_out)

        # ΔW = alpha * B^T @ A, so y_delta = alpha * X @ A^T @ B
        XA = X @ self.A.T  # (batch, r)
        delta_out = self.alpha * (XA @ B)  # (batch, d_out)

        if W_base is not None:
            output = X @ W_base.T + delta_out
        else:
            output = delta_out

        cache = {
            "X": X,
            "XA": XA,
            "B": B,
            "weights": weights,
            "W_base": W_base,
        }
        return output, cache

    def backward(
        self, d_output: np.ndarray, cache: dict[str, Any]
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Backward through LoRA.

        Returns:
            d_X: (batch, d_in)
            grads: [d_face_raw, d_A]
        """
        X = cache["X"]
        XA = cache["XA"]
        B = cache["B"]
        weights = cache["weights"]
        W_base = cache["W_base"]

        # delta_out[b, j] = alpha * sum_i XA[b, i] * B[i, j]
        d_XA = self.alpha * (d_output @ B.T)  # (batch, r)
        d_B = self.alpha * (XA.T @ d_output)  # (r, d_out)

        # XA = X @ A^T
        d_A = d_XA.T @ X  # (r, d_in)
        d_X = d_XA @ self.A  # (batch, d_in)

        if W_base is not None:
            d_X += d_output @ W_base

        # Boundary measurement backward
        d_weights = boundary_measurement_backward(weights, self.rank, self.d_out, B, d_B)
        d_face_raw = d_weights * weights

        return d_X, [d_face_raw, d_A]
