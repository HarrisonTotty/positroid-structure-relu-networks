"""Determinant MLP with configurable matrix parameterization.

Ablation variants of TropicalMLP to isolate what makes it work:
- "positroid": B from boundary measurement (identical to TropicalMLP)
- "unconstrained": B is a directly learnable k×n matrix (no positivity/TP constraint)
- "fixed_random": B is a fixed random k×n matrix (not learned)

All modes share the same architecture:
    1. Encode: Z = reshape(x @ W_enc^T, (n, k))
    2. For cell j: det_j = det(B_j @ Z)
    3. Output: y = [det_1, ..., det_m] @ W_read + b
"""

from __future__ import annotations

from typing import Any

import numpy as np

from positroid.positroid_cell.boundary_map import (
    boundary_measurement_backward,
    boundary_measurement_matrix,
)
from positroid.transformer._utils import batch_det, batch_det_grad


class DetMLP:
    """MLP using determinant nonlinearity with configurable matrix source.

    Parameters vary by mode:
        positroid:     [face_raws, W_enc, W_read, b_read]
        unconstrained: [B_raw, W_enc, W_read, b_read]
        fixed_random:  [W_enc, W_read, b_read]   (B is not learned)
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_cells: int = 4,
        k: int = 2,
        n: int = 8,
        matrix_mode: str = "positroid",
        seed: int = 42,
    ) -> None:
        if matrix_mode not in ("positroid", "unconstrained", "fixed_random"):
            raise ValueError(f"Unknown matrix_mode: {matrix_mode}")

        self.d_in = d_in
        self.d_out = d_out
        self.n_cells = n_cells
        self.k = k
        self.n = n
        self.matrix_mode = matrix_mode
        self.n_face = k * (n - k)

        rng = np.random.default_rng(seed)

        # Matrix parameters depend on mode
        if matrix_mode == "positroid":
            self.face_raws = rng.standard_normal((n_cells, self.n_face)) * 0.3
        elif matrix_mode == "unconstrained":
            scale = (2.0 / n) ** 0.5
            self.B_raw = rng.standard_normal((n_cells, k, n)) * scale
        else:  # fixed_random
            self.B_fixed = rng.standard_normal((n_cells, k, n)) * (2.0 / n) ** 0.5

        # Shared encoding and readout (same across all modes)
        enc_scale = (2.0 / d_in) ** 0.5
        self.W_enc = rng.standard_normal((n * k, d_in)) * enc_scale
        self.W_read = rng.standard_normal((n_cells, d_out)) * 0.1
        self.b_read = np.zeros(d_out)

    def params(self) -> list[np.ndarray]:
        if self.matrix_mode == "positroid":
            return [self.face_raws, self.W_enc, self.W_read, self.b_read]
        elif self.matrix_mode == "unconstrained":
            return [self.B_raw, self.W_enc, self.W_read, self.b_read]
        else:  # fixed_random
            return [self.W_enc, self.W_read, self.b_read]

    def set_params(self, param_list: list[np.ndarray]) -> None:
        if self.matrix_mode == "positroid":
            self.face_raws, self.W_enc, self.W_read, self.b_read = param_list
        elif self.matrix_mode == "unconstrained":
            self.B_raw, self.W_enc, self.W_read, self.b_read = param_list
        else:
            self.W_enc, self.W_read, self.b_read = param_list

    def param_count(self) -> int:
        return sum(p.size for p in self.params())

    def _get_matrices(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Get k×n matrices for all cells. Returns (B_matrices, weights_or_None)."""
        if self.matrix_mode == "positroid":
            matrices = np.zeros((self.n_cells, self.k, self.n))
            weights = np.zeros((self.n_cells, self.n_face))
            for j in range(self.n_cells):
                w_j = np.exp(self.face_raws[j])
                matrices[j] = boundary_measurement_matrix(w_j, self.k, self.n)
                weights[j] = w_j
            return matrices, weights
        elif self.matrix_mode == "unconstrained":
            return self.B_raw.copy(), None
        else:
            return self.B_fixed.copy(), None

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Forward pass. X: (batch, d_in) -> (batch, d_out), cache."""
        batch = X.shape[0]

        Z_flat = X @ self.W_enc.T  # (batch, n*k)
        Z = Z_flat.reshape(batch, self.n, self.k)  # (batch, n, k)

        dets = np.zeros((batch, self.n_cells))
        prods = np.zeros((self.n_cells, batch, self.k, self.k))
        B_matrices, weights = self._get_matrices()

        for j in range(self.n_cells):
            B_j = B_matrices[j]
            P_j = np.einsum("kn,bnl->bkl", B_j, Z)
            dets[:, j] = batch_det(P_j, self.k)
            prods[j] = P_j

        output = dets @ self.W_read + self.b_read

        cache = {
            "X": X,
            "Z_flat": Z_flat,
            "Z": Z,
            "dets": dets,
            "prods": prods,
            "B_matrices": B_matrices,
            "weights": weights,
        }
        return output, cache

    def backward(
        self, d_output: np.ndarray, cache: dict[str, Any]
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Backward pass. Returns (d_X, grads) matching params() order."""
        X = cache["X"]
        Z = cache["Z"]
        dets = cache["dets"]
        prods = cache["prods"]
        B_matrices = cache["B_matrices"]
        weights = cache["weights"]
        batch = X.shape[0]

        # Readout backward
        d_W_read = dets.T @ d_output  # (m, d_out)
        d_b_read = d_output.sum(axis=0)  # (d_out,)
        d_dets = d_output @ self.W_read.T  # (batch, m)

        # Per-cell backward
        d_Z = np.zeros_like(Z)
        matrix_grads: list[np.ndarray] = []

        if self.matrix_mode == "positroid":
            d_face_raws = np.zeros_like(self.face_raws)
            for j in range(self.n_cells):
                P_j = prods[j]
                B_j = B_matrices[j]
                w_j = weights[j]
                d_P_j = batch_det_grad(P_j, d_dets[:, j], self.k)
                d_B_j = np.einsum("bil,bql->iq", d_P_j, Z)
                d_Z += np.einsum("iq,bil->bql", B_j, d_P_j)
                d_w_j = boundary_measurement_backward(w_j, self.k, self.n, B_j, d_B_j)
                d_face_raws[j] = d_w_j * w_j  # chain through exp
            matrix_grads = [d_face_raws]

        elif self.matrix_mode == "unconstrained":
            d_B_raw = np.zeros_like(self.B_raw)
            for j in range(self.n_cells):
                P_j = prods[j]
                B_j = B_matrices[j]
                d_P_j = batch_det_grad(P_j, d_dets[:, j], self.k)
                d_B_j = np.einsum("bil,bql->iq", d_P_j, Z)
                d_Z += np.einsum("iq,bil->bql", B_j, d_P_j)
                d_B_raw[j] = d_B_j
            matrix_grads = [d_B_raw]

        else:  # fixed_random — no gradient for B
            for j in range(self.n_cells):
                P_j = prods[j]
                B_j = B_matrices[j]
                d_P_j = batch_det_grad(P_j, d_dets[:, j], self.k)
                d_Z += np.einsum("iq,bil->bql", B_j, d_P_j)
            matrix_grads = []

        # Encoding backward
        d_Z_flat = d_Z.reshape(batch, self.n * self.k)
        d_W_enc = d_Z_flat.T @ X
        d_X = d_Z_flat @ self.W_enc

        grads = matrix_grads + [d_W_enc, d_W_read, d_b_read]
        return d_X, grads
