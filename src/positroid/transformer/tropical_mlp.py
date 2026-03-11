"""Tropical MLP: matroid cell evaluations replacing expand-contract MLP.

Proposal C from the Cognihedron at Scale brainstorming document.

Standard MLP: y = W2 · ReLU(W1 · x + b1) + b2   (8d² params for d_ff=4d)
Tropical MLP: y = W_read · [det(B_1·Z(x)), ..., det(B_m·Z(x))] + b

Each cell j computes det(B_j · Z(x)) where:
  B_j ∈ Gr+(k,n) via boundary measurement with k*(n-k) face weights
  Z(x) = reshape(W_enc · x, (n, k)) is a shared encoding

For training, standard det is used (smooth, differentiable).
Tropical det (max over permutations of sums) is available for analysis.
"""

from __future__ import annotations

from typing import Any

from itertools import permutations

import numpy as np

from positroid.positroid_cell.boundary_map import (
    boundary_measurement_backward,
    boundary_measurement_matrix,
)
from positroid.transformer._utils import batch_det, batch_det_grad


class TropicalMLP:
    """MLP replacement using positroid cell evaluations.

    Architecture:
        1. Encode: Z = reshape(x @ W_enc^T, (n, k))    — shared across cells
        2. For cell j: P_j = B_j @ Z, det_j = det(P_j) — per-cell boundary meas.
        3. Output: y = [det_1, ..., det_m] @ W_read + b  — linear readout

    Parameters:
        face_raws: (m, n_face) raw face weights per cell
        W_enc: (n*k, d_in) encoding projection
        W_read: (m, d_out) readout weights
        b_read: (d_out,) readout bias
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_cells: int = 16,
        k: int = 2,
        n: int = 8,
        seed: int = 42,
    ) -> None:
        self.d_in = d_in
        self.d_out = d_out
        self.n_cells = n_cells
        self.k = k
        self.n = n
        self.n_face = k * (n - k)

        rng = np.random.default_rng(seed)

        self.face_raws = rng.standard_normal((n_cells, self.n_face)) * 0.3
        scale = (2.0 / d_in) ** 0.5
        self.W_enc = rng.standard_normal((n * k, d_in)) * scale
        self.W_read = rng.standard_normal((n_cells, d_out)) * 0.1
        self.b_read = np.zeros(d_out)

    def params(self) -> list[np.ndarray]:
        return [self.face_raws, self.W_enc, self.W_read, self.b_read]

    def set_params(self, param_list: list[np.ndarray]) -> None:
        self.face_raws, self.W_enc, self.W_read, self.b_read = param_list

    def param_count(self) -> int:
        return (
            self.n_cells * self.n_face
            + self.n * self.k * self.d_in
            + self.n_cells * self.d_out
            + self.d_out
        )

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Forward pass.

        Args:
            X: (batch, d_in) input features.

        Returns:
            output: (batch, d_out)
            cache: intermediates for backward pass.
        """
        batch = X.shape[0]

        # 1. Shared encoding
        Z_flat = X @ self.W_enc.T  # (batch, n*k)
        Z = Z_flat.reshape(batch, self.n, self.k)  # (batch, n, k)

        # 2. Per-cell boundary measurement + determinant
        dets = np.zeros((batch, self.n_cells))
        prods = np.zeros((self.n_cells, batch, self.k, self.k))
        bnds = np.zeros((self.n_cells, self.k, self.n))
        all_weights = np.zeros((self.n_cells, self.n_face))

        for j in range(self.n_cells):
            w_j = np.exp(self.face_raws[j])
            B_j = boundary_measurement_matrix(w_j, self.k, self.n)
            # P_j[b] = B_j @ Z[b]: (batch, k, k)
            P_j = np.einsum("kn,bnl->bkl", B_j, Z)
            dets[:, j] = batch_det(P_j, self.k)
            prods[j] = P_j
            bnds[j] = B_j
            all_weights[j] = w_j

        # 3. Linear readout
        output = dets @ self.W_read + self.b_read  # (batch, d_out)

        cache = {
            "X": X,
            "Z_flat": Z_flat,
            "Z": Z,
            "dets": dets,
            "prods": prods,
            "bnds": bnds,
            "all_weights": all_weights,
        }
        return output, cache

    def backward(
        self, d_output: np.ndarray, cache: dict[str, Any]
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Backward pass.

        Returns:
            d_X: (batch, d_in) input gradient.
            grads: [d_face_raws, d_W_enc, d_W_read, d_b_read].
        """
        X = cache["X"]
        Z = cache["Z"]
        dets = cache["dets"]
        prods = cache["prods"]
        bnds = cache["bnds"]
        all_weights = cache["all_weights"]
        batch = X.shape[0]

        # Readout backward
        d_W_read = dets.T @ d_output  # (m, d_out)
        d_b_read = d_output.sum(axis=0)  # (d_out,)
        d_dets = d_output @ self.W_read.T  # (batch, m)

        # Per-cell backward
        d_face_raws = np.zeros_like(self.face_raws)
        d_Z = np.zeros_like(Z)  # (batch, n, k)

        for j in range(self.n_cells):
            P_j = prods[j]  # (batch, k, k)
            B_j = bnds[j]  # (k, n)
            w_j = all_weights[j]  # (n_face,)

            # det gradient → product gradient
            d_P_j = batch_det_grad(P_j, d_dets[:, j], self.k)  # (batch, k, k)

            # P_j[b,i,l] = B_j[i,q] * Z[b,q,l]
            # d_B_j[i,q] = sum_{b,l} d_P_j[b,i,l] * Z[b,q,l]
            d_B_j = np.einsum("bil,bql->iq", d_P_j, Z)
            # d_Z[b,q,l] += sum_i B_j[i,q] * d_P_j[b,i,l]
            d_Z += np.einsum("iq,bil->bql", B_j, d_P_j)

            # Boundary measurement backward
            d_w_j = boundary_measurement_backward(w_j, self.k, self.n, B_j, d_B_j)
            d_face_raws[j] = d_w_j * w_j  # chain through exp

        # Encoding backward: Z_flat = X @ W_enc^T
        d_Z_flat = d_Z.reshape(batch, self.n * self.k)
        d_W_enc = d_Z_flat.T @ X  # (n*k, d_in)
        d_X = d_Z_flat @ self.W_enc  # (batch, d_in)

        grads = [d_face_raws, d_W_enc, d_W_read, d_b_read]
        return d_X, grads

    def forward_tropical(self, X: np.ndarray) -> np.ndarray:
        """Forward pass using tropical (max-plus) operations.

        Computes the Maslov dequantization limit of det(B·Z):

            trop_det(log(B) ⊕ Z) = max_σ Σ_i max_q (log B[i,q] + Z[q,σ(i)])

        where ⊕ is tropical (max-plus) matrix multiplication.  B entries are
        strictly positive from boundary measurement, so log(B) is well-defined
        (zero entries map to -∞, the tropical additive identity).  Z entries
        are treated directly as tropical semiring elements.

        The linear readout (sum over cells) remains in the standard semiring
        — only the inner cell evaluations are tropicalized.
        """
        batch = X.shape[0]
        Z_flat = X @ self.W_enc.T
        Z = Z_flat.reshape(batch, self.n, self.k)

        dets = np.zeros((batch, self.n_cells))
        perms = list(permutations(range(self.k)))

        for j in range(self.n_cells):
            w_j = np.exp(self.face_raws[j])
            B_j = boundary_measurement_matrix(w_j, self.k, self.n)
            with np.errstate(divide="ignore"):
                log_B_j = np.log(B_j)  # -inf for zero entries (tropical zero)

            # Tropical matrix product: trop_P[b,i,l] = max_q(log_B[i,q] + Z[b,q,l])
            # log_B_j: (k, n), Z: (batch, n, k)
            trop_P = np.full((batch, self.k, self.k), -np.inf)
            for q in range(self.n):
                # log_B_j[:, q] is (k,), Z[:, q, :] is (batch, k)
                # Contribution: log_B_j[i, q] + Z[b, q, l]
                contrib = log_B_j[:, q][None, :, None] + Z[:, q, :][:, None, :]
                np.maximum(trop_P, contrib, out=trop_P)

            # Tropical det: max_σ Σ_i trop_P[b, i, σ(i)]
            for b in range(batch):
                max_val = -np.inf
                for sigma in perms:
                    val = sum(trop_P[b, i, sigma[i]] for i in range(self.k))
                    if val > max_val:
                        max_val = val
                dets[b, j] = max_val

        return dets @ self.W_read + self.b_read
