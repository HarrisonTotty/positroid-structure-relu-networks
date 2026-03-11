"""Positroid cell network for classification.

A positroid network computes f(x) = det(B(t) . Z(x)) - c, where:
- B(t) is the boundary measurement matrix of the top cell of Gr+(k,n),
  parameterized by k*(n-k) positive face weights t_i = exp(r_i)
- Z(x) is an input encoding: n x k matrix whose rows z_i(x) depend on input x
- c is a learnable output bias
- The prediction is sigmoid(f(x)) for binary, softmax over C classes for multiclass

For k=2 with affine encoding z_i(x) = [1, a_i . x], the network computes
a linear classifier (det is linear in x via Cauchy-Binet).

For k=3 with encoding z_i(x) = [1, a_i^(1).x, a_i^(2).x], the network
computes a quadratic classifier (det is degree 2 in x).

References:
  Postnikov, "Total positivity, Grassmannians, and networks" (2006)
  See docs/brainstorming/positroid-network-architecture.md
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from positroid.positroid_cell.boundary_map import (
    boundary_measurement_backward,
    boundary_measurement_matrix,
)

# ── Numerics ──


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    neg = ~pos
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def _bce(y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-12) -> float:
    """Mean binary cross-entropy loss."""
    p = np.clip(y_pred.ravel(), eps, 1 - eps)
    t = y_true.ravel()
    return -float(np.mean(t * np.log(p) + (1 - t) * np.log(1 - p)))


def _softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax. z: (batch, C) -> (batch, C)."""
    z_shifted = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    result: np.ndarray = exp_z / exp_z.sum(axis=1, keepdims=True)
    return result


def _cross_entropy(y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-12) -> float:
    """Mean cross-entropy loss. y_pred: (batch, C) probs, y_true: (batch,) int labels."""
    batch = y_pred.shape[0]
    log_probs = np.log(np.clip(y_pred, eps, 1.0))
    return -float(np.sum(log_probs[np.arange(batch), y_true.astype(int)]) / batch)


def _det2x2(mat: np.ndarray) -> np.ndarray:
    """Determinant of batch of 2x2 matrices. mat: (batch, 2, 2) -> (batch,)."""
    result: np.ndarray = mat[:, 0, 0] * mat[:, 1, 1] - mat[:, 0, 1] * mat[:, 1, 0]
    return result


def _det3x3(mat: np.ndarray) -> np.ndarray:
    """Determinant of batch of 3x3 matrices. mat: (batch, 3, 3) -> (batch,)."""
    result: np.ndarray = (
        mat[:, 0, 0] * (mat[:, 1, 1] * mat[:, 2, 2] - mat[:, 1, 2] * mat[:, 2, 1])
        - mat[:, 0, 1] * (mat[:, 1, 0] * mat[:, 2, 2] - mat[:, 1, 2] * mat[:, 2, 0])
        + mat[:, 0, 2] * (mat[:, 1, 0] * mat[:, 2, 1] - mat[:, 1, 1] * mat[:, 2, 0])
    )
    return result


def _det_grad_2x2(mat: np.ndarray, d_det: np.ndarray) -> np.ndarray:
    """Gradient of det w.r.t. mat for batch of 2x2 matrices.

    d det / d mat[i,j] = cofactor(i,j).
    Returns d_mat: (batch, 2, 2).
    """
    d_mat = np.empty_like(mat)
    d_mat[:, 0, 0] = d_det * mat[:, 1, 1]
    d_mat[:, 0, 1] = d_det * (-mat[:, 1, 0])
    d_mat[:, 1, 0] = d_det * (-mat[:, 0, 1])
    d_mat[:, 1, 1] = d_det * mat[:, 0, 0]
    return d_mat


def _det_grad_3x3(mat: np.ndarray, d_det: np.ndarray) -> np.ndarray:
    """Gradient of det w.r.t. mat for batch of 3x3 matrices.

    d det / d mat[i,j] = cofactor(i,j).
    Returns d_mat: (batch, 3, 3).
    """
    d_mat = np.empty_like(mat)
    # Row 0 cofactors
    d_mat[:, 0, 0] = d_det * (mat[:, 1, 1] * mat[:, 2, 2] - mat[:, 1, 2] * mat[:, 2, 1])
    d_mat[:, 0, 1] = d_det * (-(mat[:, 1, 0] * mat[:, 2, 2] - mat[:, 1, 2] * mat[:, 2, 0]))
    d_mat[:, 0, 2] = d_det * (mat[:, 1, 0] * mat[:, 2, 1] - mat[:, 1, 1] * mat[:, 2, 0])
    # Row 1 cofactors
    d_mat[:, 1, 0] = d_det * (-(mat[:, 0, 1] * mat[:, 2, 2] - mat[:, 0, 2] * mat[:, 2, 1]))
    d_mat[:, 1, 1] = d_det * (mat[:, 0, 0] * mat[:, 2, 2] - mat[:, 0, 2] * mat[:, 2, 0])
    d_mat[:, 1, 2] = d_det * (-(mat[:, 0, 0] * mat[:, 2, 1] - mat[:, 0, 1] * mat[:, 2, 0]))
    # Row 2 cofactors
    d_mat[:, 2, 0] = d_det * (mat[:, 0, 1] * mat[:, 1, 2] - mat[:, 0, 2] * mat[:, 1, 1])
    d_mat[:, 2, 1] = d_det * (-(mat[:, 0, 0] * mat[:, 1, 2] - mat[:, 0, 2] * mat[:, 1, 0]))
    d_mat[:, 2, 2] = d_det * (mat[:, 0, 0] * mat[:, 1, 1] - mat[:, 0, 1] * mat[:, 1, 0])
    return d_mat


def _cofactor(mat: np.ndarray) -> np.ndarray:
    """Cofactor matrix of a single k*k matrix (not batched)."""
    k = mat.shape[0]
    if k == 1:
        return np.ones((1, 1))
    if k == 2:
        cof = np.empty((2, 2))
        cof[0, 0] = mat[1, 1]
        cof[0, 1] = -mat[1, 0]
        cof[1, 0] = -mat[0, 1]
        cof[1, 1] = mat[0, 0]
        return cof
    if k == 3:
        cof = np.empty((3, 3))
        cof[0, 0] = mat[1, 1] * mat[2, 2] - mat[1, 2] * mat[2, 1]
        cof[0, 1] = -(mat[1, 0] * mat[2, 2] - mat[1, 2] * mat[2, 0])
        cof[0, 2] = mat[1, 0] * mat[2, 1] - mat[1, 1] * mat[2, 0]
        cof[1, 0] = -(mat[0, 1] * mat[2, 2] - mat[0, 2] * mat[2, 1])
        cof[1, 1] = mat[0, 0] * mat[2, 2] - mat[0, 2] * mat[2, 0]
        cof[1, 2] = -(mat[0, 0] * mat[2, 1] - mat[0, 1] * mat[2, 0])
        cof[2, 0] = mat[0, 1] * mat[1, 2] - mat[0, 2] * mat[1, 1]
        cof[2, 1] = -(mat[0, 0] * mat[1, 2] - mat[0, 2] * mat[1, 0])
        cof[2, 2] = mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]
        return cof
    # General: det * inv^T (valid when det != 0, which holds for Gr+ minors)
    d = np.linalg.det(mat)
    if abs(d) > 1e-15:
        result: np.ndarray = d * np.linalg.inv(mat).T
        return result
    # Fallback for near-singular matrices
    cof = np.empty((k, k))
    for i in range(k):
        for j in range(k):
            minor_mat = np.delete(np.delete(mat, i, axis=0), j, axis=1)
            cof[i, j] = ((-1) ** (i + j)) * np.linalg.det(minor_mat)
    return cof


def _batch_det(prod: np.ndarray, k: int) -> np.ndarray:
    """Batch determinant for (batch, k, k) matrices."""
    if k == 2:
        return _det2x2(prod)
    elif k == 3:
        return _det3x3(prod)
    else:
        return np.linalg.det(prod)  # type: ignore[no-any-return]


def _batch_det_grad(prod: np.ndarray, d_dets: np.ndarray, k: int) -> np.ndarray:
    """Batch gradient of det w.r.t. prod."""
    if k == 2:
        return _det_grad_2x2(prod, d_dets)
    elif k == 3:
        return _det_grad_3x3(prod, d_dets)
    else:
        det_vals = np.linalg.det(prod)
        cof_batch = det_vals[:, None, None] * np.linalg.inv(prod).transpose(0, 2, 1)
        return d_dets[:, None, None] * cof_batch  # type: ignore[no-any-return]


# ── Positroid Network ──


class PositroidNetwork:
    """A positroid cell network for classification.

    Architecture:
      Input x in R^d
      -> Encoding enc(x): n x k matrix
      -> Boundary measurement bnd(t): k x n matrix (TP, from face weights)
      -> prod = bnd(t) . enc(x): k x k matrix
      -> f(x) = det(prod) - bias
      -> y = sigmoid(f(x))  [binary]  or  softmax over C classes [multiclass]

    Parameters:
      - face_weights_raw: k*(n-k) reals [binary] or (C, k*(n-k)) [multiclass]
      - output_bias: (1,) [binary] or (C,) [multiclass]
      - encoding_vectors: n * (k-1) * d parameters (if learnable, shared across classes)
    """

    def __init__(
        self,
        k: int,
        n: int,
        input_dim: int,
        encoding: str = "fixed",
        readout: str = "det",
        num_classes: int = 2,
        rng: np.random.Generator | None = None,
    ) -> None:
        if rng is None:
            rng = np.random.default_rng()

        self.k = k
        self.n = n
        self.input_dim = input_dim
        self.encoding = encoding  # 'fixed' or 'learnable'
        self.readout = readout  # 'det', 'plucker_ratio', 'canonical_residue'
        self.num_classes = num_classes

        # Caches for readout normalization (set during forward, binary only)
        self._cache_denom: float = 1.0
        self._cache_denom_grad: np.ndarray = np.zeros((k, n))
        self._cache_dets: np.ndarray = np.zeros(0)

        num_face_weights = k * (n - k)

        if num_classes > 2:
            # Multiclass: separate face weights per class
            self.face_weights_raw = rng.normal(0, 0.5, size=(num_classes, num_face_weights))
            self.output_bias = np.zeros(num_classes)
        else:
            # Binary: single set of face weights
            self.face_weights_raw = rng.normal(0, 0.5, size=num_face_weights)
            self.output_bias = np.zeros(1)

        # Encoding vectors: shared across classes
        self.encoding_vectors = self._init_encoding(rng)

        # Precomputed index arrays for vectorized readout
        from itertools import combinations

        self._plucker_cols = np.array(list(combinations(range(n), k)), dtype=np.intp)  # (C(n,k), k)
        self._consecutive_cols = np.array(
            [sorted([(i + j) % n for j in range(k)]) for i in range(n)],
            dtype=np.intp,
        )  # (n, k)

    def _init_encoding(self, rng: np.random.Generator) -> np.ndarray:
        """Initialize encoding vectors.

        For k=2, d=2: n direction vectors on unit circle (evenly spaced).
        For k=3, d=2: n rotation matrices (evenly spaced angles).
        General: random orthogonal directions.
        """
        n, d, km1 = self.n, self.input_dim, self.k - 1

        if d == 2 and km1 == 1:
            # k=2, d=2: evenly spaced directions on unit circle
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            vecs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
            return vecs.reshape(n, 1, d)

        if d == 2 and km1 == 2:
            # k=3, d=2: rotation matrices at evenly spaced angles
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            enc = np.zeros((n, 2, 2))
            enc[:, 0, 0] = np.cos(angles)
            enc[:, 0, 1] = np.sin(angles)
            enc[:, 1, 0] = -np.sin(angles)
            enc[:, 1, 1] = np.cos(angles)
            return enc

        # General case: random directions, normalized
        enc = rng.normal(0, 1, size=(n, km1, d))
        # Normalize each direction vector
        for i in range(n):
            for j in range(km1):
                norm = np.linalg.norm(enc[i, j])
                if norm > 1e-10:
                    enc[i, j] /= norm
        return enc

    def _build_encoding_matrix(self, x: np.ndarray) -> np.ndarray:
        """Build encoding matrix enc(x) for a batch of inputs.

        Args:
            x: (batch, d) input features.

        Returns:
            enc_mat: (batch, n, k) encoding matrix.
                     enc_mat[b, i, 0] = 1
                     enc_mat[b, i, 1:] = encoding_vectors[i] @ x[b]
        """
        batch = x.shape[0]

        # Projections: (batch, n, k-1)
        proj = np.einsum("njd,bd->bnj", self.encoding_vectors, x)

        # enc_mat: (batch, n, k) with first column = 1
        enc_mat = np.ones((batch, self.n, self.k))
        enc_mat[:, :, 1:] = proj

        return enc_mat

    def _plucker_sum_and_grad(self, bnd: np.ndarray) -> tuple[float, np.ndarray]:
        """Compute S = sum of all Plucker coordinates and dS/dB.

        S = sum_I Delta_I(B) over all C(n,k) k-minors.

        Returns:
            S: scalar sum of all maximal minors.
            dS_dbnd: (k, n) gradient of S w.r.t. bnd.
        """
        k, n = bnd.shape
        cols = self._plucker_cols  # (M, k)
        submats = bnd[:, cols].transpose(1, 0, 2)  # (M, k, k)
        m = submats.shape[0]

        det_vals = _batch_det(submats, k)
        s_val = float(det_vals.sum())

        ones = np.ones(m)
        cofs = _batch_det_grad(submats, ones, k)

        ds = np.zeros((k, n))
        for j in range(k):
            np.add.at(ds, (slice(None), cols[:, j]), cofs[:, :, j].T)  # type: ignore[arg-type]
        return s_val, ds

    def _consecutive_product_and_grad(self, bnd: np.ndarray) -> tuple[float, np.ndarray]:
        """Compute P = product of n consecutive k-minors and dP/dB.

        cyc_i = sorted({i, i+1, ..., i+k-1} mod n) for i = 0..n-1.
        P = prod_i Delta_{cyc_i}(B).

        Returns:
            P: scalar product of consecutive minors.
            dP_dbnd: (k, n) gradient of P w.r.t. bnd.
        """
        k, n = bnd.shape
        cols = self._consecutive_cols  # (n, k)
        submats = bnd[:, cols].transpose(1, 0, 2)  # (n, k, k)

        minor_vals = _batch_det(submats, k)
        p_val = float(np.prod(minor_vals))

        ones = np.ones(n)
        cofs = _batch_det_grad(submats, ones, k)

        # Scales: dP/dDelta_i = P / Delta_i
        safe = np.abs(minor_vals) > 1e-30
        scales = np.where(safe, p_val / minor_vals, 0.0)
        if not np.all(safe):
            for i in np.where(~safe)[0]:
                scales[i] = float(np.prod(minor_vals[np.arange(n) != i]))

        scaled_cofs = scales[:, None, None] * cofs  # (n, k, k)
        dp = np.zeros((k, n))
        for j in range(k):
            np.add.at(dp, (slice(None), cols[:, j]), scaled_cofs[:, :, j].T)  # type: ignore[arg-type]
        return p_val, dp

    def forward(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass for a batch of inputs.

        Args:
            x: (batch, d) input features.

        Returns (binary, num_classes <= 2):
            logits: (batch,) raw output (before sigmoid)
            prod: (batch, k, k) product matrices
            enc_mat: (batch, n, k) encoding matrices
            bnd: (k, n) boundary measurement matrix
            weights: (m,) positive face weights

        Returns (multiclass, num_classes > 2):
            logits: (batch, C) raw outputs (before softmax)
            prod: (C, batch, k, k) product matrices per class
            enc_mat: (batch, n, k) encoding matrices (shared)
            bnd: (C, k, n) boundary measurement matrices per class
            weights: (C, m) positive face weights per class
        """
        enc_mat = self._build_encoding_matrix(x)  # (batch, n, k)

        if self.num_classes <= 2:
            return self._forward_binary(enc_mat)
        else:
            return self._forward_multiclass(enc_mat)

    def _forward_binary(
        self, enc_mat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Binary forward pass (original behavior)."""
        weights = np.exp(self.face_weights_raw)
        bnd = boundary_measurement_matrix(weights, self.k, self.n)
        prod = np.einsum("kn,bnj->bkj", bnd, enc_mat)
        dets = _batch_det(prod, self.k)

        if self.readout == "plucker_ratio":
            s_val, ds_dbnd = self._plucker_sum_and_grad(bnd)
            self._cache_denom = s_val
            self._cache_denom_grad = ds_dbnd
            self._cache_dets = dets.copy()
            logits = dets / s_val - self.output_bias[0]
        elif self.readout == "canonical_residue":
            p_val, dp_dbnd = self._consecutive_product_and_grad(bnd)
            self._cache_denom = p_val
            self._cache_denom_grad = dp_dbnd
            self._cache_dets = dets.copy()
            logits = dets / p_val - self.output_bias[0]
        else:
            logits = dets - self.output_bias[0]

        return logits, prod, enc_mat, bnd, weights

    def _forward_multiclass(
        self, enc_mat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Multiclass forward pass: one boundary matrix per class."""
        batch = enc_mat.shape[0]
        c = self.num_classes
        m = self.k * (self.n - self.k)

        logits = np.empty((batch, c))
        prods = np.empty((c, batch, self.k, self.k))
        bnds = np.empty((c, self.k, self.n))
        all_weights = np.empty((c, m))

        for ci in range(c):
            weights_c = np.exp(self.face_weights_raw[ci])
            bnd_c = boundary_measurement_matrix(weights_c, self.k, self.n)
            prod_c = np.einsum("kn,bnj->bkj", bnd_c, enc_mat)
            dets_c = _batch_det(prod_c, self.k)

            logits[:, ci] = dets_c - self.output_bias[ci]
            prods[ci] = prod_c
            bnds[ci] = bnd_c
            all_weights[ci] = weights_c

        return logits, prods, enc_mat, bnds, all_weights

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict probabilities for a batch of inputs.

        Returns:
            Binary: (batch,) sigmoid probabilities.
            Multiclass: (batch, C) softmax probabilities.
        """
        logits, *_ = self.forward(x)
        if self.num_classes <= 2:
            return _sigmoid(logits)
        else:
            return _softmax(logits)

    def param_list(self) -> list[np.ndarray]:
        """List of trainable parameter arrays."""
        params = [self.face_weights_raw, self.output_bias]
        if self.encoding == "learnable":
            params.append(self.encoding_vectors)
        return params

    @property
    def num_params(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.size for p in self.param_list())

    def compute_grads(
        self,
        x: np.ndarray,
        y_true: np.ndarray,
        logits: np.ndarray,
        prod: np.ndarray,
        enc_mat: np.ndarray,
        bnd: np.ndarray,
        weights: np.ndarray,
    ) -> list[np.ndarray]:
        """Compute gradients for all trainable parameters.

        Binary chain: loss -> sigmoid -> (det - bias) -> det(bnd.enc) -> bnd(weights) -> raw
        Multiclass chain: loss -> softmax+CE -> (det - bias) -> ... same

        Args:
            x: (batch, d) inputs
            y_true: (batch,) labels (binary {0,1} or integer class indices)
            logits: forward pass logits
            prod: product matrices
            enc_mat: (batch, n, k) encoding matrices
            bnd: boundary measurement matrix/matrices
            weights: positive face weights

        Returns:
            List of gradients matching param_list() structure.
        """
        if self.num_classes <= 2:
            return self._compute_grads_binary(x, y_true, logits, prod, enc_mat, bnd, weights)
        else:
            return self._compute_grads_multiclass(x, y_true, logits, prod, enc_mat, bnd, weights)

    def _compute_grads_binary(
        self,
        x: np.ndarray,
        y_true: np.ndarray,
        logits: np.ndarray,
        prod: np.ndarray,
        enc_mat: np.ndarray,
        bnd: np.ndarray,
        weights: np.ndarray,
    ) -> list[np.ndarray]:
        """Binary classification gradients (original implementation)."""
        batch = x.shape[0]
        y_pred = _sigmoid(logits)

        # dL/d_logits: BCE + sigmoid combined gradient
        d_logits = (y_pred - y_true) / batch  # (batch,)

        # dL/d_bias = -sum(d_logits)
        d_bias = np.array([-np.sum(d_logits)])

        # dL/d_det: depends on readout mode
        d_bnd_extra = np.zeros_like(bnd)
        if self.readout in ("plucker_ratio", "canonical_residue"):
            denom = self._cache_denom
            cached_dets = self._cache_dets
            d_dets = d_logits / denom
            scalar = -np.sum(d_logits * cached_dets) / (denom**2)
            d_bnd_extra = scalar * self._cache_denom_grad
        else:
            d_dets = d_logits

        # dL/d_prod through determinant
        d_prod = _batch_det_grad(prod, d_dets, self.k)

        # prod = bnd @ enc_mat
        d_bnd = np.einsum("bkj,bnj->kn", d_prod, enc_mat)
        d_bnd += d_bnd_extra

        # Chain through boundary_measurement_matrix
        d_weights = boundary_measurement_backward(weights, self.k, self.n, bnd, d_bnd)
        d_raw = d_weights * weights

        grads: list[np.ndarray] = [d_raw, d_bias]

        if self.encoding == "learnable":
            d_enc_mat = np.einsum("kn,bkj->bnj", bnd, d_prod)
            d_enc_vectors = np.einsum("bnj,bd->njd", d_enc_mat[:, :, 1:], x)
            grads.append(d_enc_vectors)

        return grads

    def _compute_grads_multiclass(
        self,
        x: np.ndarray,
        y_true: np.ndarray,
        logits: np.ndarray,
        prod: np.ndarray,
        enc_mat: np.ndarray,
        bnd: np.ndarray,
        weights: np.ndarray,
    ) -> list[np.ndarray]:
        """Multiclass softmax + cross-entropy gradients.

        Args:
            logits: (batch, C)
            prod: (C, batch, k, k)
            enc_mat: (batch, n, k)
            bnd: (C, k, n)
            weights: (C, m)
        """
        batch = x.shape[0]
        c = self.num_classes

        # Softmax + CE combined gradient: probs - one_hot(y)
        probs = _softmax(logits)  # (batch, C)
        one_hot = np.zeros((batch, c))
        one_hot[np.arange(batch), y_true.astype(int)] = 1.0
        d_logits = (probs - one_hot) / batch  # (batch, C)

        # Bias gradient: logit = det - bias, so d/d_bias = -d_logits
        d_bias = -np.sum(d_logits, axis=0)  # (C,)

        # Gradient for each class's face weights
        d_raw_all = np.zeros_like(self.face_weights_raw)  # (C, m)
        d_enc_total = np.zeros_like(enc_mat)  # (batch, n, k)

        for ci in range(c):
            d_dets_c = d_logits[:, ci]  # (batch,)
            prod_c = prod[ci]  # (batch, k, k)
            bnd_c = bnd[ci]  # (k, n)
            weights_c = weights[ci]  # (m,)

            d_prod_c = _batch_det_grad(prod_c, d_dets_c, self.k)
            d_bnd_c = np.einsum("bkj,bnj->kn", d_prod_c, enc_mat)
            d_weights_c = boundary_measurement_backward(weights_c, self.k, self.n, bnd_c, d_bnd_c)
            d_raw_all[ci] = d_weights_c * weights_c

            if self.encoding == "learnable":
                d_enc_mat_c = np.einsum("kn,bkj->bnj", bnd_c, d_prod_c)
                d_enc_total += d_enc_mat_c

        grads: list[np.ndarray] = [d_raw_all, d_bias]

        if self.encoding == "learnable":
            d_enc_vectors = np.einsum("bnj,bd->njd", d_enc_total[:, :, 1:], x)
            grads.append(d_enc_vectors)

        return grads


# ── Adam Optimizer ──


class _Adam:
    """Adam optimizer (local copy to avoid circular imports)."""

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


# ── Training ──


@dataclass
class PositroidTrainConfig:
    """Configuration for training a positroid network."""

    k: int = 2
    n: int = 6
    encoding: str = "fixed"  # 'fixed' or 'learnable'
    readout: str = "det"  # 'det', 'plucker_ratio', 'canonical_residue'
    num_classes: int = 2
    learning_rate: float = 0.01
    epochs: int = 200
    batch_size: int = 32
    seed: int = 42


@dataclass
class PositroidTrainHistory:
    """Training history for a positroid network."""

    losses: list[float] = field(default_factory=list)
    accuracies: list[float] = field(default_factory=list)


def train_positroid(
    x: np.ndarray,
    y: np.ndarray,
    config: PositroidTrainConfig,
) -> tuple[PositroidNetwork, PositroidTrainHistory]:
    """Train a positroid network.

    Supports binary (num_classes=2, sigmoid+BCE) and multiclass
    (num_classes>2, softmax+CE) classification.

    Args:
        x: (n_samples, input_dim) training features.
        y: (n_samples,) labels. Binary {0,1} or integer class indices.
        config: Training configuration.

    Returns:
        Trained PositroidNetwork and training history.
    """
    rng = np.random.default_rng(config.seed)
    input_dim = x.shape[1]
    multiclass = config.num_classes > 2

    net = PositroidNetwork(
        k=config.k,
        n=config.n,
        input_dim=input_dim,
        encoding=config.encoding,
        readout=config.readout,
        num_classes=config.num_classes,
        rng=rng,
    )

    opt = _Adam(net.param_list(), config.learning_rate)
    history = PositroidTrainHistory()
    n_samples = x.shape[0]

    for _epoch in range(config.epochs):
        perm = rng.permutation(n_samples)
        x_shuf, y_shuf = x[perm], y[perm]

        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, config.batch_size):
            x_batch = x_shuf[start : start + config.batch_size]
            y_batch = y_shuf[start : start + config.batch_size]

            logits, prod, enc_mat, bnd, weights = net.forward(x_batch)

            if multiclass:
                probs = _softmax(logits)
                loss = _cross_entropy(probs, y_batch)
            else:
                y_pred = _sigmoid(logits)
                loss = _bce(y_pred, y_batch)

            epoch_loss += loss
            n_batches += 1

            grads = net.compute_grads(x_batch, y_batch, logits, prod, enc_mat, bnd, weights)
            opt.step(grads)

        history.losses.append(epoch_loss / n_batches)

        # Full-dataset accuracy
        logits_full, *_ = net.forward(x)
        if multiclass:
            preds = np.argmax(logits_full, axis=1)
            accuracy = float(np.mean(preds == y.astype(int)))
        else:
            preds = (logits_full > 0.0).astype(float)
            accuracy = float(np.mean(preds == y))
        history.accuracies.append(accuracy)

    return net, history
