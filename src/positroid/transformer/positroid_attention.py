"""Positroid attention: Plücker coordinate-based attention mechanism.

Proposal A from the Cognihedron at Scale brainstorming document.

Extends Grassmann Flows (Zhang, arXiv:2512.19428) with positroid structure
via boundary measurement matrices from Gr+(k,n).

For k=2: score(t,t') = det(B · [z_t | z_{t'}]) — antisymmetric bilinear form.
  This is the Plücker coordinate p_{01} of the 2-plane spanned by B·z_t and B·z_{t'}.
  score(t,t) = 0 always, so a learnable self-bias is added.

For k>=3: tokens are projected to R^k via B·z, then standard dot-product
  attention is used on the k-dimensional representations. The boundary
  measurement constrains the projection to live on Gr+(k,n).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from positroid.positroid_cell.boundary_map import (
    boundary_measurement_backward,
    boundary_measurement_matrix,
)
from positroid.transformer._utils import softmax


class PositroidAttentionHead:
    """Single attention head using Plücker coordinates of Gr+(k,n).

    Parameters:
        face_weight_raw: k*(n-k) reals (exp → positive face weights)
        W_proj: (d_model, n) token projection
        W_V: (d_model, d_v) value projection
        W_O: (d_v, d_model) output projection
        self_bias: scalar (compensates for score(t,t)=0 in k=2 mode)
    """

    def __init__(
        self,
        d_model: int,
        n: int,
        k: int = 2,
        d_v: int | None = None,
        seed: int = 42,
    ) -> None:
        if d_v is None:
            d_v = d_model
        self.d_model = d_model
        self.n = n
        self.k = k
        self.d_v = d_v
        self.n_face = k * (n - k)

        rng = np.random.default_rng(seed)

        self.face_weight_raw = rng.standard_normal(self.n_face) * 0.1
        scale = (2.0 / d_model) ** 0.5
        self.W_proj = rng.standard_normal((d_model, n)) * scale
        self.W_V = rng.standard_normal((d_model, d_v)) * scale
        self.W_O = rng.standard_normal((d_v, d_model)) * (2.0 / d_v) ** 0.5
        self.self_bias = np.zeros(1)

    def params(self) -> list[np.ndarray]:
        return [
            self.face_weight_raw,
            self.W_proj,
            self.W_V,
            self.W_O,
            self.self_bias,
        ]

    def set_params(self, param_list: list[np.ndarray]) -> None:
        self.face_weight_raw = param_list[0]
        self.W_proj = param_list[1]
        self.W_V = param_list[2]
        self.W_O = param_list[3]
        self.self_bias = param_list[4]

    def param_count(self) -> int:
        return (
            self.n_face
            + self.d_model * self.n
            + self.d_model * self.d_v
            + self.d_v * self.d_model
            + 1
        )

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Forward pass.

        Args:
            X: (T, d_model) input sequence.

        Returns:
            output: (T, d_model)
            cache: intermediates for backward pass.
        """
        T = X.shape[0]

        weights = np.exp(self.face_weight_raw)
        B = boundary_measurement_matrix(weights, self.k, self.n)  # (k, n)
        Z = X @ self.W_proj  # (T, n)
        Q = Z @ B.T  # (T, k)

        if self.k == 2:
            # Antisymmetric Plücker scores
            scores = Q[:, 0:1] @ Q[:, 1:2].T - Q[:, 1:2] @ Q[:, 0:1].T
        else:
            # Dot-product on positroid-projected representations
            scores = Q @ Q.T / (self.k**0.5)

        scores = scores + self.self_bias[0] * np.eye(T)
        attn = softmax(scores, axis=-1)

        V = X @ self.W_V  # (T, d_v)
        context = attn @ V  # (T, d_v)
        output = context @ self.W_O  # (T, d_model)

        cache = {
            "X": X,
            "Z": Z,
            "B": B,
            "Q": Q,
            "scores": scores,
            "attn": attn,
            "V": V,
            "context": context,
            "weights": weights,
        }
        return output, cache

    def backward(
        self, d_output: np.ndarray, cache: dict[str, Any]
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Backward pass.

        Returns:
            d_X: (T, d_model) input gradient.
            grads: parameter gradients matching params() order.
        """
        X = cache["X"]
        Z = cache["Z"]
        B = cache["B"]
        Q = cache["Q"]
        attn = cache["attn"]
        V = cache["V"]
        context = cache["context"]
        weights = cache["weights"]
        T = X.shape[0]

        # output = context @ W_O
        d_W_O = context.T @ d_output
        d_context = d_output @ self.W_O.T

        # context = attn @ V
        d_attn = d_context @ V.T
        d_V = attn.T @ d_context

        # V = X @ W_V
        d_W_V = X.T @ d_V
        d_X_v = d_V @ self.W_V.T

        # Softmax backward
        s = (attn * d_attn).sum(axis=-1, keepdims=True)
        d_scores = attn * (d_attn - s)

        # Self-bias gradient
        d_self_bias = np.array([(d_scores * np.eye(T)).sum()])

        # Scores → Q backward
        if self.k == 2:
            d_antisym = d_scores - d_scores.T
            d_Q = np.zeros_like(Q)
            d_Q[:, 0] = d_antisym @ Q[:, 1]
            d_Q[:, 1] = -d_antisym @ Q[:, 0]
        else:
            d_Q = (d_scores + d_scores.T) @ Q / (self.k**0.5)

        # Q = Z @ B.T
        d_Z = d_Q @ B
        d_B = d_Q.T @ Z  # (k, n)

        # Z = X @ W_proj
        d_W_proj = X.T @ d_Z
        d_X_z = d_Z @ self.W_proj.T

        # Boundary measurement backward
        d_weights = boundary_measurement_backward(weights, self.k, self.n, B, d_B)
        d_face_raw = d_weights * weights

        d_X = d_X_v + d_X_z
        grads = [d_face_raw, d_W_proj, d_W_V, d_W_O, d_self_bias]
        return d_X, grads


class PositroidMultiHeadAttention:
    """Multi-head positroid attention with heterogeneous k values.

    Different heads can use different k values:
      k=2: bilinear (antisymmetric Plücker)
      k=3: cubic interaction
      k=4: quartic, etc.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n: int,
        k_values: list[int] | None = None,
        seed: int = 42,
    ) -> None:
        if k_values is None:
            k_values = [2] * n_heads
        assert len(k_values) == n_heads
        self.d_model = d_model
        self.n_heads = n_heads
        d_v = d_model // n_heads

        rng = np.random.default_rng(seed)
        self.heads = [
            PositroidAttentionHead(
                d_model, n, k=k_values[i], d_v=d_v, seed=int(rng.integers(2**31))
            )
            for i in range(n_heads)
        ]

    def params(self) -> list[np.ndarray]:
        result: list[np.ndarray] = []
        for h in self.heads:
            result.extend(h.params())
        return result

    def set_params(self, param_list: list[np.ndarray]) -> None:
        n_per = 5
        for i, h in enumerate(self.heads):
            h.set_params(param_list[i * n_per : (i + 1) * n_per])

    def param_count(self) -> int:
        return sum(h.param_count() for h in self.heads)

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Forward: sum of all head outputs."""
        output = np.zeros_like(X)
        caches = []
        for h in self.heads:
            h_out, h_cache = h.forward(X)
            output += h_out
            caches.append(h_cache)
        return output, caches

    def backward(
        self, d_output: np.ndarray, caches: list[dict[str, Any]]
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Backward through all heads."""
        d_X = np.zeros((caches[0]["X"].shape[0], self.d_model))
        all_grads: list[np.ndarray] = []
        for h, cache in zip(self.heads, caches):
            d_X_h, grads_h = h.backward(d_output, cache)
            d_X += d_X_h
            all_grads.extend(grads_h)
        return d_X, all_grads
