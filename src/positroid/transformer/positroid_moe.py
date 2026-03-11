"""Positroid Mixture-of-Experts: Plücker coordinate routing.

Proposal B from the Cognihedron at Scale brainstorming document.

Routes tokens to experts via Plücker coordinate features:
  1. Project token x → k×n matrix Z
  2. Compute C(n,k) Plücker coordinates (k-minors of Z)
  3. Linear projection of Plücker coords → expert scores
  4. Soft routing via softmax

The Plücker coordinates are polynomial (degree k) in the input,
giving nonlinear routing without a learned gating network.
For k=2, routing is quadratic in the token representation.
"""

from __future__ import annotations

from typing import Any

from itertools import combinations

import numpy as np

from positroid.transformer._utils import batch_det, batch_det_grad, softmax


class PositroidRouter:
    """Routes tokens to experts via Plücker coordinate sign patterns.

    Parameters:
        W_route: (k*n, d_model) projection to k×n matrix
        W_gate: (n_experts, n_plucker) Plücker-to-expert projection
        b_gate: (n_experts,) gate bias
    """

    def __init__(
        self,
        d_model: int,
        n: int,
        k: int,
        n_experts: int,
        seed: int = 42,
    ) -> None:
        self.d_model = d_model
        self.n = n
        self.k = k
        self.n_experts = n_experts
        self._col_sets = list(combinations(range(n), k))
        self.n_plucker = len(self._col_sets)

        rng = np.random.default_rng(seed)
        scale = (2.0 / d_model) ** 0.5
        self.W_route = rng.standard_normal((k * n, d_model)) * scale
        self.W_gate = rng.standard_normal((n_experts, self.n_plucker)) * 0.1
        self.b_gate = np.zeros(n_experts)

    def params(self) -> list[np.ndarray]:
        return [self.W_route, self.W_gate, self.b_gate]

    def param_count(self) -> int:
        return self.k * self.n * self.d_model + self.n_experts * self.n_plucker + self.n_experts

    def route(self, X: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Compute soft routing probabilities.

        Args:
            X: (batch, d_model)

        Returns:
            probs: (batch, n_experts) routing probabilities.
            cache: intermediates for backward.
        """
        batch = X.shape[0]

        Z_flat = X @ self.W_route.T  # (batch, k*n)
        Z = Z_flat.reshape(batch, self.k, self.n)  # (batch, k, n)

        # Plücker coordinates: C(n,k) k-minors
        plucker = np.zeros((batch, self.n_plucker))
        for idx, cols in enumerate(self._col_sets):
            submat = Z[:, :, list(cols)]  # (batch, k, k)
            plucker[:, idx] = batch_det(submat, self.k)

        scores = plucker @ self.W_gate.T + self.b_gate  # (batch, n_experts)
        probs = softmax(scores, axis=-1)

        cache = {
            "X": X,
            "Z_flat": Z_flat,
            "Z": Z,
            "plucker": plucker,
            "scores": scores,
            "probs": probs,
        }
        return probs, cache

    def backward(
        self, d_probs: np.ndarray, cache: dict[str, Any]
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Backward through routing.

        Args:
            d_probs: (batch, n_experts) gradient w.r.t. routing probabilities.

        Returns:
            d_X: (batch, d_model)
            grads: [d_W_route, d_W_gate, d_b_gate]
        """
        X = cache["X"]
        Z = cache["Z"]
        plucker = cache["plucker"]
        probs = cache["probs"]
        batch = X.shape[0]

        # Softmax backward
        s = (probs * d_probs).sum(axis=-1, keepdims=True)
        d_scores = probs * (d_probs - s)  # (batch, n_experts)

        # Gate backward
        d_W_gate = d_scores.T @ plucker  # (n_experts, n_plucker)
        d_b_gate = d_scores.sum(axis=0)
        d_plucker = d_scores @ self.W_gate  # (batch, n_plucker)

        # Plücker → Z backward (through determinants)
        d_Z = np.zeros_like(Z)  # (batch, k, n)
        for idx, cols in enumerate(self._col_sets):
            col_list = list(cols)
            submat = Z[:, :, col_list]  # (batch, k, k)
            d_submat = batch_det_grad(submat, d_plucker[:, idx], self.k)
            for ci, c in enumerate(col_list):
                d_Z[:, :, c] += d_submat[:, :, ci]

        # Z → X backward
        d_Z_flat = d_Z.reshape(batch, self.k * self.n)
        d_W_route = d_Z_flat.T @ X  # (k*n, d_model)
        d_X = d_Z_flat @ self.W_route  # (batch, d_model)

        return d_X, [d_W_route, d_W_gate, d_b_gate]


class _Expert:
    """Simple feedforward expert (ReLU MLP)."""

    def __init__(self, d_model: int, d_ff: int, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        scale = (2.0 / d_model) ** 0.5
        self.W1 = rng.standard_normal((d_ff, d_model)) * scale
        self.b1 = np.zeros(d_ff)
        self.W2 = rng.standard_normal((d_model, d_ff)) * (2.0 / d_ff) ** 0.5
        self.b2 = np.zeros(d_model)

    def params(self) -> list[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2]

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        z1 = X @ self.W1.T + self.b1
        h1 = np.maximum(0, z1)
        out = h1 @ self.W2.T + self.b2
        return out, {"X": X, "z1": z1, "h1": h1}

    def backward(
        self, d_out: np.ndarray, cache: dict[str, Any]
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        X, z1, h1 = cache["X"], cache["z1"], cache["h1"]
        d_W2 = d_out.T @ h1
        d_b2 = d_out.sum(0)
        d_h1 = d_out @ self.W2
        d_z1 = d_h1 * (z1 > 0).astype(float)
        d_W1 = d_z1.T @ X
        d_b1 = d_z1.sum(0)
        d_X = d_z1 @ self.W1
        return d_X, [d_W1, d_b1, d_W2, d_b2]


class PositroidMoE:
    """Positroid Mixture-of-Experts layer.

    Soft routing: output = sum_e probs[e] * expert_e(x)

    Parameters:
        Router parameters + n_experts * expert parameters.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int = 0,
        n_experts: int = 4,
        n: int = 6,
        k: int = 2,
        seed: int = 42,
    ) -> None:
        if d_ff == 0:
            d_ff = 2 * d_model
        self.d_model = d_model
        self.n_experts = n_experts

        rng = np.random.default_rng(seed)
        self.router = PositroidRouter(d_model, n, k, n_experts, seed=int(rng.integers(2**31)))
        self.experts = [
            _Expert(d_model, d_ff, seed=int(rng.integers(2**31))) for _ in range(n_experts)
        ]

    def params(self) -> list[np.ndarray]:
        p = self.router.params()
        for e in self.experts:
            p.extend(e.params())
        return p

    def param_count(self) -> int:
        return self.router.param_count() + sum(
            sum(p.size for p in e.params()) for e in self.experts
        )

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Forward pass with soft routing.

        Args:
            X: (batch, d_model)

        Returns:
            output: (batch, d_model)
            cache: intermediates for backward.
        """
        probs, router_cache = self.router.route(X)  # (batch, n_experts)

        expert_outputs = []
        expert_caches = []
        for e in self.experts:
            e_out, e_cache = e.forward(X)
            expert_outputs.append(e_out)
            expert_caches.append(e_cache)

        # Weighted combination: output[b] = sum_e probs[b,e] * expert_e(X)[b]
        stacked = np.stack(expert_outputs, axis=1)  # (batch, n_experts, d_model)
        output = np.einsum("be,bed->bd", probs, stacked)

        cache = {
            "router_cache": router_cache,
            "probs": probs,
            "expert_outputs": expert_outputs,
            "expert_caches": expert_caches,
            "stacked": stacked,
        }
        return output, cache

    def backward(
        self, d_output: np.ndarray, cache: dict[str, Any]
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Backward through MoE.

        Returns:
            d_X: (batch, d_model)
            grads: all parameter gradients.
        """
        probs = cache["probs"]
        stacked = cache["stacked"]
        expert_caches = cache["expert_caches"]
        router_cache = cache["router_cache"]

        # output = einsum('be,bed->bd', probs, stacked)
        d_probs = np.einsum("bd,bed->be", d_output, stacked)
        d_stacked = np.einsum("be,bd->bed", probs, d_output)

        # Router backward
        d_X_router, router_grads = self.router.backward(d_probs, router_cache)

        # Expert backward
        d_X = d_X_router.copy()
        all_grads = router_grads
        for e_idx, (expert, e_cache) in enumerate(zip(self.experts, expert_caches)):
            d_e_out = d_stacked[:, e_idx, :]  # (batch, d_model)
            d_X_e, e_grads = expert.backward(d_e_out, e_cache)
            d_X += d_X_e
            all_grads.extend(e_grads)

        return d_X, all_grads
