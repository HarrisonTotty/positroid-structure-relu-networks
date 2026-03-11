"""Tests for positroid MoE (Proposal B)."""

import numpy as np
import pytest

from positroid.transformer.positroid_moe import PositroidMoE, PositroidRouter


class TestPositroidRouter:
    def test_routing_sums_to_one(self):
        router = PositroidRouter(d_model=16, n=6, k=2, n_experts=4, seed=42)
        X = np.random.default_rng(0).standard_normal((8, 16))
        probs, cache = router.route(X)
        assert probs.shape == (8, 4)
        np.testing.assert_allclose(probs.sum(axis=-1), 1.0, atol=1e-10)

    def test_routing_positive(self):
        router = PositroidRouter(d_model=16, n=6, k=2, n_experts=4, seed=42)
        X = np.random.default_rng(0).standard_normal((5, 16))
        probs, _ = router.route(X)
        assert np.all(probs >= 0)

    def test_plucker_count(self):
        """C(n,k) Plücker coordinates computed."""
        router = PositroidRouter(d_model=8, n=6, k=2, n_experts=3, seed=42)
        assert router.n_plucker == 15  # C(6,2) = 15

    def test_backward(self):
        router = PositroidRouter(d_model=8, n=5, k=2, n_experts=3, seed=42)
        X = np.random.default_rng(0).standard_normal((4, 8))
        probs, cache = router.route(X)
        d_probs = np.ones_like(probs)
        d_X, grads = router.backward(d_probs, cache)
        assert d_X.shape == X.shape
        assert len(grads) == 3


class TestPositroidMoE:
    def test_forward_shapes(self):
        moe = PositroidMoE(d_model=16, d_ff=8, n_experts=3, n=6, k=2, seed=42)
        X = np.random.default_rng(0).standard_normal((5, 16))
        out, cache = moe.forward(X)
        assert out.shape == (5, 16)

    def test_gradient_finite_difference(self):
        rng = np.random.default_rng(42)
        moe = PositroidMoE(d_model=8, d_ff=4, n_experts=2, n=5, k=2, seed=42)
        X = rng.standard_normal((3, 8))

        def loss_fn():
            out, _ = moe.forward(X)
            return np.sum(out**2)

        out, cache = moe.forward(X)
        d_out = 2.0 * out
        _, grads = moe.backward(d_out, cache)

        params = moe.params()
        eps = 1e-5
        for p_idx, (p, g) in enumerate(zip(params, grads)):
            p_flat = p.ravel()
            g_flat = g.ravel()
            for j in range(min(2, len(p_flat))):
                old = p_flat[j]
                p_flat[j] = old + eps
                f_plus = loss_fn()
                p_flat[j] = old - eps
                f_minus = loss_fn()
                p_flat[j] = old
                fd = (f_plus - f_minus) / (2 * eps)
                assert abs(fd - g_flat[j]) < 1e-2 + 1e-2 * abs(fd), (
                    f"Param {p_idx}[{j}]: fd={fd:.6f}, analytical={g_flat[j]:.6f}"
                )

    def test_different_experts_produce_different_outputs(self):
        """Each expert should contribute differently."""
        moe = PositroidMoE(d_model=8, d_ff=4, n_experts=3, n=5, k=2, seed=42)
        X = np.random.default_rng(0).standard_normal((1, 8))
        _, cache = moe.forward(X)
        outputs = cache["expert_outputs"]
        # At least one pair of experts should differ
        assert not np.allclose(outputs[0], outputs[1])
