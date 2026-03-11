"""Tests for positroid attention (Proposal A)."""

import numpy as np
import pytest

from positroid.transformer.positroid_attention import (
    PositroidAttentionHead,
    PositroidMultiHeadAttention,
)


class TestPositroidAttentionHead:
    def test_forward_shapes(self):
        head = PositroidAttentionHead(d_model=16, n=8, k=2, d_v=16, seed=42)
        X = np.random.default_rng(0).standard_normal((5, 16))
        out, cache = head.forward(X)
        assert out.shape == (5, 16)
        assert cache["attn"].shape == (5, 5)

    def test_attention_sums_to_one(self):
        head = PositroidAttentionHead(d_model=16, n=8, k=2, seed=42)
        X = np.random.default_rng(0).standard_normal((4, 16))
        _, cache = head.forward(X)
        row_sums = cache["attn"].sum(axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_k2_antisymmetric_scores(self):
        """For k=2, raw scores (before self-bias) should be antisymmetric."""
        head = PositroidAttentionHead(d_model=16, n=8, k=2, seed=42)
        head.self_bias[:] = 0.0  # Remove self-bias
        X = np.random.default_rng(0).standard_normal((4, 16))
        _, cache = head.forward(X)
        scores = cache["scores"]
        np.testing.assert_allclose(scores, -scores.T, atol=1e-12)

    def test_k3_forward(self):
        head = PositroidAttentionHead(d_model=16, n=8, k=3, seed=42)
        X = np.random.default_rng(0).standard_normal((4, 16))
        out, cache = head.forward(X)
        assert out.shape == (4, 16)

    def test_param_count(self):
        head = PositroidAttentionHead(d_model=16, n=8, k=2, d_v=8, seed=42)
        count = head.param_count()
        # face: 2*6=12, W_proj: 16*8=128, W_V: 16*8=128, W_O: 8*16=128, bias: 1
        assert count == 12 + 128 + 128 + 128 + 1

    def test_gradient_finite_difference(self):
        """Verify analytical gradients against finite differences."""
        rng = np.random.default_rng(42)
        head = PositroidAttentionHead(d_model=8, n=6, k=2, d_v=8, seed=42)
        X = rng.standard_normal((3, 8))

        def loss_fn():
            out, _ = head.forward(X)
            return np.sum(out**2)

        out, cache = head.forward(X)
        d_out = 2.0 * out
        _, grads = head.backward(d_out, cache)

        params = head.params()
        eps = 1e-5
        for p_idx, (p, g) in enumerate(zip(params, grads)):
            p_flat = p.ravel()
            g_flat = g.ravel()
            # Check up to 3 elements per parameter
            for j in range(min(3, len(p_flat))):
                old = p_flat[j]
                p_flat[j] = old + eps
                f_plus = loss_fn()
                p_flat[j] = old - eps
                f_minus = loss_fn()
                p_flat[j] = old
                fd = (f_plus - f_minus) / (2 * eps)
                assert abs(fd - g_flat[j]) < 1e-3 + 1e-3 * abs(fd), (
                    f"Param {p_idx}[{j}]: fd={fd:.6f}, analytical={g_flat[j]:.6f}"
                )

    def test_backward_shapes(self):
        head = PositroidAttentionHead(d_model=8, n=6, k=2, d_v=8, seed=42)
        X = np.random.default_rng(0).standard_normal((3, 8))
        out, cache = head.forward(X)
        d_X, grads = head.backward(np.ones_like(out), cache)
        assert d_X.shape == X.shape
        assert len(grads) == 5


class TestPositroidMultiHeadAttention:
    def test_forward_shapes(self):
        mha = PositroidMultiHeadAttention(d_model=16, n_heads=2, n=8, k_values=[2, 2], seed=42)
        X = np.random.default_rng(0).standard_normal((5, 16))
        out, caches = mha.forward(X)
        assert out.shape == (5, 16)
        assert len(caches) == 2

    def test_heterogeneous_k(self):
        mha = PositroidMultiHeadAttention(d_model=16, n_heads=2, n=8, k_values=[2, 3], seed=42)
        X = np.random.default_rng(0).standard_normal((4, 16))
        out, caches = mha.forward(X)
        assert out.shape == (4, 16)

    def test_backward(self):
        mha = PositroidMultiHeadAttention(d_model=8, n_heads=2, n=6, k_values=[2, 2], seed=42)
        X = np.random.default_rng(0).standard_normal((3, 8))
        out, caches = mha.forward(X)
        d_X, grads = mha.backward(np.ones_like(out), caches)
        assert d_X.shape == X.shape
        assert len(grads) == 10  # 5 per head × 2 heads
