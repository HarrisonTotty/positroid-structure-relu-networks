"""Tests for positroid LoRA (Proposal D)."""

import numpy as np
import pytest

from positroid.transformer.positroid_lora import PositroidLoRA


class TestPositroidLoRA:
    def test_forward_shapes(self):
        lora = PositroidLoRA(d_in=16, d_out=8, rank=4, seed=42)
        X = np.random.default_rng(0).standard_normal((5, 16))
        out, cache = lora.forward(X)
        assert out.shape == (5, 8)

    def test_forward_with_base_weight(self):
        lora = PositroidLoRA(d_in=16, d_out=8, rank=4, seed=42)
        rng = np.random.default_rng(0)
        X = rng.standard_normal((5, 16))
        W_base = rng.standard_normal((8, 16))
        out, cache = lora.forward(X, W_base=W_base)
        # Should equal base output + delta
        base_out = X @ W_base.T
        delta_out, _ = lora.forward(X)
        np.testing.assert_allclose(out, base_out + delta_out, atol=1e-12)

    def test_delta_w_rank(self):
        """ΔW should have rank ≤ r."""
        lora = PositroidLoRA(d_in=16, d_out=8, rank=3, seed=42)
        dw = lora.get_delta_w()
        assert dw.shape == (8, 16)
        sv = np.linalg.svd(dw, compute_uv=False)
        # Rank should be at most 3
        assert sv[3] < 1e-10 * sv[0]

    def test_delta_w_starts_small(self):
        """With small initialization, ΔW should be small."""
        lora = PositroidLoRA(d_in=16, d_out=8, rank=4, seed=42)
        dw = lora.get_delta_w()
        assert np.max(np.abs(dw)) < 0.1

    def test_gradient_finite_difference(self):
        rng = np.random.default_rng(42)
        lora = PositroidLoRA(d_in=8, d_out=6, rank=3, seed=42)
        X = rng.standard_normal((4, 8))

        def loss_fn():
            out, _ = lora.forward(X)
            return np.sum(out**2)

        out, cache = lora.forward(X)
        d_out = 2.0 * out
        _, grads = lora.backward(d_out, cache)

        params = lora.params()
        eps = 1e-5
        for p_idx, (p, g) in enumerate(zip(params, grads)):
            p_flat = p.ravel()
            g_flat = g.ravel()
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

    def test_gradient_with_base_weight(self):
        rng = np.random.default_rng(42)
        lora = PositroidLoRA(d_in=8, d_out=6, rank=3, seed=42)
        X = rng.standard_normal((4, 8))
        W_base = rng.standard_normal((6, 8))

        def loss_fn():
            out, _ = lora.forward(X, W_base=W_base)
            return np.sum(out**2)

        out, cache = lora.forward(X, W_base=W_base)
        d_out = 2.0 * out
        _, grads = lora.backward(d_out, cache)

        params = lora.params()
        eps = 1e-5
        for p_idx, (p, g) in enumerate(zip(params, grads)):
            p_flat = p.ravel()
            g_flat = g.ravel()
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

    def test_param_count(self):
        lora = PositroidLoRA(d_in=16, d_out=8, rank=3, seed=42)
        # face: 3*(8-3)=15, A: 3*16=48
        assert lora.param_count() == 15 + 48
