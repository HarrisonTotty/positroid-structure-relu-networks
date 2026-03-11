"""Tests for tropical MLP (Proposal C)."""

import numpy as np
import pytest

from positroid.transformer.tropical_mlp import TropicalMLP


class TestTropicalMLP:
    def test_forward_shapes(self):
        mlp = TropicalMLP(d_in=16, d_out=8, n_cells=4, k=2, n=6, seed=42)
        X = np.random.default_rng(0).standard_normal((10, 16))
        out, cache = mlp.forward(X)
        assert out.shape == (10, 8)

    def test_forward_shapes_k3(self):
        mlp = TropicalMLP(d_in=16, d_out=8, n_cells=4, k=3, n=6, seed=42)
        X = np.random.default_rng(0).standard_normal((10, 16))
        out, cache = mlp.forward(X)
        assert out.shape == (10, 8)

    def test_param_count(self):
        mlp = TropicalMLP(d_in=16, d_out=8, n_cells=4, k=2, n=6, seed=42)
        count = mlp.param_count()
        # face: 4 * 2*(6-2) = 32
        # enc: 6*2 * 16 = 192
        # read: 4 * 8 = 32
        # bias: 8
        assert count == 32 + 192 + 32 + 8

    def test_gradient_finite_difference(self):
        rng = np.random.default_rng(42)
        mlp = TropicalMLP(d_in=8, d_out=4, n_cells=3, k=2, n=5, seed=42)
        X = rng.standard_normal((4, 8))

        def loss_fn():
            out, _ = mlp.forward(X)
            return np.sum(out**2)

        out, cache = mlp.forward(X)
        d_out = 2.0 * out
        _, grads = mlp.backward(d_out, cache)

        params = mlp.params()
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

    def test_backward_shapes(self):
        mlp = TropicalMLP(d_in=8, d_out=4, n_cells=3, k=2, n=5, seed=42)
        X = np.random.default_rng(0).standard_normal((4, 8))
        out, cache = mlp.forward(X)
        d_X, grads = mlp.backward(np.ones_like(out), cache)
        assert d_X.shape == X.shape
        assert len(grads) == 4

    def test_tropical_forward(self):
        """Tropical forward should produce finite values."""
        mlp = TropicalMLP(d_in=8, d_out=4, n_cells=3, k=2, n=5, seed=42)
        X = np.random.default_rng(0).standard_normal((4, 8))
        out_trop = mlp.forward_tropical(X)
        assert out_trop.shape == (4, 4)
        assert np.all(np.isfinite(out_trop))

    def test_single_cell(self):
        """Single cell should produce non-zero output."""
        mlp = TropicalMLP(d_in=4, d_out=2, n_cells=1, k=2, n=4, seed=42)
        X = np.random.default_rng(0).standard_normal((2, 4))
        out, _ = mlp.forward(X)
        assert not np.allclose(out, 0.0)
