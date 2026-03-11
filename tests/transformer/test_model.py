"""Tests for positroid transformer model integration."""

import numpy as np
import pytest

from positroid.transformer.model import (
    LayerNorm,
    PositroidClassifier,
    PositroidTransformerBlock,
    StandardMLP,
    train_classifier,
)


class TestLayerNorm:
    def test_forward_shapes(self):
        ln = LayerNorm(16)
        X = np.random.default_rng(0).standard_normal((5, 16))
        out, cache = ln.forward(X)
        assert out.shape == (5, 16)

    def test_output_normalized(self):
        ln = LayerNorm(16)
        X = np.random.default_rng(0).standard_normal((5, 16))
        out, _ = ln.forward(X)
        # With default gamma=1, beta=0: each row should have mean≈0, var≈1
        np.testing.assert_allclose(out.mean(axis=-1), 0.0, atol=1e-10)
        np.testing.assert_allclose(out.var(axis=-1), 1.0, atol=1e-2)

    def test_gradient_finite_difference(self):
        rng = np.random.default_rng(42)
        ln = LayerNorm(8)
        X = rng.standard_normal((3, 8))

        def loss_fn():
            out, _ = ln.forward(X)
            return np.sum(out**2)

        out, cache = ln.forward(X)
        d_out = 2.0 * out
        d_X, grads = ln.backward(d_out, cache)

        # Check gamma and beta gradients
        params = ln.params()
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


class TestStandardMLP:
    def test_forward_shapes(self):
        mlp = StandardMLP(16, d_ff=32, seed=42)
        X = np.random.default_rng(0).standard_normal((5, 16))
        out, cache = mlp.forward(X)
        assert out.shape == (5, 16)

    def test_gradient_finite_difference(self):
        rng = np.random.default_rng(42)
        mlp = StandardMLP(8, d_ff=16, seed=42)
        X = rng.standard_normal((3, 8))

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


class TestPositroidTransformerBlock:
    def test_forward_shapes(self):
        block = PositroidTransformerBlock(d_model=16, n_heads=2, n=8, k_values=[2, 2], seed=42)
        X = np.random.default_rng(0).standard_normal((4, 16))
        out, cache = block.forward(X)
        assert out.shape == (4, 16)

    def test_tropical_mlp_variant(self):
        block = PositroidTransformerBlock(
            d_model=16,
            n_heads=2,
            n=8,
            k_values=[2, 2],
            mlp_type="tropical",
            mlp_kwargs={"n_cells": 4, "k": 2, "n": 6},
            seed=42,
        )
        X = np.random.default_rng(0).standard_normal((4, 16))
        out, cache = block.forward(X)
        assert out.shape == (4, 16)

    def test_residual_connection(self):
        """Output should differ from input (residual adds non-trivial signal)."""
        block = PositroidTransformerBlock(d_model=16, n_heads=2, n=8, seed=42)
        X = np.random.default_rng(0).standard_normal((4, 16))
        out, _ = block.forward(X)
        assert not np.allclose(out, X)


class TestPositroidClassifier:
    def test_forward_shapes(self):
        clf = PositroidClassifier(
            d_input=10,
            n_classes=3,
            d_model=8,
            n_tokens=2,
            n_layers=1,
            n_heads=2,
            n=6,
            seed=42,
        )
        X = np.random.default_rng(0).standard_normal((5, 10))
        logits, cache = clf.forward(X)
        assert logits.shape == (5, 3)

    def test_predict_probabilities(self):
        clf = PositroidClassifier(
            d_input=10,
            n_classes=3,
            d_model=8,
            n_tokens=2,
            n_layers=1,
            n_heads=2,
            n=6,
            seed=42,
        )
        X = np.random.default_rng(0).standard_normal((5, 10))
        probs = clf.predict(X)
        assert probs.shape == (5, 3)
        np.testing.assert_allclose(probs.sum(axis=-1), 1.0, atol=1e-10)
        assert np.all(probs >= 0)

    def test_training_reduces_loss(self):
        """Training should reduce the loss (at least a little)."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 10))
        y = rng.integers(0, 3, size=30)

        clf = PositroidClassifier(
            d_input=10,
            n_classes=3,
            d_model=8,
            n_tokens=2,
            n_layers=1,
            n_heads=1,
            n=6,
            seed=42,
        )
        history = train_classifier(clf, X, y, epochs=10, lr=0.01, batch_size=15)
        assert history["losses"][-1] < history["losses"][0]

    def test_param_count(self):
        clf = PositroidClassifier(
            d_input=10,
            n_classes=3,
            d_model=8,
            n_tokens=2,
            n_layers=1,
            n_heads=1,
            n=6,
            seed=42,
        )
        count = clf.param_count()
        assert count > 0
        assert count == sum(p.size for p in clf.params())
