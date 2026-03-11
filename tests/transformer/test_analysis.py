"""Tests for empirical analysis tools (Proposal E)."""

import numpy as np
import pytest

from positroid.transformer.analysis import (
    analyze_weight_matrix,
    check_approximate_tp,
    check_attention_positroid,
    fit_boundary_measurement,
    weight_effective_rank,
)


class TestWeightEffectiveRank:
    def test_full_rank(self):
        rng = np.random.default_rng(42)
        W = rng.standard_normal((8, 8))
        result = weight_effective_rank(W)
        assert result["effective_rank"] == 8
        assert result["full_rank"] == 8

    def test_low_rank(self):
        rng = np.random.default_rng(42)
        A = rng.standard_normal((8, 3))
        B = rng.standard_normal((3, 8))
        W = A @ B  # Rank 3
        result = weight_effective_rank(W, threshold=0.01)
        assert result["effective_rank"] <= 4  # Should be close to 3

    def test_singular_values_sorted(self):
        rng = np.random.default_rng(42)
        W = rng.standard_normal((5, 8))
        result = weight_effective_rank(W)
        sv = result["singular_values"]
        assert np.all(sv[:-1] >= sv[1:])


class TestCheckApproximateTP:
    def test_tp_matrix(self):
        """Known TP matrix from exponential kernel."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([0.5, 1.0, 1.5])
        W = np.exp(np.outer(a, b))
        result = check_approximate_tp(W)
        assert result["is_tp"]
        assert result["is_tn"]
        assert result["n_negative"] == 0
        assert result["fraction_positive"] == 1.0

    def test_random_matrix_not_tp(self):
        rng = np.random.default_rng(42)
        W = rng.standard_normal((4, 4))
        result = check_approximate_tp(W)
        assert not result["is_tp"]
        assert result["n_negative"] > 0

    def test_max_order(self):
        rng = np.random.default_rng(42)
        W = rng.standard_normal((4, 4))
        result_full = check_approximate_tp(W)
        result_order2 = check_approximate_tp(W, max_order=2)
        assert result_order2["total_minors"] <= result_full["total_minors"]


class TestCheckAttentionPositroid:
    def test_identity_attention(self):
        """Identity attention (each query attends to itself)."""
        attn = np.eye(4)
        result = check_attention_positroid(attn)
        assert "error" not in result
        assert result["rank"] >= 1

    def test_uniform_attention(self):
        """Uniform attention."""
        T = 4
        attn = np.ones((T, T)) / T
        result = check_attention_positroid(attn)
        # Hard attention picks first token for all queries
        assert "error" not in result


class TestFitBoundaryMeasurement:
    def test_fit_known_matrix(self):
        """Fit to a matrix that IS a boundary measurement."""
        from positroid.positroid_cell.boundary_map import boundary_measurement_matrix

        k, n = 2, 5
        true_weights = np.array([1.5, 2.0, 0.5, 1.0, 3.0, 0.8])
        target = boundary_measurement_matrix(true_weights, k, n)

        result = fit_boundary_measurement(target, k, max_iter=2000, lr=0.05)
        assert result["relative_error"] < 0.01

    def test_fit_reduces_error(self):
        """Fitting should reduce error vs initial guess."""
        rng = np.random.default_rng(42)
        k, n = 2, 4
        target = rng.standard_normal((k, n))
        target = np.abs(target) + 0.1  # Make positive

        result = fit_boundary_measurement(target, k, max_iter=500, lr=0.01)
        # Initial error (B ≈ [I|0]) should be much larger than fitted
        from positroid.positroid_cell.boundary_map import boundary_measurement_matrix

        init_approx = boundary_measurement_matrix(np.ones(k * (n - k)), k, n)
        init_error = float(np.sum((init_approx - target) ** 2))
        assert result["error"] < init_error


class TestAnalyzeWeightMatrix:
    def test_small_matrix(self):
        rng = np.random.default_rng(42)
        W = rng.standard_normal((4, 6))
        result = analyze_weight_matrix(W)
        assert "rank_info" in result
        assert "tp_info" in result
        assert "fit_info" in result

    def test_large_matrix(self):
        """Large matrix should only check order-2 minors."""
        rng = np.random.default_rng(42)
        W = rng.standard_normal((20, 20))
        result = analyze_weight_matrix(W)
        assert "rank_info" in result
        assert "tp_info" in result
        # Should not have fit_info (too large)
        assert "fit_info" not in result
