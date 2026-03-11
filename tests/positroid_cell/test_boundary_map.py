"""Tests for the boundary measurement map."""

import numpy as np
import pytest

from positroid.positroid_cell.boundary_map import (
    boundary_measurement_backward,
    boundary_measurement_matrix,
    plucker_coordinates,
)


class TestBoundaryMeasurementMatrix:
    """Tests for boundary_measurement_matrix."""

    def test_shape_k2_n4(self) -> None:
        weights = np.ones(4)
        mat = boundary_measurement_matrix(weights, k=2, n=4)
        assert mat.shape == (2, 4)

    def test_shape_k2_n6(self) -> None:
        weights = np.ones(8)
        mat = boundary_measurement_matrix(weights, k=2, n=6)
        assert mat.shape == (2, 6)

    def test_shape_k3_n6(self) -> None:
        weights = np.ones(9)
        mat = boundary_measurement_matrix(weights, k=3, n=6)
        assert mat.shape == (3, 6)

    def test_wrong_weight_count_raises(self) -> None:
        with pytest.raises(ValueError, match="Expected 8"):
            boundary_measurement_matrix(np.ones(5), k=2, n=6)

    def test_identity_at_zero_weights(self) -> None:
        """With very small positive weights, mat should be ~[I_k | 0]."""
        weights = np.full(8, 1e-10)
        mat = boundary_measurement_matrix(weights, k=2, n=6)
        np.testing.assert_allclose(mat[:, :2], np.eye(2), atol=1e-8)
        np.testing.assert_allclose(mat[:, 2:], 0.0, atol=1e-6)

    def test_all_plucker_positive_k2_n4(self) -> None:
        """All 2x2 minors should be strictly positive."""
        rng = np.random.default_rng(42)
        weights = np.exp(rng.normal(0, 0.5, size=4))
        mat = boundary_measurement_matrix(weights, k=2, n=4)
        pluckers = plucker_coordinates(mat)
        for idx, val in pluckers.items():
            assert val > 0, f"Plucker coord {idx} = {val} is not positive"

    def test_all_plucker_positive_k2_n6(self) -> None:
        """All 2x2 minors should be strictly positive for Gr+(2,6)."""
        rng = np.random.default_rng(42)
        weights = np.exp(rng.normal(0, 0.5, size=8))
        mat = boundary_measurement_matrix(weights, k=2, n=6)
        pluckers = plucker_coordinates(mat)
        assert len(pluckers) == 15  # C(6,2)
        for idx, val in pluckers.items():
            assert val > 0, f"Plucker coord {idx} = {val} is not positive"

    def test_all_plucker_positive_k3_n6(self) -> None:
        """All 3x3 minors should be strictly positive for Gr+(3,6)."""
        rng = np.random.default_rng(42)
        weights = np.exp(rng.normal(0, 0.5, size=9))
        mat = boundary_measurement_matrix(weights, k=3, n=6)
        pluckers = plucker_coordinates(mat)
        assert len(pluckers) == 20  # C(6,3)
        for idx, val in pluckers.items():
            assert val > 0, f"Plucker coord {idx} = {val} is not positive"

    def test_plucker_relation_k2_n4(self) -> None:
        """Verify the Plucker relation p13*p24 = p12*p34 + p14*p23."""
        rng = np.random.default_rng(123)
        weights = np.exp(rng.normal(0, 0.5, size=4))
        mat = boundary_measurement_matrix(weights, k=2, n=4)
        p = plucker_coordinates(mat)
        lhs = p[(0, 2)] * p[(1, 3)]
        rhs = p[(0, 1)] * p[(2, 3)] + p[(0, 3)] * p[(1, 2)]
        assert lhs == pytest.approx(rhs, rel=1e-10)

    def test_plucker_relation_k2_n6(self) -> None:
        """Verify Plucker relations for Gr(2,6)."""
        rng = np.random.default_rng(456)
        weights = np.exp(rng.normal(0, 0.5, size=8))
        mat = boundary_measurement_matrix(weights, k=2, n=6)
        p = plucker_coordinates(mat)
        # Check p_{ik} * p_{jl} = p_{ij} * p_{kl} + p_{il} * p_{jk}
        for i, j, k, ell in [(0, 1, 2, 3), (0, 2, 3, 5), (1, 2, 4, 5)]:
            lhs = p[(i, k)] * p[(j, ell)]
            rhs = p[(i, j)] * p[(k, ell)] + p[(i, ell)] * p[(j, k)]
            assert lhs == pytest.approx(rhs, rel=1e-10), (
                f"Plucker relation failed for ({i},{j},{k},{ell})"
            )

    def test_multiple_random_seeds_positive(self) -> None:
        """TP property holds for many random weight vectors."""
        for seed in range(20):
            rng = np.random.default_rng(seed)
            weights = np.exp(rng.normal(0, 1.0, size=8))
            mat = boundary_measurement_matrix(weights, k=2, n=6)
            pluckers = plucker_coordinates(mat)
            for idx, val in pluckers.items():
                assert val > 0, f"seed={seed}: Plucker coord {idx} = {val} not positive"


class TestBoundaryMeasurementBackward:
    """Tests for gradient computation."""

    def test_gradient_finite_difference_k2_n4(self) -> None:
        """Gradient matches finite differences for k=2, n=4."""
        rng = np.random.default_rng(42)
        weights = np.exp(rng.normal(0, 0.3, size=4))

        mat = boundary_measurement_matrix(weights, k=2, n=4)
        d_mat = rng.normal(0, 1, size=(2, 4))
        grad = boundary_measurement_backward(weights, 2, 4, mat, d_mat)

        eps = 1e-6
        grad_fd = np.zeros_like(weights)
        for i in range(len(weights)):
            wp = weights.copy()
            wm = weights.copy()
            wp[i] += eps
            wm[i] -= eps
            mat_p = boundary_measurement_matrix(wp, 2, 4)
            mat_m = boundary_measurement_matrix(wm, 2, 4)
            grad_fd[i] = np.sum(d_mat * (mat_p - mat_m)) / (2 * eps)

        np.testing.assert_allclose(grad, grad_fd, rtol=1e-5, atol=1e-8)

    def test_gradient_finite_difference_k2_n6(self) -> None:
        """Gradient matches finite differences for k=2, n=6."""
        rng = np.random.default_rng(42)
        weights = np.exp(rng.normal(0, 0.3, size=8))

        mat = boundary_measurement_matrix(weights, k=2, n=6)
        d_mat = rng.normal(0, 1, size=(2, 6))
        grad = boundary_measurement_backward(weights, 2, 6, mat, d_mat)

        eps = 1e-6
        grad_fd = np.zeros_like(weights)
        for i in range(len(weights)):
            wp = weights.copy()
            wm = weights.copy()
            wp[i] += eps
            wm[i] -= eps
            mat_p = boundary_measurement_matrix(wp, 2, 6)
            mat_m = boundary_measurement_matrix(wm, 2, 6)
            grad_fd[i] = np.sum(d_mat * (mat_p - mat_m)) / (2 * eps)

        np.testing.assert_allclose(grad, grad_fd, rtol=1e-5, atol=1e-8)

    def test_gradient_finite_difference_k3_n6(self) -> None:
        """Gradient matches finite differences for k=3, n=6."""
        rng = np.random.default_rng(42)
        weights = np.exp(rng.normal(0, 0.3, size=9))

        mat = boundary_measurement_matrix(weights, k=3, n=6)
        d_mat = rng.normal(0, 1, size=(3, 6))
        grad = boundary_measurement_backward(weights, 3, 6, mat, d_mat)

        eps = 1e-6
        grad_fd = np.zeros_like(weights)
        for i in range(len(weights)):
            wp = weights.copy()
            wm = weights.copy()
            wp[i] += eps
            wm[i] -= eps
            mat_p = boundary_measurement_matrix(wp, 3, 6)
            mat_m = boundary_measurement_matrix(wm, 3, 6)
            grad_fd[i] = np.sum(d_mat * (mat_p - mat_m)) / (2 * eps)

        np.testing.assert_allclose(grad, grad_fd, rtol=1e-5, atol=1e-8)
