import numpy as np
import pytest

from positroid.datasets.toy2d import make_moons
from positroid.linalg.totally_positive import is_totally_positive
from positroid.network.train import (
    SGD,
    Adam,
    CauchyConstrainedParams,
    FixedConvergentBiasOnlyParams,
    LoewnerWhitneyConstrainedParams,
    NegatedBidiagonalConstrainedParams,
    PermutedExponentialConstrainedParams,
    QuadraticDistanceConstrainedParams,
    SinusoidalConstrainedParams,
    TPConstrainedParams,
    TrainConfig,
    backward_pass,
    backward_pass_multiclass,
    binary_cross_entropy,
    forward_pass,
    forward_pass_multiclass,
    sigmoid,
    train,
    train_multiclass,
)


class TestSigmoid:
    def test_basic_values(self):
        result = sigmoid(np.array([0.0]))
        assert result[0] == pytest.approx(0.5)

    def test_large_positive(self):
        result = sigmoid(np.array([100.0]))
        assert result[0] == pytest.approx(1.0)

    def test_large_negative(self):
        result = sigmoid(np.array([-100.0]))
        assert result[0] == pytest.approx(0.0)

    def test_numerical_stability(self):
        result = sigmoid(np.array([1000.0, -1000.0]))
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class TestBCE:
    def test_perfect_prediction(self):
        loss = binary_cross_entropy(np.array([0.99]), np.array([1.0]))
        assert loss < 0.05

    def test_wrong_prediction(self):
        loss = binary_cross_entropy(np.array([0.01]), np.array([1.0]))
        assert loss > 2.0

    def test_symmetric(self):
        loss0 = binary_cross_entropy(np.array([0.99]), np.array([1.0]))
        loss1 = binary_cross_entropy(np.array([0.01]), np.array([0.0]))
        assert loss0 == pytest.approx(loss1, abs=1e-6)


class TestForwardPass:
    def test_shapes(self):
        rng = np.random.default_rng(42)
        w1 = rng.standard_normal((4, 2))
        b1 = np.zeros(4)
        w2 = rng.standard_normal((1, 4))
        b2 = np.zeros(1)
        x = rng.standard_normal((10, 2))

        y_pred, z1_pre, z1_post, z2_pre = forward_pass(x, w1, b1, w2, b2)
        assert y_pred.shape == (10, 1)
        assert z1_pre.shape == (10, 4)
        assert z1_post.shape == (10, 4)
        assert z2_pre.shape == (10, 1)

    def test_relu_applied(self):
        w1 = np.eye(2)
        b1 = np.zeros(2)
        w2 = np.ones((1, 2))
        b2 = np.zeros(1)
        x = np.array([[-1.0, 2.0]])

        _, z1_pre, z1_post, _ = forward_pass(x, w1, b1, w2, b2)
        np.testing.assert_allclose(z1_pre, [[-1.0, 2.0]])
        np.testing.assert_allclose(z1_post, [[0.0, 2.0]])

    def test_output_in_zero_one(self):
        rng = np.random.default_rng(42)
        w1 = rng.standard_normal((4, 2))
        b1 = rng.standard_normal(4)
        w2 = rng.standard_normal((1, 4))
        b2 = rng.standard_normal(1)
        x = rng.standard_normal((20, 2))

        y_pred, _, _, _ = forward_pass(x, w1, b1, w2, b2)
        assert np.all(y_pred >= 0)
        assert np.all(y_pred <= 1)


class TestBackwardPass:
    def test_gradient_check_w1(self):
        """Numerical gradient check for dW1."""
        rng = np.random.default_rng(42)
        w1 = rng.standard_normal((4, 2))
        b1 = rng.standard_normal(4)
        w2 = rng.standard_normal((1, 4))
        b2 = rng.standard_normal(1)
        x = rng.standard_normal((5, 2))
        y_true = rng.integers(0, 2, size=5).astype(float)

        y_pred, z1_pre, z1_post, _ = forward_pass(x, w1, b1, w2, b2)
        dw1, _, _, _ = backward_pass(x, y_true, y_pred, z1_pre, z1_post, w2)

        eps = 1e-5
        for i in range(w1.shape[0]):
            for j in range(w1.shape[1]):
                w1p = w1.copy()
                w1p[i, j] += eps
                w1m = w1.copy()
                w1m[i, j] -= eps
                loss_p = binary_cross_entropy(
                    forward_pass(x, w1p, b1, w2, b2)[0],
                    y_true,
                )
                loss_m = binary_cross_entropy(
                    forward_pass(x, w1m, b1, w2, b2)[0],
                    y_true,
                )
                numerical = (loss_p - loss_m) / (2 * eps)
                assert dw1[i, j] == pytest.approx(numerical, abs=1e-4), (
                    f"dW1[{i},{j}]: analytical={dw1[i, j]:.6f}, numerical={numerical:.6f}"
                )

    def test_gradient_check_w2(self):
        """Numerical gradient check for dW2."""
        rng = np.random.default_rng(42)
        w1 = rng.standard_normal((4, 2))
        b1 = rng.standard_normal(4)
        w2 = rng.standard_normal((1, 4))
        b2 = rng.standard_normal(1)
        x = rng.standard_normal((5, 2))
        y_true = rng.integers(0, 2, size=5).astype(float)

        y_pred, z1_pre, z1_post, _ = forward_pass(x, w1, b1, w2, b2)
        _, _, dw2, _ = backward_pass(x, y_true, y_pred, z1_pre, z1_post, w2)

        eps = 1e-5
        for j in range(w2.shape[1]):
            w2p = w2.copy()
            w2p[0, j] += eps
            w2m = w2.copy()
            w2m[0, j] -= eps
            loss_p = binary_cross_entropy(
                forward_pass(x, w1, b1, w2p, b2)[0],
                y_true,
            )
            loss_m = binary_cross_entropy(
                forward_pass(x, w1, b1, w2m, b2)[0],
                y_true,
            )
            numerical = (loss_p - loss_m) / (2 * eps)
            assert dw2[0, j] == pytest.approx(numerical, abs=1e-4)


class TestTPConstrainedParams:
    def test_weight_is_tp(self):
        rng = np.random.default_rng(42)
        params = TPConstrainedParams(input_dim=2, hidden_dim=6, rng=rng)
        w1 = params.weight_matrix()
        assert is_totally_positive(w1)

    def test_sorted_params_increasing(self):
        rng = np.random.default_rng(42)
        params = TPConstrainedParams(input_dim=2, hidden_dim=6, rng=rng)
        a, b = params.sorted_params()
        assert all(a[i] < a[i + 1] for i in range(len(a) - 1))
        assert all(b[i] < b[i + 1] for i in range(len(b) - 1))


class TestCauchyConstrainedParams:
    def test_weight_is_tp(self):
        rng = np.random.default_rng(42)
        params = CauchyConstrainedParams(input_dim=2, hidden_dim=6, rng=rng)
        w1 = params.weight_matrix()
        assert is_totally_positive(w1)

    def test_sorted_params_increasing_and_positive(self):
        rng = np.random.default_rng(42)
        params = CauchyConstrainedParams(input_dim=2, hidden_dim=6, rng=rng)
        a, b = params.sorted_params()
        assert all(a[i] < a[i + 1] for i in range(len(a) - 1))
        assert all(b[i] < b[i + 1] for i in range(len(b) - 1))
        assert np.all(a > 0)
        assert np.all(b > 0)

    def test_higher_dim_is_tp(self):
        """Cauchy minors decay rapidly with matrix size; use relaxed tol."""
        rng = np.random.default_rng(42)
        params = CauchyConstrainedParams(input_dim=5, hidden_dim=8, rng=rng)
        w1 = params.weight_matrix()
        assert w1.shape == (8, 5)
        assert np.all(w1 > 0)
        # Cauchy 8x5 has minors ~1e-20; provably TP by construction
        assert is_totally_positive(w1, tol=1e-25)

    def test_gradient_check_a_raw(self):
        """Numerical gradient check for Cauchy da_raw."""
        rng = np.random.default_rng(42)
        params = CauchyConstrainedParams(input_dim=2, hidden_dim=4, rng=rng)
        x = rng.standard_normal((5, 2))
        y_true = rng.integers(0, 2, size=5).astype(float)

        w1, b1, w2, b2 = params.weights()
        y_pred, z1_pre, z1_post, _ = forward_pass(x, w1, b1, w2, b2)
        grads = params.compute_grads(x, y_true, y_pred, z1_pre, z1_post)
        da_raw = grads[0]

        eps = 1e-5
        for k in range(len(params.a_raw)):
            orig = params.a_raw[k]
            params.a_raw[k] = orig + eps
            w1p, b1p, w2p, b2p = params.weights()
            loss_p = binary_cross_entropy(
                forward_pass(x, w1p, b1p, w2p, b2p)[0],
                y_true,
            )
            params.a_raw[k] = orig - eps
            w1m, b1m, w2m, b2m = params.weights()
            loss_m = binary_cross_entropy(
                forward_pass(x, w1m, b1m, w2m, b2m)[0],
                y_true,
            )
            params.a_raw[k] = orig
            numerical = (loss_p - loss_m) / (2 * eps)
            assert da_raw[k] == pytest.approx(numerical, abs=1e-4), (
                f"da_raw[{k}]: analytical={da_raw[k]:.6f}, numerical={numerical:.6f}"
            )


class TestSinusoidalConstrainedParams:
    def test_weight_is_not_tp(self):
        rng = np.random.default_rng(42)
        params = SinusoidalConstrainedParams(input_dim=2, hidden_dim=6, rng=rng)
        w1 = params.weight_matrix()
        assert not is_totally_positive(w1)

    def test_weight_entries_in_range(self):
        rng = np.random.default_rng(42)
        params = SinusoidalConstrainedParams(input_dim=2, hidden_dim=6, rng=rng)
        w1 = params.weight_matrix()
        assert np.all(w1 >= 1.0)
        assert np.all(w1 <= 3.0)

    def test_sorted_params_increasing(self):
        rng = np.random.default_rng(42)
        params = SinusoidalConstrainedParams(input_dim=2, hidden_dim=6, rng=rng)
        a, b = params.sorted_params()
        assert all(a[i] < a[i + 1] for i in range(len(a) - 1))
        assert all(b[i] < b[i + 1] for i in range(len(b) - 1))

    def test_gradient_check_a_raw(self):
        """Numerical gradient check for sinusoidal da_raw."""
        rng = np.random.default_rng(42)
        params = SinusoidalConstrainedParams(input_dim=2, hidden_dim=4, rng=rng)
        x = rng.standard_normal((5, 2))
        y_true = rng.integers(0, 2, size=5).astype(float)

        w1, b1, w2, b2 = params.weights()
        y_pred, z1_pre, z1_post, _ = forward_pass(x, w1, b1, w2, b2)
        grads = params.compute_grads(x, y_true, y_pred, z1_pre, z1_post)
        da_raw = grads[0]

        eps = 1e-5
        for k in range(len(params.a_raw)):
            orig = params.a_raw[k]
            params.a_raw[k] = orig + eps
            w1p, b1p, w2p, b2p = params.weights()
            loss_p = binary_cross_entropy(
                forward_pass(x, w1p, b1p, w2p, b2p)[0],
                y_true,
            )
            params.a_raw[k] = orig - eps
            w1m, b1m, w2m, b2m = params.weights()
            loss_m = binary_cross_entropy(
                forward_pass(x, w1m, b1m, w2m, b2m)[0],
                y_true,
            )
            params.a_raw[k] = orig
            numerical = (loss_p - loss_m) / (2 * eps)
            assert da_raw[k] == pytest.approx(numerical, abs=1e-4), (
                f"da_raw[{k}]: analytical={da_raw[k]:.6f}, numerical={numerical:.6f}"
            )


class TestQuadraticDistanceConstrainedParams:
    def test_weight_is_not_tp(self):
        rng = np.random.default_rng(42)
        params = QuadraticDistanceConstrainedParams(input_dim=2, hidden_dim=6, rng=rng)
        w1 = params.weight_matrix()
        assert not is_totally_positive(w1)

    def test_weight_entries_positive(self):
        rng = np.random.default_rng(42)
        params = QuadraticDistanceConstrainedParams(input_dim=2, hidden_dim=6, rng=rng)
        w1 = params.weight_matrix()
        assert np.all(w1 >= 1.0)

    def test_sorted_params_increasing(self):
        rng = np.random.default_rng(42)
        params = QuadraticDistanceConstrainedParams(input_dim=2, hidden_dim=6, rng=rng)
        a, b = params.sorted_params()
        assert all(a[i] < a[i + 1] for i in range(len(a) - 1))
        assert all(b[i] < b[i + 1] for i in range(len(b) - 1))

    def test_gradient_check_a_raw(self):
        """Numerical gradient check for quadratic distance da_raw."""
        rng = np.random.default_rng(42)
        params = QuadraticDistanceConstrainedParams(input_dim=2, hidden_dim=4, rng=rng)
        x = rng.standard_normal((5, 2))
        y_true = rng.integers(0, 2, size=5).astype(float)

        w1, b1, w2, b2 = params.weights()
        y_pred, z1_pre, z1_post, _ = forward_pass(x, w1, b1, w2, b2)
        grads = params.compute_grads(x, y_true, y_pred, z1_pre, z1_post)
        da_raw = grads[0]

        eps = 1e-5
        for k in range(len(params.a_raw)):
            orig = params.a_raw[k]
            params.a_raw[k] = orig + eps
            w1p, b1p, w2p, b2p = params.weights()
            loss_p = binary_cross_entropy(
                forward_pass(x, w1p, b1p, w2p, b2p)[0],
                y_true,
            )
            params.a_raw[k] = orig - eps
            w1m, b1m, w2m, b2m = params.weights()
            loss_m = binary_cross_entropy(
                forward_pass(x, w1m, b1m, w2m, b2m)[0],
                y_true,
            )
            params.a_raw[k] = orig
            numerical = (loss_p - loss_m) / (2 * eps)
            assert da_raw[k] == pytest.approx(numerical, abs=1e-4), (
                f"da_raw[{k}]: analytical={da_raw[k]:.6f}, numerical={numerical:.6f}"
            )


class TestPermutedExponentialConstrainedParams:
    def test_weight_is_not_tp(self):
        rng = np.random.default_rng(42)
        params = PermutedExponentialConstrainedParams(input_dim=2, hidden_dim=6, rng=rng)
        w1 = params.weight_matrix()
        assert not is_totally_positive(w1)

    def test_weight_entries_positive(self):
        rng = np.random.default_rng(42)
        params = PermutedExponentialConstrainedParams(input_dim=2, hidden_dim=6, rng=rng)
        w1 = params.weight_matrix()
        assert np.all(w1 > 0)

    def test_has_negative_2x2_minor(self):
        """Reversed columns should produce at least one negative 2x2 minor."""
        rng = np.random.default_rng(42)
        params = PermutedExponentialConstrainedParams(input_dim=2, hidden_dim=6, rng=rng)
        w1 = params.weight_matrix()
        has_neg = False
        for i in range(w1.shape[0]):
            for j in range(i + 1, w1.shape[0]):
                det = w1[i, 0] * w1[j, 1] - w1[i, 1] * w1[j, 0]
                if det < 0:
                    has_neg = True
                    break
            if has_neg:
                break
        assert has_neg

    def test_gradient_check_a_raw(self):
        """Numerical gradient check with small params to avoid sigmoid saturation."""
        rng = np.random.default_rng(42)
        params = PermutedExponentialConstrainedParams(input_dim=2, hidden_dim=4, rng=rng)
        # Use small params so exp(a*b) entries stay near 1, avoiding saturation
        params.a_raw[:] = rng.uniform(0.01, 0.05, size=len(params.a_raw))
        params.b_raw[:] = rng.uniform(0.01, 0.05, size=len(params.b_raw))
        x = rng.standard_normal((5, 2))
        y_true = rng.integers(0, 2, size=5).astype(float)

        w1, b1, w2, b2 = params.weights()
        y_pred, z1_pre, z1_post, _ = forward_pass(x, w1, b1, w2, b2)
        grads = params.compute_grads(x, y_true, y_pred, z1_pre, z1_post)
        da_raw = grads[0]

        eps = 1e-5
        for k in range(len(params.a_raw)):
            orig = params.a_raw[k]
            params.a_raw[k] = orig + eps
            w1p, b1p, w2p, b2p = params.weights()
            loss_p = binary_cross_entropy(
                forward_pass(x, w1p, b1p, w2p, b2p)[0],
                y_true,
            )
            params.a_raw[k] = orig - eps
            w1m, b1m, w2m, b2m = params.weights()
            loss_m = binary_cross_entropy(
                forward_pass(x, w1m, b1m, w2m, b2m)[0],
                y_true,
            )
            params.a_raw[k] = orig
            numerical = (loss_p - loss_m) / (2 * eps)
            assert da_raw[k] == pytest.approx(numerical, abs=1e-4), (
                f"da_raw[{k}]: analytical={da_raw[k]:.6f}, numerical={numerical:.6f}"
            )

    def test_gradient_check_b_raw(self):
        """Gradient check for b_raw with input_dim=3 to exercise permutation inverse."""
        rng = np.random.default_rng(42)
        params = PermutedExponentialConstrainedParams(input_dim=3, hidden_dim=4, rng=rng)
        params.a_raw[:] = rng.uniform(0.01, 0.05, size=len(params.a_raw))
        params.b_raw[:] = rng.uniform(0.01, 0.05, size=len(params.b_raw))
        x = rng.standard_normal((5, 3))
        y_true = rng.integers(0, 2, size=5).astype(float)

        w1, b1, w2, b2 = params.weights()
        y_pred, z1_pre, z1_post, _ = forward_pass(x, w1, b1, w2, b2)
        grads = params.compute_grads(x, y_true, y_pred, z1_pre, z1_post)
        db_raw = grads[1]

        eps = 1e-5
        for k in range(len(params.b_raw)):
            orig = params.b_raw[k]
            params.b_raw[k] = orig + eps
            w1p, b1p, w2p, b2p = params.weights()
            loss_p = binary_cross_entropy(
                forward_pass(x, w1p, b1p, w2p, b2p)[0],
                y_true,
            )
            params.b_raw[k] = orig - eps
            w1m, b1m, w2m, b2m = params.weights()
            loss_m = binary_cross_entropy(
                forward_pass(x, w1m, b1m, w2m, b2m)[0],
                y_true,
            )
            params.b_raw[k] = orig
            numerical = (loss_p - loss_m) / (2 * eps)
            assert db_raw[k] == pytest.approx(numerical, abs=1e-4), (
                f"db_raw[{k}]: analytical={db_raw[k]:.6f}, numerical={numerical:.6f}"
            )


class TestNegatedBidiagonalConstrainedParams:
    def test_weight_is_not_tp(self):
        """B@E should not be TP due to negative subdiagonal entries in B."""
        rng = np.random.default_rng(42)
        params = NegatedBidiagonalConstrainedParams(input_dim=2, hidden_dim=6, rng=rng)
        w1 = params.weight_matrix()
        # Check that at least one 2x2 minor is negative
        from itertools import combinations

        has_neg_minor = False
        for i, j in combinations(range(w1.shape[0]), 2):
            det = w1[i, 0] * w1[j, 1] - w1[i, 1] * w1[j, 0]
            if det < -1e-12:
                has_neg_minor = True
                break
        assert has_neg_minor, "Expected at least one negative 2x2 minor"

    def test_bidiagonal_structure(self):
        """B should have 1s on diagonal and alternating signs on subdiagonal."""
        rng = np.random.default_rng(42)
        params = NegatedBidiagonalConstrainedParams(input_dim=2, hidden_dim=4, rng=rng)
        bidiag = params._B
        np.testing.assert_allclose(np.diag(bidiag), 1.0)
        expected_subdiag = [(-1) ** (i + 1) * 1.5 for i in range(3)]
        actual_subdiag = [bidiag[i + 1, i] for i in range(3)]
        np.testing.assert_allclose(actual_subdiag, expected_subdiag)

    def test_gradient_check_a_raw(self):
        """Numerical gradient check with small params to avoid sigmoid saturation."""
        rng = np.random.default_rng(42)
        params = NegatedBidiagonalConstrainedParams(input_dim=2, hidden_dim=4, rng=rng)
        params.a_raw[:] = rng.uniform(0.01, 0.05, size=len(params.a_raw))
        params.b_raw[:] = rng.uniform(0.01, 0.05, size=len(params.b_raw))
        x = rng.standard_normal((5, 2))
        y_true = rng.integers(0, 2, size=5).astype(float)

        w1, b1, w2, b2 = params.weights()
        y_pred, z1_pre, z1_post, _ = forward_pass(x, w1, b1, w2, b2)
        grads = params.compute_grads(x, y_true, y_pred, z1_pre, z1_post)
        da_raw = grads[0]

        eps = 1e-5
        for k in range(len(params.a_raw)):
            orig = params.a_raw[k]
            params.a_raw[k] = orig + eps
            w1p, b1p, w2p, b2p = params.weights()
            loss_p = binary_cross_entropy(
                forward_pass(x, w1p, b1p, w2p, b2p)[0],
                y_true,
            )
            params.a_raw[k] = orig - eps
            w1m, b1m, w2m, b2m = params.weights()
            loss_m = binary_cross_entropy(
                forward_pass(x, w1m, b1m, w2m, b2m)[0],
                y_true,
            )
            params.a_raw[k] = orig
            numerical = (loss_p - loss_m) / (2 * eps)
            assert da_raw[k] == pytest.approx(numerical, abs=1e-4), (
                f"da_raw[{k}]: analytical={da_raw[k]:.6f}, numerical={numerical:.6f}"
            )

    def test_gradient_check_b_raw(self):
        """Numerical gradient check with small params to avoid sigmoid saturation."""
        rng = np.random.default_rng(42)
        params = NegatedBidiagonalConstrainedParams(input_dim=2, hidden_dim=4, rng=rng)
        params.a_raw[:] = rng.uniform(0.01, 0.05, size=len(params.a_raw))
        params.b_raw[:] = rng.uniform(0.01, 0.05, size=len(params.b_raw))
        x = rng.standard_normal((5, 2))
        y_true = rng.integers(0, 2, size=5).astype(float)

        w1, b1, w2, b2 = params.weights()
        y_pred, z1_pre, z1_post, _ = forward_pass(x, w1, b1, w2, b2)
        grads = params.compute_grads(x, y_true, y_pred, z1_pre, z1_post)
        db_raw = grads[1]

        eps = 1e-5
        for k in range(len(params.b_raw)):
            orig = params.b_raw[k]
            params.b_raw[k] = orig + eps
            w1p, b1p, w2p, b2p = params.weights()
            loss_p = binary_cross_entropy(
                forward_pass(x, w1p, b1p, w2p, b2p)[0],
                y_true,
            )
            params.b_raw[k] = orig - eps
            w1m, b1m, w2m, b2m = params.weights()
            loss_m = binary_cross_entropy(
                forward_pass(x, w1m, b1m, w2m, b2m)[0],
                y_true,
            )
            params.b_raw[k] = orig
            numerical = (loss_p - loss_m) / (2 * eps)
            assert db_raw[k] == pytest.approx(numerical, abs=1e-4), (
                f"db_raw[{k}]: analytical={db_raw[k]:.6f}, numerical={numerical:.6f}"
            )


class TestFixedConvergentBiasOnlyParams:
    def test_weight_is_not_tp(self):
        rng = np.random.default_rng(42)
        params = FixedConvergentBiasOnlyParams(input_dim=2, hidden_dim=6, rng=rng)
        w1 = params.weight_matrix()
        assert not is_totally_positive(w1)

    def test_weight_entries_positive(self):
        rng = np.random.default_rng(42)
        params = FixedConvergentBiasOnlyParams(input_dim=2, hidden_dim=6, rng=rng)
        w1 = params.weight_matrix()
        assert np.all(w1 > 0)

    def test_w1_not_in_param_list(self):
        rng = np.random.default_rng(42)
        params = FixedConvergentBiasOnlyParams(input_dim=2, hidden_dim=6, rng=rng)
        plist = params.param_list()
        assert len(plist) == 3  # b1, w2, b2 only

    def test_w1_frozen_during_training(self):
        """W1 should not change after training."""
        x, y = make_moons(n_samples=100, rng=np.random.default_rng(42))
        config = TrainConfig(
            hidden_dim=6,
            epochs=30,
            param_mode="fixed_convergent_bias_only",
            seed=42,
        )
        # Build params manually to capture initial w1
        # Train via the main train() function
        net, _ = train(x, y, config)
        w1_after = net.layers[0].weight
        # Build with same seed to get expected frozen W1
        rng2 = np.random.default_rng(config.seed)
        params2 = FixedConvergentBiasOnlyParams(x.shape[1], config.hidden_dim, rng2)
        np.testing.assert_array_equal(w1_after, params2.weight_matrix())

    def test_grads_length_matches_params(self):
        rng = np.random.default_rng(42)
        params = FixedConvergentBiasOnlyParams(input_dim=2, hidden_dim=4, rng=rng)
        x = rng.standard_normal((5, 2))
        y_true = rng.integers(0, 2, size=5).astype(float)

        w1, b1, w2, b2 = params.weights()
        y_pred, z1_pre, z1_post, _ = forward_pass(x, w1, b1, w2, b2)
        grads = params.compute_grads(x, y_true, y_pred, z1_pre, z1_post)
        assert len(grads) == len(params.param_list()) == 3


class TestOptimizers:
    def test_sgd_decreases_param(self):
        params = [np.array([1.0, 2.0])]
        opt = SGD(params, lr=0.1)
        opt.step([np.array([1.0, 1.0])])
        np.testing.assert_allclose(params[0], [0.9, 1.9])

    def test_adam_updates(self):
        params = [np.array([1.0, 2.0])]
        opt = Adam(params, lr=0.01)
        opt.step([np.array([1.0, 1.0])])
        assert params[0][0] < 1.0
        assert params[0][1] < 2.0


class TestTrain:
    def test_unconstrained_loss_decreases(self):
        x, y = make_moons(n_samples=100, rng=np.random.default_rng(42))
        config = TrainConfig(hidden_dim=8, epochs=50, learning_rate=0.1, seed=42)
        _, history = train(x, y, config)
        assert len(history.losses) == 50
        assert history.losses[-1] < history.losses[0]

    def test_tp_constrained_trains(self):
        x, y = make_moons(n_samples=100, rng=np.random.default_rng(42))
        config = TrainConfig(
            hidden_dim=8,
            epochs=50,
            learning_rate=0.1,
            tp_constrained=True,
            seed=42,
        )
        net, history = train(x, y, config)
        assert len(history.losses) == 50
        # Weights should still be TP
        assert is_totally_positive(net.layers[0].weight)

    def test_network_output_shape(self):
        x, y = make_moons(n_samples=100, rng=np.random.default_rng(42))
        config = TrainConfig(hidden_dim=6, epochs=10, seed=42)
        net, _ = train(x, y, config)
        assert net.input_dim == 2
        assert net.output_dim == 1
        assert net.hidden_dims == [6]

    def test_snapshots(self):
        x, y = make_moons(n_samples=100, rng=np.random.default_rng(42))
        config = TrainConfig(hidden_dim=6, epochs=20, seed=42)
        _, history = train(x, y, config, snapshot_epochs=[0, 10, 19])
        assert 0 in history.snapshots
        assert 10 in history.snapshots
        assert 19 in history.snapshots

    def test_cauchy_constrained_trains(self):
        x, y = make_moons(n_samples=100, rng=np.random.default_rng(42))
        config = TrainConfig(
            hidden_dim=8,
            epochs=50,
            learning_rate=0.1,
            tp_constrained=True,
            tp_kernel="cauchy",
            seed=42,
        )
        net, history = train(x, y, config)
        assert len(history.losses) == 50
        assert is_totally_positive(net.layers[0].weight)

    def test_sinusoidal_trains(self):
        x, y = make_moons(n_samples=100, rng=np.random.default_rng(42))
        config = TrainConfig(
            hidden_dim=8,
            epochs=50,
            learning_rate=0.1,
            param_mode="sinusoidal",
            seed=42,
        )
        net, history = train(x, y, config)
        assert len(history.losses) == 50
        assert history.losses[-1] < history.losses[0]

    def test_quadratic_distance_trains(self):
        x, y = make_moons(n_samples=100, rng=np.random.default_rng(42))
        config = TrainConfig(
            hidden_dim=8,
            epochs=50,
            learning_rate=0.1,
            param_mode="quadratic_distance",
            seed=42,
        )
        net, history = train(x, y, config)
        assert len(history.losses) == 50
        assert history.losses[-1] < history.losses[0]

    def test_param_mode_overrides_tp_fields(self):
        """param_mode takes priority over tp_constrained/tp_kernel."""
        x, y = make_moons(n_samples=100, rng=np.random.default_rng(42))
        config = TrainConfig(
            hidden_dim=6,
            epochs=10,
            param_mode="sinusoidal",
            tp_constrained=True,
            tp_kernel="exponential",
            seed=42,
        )
        net, _ = train(x, y, config)
        # Sinusoidal weights should NOT be TP
        assert not is_totally_positive(net.layers[0].weight)

    def test_permuted_exponential_trains(self):
        x, y = make_moons(n_samples=100, rng=np.random.default_rng(42))
        config = TrainConfig(
            hidden_dim=8,
            epochs=50,
            learning_rate=0.1,
            param_mode="permuted_exponential",
            seed=42,
        )
        net, history = train(x, y, config)
        assert len(history.losses) == 50
        assert history.losses[-1] < history.losses[0]

    def test_negated_bidiagonal_trains(self):
        x, y = make_moons(n_samples=100, rng=np.random.default_rng(42))
        config = TrainConfig(
            hidden_dim=8,
            epochs=50,
            learning_rate=0.1,
            param_mode="negated_bidiagonal",
            seed=42,
        )
        net, history = train(x, y, config)
        assert len(history.losses) == 50
        assert history.losses[-1] < history.losses[0]

    def test_fixed_convergent_bias_only_trains(self):
        x, y = make_moons(n_samples=100, rng=np.random.default_rng(42))
        config = TrainConfig(
            hidden_dim=8,
            epochs=50,
            learning_rate=0.1,
            param_mode="fixed_convergent_bias_only",
            seed=42,
        )
        net, history = train(x, y, config)
        assert len(history.losses) == 50
        assert history.losses[-1] < history.losses[0]

    def test_sgd_optimizer(self):
        x, y = make_moons(n_samples=100, rng=np.random.default_rng(42))
        config = TrainConfig(
            hidden_dim=6,
            epochs=30,
            learning_rate=0.5,
            optimizer="sgd",
            seed=42,
        )
        _, history = train(x, y, config)
        assert history.losses[-1] < history.losses[0]

    def test_lw_trains(self):
        x, y = make_moons(n_samples=100, rng=np.random.default_rng(42))
        config = TrainConfig(
            hidden_dim=8,
            epochs=50,
            learning_rate=0.1,
            param_mode="tp_loewner_whitney",
            seed=42,
        )
        net, history = train(x, y, config)
        assert len(history.losses) == 50
        assert history.losses[-1] < history.losses[0]


class TestLoewnerWhitneyConstrainedParams:
    def test_weight_is_tp(self):
        rng = np.random.default_rng(42)
        params = LoewnerWhitneyConstrainedParams(input_dim=2, hidden_dim=6, rng=rng)
        w1 = params.weight_matrix()
        assert w1.shape == (6, 2)
        assert is_totally_positive(w1)

    def test_weight_shape(self):
        rng = np.random.default_rng(42)
        params = LoewnerWhitneyConstrainedParams(input_dim=3, hidden_dim=8, rng=rng)
        w1 = params.weight_matrix()
        assert w1.shape == (8, 3)

    def test_higher_dim_is_tp(self):
        rng = np.random.default_rng(42)
        params = LoewnerWhitneyConstrainedParams(input_dim=3, hidden_dim=5, rng=rng)
        w1 = params.weight_matrix()
        assert w1.shape == (5, 3)
        assert is_totally_positive(w1)

    def test_gradient_check_all_params(self):
        """Numerical gradient check for all LW params (diag, upper, lower)."""
        rng = np.random.default_rng(42)
        params = LoewnerWhitneyConstrainedParams(input_dim=2, hidden_dim=4, rng=rng)
        # Small params to avoid numerical issues
        params._diag_raw[:] = rng.uniform(-0.5, 0.0, size=len(params._diag_raw))
        params._upper_raw[:] = rng.uniform(-0.5, 0.0, size=len(params._upper_raw))
        params._lower_raw[:] = rng.uniform(-0.5, 0.0, size=len(params._lower_raw))
        x = rng.standard_normal((5, 2))
        y_true = rng.integers(0, 2, size=5).astype(float)

        w1, b1, w2, b2 = params.weights()
        y_pred, z1_pre, z1_post, _ = forward_pass(x, w1, b1, w2, b2)
        grads = params.compute_grads(x, y_true, y_pred, z1_pre, z1_post)

        eps = 1e-5
        raw_arrays = [params._diag_raw, params._upper_raw, params._lower_raw]
        names = ["diag_raw", "upper_raw", "lower_raw"]
        for arr_idx, (arr, name) in enumerate(zip(raw_arrays, names, strict=True)):
            d_raw = grads[arr_idx]
            for k in range(len(arr)):
                orig = arr[k]
                arr[k] = orig + eps
                w1p, b1p, w2p, b2p = params.weights()
                loss_p = binary_cross_entropy(
                    forward_pass(x, w1p, b1p, w2p, b2p)[0],
                    y_true,
                )
                arr[k] = orig - eps
                w1m, b1m, w2m, b2m = params.weights()
                loss_m = binary_cross_entropy(
                    forward_pass(x, w1m, b1m, w2m, b2m)[0],
                    y_true,
                )
                arr[k] = orig
                numerical = (loss_p - loss_m) / (2 * eps)
                assert d_raw[k] == pytest.approx(numerical, abs=1e-4), (
                    f"d_{name}[{k}]: analytical={d_raw[k]:.6f}, numerical={numerical:.6f}"
                )

    def test_gradient_check_3d(self):
        """Gradient check with d=3 to exercise upper bidiag wiring."""
        rng = np.random.default_rng(42)
        params = LoewnerWhitneyConstrainedParams(input_dim=3, hidden_dim=5, rng=rng)
        params._diag_raw[:] = rng.uniform(-0.5, 0.0, size=len(params._diag_raw))
        params._upper_raw[:] = rng.uniform(-0.5, 0.0, size=len(params._upper_raw))
        params._lower_raw[:] = rng.uniform(-0.5, 0.0, size=len(params._lower_raw))
        x = rng.standard_normal((5, 3))
        y_true = rng.integers(0, 2, size=5).astype(float)

        w1, b1, w2, b2 = params.weights()
        y_pred, z1_pre, z1_post, _ = forward_pass(x, w1, b1, w2, b2)
        grads = params.compute_grads(x, y_true, y_pred, z1_pre, z1_post)

        eps = 1e-5
        raw_arrays = [params._diag_raw, params._upper_raw, params._lower_raw]
        names = ["diag_raw", "upper_raw", "lower_raw"]
        for arr_idx, (arr, name) in enumerate(zip(raw_arrays, names, strict=True)):
            d_raw = grads[arr_idx]
            for k in range(len(arr)):
                orig = arr[k]
                arr[k] = orig + eps
                w1p, b1p, w2p, b2p = params.weights()
                loss_p = binary_cross_entropy(
                    forward_pass(x, w1p, b1p, w2p, b2p)[0],
                    y_true,
                )
                arr[k] = orig - eps
                w1m, b1m, w2m, b2m = params.weights()
                loss_m = binary_cross_entropy(
                    forward_pass(x, w1m, b1m, w2m, b2m)[0],
                    y_true,
                )
                arr[k] = orig
                numerical = (loss_p - loss_m) / (2 * eps)
                assert d_raw[k] == pytest.approx(numerical, abs=1e-4), (
                    f"d_{name}[{k}]: analytical={d_raw[k]:.6f}, numerical={numerical:.6f}"
                )


class TestMulticlassReLU:
    """Tests for multiclass ReLU training."""

    def test_gradient_fd_multiclass_backward(self) -> None:
        """Finite-difference check for backward_pass_multiclass."""
        from positroid.network.train import _cross_entropy

        rng = np.random.default_rng(42)
        batch, d, h, c = 8, 3, 5, 4
        x = rng.normal(0, 0.5, size=(batch, d))
        y = rng.integers(0, c, size=batch).astype(float)

        w1 = rng.normal(0, 0.3, (h, d))
        b1 = rng.normal(0, 0.1, h)
        w2 = rng.normal(0, 0.3, (c, h))
        b2 = rng.normal(0, 0.1, c)

        probs, z1_pre, z1_post, _ = forward_pass_multiclass(x, w1, b1, w2, b2)
        dw1, db1_g, dw2, db2_g = backward_pass_multiclass(x, y, probs, z1_pre, z1_post, w2)

        eps = 1e-6
        # Check all four parameter arrays
        for name, arr, grad in [
            ("w1", w1, dw1),
            ("b1", b1, db1_g),
            ("w2", w2, dw2),
            ("b2", b2, db2_g),
        ]:
            flat_arr = arr.ravel()
            flat_grad = grad.ravel()
            for i in range(min(5, len(flat_arr))):
                orig = flat_arr[i]
                flat_arr[i] = orig + eps
                probs_p, _, _, _ = forward_pass_multiclass(x, w1, b1, w2, b2)
                loss_p = _cross_entropy(probs_p, y)

                flat_arr[i] = orig - eps
                probs_m, _, _, _ = forward_pass_multiclass(x, w1, b1, w2, b2)
                loss_m = _cross_entropy(probs_m, y)

                flat_arr[i] = orig

                fd = (loss_p - loss_m) / (2 * eps)
                assert flat_grad[i] == pytest.approx(fd, rel=1e-3, abs=1e-6), (
                    f"{name}[{i}]: analytic={flat_grad[i]:.6f}, fd={fd:.6f}"
                )

    def test_multiclass_training_converges(self) -> None:
        """Multiclass ReLU should learn separable clusters."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, size=(150, 3))
        y = np.zeros(150)
        y[50:100] = 1
        y[100:] = 2
        x[:50] += np.array([3, 0, 0])
        x[50:100] += np.array([0, 3, 0])
        x[100:] += np.array([0, 0, 3])

        config = TrainConfig(
            hidden_dim=10,
            learning_rate=0.01,
            epochs=100,
            batch_size=32,
            seed=42,
        )
        _, history = train_multiclass(x, y, config, num_classes=3)

        assert history.losses[-1] < history.losses[0]
        assert history.accuracies[-1] > 0.8
