import numpy as np
import pytest

from positroid.datasets.toy2d import make_moons
from positroid.linalg.totally_positive import is_totally_positive
from positroid.network.train import (
    SGD,
    Adam,
    CauchyConstrainedParams,
    TPConstrainedParams,
    TrainConfig,
    backward_pass,
    binary_cross_entropy,
    forward_pass,
    sigmoid,
    train,
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
                    forward_pass(x, w1p, b1, w2, b2)[0], y_true,
                )
                loss_m = binary_cross_entropy(
                    forward_pass(x, w1m, b1, w2, b2)[0], y_true,
                )
                numerical = (loss_p - loss_m) / (2 * eps)
                assert dw1[i, j] == pytest.approx(numerical, abs=1e-4), (
                    f"dW1[{i},{j}]: analytical={dw1[i,j]:.6f}, numerical={numerical:.6f}"
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
                forward_pass(x, w1, b1, w2p, b2)[0], y_true,
            )
            loss_m = binary_cross_entropy(
                forward_pass(x, w1, b1, w2m, b2)[0], y_true,
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
                forward_pass(x, w1p, b1p, w2p, b2p)[0], y_true,
            )
            params.a_raw[k] = orig - eps
            w1m, b1m, w2m, b2m = params.weights()
            loss_m = binary_cross_entropy(
                forward_pass(x, w1m, b1m, w2m, b2m)[0], y_true,
            )
            params.a_raw[k] = orig
            numerical = (loss_p - loss_m) / (2 * eps)
            assert da_raw[k] == pytest.approx(numerical, abs=1e-4), (
                f"da_raw[{k}]: analytical={da_raw[k]:.6f}, numerical={numerical:.6f}"
            )


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
            hidden_dim=8, epochs=50, learning_rate=0.1,
            tp_constrained=True, seed=42,
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
            hidden_dim=8, epochs=50, learning_rate=0.1,
            tp_constrained=True, tp_kernel="cauchy", seed=42,
        )
        net, history = train(x, y, config)
        assert len(history.losses) == 50
        assert is_totally_positive(net.layers[0].weight)

    def test_sgd_optimizer(self):
        x, y = make_moons(n_samples=100, rng=np.random.default_rng(42))
        config = TrainConfig(
            hidden_dim=6, epochs=30, learning_rate=0.5,
            optimizer="sgd", seed=42,
        )
        _, history = train(x, y, config)
        assert history.losses[-1] < history.losses[0]
