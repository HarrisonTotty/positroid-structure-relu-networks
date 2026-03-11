"""Tests for the positroid network."""

import numpy as np
import pytest

from positroid.network.positroid_network import (
    PositroidNetwork,
    PositroidTrainConfig,
    _cross_entropy,
    _softmax,
    train_positroid,
)


class TestPositroidNetworkForward:
    """Tests for forward pass."""

    def test_forward_shape_k2(self) -> None:
        rng = np.random.default_rng(42)
        net = PositroidNetwork(k=2, n=6, input_dim=2, rng=rng)
        x = rng.normal(0, 1, size=(10, 2))
        logits, prod, enc_mat, bnd, weights = net.forward(x)
        assert logits.shape == (10,)
        assert prod.shape == (10, 2, 2)
        assert enc_mat.shape == (10, 6, 2)
        assert bnd.shape == (2, 6)
        assert weights.shape == (8,)

    def test_forward_shape_k3(self) -> None:
        rng = np.random.default_rng(42)
        net = PositroidNetwork(k=3, n=6, input_dim=2, rng=rng)
        x = rng.normal(0, 1, size=(10, 2))
        logits, prod, enc_mat, bnd, weights = net.forward(x)
        assert logits.shape == (10,)
        assert prod.shape == (10, 3, 3)
        assert enc_mat.shape == (10, 6, 3)
        assert bnd.shape == (3, 6)
        assert weights.shape == (9,)

    def test_predict_in_01(self) -> None:
        rng = np.random.default_rng(42)
        net = PositroidNetwork(k=2, n=6, input_dim=2, rng=rng)
        x = rng.normal(0, 1, size=(20, 2))
        preds = net.predict(x)
        assert np.all(preds >= 0) and np.all(preds <= 1)

    def test_encoding_matrix_structure(self) -> None:
        """First column of Z should be all ones."""
        rng = np.random.default_rng(42)
        net = PositroidNetwork(k=2, n=6, input_dim=2, rng=rng)
        x = rng.normal(0, 1, size=(5, 2))
        enc_mat = net._build_encoding_matrix(x)
        np.testing.assert_allclose(enc_mat[:, :, 0], 1.0)


class TestPositroidNetworkGradients:
    """Tests for gradient computation."""

    def test_gradient_finite_difference_fixed_k2(self) -> None:
        """Gradient check for fixed encoding, k=2."""
        rng = np.random.default_rng(42)
        net = PositroidNetwork(k=2, n=4, input_dim=2, encoding="fixed", rng=rng)
        # Use small x to avoid sigmoid saturation / high-curvature regions
        x = rng.normal(0, 0.5, size=(10, 2))
        y = (x[:, 0] > 0).astype(float)

        logits, prod, enc_mat, bnd, weights = net.forward(x)
        grads = net.compute_grads(x, y, logits, prod, enc_mat, bnd, weights)

        from positroid.network.positroid_network import _bce, _sigmoid

        eps = 1e-6
        for i in range(len(net.face_weights_raw)):
            net.face_weights_raw[i] += eps
            logits_p, *_ = net.forward(x)
            loss_p = _bce(_sigmoid(logits_p), y)

            net.face_weights_raw[i] -= 2 * eps
            logits_m, *_ = net.forward(x)
            loss_m = _bce(_sigmoid(logits_m), y)

            net.face_weights_raw[i] += eps  # restore

            fd = (loss_p - loss_m) / (2 * eps)
            assert grads[0][i] == pytest.approx(fd, rel=1e-3, abs=1e-6), (
                f"face_weight_raw[{i}]: analytic={grads[0][i]:.6f}, fd={fd:.6f}"
            )

    def test_gradient_finite_difference_learnable_k2(self) -> None:
        """Gradient check for learnable encoding, k=2."""
        rng = np.random.default_rng(42)
        net = PositroidNetwork(k=2, n=4, input_dim=2, encoding="learnable", rng=rng)
        x = rng.normal(0, 0.5, size=(10, 2))
        y = (x[:, 0] > 0).astype(float)

        logits, prod, enc_mat, bnd, weights = net.forward(x)
        grads = net.compute_grads(x, y, logits, prod, enc_mat, bnd, weights)

        from positroid.network.positroid_network import _bce, _sigmoid

        eps = 1e-6
        enc_flat = net.encoding_vectors.ravel()
        for i in range(min(6, len(enc_flat))):  # Check first 6 entries
            enc_flat[i] += eps
            net.encoding_vectors = enc_flat.reshape(net.encoding_vectors.shape)
            logits_p, *_ = net.forward(x)
            loss_p = _bce(_sigmoid(logits_p), y)

            enc_flat[i] -= 2 * eps
            net.encoding_vectors = enc_flat.reshape(net.encoding_vectors.shape)
            logits_m, *_ = net.forward(x)
            loss_m = _bce(_sigmoid(logits_m), y)

            enc_flat[i] += eps  # restore
            net.encoding_vectors = enc_flat.reshape(net.encoding_vectors.shape)

            fd = (loss_p - loss_m) / (2 * eps)
            analytic = grads[2].ravel()[i]
            assert analytic == pytest.approx(fd, rel=1e-3, abs=1e-6), (
                f"encoding[{i}]: analytic={analytic:.6f}, fd={fd:.6f}"
            )

    def test_gradient_finite_difference_k3(self) -> None:
        """Gradient check for k=3."""
        rng = np.random.default_rng(42)
        net = PositroidNetwork(k=3, n=5, input_dim=2, encoding="fixed", rng=rng)
        # Small x to keep logits in non-saturated regime
        x = rng.normal(0, 0.3, size=(5, 2))
        y = (x[:, 0] > 0).astype(float)

        logits, prod, enc_mat, bnd, weights = net.forward(x)
        grads = net.compute_grads(x, y, logits, prod, enc_mat, bnd, weights)

        from positroid.network.positroid_network import _bce, _sigmoid

        eps = 1e-6
        for i in range(len(net.face_weights_raw)):
            net.face_weights_raw[i] += eps
            logits_p, *_ = net.forward(x)
            loss_p = _bce(_sigmoid(logits_p), y)

            net.face_weights_raw[i] -= 2 * eps
            logits_m, *_ = net.forward(x)
            loss_m = _bce(_sigmoid(logits_m), y)

            net.face_weights_raw[i] += eps

            fd = (loss_p - loss_m) / (2 * eps)
            assert grads[0][i] == pytest.approx(fd, rel=1e-3, abs=1e-6), (
                f"face_weight_raw[{i}]: analytic={grads[0][i]:.6f}, fd={fd:.6f}"
            )


class TestPositroidNetworkParams:
    """Tests for parameter management."""

    def test_param_list_fixed(self) -> None:
        net = PositroidNetwork(k=2, n=6, input_dim=2, encoding="fixed")
        params = net.param_list()
        assert len(params) == 2  # face_weights_raw, output_bias

    def test_param_list_learnable(self) -> None:
        net = PositroidNetwork(k=2, n=6, input_dim=2, encoding="learnable")
        params = net.param_list()
        assert len(params) == 3  # face_weights_raw, output_bias, encoding_vectors

    def test_num_params_k2_n6_fixed(self) -> None:
        net = PositroidNetwork(k=2, n=6, input_dim=2, encoding="fixed")
        # 8 face weights + 1 bias = 9
        assert net.num_params == 9

    def test_num_params_k2_n6_learnable(self) -> None:
        net = PositroidNetwork(k=2, n=6, input_dim=2, encoding="learnable")
        # 8 face weights + 1 bias + 6*(1*2) encoding = 9 + 12 = 21
        assert net.num_params == 21

    def test_num_params_k3_n6_fixed(self) -> None:
        net = PositroidNetwork(k=3, n=6, input_dim=2, encoding="fixed")
        # 9 face weights + 1 bias = 10
        assert net.num_params == 10


class TestTrainPositroid:
    """Tests for the training function."""

    def test_loss_decreases_k2(self) -> None:
        """Loss should decrease over training."""
        from positroid.datasets.toy2d import make_moons

        rng = np.random.default_rng(42)
        x, y = make_moons(n_samples=100, rng=rng)

        config = PositroidTrainConfig(
            k=2, n=6, encoding="fixed", epochs=50, learning_rate=0.01, seed=42
        )
        _, history = train_positroid(x, y, config)

        # Loss should decrease (first 5 epochs avg vs last 5 epochs avg)
        early_loss = np.mean(history.losses[:5])
        late_loss = np.mean(history.losses[-5:])
        assert late_loss < early_loss, f"Loss did not decrease: {early_loss:.4f} -> {late_loss:.4f}"

    def test_loss_decreases_k3(self) -> None:
        """Loss should decrease for k=3 as well."""
        from positroid.datasets.toy2d import make_moons

        rng = np.random.default_rng(42)
        x, y = make_moons(n_samples=100, rng=rng)

        config = PositroidTrainConfig(
            k=3, n=6, encoding="fixed", epochs=50, learning_rate=0.01, seed=42
        )
        _, history = train_positroid(x, y, config)

        early_loss = np.mean(history.losses[:5])
        late_loss = np.mean(history.losses[-5:])
        assert late_loss < early_loss

    def test_learnable_encoding_trains(self) -> None:
        """Learnable encoding should also reduce loss."""
        from positroid.datasets.toy2d import make_moons

        rng = np.random.default_rng(42)
        x, y = make_moons(n_samples=100, rng=rng)

        config = PositroidTrainConfig(
            k=2, n=6, encoding="learnable", epochs=50, learning_rate=0.01, seed=42
        )
        _, history = train_positroid(x, y, config)

        early_loss = np.mean(history.losses[:5])
        late_loss = np.mean(history.losses[-5:])
        assert late_loss < early_loss

    def test_history_populated(self) -> None:
        """History should have entries for each epoch."""
        from positroid.datasets.toy2d import make_moons

        rng = np.random.default_rng(42)
        x, y = make_moons(n_samples=50, rng=rng)

        config = PositroidTrainConfig(k=2, n=4, epochs=10, seed=42)
        _, history = train_positroid(x, y, config)

        assert len(history.losses) == 10
        assert len(history.accuracies) == 10


class TestReadoutForward:
    """Forward pass tests for plucker_ratio and canonical_residue readouts."""

    def test_forward_shape_plucker_ratio(self) -> None:
        rng = np.random.default_rng(42)
        net = PositroidNetwork(k=2, n=6, input_dim=2, readout="plucker_ratio", rng=rng)
        x = rng.normal(0, 1, size=(10, 2))
        logits, prod, enc_mat, bnd, weights = net.forward(x)
        assert logits.shape == (10,)
        assert prod.shape == (10, 2, 2)

    def test_forward_shape_canonical_residue(self) -> None:
        rng = np.random.default_rng(42)
        net = PositroidNetwork(k=2, n=6, input_dim=2, readout="canonical_residue", rng=rng)
        x = rng.normal(0, 1, size=(10, 2))
        logits, prod, enc_mat, bnd, weights = net.forward(x)
        assert logits.shape == (10,)
        assert prod.shape == (10, 2, 2)

    def test_plucker_ratio_scales_logits(self) -> None:
        """Plucker ratio logits should differ from det logits by division by S."""
        rng = np.random.default_rng(42)
        net_det = PositroidNetwork(k=2, n=6, input_dim=2, readout="det", rng=rng)
        rng2 = np.random.default_rng(42)
        net_pr = PositroidNetwork(k=2, n=6, input_dim=2, readout="plucker_ratio", rng=rng2)
        x = rng.normal(0, 1, size=(5, 2))
        logits_det, *_ = net_det.forward(x)
        logits_pr, _, _, bnd, _ = net_pr.forward(x)
        # logits_pr = dets / S - bias, logits_det = dets - bias
        # With same init (bias=0), logits_pr = logits_det / S
        s_val = net_pr._cache_denom
        np.testing.assert_allclose(logits_pr, logits_det / s_val, rtol=1e-10)


class TestReadoutGradients:
    """Finite-difference gradient checks for new readout modes."""

    @staticmethod
    def _fd_check_face_weights(
        net: PositroidNetwork, x: np.ndarray, y: np.ndarray, eps: float = 1e-6
    ) -> None:
        """Check analytic vs FD gradient for face_weights_raw."""
        from positroid.network.positroid_network import _bce, _sigmoid

        logits, prod, enc_mat, bnd, weights = net.forward(x)
        grads = net.compute_grads(x, y, logits, prod, enc_mat, bnd, weights)

        for i in range(len(net.face_weights_raw)):
            net.face_weights_raw[i] += eps
            logits_p, *_ = net.forward(x)
            loss_p = _bce(_sigmoid(logits_p), y)

            net.face_weights_raw[i] -= 2 * eps
            logits_m, *_ = net.forward(x)
            loss_m = _bce(_sigmoid(logits_m), y)

            net.face_weights_raw[i] += eps  # restore

            fd = (loss_p - loss_m) / (2 * eps)
            assert grads[0][i] == pytest.approx(fd, rel=1e-3, abs=1e-6), (
                f"face_weight_raw[{i}]: analytic={grads[0][i]:.6f}, fd={fd:.6f}"
            )

    def test_gradient_fd_plucker_ratio_k2(self) -> None:
        rng = np.random.default_rng(42)
        net = PositroidNetwork(k=2, n=4, input_dim=2, readout="plucker_ratio", rng=rng)
        x = rng.normal(0, 0.5, size=(10, 2))
        y = (x[:, 0] > 0).astype(float)
        self._fd_check_face_weights(net, x, y)

    def test_gradient_fd_plucker_ratio_k3(self) -> None:
        rng = np.random.default_rng(42)
        net = PositroidNetwork(k=3, n=5, input_dim=2, readout="plucker_ratio", rng=rng)
        x = rng.normal(0, 0.3, size=(5, 2))
        y = (x[:, 0] > 0).astype(float)
        self._fd_check_face_weights(net, x, y)

    def test_gradient_fd_canonical_residue_k2(self) -> None:
        rng = np.random.default_rng(42)
        net = PositroidNetwork(k=2, n=4, input_dim=2, readout="canonical_residue", rng=rng)
        x = rng.normal(0, 0.5, size=(10, 2))
        y = (x[:, 0] > 0).astype(float)
        self._fd_check_face_weights(net, x, y)

    def test_gradient_fd_canonical_residue_k3(self) -> None:
        rng = np.random.default_rng(42)
        net = PositroidNetwork(k=3, n=5, input_dim=2, readout="canonical_residue", rng=rng)
        x = rng.normal(0, 0.3, size=(5, 2))
        y = (x[:, 0] > 0).astype(float)
        self._fd_check_face_weights(net, x, y)


class TestReadoutTraining:
    """Training tests for new readout modes."""

    def test_loss_decreases_plucker_ratio(self) -> None:
        from positroid.datasets.toy2d import make_moons

        rng = np.random.default_rng(42)
        x, y = make_moons(n_samples=100, rng=rng)

        config = PositroidTrainConfig(
            k=2, n=6, readout="plucker_ratio", epochs=50, learning_rate=0.01, seed=42
        )
        _, history = train_positroid(x, y, config)

        early_loss = np.mean(history.losses[:5])
        late_loss = np.mean(history.losses[-5:])
        assert late_loss < early_loss, f"Loss did not decrease: {early_loss:.4f} -> {late_loss:.4f}"

    def test_loss_decreases_canonical_residue(self) -> None:
        from positroid.datasets.toy2d import make_moons

        rng = np.random.default_rng(42)
        x, y = make_moons(n_samples=100, rng=rng)

        config = PositroidTrainConfig(
            k=2,
            n=6,
            readout="canonical_residue",
            epochs=50,
            learning_rate=0.01,
            seed=42,
        )
        _, history = train_positroid(x, y, config)

        early_loss = np.mean(history.losses[:5])
        late_loss = np.mean(history.losses[-5:])
        assert late_loss < early_loss, f"Loss did not decrease: {early_loss:.4f} -> {late_loss:.4f}"


class TestMulticlassForward:
    """Tests for multiclass forward pass."""

    def test_forward_shape_multiclass_k2(self) -> None:
        rng = np.random.default_rng(42)
        net = PositroidNetwork(k=2, n=8, input_dim=5, num_classes=5, rng=rng)
        x = rng.normal(0, 1, size=(10, 5))
        logits, prod, enc_mat, bnd, weights = net.forward(x)
        assert logits.shape == (10, 5)
        assert prod.shape == (5, 10, 2, 2)
        assert enc_mat.shape == (10, 8, 2)
        assert bnd.shape == (5, 2, 8)
        assert weights.shape == (5, 12)  # k*(n-k) = 2*6 = 12

    def test_forward_shape_multiclass_k3(self) -> None:
        rng = np.random.default_rng(42)
        net = PositroidNetwork(k=3, n=6, input_dim=3, num_classes=4, rng=rng)
        x = rng.normal(0, 1, size=(8, 3))
        logits, prod, enc_mat, bnd, weights = net.forward(x)
        assert logits.shape == (8, 4)
        assert prod.shape == (4, 8, 3, 3)
        assert enc_mat.shape == (8, 6, 3)
        assert bnd.shape == (4, 3, 6)
        assert weights.shape == (4, 9)

    def test_predict_multiclass_sums_to_one(self) -> None:
        rng = np.random.default_rng(42)
        net = PositroidNetwork(k=2, n=6, input_dim=3, num_classes=5, rng=rng)
        x = rng.normal(0, 1, size=(10, 3))
        probs = net.predict(x)
        assert probs.shape == (10, 5)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-6)

    def test_num_params_multiclass(self) -> None:
        net = PositroidNetwork(k=2, n=8, input_dim=5, num_classes=10, encoding="fixed")
        # face_weights: 10 * (2*6) = 120, bias: 10 = 130 total
        assert net.num_params == 130

    def test_num_params_multiclass_learnable(self) -> None:
        net = PositroidNetwork(k=2, n=8, input_dim=5, num_classes=10, encoding="learnable")
        # face_weights: 120, bias: 10, encoding: 8*1*5 = 40
        assert net.num_params == 170

    def test_binary_backward_compat(self) -> None:
        """Default num_classes=2 should produce same shapes as original."""
        rng = np.random.default_rng(42)
        net = PositroidNetwork(k=2, n=6, input_dim=2, rng=rng)
        assert net.num_classes == 2
        x = rng.normal(0, 1, size=(5, 2))
        logits, prod, enc_mat, bnd, weights = net.forward(x)
        assert logits.shape == (5,)
        assert prod.shape == (5, 2, 2)
        assert bnd.shape == (2, 6)
        assert weights.shape == (8,)


class TestMulticlassGradients:
    """Finite-difference gradient checks for multiclass."""

    def test_gradient_fd_multiclass_k2(self) -> None:
        rng = np.random.default_rng(42)
        net = PositroidNetwork(k=2, n=6, input_dim=3, num_classes=4, encoding="fixed", rng=rng)
        x = rng.normal(0, 0.3, size=(8, 3))
        y = rng.integers(0, 4, size=8).astype(float)

        logits, prod, enc_mat, bnd, weights = net.forward(x)
        grads = net.compute_grads(x, y, logits, prod, enc_mat, bnd, weights)

        eps = 1e-6
        # Check a subset of face weight gradients (flat over all classes)
        flat_raw = net.face_weights_raw.ravel()
        flat_grad = grads[0].ravel()
        for i in range(min(10, len(flat_raw))):
            flat_raw[i] += eps
            net.face_weights_raw = flat_raw.reshape(net.face_weights_raw.shape)
            logits_p, *_ = net.forward(x)
            loss_p = _cross_entropy(_softmax(logits_p), y)

            flat_raw[i] -= 2 * eps
            net.face_weights_raw = flat_raw.reshape(net.face_weights_raw.shape)
            logits_m, *_ = net.forward(x)
            loss_m = _cross_entropy(_softmax(logits_m), y)

            flat_raw[i] += eps
            net.face_weights_raw = flat_raw.reshape(net.face_weights_raw.shape)

            fd = (loss_p - loss_m) / (2 * eps)
            assert flat_grad[i] == pytest.approx(fd, rel=1e-3, abs=1e-6), (
                f"face_weight_raw[{i}]: analytic={flat_grad[i]:.6f}, fd={fd:.6f}"
            )

    def test_gradient_fd_multiclass_k3(self) -> None:
        rng = np.random.default_rng(42)
        net = PositroidNetwork(k=3, n=5, input_dim=2, num_classes=3, encoding="fixed", rng=rng)
        x = rng.normal(0, 0.2, size=(5, 2))
        y = rng.integers(0, 3, size=5).astype(float)

        logits, prod, enc_mat, bnd, weights = net.forward(x)
        grads = net.compute_grads(x, y, logits, prod, enc_mat, bnd, weights)

        eps = 1e-6
        flat_raw = net.face_weights_raw.ravel()
        flat_grad = grads[0].ravel()
        for i in range(min(10, len(flat_raw))):
            flat_raw[i] += eps
            net.face_weights_raw = flat_raw.reshape(net.face_weights_raw.shape)
            logits_p, *_ = net.forward(x)
            loss_p = _cross_entropy(_softmax(logits_p), y)

            flat_raw[i] -= 2 * eps
            net.face_weights_raw = flat_raw.reshape(net.face_weights_raw.shape)
            logits_m, *_ = net.forward(x)
            loss_m = _cross_entropy(_softmax(logits_m), y)

            flat_raw[i] += eps
            net.face_weights_raw = flat_raw.reshape(net.face_weights_raw.shape)

            fd = (loss_p - loss_m) / (2 * eps)
            assert flat_grad[i] == pytest.approx(fd, rel=1e-3, abs=1e-6), (
                f"face_weight_raw[{i}]: analytic={flat_grad[i]:.6f}, fd={fd:.6f}"
            )

    def test_gradient_fd_multiclass_learnable(self) -> None:
        rng = np.random.default_rng(42)
        net = PositroidNetwork(k=2, n=5, input_dim=3, num_classes=3, encoding="learnable", rng=rng)
        x = rng.normal(0, 0.3, size=(6, 3))
        y = rng.integers(0, 3, size=6).astype(float)

        logits, prod, enc_mat, bnd, weights = net.forward(x)
        grads = net.compute_grads(x, y, logits, prod, enc_mat, bnd, weights)

        eps = 1e-6
        enc_flat = net.encoding_vectors.ravel()
        enc_grad = grads[2].ravel()
        for i in range(min(6, len(enc_flat))):
            enc_flat[i] += eps
            net.encoding_vectors = enc_flat.reshape(net.encoding_vectors.shape)
            logits_p, *_ = net.forward(x)
            loss_p = _cross_entropy(_softmax(logits_p), y)

            enc_flat[i] -= 2 * eps
            net.encoding_vectors = enc_flat.reshape(net.encoding_vectors.shape)
            logits_m, *_ = net.forward(x)
            loss_m = _cross_entropy(_softmax(logits_m), y)

            enc_flat[i] += eps
            net.encoding_vectors = enc_flat.reshape(net.encoding_vectors.shape)

            fd = (loss_p - loss_m) / (2 * eps)
            assert enc_grad[i] == pytest.approx(fd, rel=1e-3, abs=1e-6), (
                f"encoding[{i}]: analytic={enc_grad[i]:.6f}, fd={fd:.6f}"
            )

    def test_gradient_fd_multiclass_bias(self) -> None:
        """Check bias gradients for multiclass."""
        rng = np.random.default_rng(42)
        net = PositroidNetwork(k=2, n=5, input_dim=3, num_classes=4, encoding="fixed", rng=rng)
        x = rng.normal(0, 0.3, size=(8, 3))
        y = rng.integers(0, 4, size=8).astype(float)

        logits, prod, enc_mat, bnd, weights = net.forward(x)
        grads = net.compute_grads(x, y, logits, prod, enc_mat, bnd, weights)

        eps = 1e-6
        for i in range(4):
            net.output_bias[i] += eps
            logits_p, *_ = net.forward(x)
            loss_p = _cross_entropy(_softmax(logits_p), y)

            net.output_bias[i] -= 2 * eps
            logits_m, *_ = net.forward(x)
            loss_m = _cross_entropy(_softmax(logits_m), y)

            net.output_bias[i] += eps

            fd = (loss_p - loss_m) / (2 * eps)
            assert grads[1][i] == pytest.approx(fd, rel=1e-3, abs=1e-6), (
                f"bias[{i}]: analytic={grads[1][i]:.6f}, fd={fd:.6f}"
            )


class TestMulticlassTraining:
    """Training smoke tests for multiclass."""

    def test_loss_decreases_multiclass(self) -> None:
        rng = np.random.default_rng(42)
        # Simple 3-class synthetic data
        x = rng.normal(0, 1, size=(150, 3))
        y = np.zeros(150)
        y[50:100] = 1
        y[100:] = 2
        # Shift class centers for separability
        x[:50] += np.array([2, 0, 0])
        x[50:100] += np.array([0, 2, 0])
        x[100:] += np.array([0, 0, 2])

        config = PositroidTrainConfig(
            k=2, n=8, encoding="fixed", num_classes=3, epochs=50, learning_rate=0.01, seed=42
        )
        _, history = train_positroid(x, y, config)

        early_loss = np.mean(history.losses[:5])
        late_loss = np.mean(history.losses[-5:])
        assert late_loss < early_loss, f"Loss did not decrease: {early_loss:.4f} -> {late_loss:.4f}"

    def test_accuracy_above_random_multiclass(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, size=(150, 3))
        y = np.zeros(150)
        y[50:100] = 1
        y[100:] = 2
        x[:50] += np.array([3, 0, 0])
        x[50:100] += np.array([0, 3, 0])
        x[100:] += np.array([0, 0, 3])

        config = PositroidTrainConfig(
            k=2, n=8, encoding="fixed", num_classes=3, epochs=100, learning_rate=0.01, seed=42
        )
        _, history = train_positroid(x, y, config)

        # With well-separated classes, should beat random (33%)
        assert history.accuracies[-1] > 0.5
