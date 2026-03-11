"""Tests for multi-layer positroid experiment."""

from __future__ import annotations

import numpy as np

from positroid.datasets.toy2d import make_moons
from positroid.experiments.multilayer import (
    activation_pattern_matroid,
    compute_effective_matrix,
    is_activation_contiguous,
    region_adjacency_graph,
    run_multilayer_experiment,
    sample_activation_regions,
)
from positroid.network.relu_network import ReluLayer, ReluNetwork
from positroid.network.train_multilayer import (
    MultiLayerTPExponentialParams,
    MultiLayerTrainConfig,
    MultiLayerUnconstrainedParams,
    multilayer_forward,
    train_multilayer,
)

# ── Forward Pass Tests ──


class TestMultilayerForward:
    def test_output_shapes(self):
        """Correct output shapes for [d=2, H1=4, H2=3, 1]."""
        rng = np.random.default_rng(0)
        w0 = rng.normal(size=(4, 2))
        b0 = np.zeros(4)
        w1 = rng.normal(size=(3, 4))
        b1 = np.zeros(3)
        w2 = rng.normal(size=(1, 3))
        b2 = np.zeros(1)
        weights = [(w0, b0), (w1, b1), (w2, b2)]

        x = rng.normal(size=(10, 2))
        y_pred, pre_acts, post_acts = multilayer_forward(x, weights)

        assert y_pred.shape == (10, 1)
        assert len(pre_acts) == 3
        assert len(post_acts) == 3
        assert pre_acts[0].shape == (10, 4)
        assert pre_acts[1].shape == (10, 3)
        assert pre_acts[2].shape == (10, 1)

    def test_relu_on_hidden(self):
        """Hidden layers have ReLU applied (no negative post-activations)."""
        rng = np.random.default_rng(1)
        weights = [
            (rng.normal(size=(4, 2)), np.zeros(4)),
            (rng.normal(size=(3, 4)), np.zeros(3)),
            (rng.normal(size=(1, 3)), np.zeros(1)),
        ]
        x = rng.normal(size=(20, 2))
        _, _, post_acts = multilayer_forward(x, weights)

        # Hidden layers: no negative values
        assert np.all(post_acts[0] >= 0)
        assert np.all(post_acts[1] >= 0)

    def test_sigmoid_on_output(self):
        """Output layer has sigmoid applied (values in (0, 1))."""
        rng = np.random.default_rng(2)
        weights = [
            (rng.normal(size=(4, 2)), np.zeros(4)),
            (rng.normal(size=(1, 4)), np.zeros(1)),
        ]
        x = rng.normal(size=(20, 2))
        y_pred, _, _ = multilayer_forward(x, weights)

        assert np.all(y_pred > 0)
        assert np.all(y_pred < 1)


# ── Backward Pass Tests ──


class TestMultilayerBackward:
    def test_gradient_check_unconstrained(self):
        """Numerical gradient check for unconstrained multi-layer."""
        rng = np.random.default_rng(42)
        x = rng.normal(size=(8, 2))
        y = rng.integers(0, 2, size=8).astype(float)

        params = MultiLayerUnconstrainedParams(2, [4, 3], rng)
        all_weights = params.weights()
        y_pred, pre_acts, post_acts = multilayer_forward(x, all_weights)
        grads = params.compute_grads(x, y, y_pred, pre_acts, post_acts)

        from positroid.network.train import binary_cross_entropy

        eps = 1e-5
        param_list = params.param_list()
        for p_idx, (param, grad) in enumerate(zip(param_list, grads, strict=True)):
            flat_param = param.ravel()
            flat_grad = grad.ravel()
            # Check a few random indices
            check_indices = rng.choice(len(flat_param), min(3, len(flat_param)), replace=False)
            for idx in check_indices:
                old_val = flat_param[idx]

                flat_param[idx] = old_val + eps
                y_plus, _, _ = multilayer_forward(x, params.weights())
                loss_plus = binary_cross_entropy(y_plus, y)

                flat_param[idx] = old_val - eps
                y_minus, _, _ = multilayer_forward(x, params.weights())
                loss_minus = binary_cross_entropy(y_minus, y)

                flat_param[idx] = old_val
                numerical = (loss_plus - loss_minus) / (2 * eps)
                np.testing.assert_allclose(
                    flat_grad[idx],
                    numerical,
                    atol=1e-4,
                    rtol=1e-3,
                    err_msg=f"Gradient mismatch at param {p_idx}, index {idx}",
                )

    def test_gradient_check_tp_exponential(self):
        """Numerical gradient check for TP exponential multi-layer.

        Key constraints for accurate finite-difference checking:
        1. All pre-activations far from zero (avoid ReLU non-differentiability)
        2. Output logits moderate (avoid sigmoid saturation in BCE clipping)
        Uses positive inputs + biases for (1), tiny output weights for (2).
        """
        rng = np.random.default_rng(42)
        # Positive inputs: ensures x @ W.T > 0 when W > 0 (exp kernel)
        x = np.abs(rng.standard_normal((5, 2))) + 0.1
        y = rng.integers(0, 2, size=5).astype(float)

        params = MultiLayerTPExponentialParams(2, [4, 3], rng)
        for a_raw in params._a_raws:
            a_raw[:] = rng.uniform(0.01, 0.05, size=len(a_raw))
        for b_raw in params._b_raws:
            b_raw[:] = rng.uniform(0.01, 0.05, size=len(b_raw))
        for bias in params._biases:
            bias[:] = 0.5
        # Tiny output weights to prevent sigmoid saturation
        params._w_out[:] = rng.uniform(-0.001, 0.001, size=params._w_out.shape)
        params._b_out[:] = 0.0

        from positroid.network.train import binary_cross_entropy

        eps = 1e-5
        param_list = params.param_list()

        all_weights = params.weights()
        y_pred, pre_acts, post_acts = multilayer_forward(x, all_weights)

        grads = params.compute_grads(x, y, y_pred, pre_acts, post_acts)

        for p_idx, (param, grad) in enumerate(zip(param_list, grads, strict=True)):
            flat_param = param.ravel()
            flat_grad = grad.ravel()
            for idx in range(len(flat_param)):
                old_val = flat_param[idx]

                flat_param[idx] = old_val + eps
                y_plus, _, _ = multilayer_forward(x, params.weights())
                loss_plus = binary_cross_entropy(y_plus, y)

                flat_param[idx] = old_val - eps
                y_minus, _, _ = multilayer_forward(x, params.weights())
                loss_minus = binary_cross_entropy(y_minus, y)

                flat_param[idx] = old_val
                numerical = (loss_plus - loss_minus) / (2 * eps)
                np.testing.assert_allclose(
                    flat_grad[idx],
                    numerical,
                    atol=1e-4,
                    rtol=1e-3,
                    err_msg=f"Gradient mismatch at param {p_idx}, index {idx}",
                )


# ── Training Tests ──


class TestTrainMultilayer:
    def test_loss_decreases_unconstrained(self):
        """Loss decreases over 50 epochs (unconstrained)."""
        x, y = make_moons(n_samples=100, rng=np.random.default_rng(0))
        config = MultiLayerTrainConfig(
            layer_dims=[6, 4],
            epochs=50,
            param_mode="unconstrained",
            seed=42,
        )
        net, history = train_multilayer(x, y, config)
        assert history.losses[-1] < history.losses[0]
        assert net.num_layers == 3  # 2 hidden + 1 output

    def test_loss_decreases_tp(self):
        """Loss decreases over 50 epochs (tp_exponential)."""
        x, y = make_moons(n_samples=100, rng=np.random.default_rng(0))
        config = MultiLayerTrainConfig(
            layer_dims=[6, 4],
            epochs=50,
            param_mode="tp_exponential",
            seed=42,
        )
        net, history = train_multilayer(x, y, config)
        assert history.losses[-1] < history.losses[0]
        assert net.num_layers == 3

    def test_network_structure(self):
        """Network has correct layer dimensions."""
        x, y = make_moons(n_samples=50, rng=np.random.default_rng(0))
        config = MultiLayerTrainConfig(
            layer_dims=[8, 6],
            epochs=5,
            param_mode="unconstrained",
            seed=0,
        )
        net, _ = train_multilayer(x, y, config)
        assert net.input_dim == 2
        assert net.hidden_dims == [8, 6]
        assert net.output_dim == 1
        assert net.layers[0].weight.shape == (8, 2)
        assert net.layers[1].weight.shape == (6, 8)
        assert net.layers[2].weight.shape == (1, 6)


# ── Effective Matrix Tests ──


class TestComputeEffectiveMatrix:
    def _make_net(self) -> ReluNetwork:
        """2 hidden layers: [2, 4, 3, 1]."""
        rng = np.random.default_rng(42)
        return ReluNetwork(
            [
                ReluLayer(rng.normal(size=(4, 2)), rng.normal(size=4)),
                ReluLayer(rng.normal(size=(3, 4)), rng.normal(size=3)),
                ReluLayer(rng.normal(size=(1, 3)), rng.normal(size=1)),
            ]
        )

    def test_all_active(self):
        """All-active pattern: W_eff = W2 @ W1."""
        net = self._make_net()
        pattern = np.ones(4, dtype=bool)
        aug = compute_effective_matrix(net, pattern)
        assert aug.shape == (3, 3)  # H2=3, d+1=3

        # Verify against direct computation (before normalization)
        w1 = net.layers[0].weight
        b1 = net.layers[0].bias
        w2 = net.layers[1].weight
        b2 = net.layers[1].bias
        w_eff_expected = w2 @ w1
        b_eff_expected = w2 @ b1 + b2
        aug_expected = np.hstack([w_eff_expected, b_eff_expected.reshape(-1, 1)])

        # Rank preserved by normalization
        assert int(np.linalg.matrix_rank(aug)) == int(np.linalg.matrix_rank(aug_expected))

        # Row spaces match: each row should be proportional to expected
        for row_idx in range(3):
            expected_norm = np.linalg.norm(aug_expected[row_idx])
            if expected_norm > 1e-10:
                expected_dir = aug_expected[row_idx] / expected_norm
                # Normalized row should match direction (up to sign)
                dot = abs(np.dot(aug[row_idx], expected_dir))
                np.testing.assert_allclose(dot, 1.0, atol=1e-10)

    def test_all_inactive(self):
        """All-inactive pattern: W_eff = 0, b_eff = b2."""
        net = self._make_net()
        pattern = np.zeros(4, dtype=bool)
        aug = compute_effective_matrix(net, pattern)
        assert aug.shape == (3, 3)

        # After normalization, weight columns should be zero and
        # bias column should be ±1 (all rows point along bias axis)
        np.testing.assert_allclose(aug[:, :2], 0.0, atol=1e-12)
        assert np.all(np.abs(aug[:, 2]) > 0.99)  # normalized bias ≈ ±1

    def test_partial_pattern(self):
        """Partial activation pattern gives correct shape."""
        net = self._make_net()
        pattern = np.array([1, 0, 1, 0], dtype=bool)
        aug = compute_effective_matrix(net, pattern)
        assert aug.shape == (3, 3)


# ── Activation Contiguity Tests ──


class TestIsActivationContiguous:
    def test_contiguous_block(self):
        assert is_activation_contiguous((0, 0, 1, 1, 1, 0, 0)) is True

    def test_non_contiguous(self):
        assert is_activation_contiguous((0, 1, 0, 1, 0)) is False

    def test_all_active(self):
        assert is_activation_contiguous((1, 1, 1, 1)) is True

    def test_all_inactive(self):
        assert is_activation_contiguous((0, 0, 0, 0)) is True

    def test_single_active(self):
        assert is_activation_contiguous((0, 0, 1, 0)) is True

    def test_two_separate(self):
        assert is_activation_contiguous((1, 0, 0, 1)) is False


# ── Activation Region Sampling Tests ──


class TestSampleActivationRegions:
    def test_basic(self):
        """Regions are non-empty and patterns have correct length."""
        x, y = make_moons(n_samples=50, rng=np.random.default_rng(0))
        config = MultiLayerTrainConfig(
            layer_dims=[4, 3],
            epochs=10,
            param_mode="unconstrained",
            seed=0,
        )
        net, _ = train_multilayer(x, y, config)
        regions = sample_activation_regions(net, x)
        assert len(regions) > 0
        for pattern, indices in regions.items():
            assert len(pattern) == 4  # H1 = 4
            assert len(indices) > 0
            assert all(0 <= idx < 50 for idx in indices)


# ── Experiment Smoke Test ──


class TestRunMultilayerExperiment:
    def test_smoke(self):
        """Smoke test: small experiment completes and has correct structure."""
        result = run_multilayer_experiment(
            dataset_name="moons",
            layer_dims=[4, 3],
            num_trials=2,
            param_modes=["tp_exponential", "unconstrained"],
            n_samples=50,
            epochs=20,
            seed=42,
        )

        assert "tp_exponential" in result.trials_by_mode
        assert "unconstrained" in result.trials_by_mode
        assert len(result.trials_by_mode["tp_exponential"]) == 2
        assert len(result.trials_by_mode["unconstrained"]) == 2

        for mode, trials in result.trials_by_mode.items():
            for trial in trials:
                assert trial.dataset_name == "moons"
                assert trial.layer_dims == [4, 3]
                assert trial.param_mode == mode
                assert 0.0 <= trial.final_accuracy <= 1.0
                assert trial.n_activation_regions > 0
                assert len(trial.effective_results) > 0

    def test_pattern_matroid_populated(self):
        """Pattern matroid fields are populated in experiment results."""
        result = run_multilayer_experiment(
            dataset_name="moons",
            layer_dims=[4, 3],
            num_trials=1,
            param_modes=["unconstrained"],
            n_samples=50,
            epochs=20,
            seed=42,
        )
        trial = result.trials_by_mode["unconstrained"][0]
        # Should have computed pattern matroid
        assert trial.pattern_matroid_rank > 0
        assert trial.n_adjacent_pairs >= 0
        # is_positroid should be bool (not None) for small experiments
        assert trial.pattern_matroid_is_positroid is not None


# ── Activation Pattern Matroid Tests ──


class TestActivationPatternMatroid:
    def test_basic(self):
        """Column matroid of known binary patterns."""
        regions: dict[tuple[int, ...], list[int]] = {
            (1, 0, 1): [0],
            (0, 1, 1): [1],
            (1, 1, 0): [2],
        }
        mat = activation_pattern_matroid(regions)
        assert mat is not None
        # 3 patterns, 3 neurons → 3×3 binary matrix
        # Rank should be at most 3
        assert mat.rank <= 3
        assert mat.rank >= 1

    def test_returns_none_for_single_pattern(self):
        regions: dict[tuple[int, ...], list[int]] = {
            (1, 0, 1): [0, 1, 2],
        }
        mat = activation_pattern_matroid(regions)
        assert mat is None

    def test_dependent_neurons(self):
        """Neurons with identical activation vectors are dependent."""
        regions: dict[tuple[int, ...], list[int]] = {
            (1, 0, 1): [0],
            (1, 1, 1): [1],
        }
        mat = activation_pattern_matroid(regions)
        assert mat is not None
        # Neuron 0=[1,1], neuron 1=[0,1], neuron 2=[1,1] in R^2
        # Neurons 0 and 2 are parallel → dependent. Rank = 2.
        assert mat.rank == 2
        # {0, 2} should not be a basis (parallel vectors)
        assert frozenset({0, 2}) not in mat.bases


# ── Region Adjacency Graph Tests ──


class TestRegionAdjacencyGraph:
    def test_basic_adjacency(self):
        """Patterns at Hamming distance 1 are adjacent."""
        regions: dict[tuple[int, ...], list[int]] = {
            (0, 0): [0],
            (1, 0): [1],
            (0, 1): [2],
            (1, 1): [3],
        }
        adj, n_edges = region_adjacency_graph(regions)
        # (0,0)↔(1,0), (0,0)↔(0,1), (1,0)↔(1,1), (0,1)↔(1,1)
        assert n_edges == 4
        assert (1, 0) in adj[(0, 0)]
        assert (0, 1) in adj[(0, 0)]
        assert (1, 1) not in adj[(0, 0)]  # Hamming distance 2

    def test_no_adjacency(self):
        """Patterns far apart have no edges."""
        regions: dict[tuple[int, ...], list[int]] = {
            (0, 0, 0): [0],
            (1, 1, 1): [1],
        }
        _, n_edges = region_adjacency_graph(regions)
        assert n_edges == 0
