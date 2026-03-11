import numpy as np
import pytest

from positroid.datasets.toy2d import make_moons
from positroid.experiments.pruning import (
    PruningExperimentResult,
    PruningTrialResult,
    evaluate_network,
    identify_essential_and_tail,
    prune_direction_replacement,
    prune_full_removal,
    run_pruning_experiment,
    run_pruning_trial,
)
from positroid.network.relu_network import ReluNetwork
from positroid.network.train import TrainConfig, train


def _make_trained_net(
    hidden_dim: int = 10,
    param_mode: str = "tp_exponential",
    seed: int = 42,
) -> tuple[ReluNetwork, np.ndarray, np.ndarray]:
    """Train a small network and return (net, x, y)."""
    x, y = make_moons(n_samples=100, rng=np.random.default_rng(seed))
    config = TrainConfig(
        hidden_dim=hidden_dim,
        epochs=50,
        param_mode=param_mode,
        seed=seed,
    )
    net, _ = train(x, y, config)
    return net, x, y


# ---------------------------------------------------------------------------
# prune_full_removal
# ---------------------------------------------------------------------------


class TestPruneFullRemoval:
    def test_remove_nothing(self):
        """Removing no neurons returns a copy with identical shapes."""
        net, _, _ = _make_trained_net()
        pruned = prune_full_removal(net, [])
        assert pruned.layers[0].weight.shape == net.layers[0].weight.shape
        assert pruned.layers[0].bias.shape == net.layers[0].bias.shape
        assert pruned.layers[1].weight.shape == net.layers[1].weight.shape
        np.testing.assert_array_equal(pruned.layers[0].weight, net.layers[0].weight)
        np.testing.assert_array_equal(pruned.layers[0].bias, net.layers[0].bias)

    def test_remove_one(self):
        """Removing one neuron shrinks shapes correctly."""
        net, _, _ = _make_trained_net(hidden_dim=8)
        h = net.layers[0].weight.shape[0]
        pruned = prune_full_removal(net, [3])
        assert pruned.layers[0].weight.shape[0] == h - 1
        assert pruned.layers[0].bias.shape[0] == h - 1
        assert pruned.layers[1].weight.shape[1] == h - 1

    def test_remove_all(self):
        """Removing all neurons gives 0-width hidden layer."""
        net, _, _ = _make_trained_net(hidden_dim=6)
        h = net.layers[0].weight.shape[0]
        pruned = prune_full_removal(net, list(range(h)))
        assert pruned.layers[0].weight.shape[0] == 0
        assert pruned.layers[0].bias.shape[0] == 0
        assert pruned.layers[1].weight.shape[1] == 0

    def test_forward_after_removal(self):
        """Forward pass works after pruning (including edge case of all removed)."""
        net, x, y = _make_trained_net(hidden_dim=8)
        pruned = prune_full_removal(net, [0, 1])
        acc, preds = evaluate_network(pruned, x, y)
        assert 0.0 <= acc <= 1.0
        assert preds.shape == (x.shape[0],)

    def test_forward_all_removed(self):
        """Removing all neurons: output is constant sigmoid(b2)."""
        net, x, y = _make_trained_net(hidden_dim=6)
        h = net.layers[0].weight.shape[0]
        pruned = prune_full_removal(net, list(range(h)))
        acc, preds = evaluate_network(pruned, x, y)
        assert 0.0 <= acc <= 1.0
        # All predictions should be identical (constant output)
        assert np.allclose(preds, preds[0])


# ---------------------------------------------------------------------------
# prune_direction_replacement
# ---------------------------------------------------------------------------


class TestPruneDirectionReplacement:
    def test_replace_nothing(self):
        """Replacing no neurons returns identical weights."""
        net, _, _ = _make_trained_net()
        essential = list(range(3))
        pruned = prune_direction_replacement(net, [], essential)
        np.testing.assert_array_equal(pruned.layers[0].weight, net.layers[0].weight)
        np.testing.assert_array_equal(pruned.layers[0].bias, net.layers[0].bias)

    def test_preserves_biases(self):
        """Biases are unchanged after direction replacement."""
        net, _, _ = _make_trained_net(hidden_dim=8)
        pruned = prune_direction_replacement(net, [4, 5, 6, 7], [0, 1, 2])
        np.testing.assert_array_equal(pruned.layers[0].bias, net.layers[0].bias)
        np.testing.assert_array_equal(pruned.layers[1].weight, net.layers[1].weight)
        np.testing.assert_array_equal(pruned.layers[1].bias, net.layers[1].bias)

    def test_preserves_essential_rows(self):
        """Essential rows are unchanged."""
        net, _, _ = _make_trained_net(hidden_dim=8)
        essential = [0, 1, 2]
        pruned = prune_direction_replacement(net, [5, 6, 7], essential)
        for idx in essential:
            np.testing.assert_array_equal(
                pruned.layers[0].weight[idx],
                net.layers[0].weight[idx],
            )

    def test_replaced_rows_in_span(self):
        """Replaced rows should lie in the span of essential rows."""
        net, _, _ = _make_trained_net(hidden_dim=8)
        essential = [0, 1, 2]
        to_replace = [5, 6, 7]
        pruned = prune_direction_replacement(net, to_replace, essential)

        # Build basis from essential rows
        ess_rows = net.layers[0].weight[essential]
        u, s, vt = np.linalg.svd(ess_rows, full_matrices=False)
        tol = max(ess_rows.shape) * np.finfo(float).eps * s[0]
        r = int(np.sum(s > tol))
        basis = vt[:r]

        for idx in to_replace:
            row = pruned.layers[0].weight[idx]
            # Project and check it's unchanged
            projected = basis.T @ (basis @ row)
            np.testing.assert_allclose(row, projected, atol=1e-10)

    def test_shape_unchanged(self):
        """Network size is preserved after direction replacement."""
        net, _, _ = _make_trained_net(hidden_dim=8)
        pruned = prune_direction_replacement(net, [5, 6, 7], [0, 1, 2])
        assert pruned.layers[0].weight.shape == net.layers[0].weight.shape
        assert pruned.layers[0].bias.shape == net.layers[0].bias.shape
        assert pruned.layers[1].weight.shape == net.layers[1].weight.shape

    def test_empty_essential_raises(self):
        """Empty essential_indices raises ValueError."""
        net, _, _ = _make_trained_net(hidden_dim=6)
        with pytest.raises(ValueError, match="empty essential subspace"):
            prune_direction_replacement(net, [0, 1], [])


# ---------------------------------------------------------------------------
# identify_essential_and_tail
# ---------------------------------------------------------------------------


class TestIdentifyEssentialAndTail:
    def test_partitions_ground_set(self):
        """Essential and tail partition [0, H)."""
        net, _, _ = _make_trained_net(hidden_dim=10)
        essential, tail, k = identify_essential_and_tail(net)
        h = net.layers[0].weight.shape[0]
        assert sorted(essential + tail) == list(range(h))
        assert set(essential) & set(tail) == set()

    def test_sorted(self):
        """Both lists are sorted."""
        net, _, _ = _make_trained_net(hidden_dim=10)
        essential, tail, k = identify_essential_and_tail(net)
        assert essential == sorted(essential)
        assert tail == sorted(tail)

    def test_rank_positive(self):
        """Rank should be positive."""
        net, _, _ = _make_trained_net(hidden_dim=10)
        _, _, k = identify_essential_and_tail(net)
        assert k >= 1


# ---------------------------------------------------------------------------
# evaluate_network
# ---------------------------------------------------------------------------


class TestEvaluateNetwork:
    def test_accuracy_range(self):
        net, x, y = _make_trained_net()
        acc, preds = evaluate_network(net, x, y)
        assert 0.0 <= acc <= 1.0

    def test_prediction_shape(self):
        net, x, y = _make_trained_net()
        _, preds = evaluate_network(net, x, y)
        assert preds.shape == (x.shape[0],)


# ---------------------------------------------------------------------------
# run_pruning_trial
# ---------------------------------------------------------------------------


class TestRunPruningTrial:
    def test_valid_result(self):
        net, x, y = _make_trained_net(hidden_dim=10)
        result = run_pruning_trial(
            net,
            x,
            y,
            "moons",
            "tp_exponential",
            "full_removal",
            seed=42,
        )
        assert result is not None
        assert isinstance(result, PruningTrialResult)
        assert result.dataset_name == "moons"
        assert result.strategy == "full_removal"
        assert len(result.prune_curve) == 5  # default fractions

    def test_zero_prune_has_zero_delta(self):
        net, x, y = _make_trained_net(hidden_dim=10)
        result = run_pruning_trial(
            net,
            x,
            y,
            "moons",
            "tp_exponential",
            "full_removal",
            seed=42,
        )
        assert result is not None
        p0 = result.prune_curve[0]
        assert p0.prune_fraction == 0.0
        assert p0.accuracy_delta == 0.0
        assert p0.prediction_l2_dist == 0.0

    def test_direction_replacement_trial(self):
        net, x, y = _make_trained_net(hidden_dim=10)
        result = run_pruning_trial(
            net,
            x,
            y,
            "moons",
            "tp_exponential",
            "direction_replacement",
            seed=42,
        )
        assert result is not None
        assert result.strategy == "direction_replacement"
        assert len(result.prune_curve) > 0

    def test_random_removal_trial(self):
        net, x, y = _make_trained_net(hidden_dim=10)
        result = run_pruning_trial(
            net,
            x,
            y,
            "moons",
            "tp_exponential",
            "random_removal",
            seed=42,
            n_random_samples=5,
        )
        assert result is not None
        assert result.strategy == "random_removal"
        assert len(result.prune_curve) == 5  # default fractions

    def test_random_removal_zero_prune_has_zero_delta(self):
        net, x, y = _make_trained_net(hidden_dim=10)
        result = run_pruning_trial(
            net,
            x,
            y,
            "moons",
            "tp_exponential",
            "random_removal",
            seed=42,
            n_random_samples=5,
        )
        assert result is not None
        p0 = result.prune_curve[0]
        assert p0.prune_fraction == 0.0
        assert p0.accuracy_delta == 0.0


# ---------------------------------------------------------------------------
# New pruning strategies
# ---------------------------------------------------------------------------


class TestMagnitudePruning:
    def test_valid_result(self):
        net, x, y = _make_trained_net(hidden_dim=10)
        result = run_pruning_trial(
            net,
            x,
            y,
            "moons",
            "tp_exponential",
            "magnitude_pruning",
            seed=42,
        )
        assert result is not None
        assert result.strategy == "magnitude_pruning"
        assert len(result.prune_curve) == 5

    def test_zero_prune_has_zero_delta(self):
        net, x, y = _make_trained_net(hidden_dim=10)
        result = run_pruning_trial(
            net,
            x,
            y,
            "moons",
            "tp_exponential",
            "magnitude_pruning",
            seed=42,
        )
        assert result is not None
        p0 = result.prune_curve[0]
        assert p0.prune_fraction == 0.0
        assert p0.accuracy_delta == 0.0


class TestActivationPruning:
    def test_valid_result(self):
        net, x, y = _make_trained_net(hidden_dim=10)
        result = run_pruning_trial(
            net,
            x,
            y,
            "moons",
            "tp_exponential",
            "activation_pruning",
            seed=42,
        )
        assert result is not None
        assert result.strategy == "activation_pruning"
        assert len(result.prune_curve) == 5

    def test_zero_prune_has_zero_delta(self):
        net, x, y = _make_trained_net(hidden_dim=10)
        result = run_pruning_trial(
            net,
            x,
            y,
            "moons",
            "tp_exponential",
            "activation_pruning",
            seed=42,
        )
        assert result is not None
        p0 = result.prune_curve[0]
        assert p0.prune_fraction == 0.0
        assert p0.accuracy_delta == 0.0


class TestSensitivityPruning:
    def test_valid_result(self):
        net, x, y = _make_trained_net(hidden_dim=10)
        result = run_pruning_trial(
            net,
            x,
            y,
            "moons",
            "tp_exponential",
            "sensitivity_pruning",
            seed=42,
        )
        assert result is not None
        assert result.strategy == "sensitivity_pruning"
        assert len(result.prune_curve) == 5

    def test_zero_prune_has_zero_delta(self):
        net, x, y = _make_trained_net(hidden_dim=10)
        result = run_pruning_trial(
            net,
            x,
            y,
            "moons",
            "tp_exponential",
            "sensitivity_pruning",
            seed=42,
        )
        assert result is not None
        p0 = result.prune_curve[0]
        assert p0.prune_fraction == 0.0
        assert p0.accuracy_delta == 0.0


class TestNewStrategiesOnUnconstrained:
    def test_unconstrained_all_strategies(self):
        """All strategies work on unconstrained networks."""
        net, x, y = _make_trained_net(hidden_dim=10, param_mode="unconstrained")
        for strategy in [
            "full_removal",
            "magnitude_pruning",
            "activation_pruning",
            "sensitivity_pruning",
        ]:
            result = run_pruning_trial(
                net,
                x,
                y,
                "moons",
                "unconstrained",
                strategy,
                seed=42,
            )
            assert result is not None, f"{strategy} returned None"
            assert len(result.prune_curve) == 5


# ---------------------------------------------------------------------------
# run_pruning_experiment (smoke test)
# ---------------------------------------------------------------------------


class TestRunPruningExperiment:
    def test_smoke(self):
        """Smoke test: 2 trials, H=6, both strategies."""
        strategies = ["full_removal", "direction_replacement"]
        modes = ["tp_exponential"]
        result = run_pruning_experiment(
            dataset_name="moons",
            hidden_dim=6,
            num_trials=2,
            param_modes=modes,
            strategies=strategies,
            n_samples=50,
            epochs=20,
            seed=42,
        )
        assert isinstance(result, PruningExperimentResult)
        for mode in modes:
            for strategy in strategies:
                ts = result.trials(mode, strategy)
                assert len(ts) == 2
                for t in ts:
                    assert t.original_accuracy >= 0.0
                    assert len(t.prune_curve) > 0

    def test_random_vs_matroid(self):
        """Smoke test: random_removal and full_removal on same network."""
        strategies = ["full_removal", "random_removal"]
        modes = ["tp_exponential"]
        result = run_pruning_experiment(
            dataset_name="moons",
            hidden_dim=6,
            num_trials=2,
            param_modes=modes,
            strategies=strategies,
            n_samples=50,
            epochs=20,
            n_random_samples=5,
            seed=42,
        )
        for mode in modes:
            for strategy in strategies:
                ts = result.trials(mode, strategy)
                assert len(ts) == 2

    def test_both_modes(self):
        """Smoke test with both TP and non-TP modes."""
        modes = ["tp_exponential", "negated_bidiagonal"]
        result = run_pruning_experiment(
            dataset_name="moons",
            hidden_dim=6,
            num_trials=2,
            param_modes=modes,
            strategies=["full_removal"],
            n_samples=50,
            epochs=20,
            seed=42,
        )
        for mode in modes:
            ts = result.trials(mode, "full_removal")
            assert len(ts) == 2
