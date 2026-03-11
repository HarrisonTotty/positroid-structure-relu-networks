import numpy as np

from positroid.datasets.toy2d import make_moons
from positroid.experiments.non_tp_baseline import (
    BaselineTrialResult,
    analyze_trial,
    run_baseline_experiment,
)
from positroid.linalg.totally_positive import is_totally_positive
from positroid.network.train import TrainConfig, train


class TestAnalyzeTrial:
    def test_returns_valid_result(self):
        x, y = make_moons(n_samples=100, rng=np.random.default_rng(42))
        config = TrainConfig(hidden_dim=6, epochs=30, param_mode="sinusoidal", seed=42)
        net, history = train(x, y, config)

        result = analyze_trial(
            net,
            "moons",
            param_mode="sinusoidal",
            final_loss=history.losses[-1],
            final_accuracy=history.accuracies[-1],
        )
        assert result is not None
        assert isinstance(result, BaselineTrialResult)
        assert result.dataset_name == "moons"
        assert result.param_mode == "sinusoidal"
        assert result.affine_matroid_rank >= 1
        assert isinstance(result.non_bases, list)
        assert isinstance(result.has_non_interval_nonbases, bool)


class TestRunBaselineExperiment:
    def test_small_experiment_runs(self):
        """Smoke test: 2 trials, 3 modes, moons, H=6, 20 epochs."""
        modes = ["tp_exponential", "sinusoidal", "quadratic_distance"]
        result = run_baseline_experiment(
            dataset_name="moons",
            hidden_dim=6,
            num_trials=2,
            param_modes=modes,
            n_samples=50,
            epochs=20,
            seed=42,
        )
        for mode in modes:
            assert len(result.trials_by_mode[mode]) == 2
            assert 0.0 <= result.positroid_rate(mode) <= 1.0
            assert 0.0 <= result.uniform_rate(mode) <= 1.0
            assert 0.0 <= result.non_interval_rate(mode) <= 1.0


class TestNewModesSmoke:
    def test_permuted_exponential_experiment(self):
        """Smoke test: permuted_exponential mode runs through experiment pipeline."""
        result = run_baseline_experiment(
            dataset_name="moons",
            hidden_dim=6,
            num_trials=2,
            param_modes=["permuted_exponential"],
            n_samples=50,
            epochs=20,
            seed=42,
        )
        assert len(result.trials_by_mode["permuted_exponential"]) == 2

    def test_negated_bidiagonal_experiment(self):
        """Smoke test: negated_bidiagonal mode runs through experiment pipeline."""
        result = run_baseline_experiment(
            dataset_name="moons",
            hidden_dim=6,
            num_trials=2,
            param_modes=["negated_bidiagonal"],
            n_samples=50,
            epochs=20,
            seed=42,
        )
        assert len(result.trials_by_mode["negated_bidiagonal"]) == 2

    def test_fixed_convergent_bias_only_experiment(self):
        """Smoke test: fixed_convergent_bias_only mode runs through pipeline."""
        result = run_baseline_experiment(
            dataset_name="moons",
            hidden_dim=6,
            num_trials=2,
            param_modes=["fixed_convergent_bias_only"],
            n_samples=50,
            epochs=20,
            seed=42,
        )
        assert len(result.trials_by_mode["fixed_convergent_bias_only"]) == 2


class TestNonTPMatrixIsNotTP:
    def test_sinusoidal_not_tp(self):
        """Sinusoidal kernel should produce non-TP weight matrices."""
        x, y = make_moons(n_samples=100, rng=np.random.default_rng(42))
        config = TrainConfig(hidden_dim=6, epochs=10, param_mode="sinusoidal", seed=42)
        net, _ = train(x, y, config)
        assert not is_totally_positive(net.layers[0].weight)

    def test_quadratic_distance_not_tp(self):
        """Quadratic distance kernel should produce non-TP weight matrices."""
        x, y = make_moons(n_samples=100, rng=np.random.default_rng(42))
        config = TrainConfig(hidden_dim=6, epochs=10, param_mode="quadratic_distance", seed=42)
        net, _ = train(x, y, config)
        assert not is_totally_positive(net.layers[0].weight)
