import numpy as np

from positroid.datasets.toy2d import make_moons
from positroid.experiments.matroid_evolution import (
    EpochSnapshot,
    EvolutionTrialResult,
    analyze_snapshot,
    run_evolution_experiment,
    run_single_trial,
)
from positroid.network.train import TrainConfig, train


class TestAnalyzeSnapshot:
    def test_returns_valid_snapshot(self):
        x, y = make_moons(n_samples=100, rng=np.random.default_rng(42))
        config = TrainConfig(hidden_dim=6, epochs=30, param_mode="tp_exponential", seed=42)
        net, history = train(x, y, config)
        snap = analyze_snapshot(net, 29, history.losses[-1], history.accuracies[-1])
        assert snap is not None
        assert isinstance(snap, EpochSnapshot)
        assert snap.epoch == 29
        assert isinstance(snap.is_positroid, bool)
        assert isinstance(snap.support, tuple)
        assert snap.support_size == len(snap.support)
        assert snap.num_bases + snap.num_non_bases > 0

    def test_uniform_snapshot_properties(self):
        """Early training typically produces uniform matroids."""
        x, y = make_moons(n_samples=100, rng=np.random.default_rng(42))
        config = TrainConfig(
            hidden_dim=6,
            epochs=5,
            param_mode="tp_exponential",
            seed=42,
        )
        _, history = train(x, y, config, snapshot_epochs=[0])
        if 0 in history.snapshots:
            snap = analyze_snapshot(history.snapshots[0], 0, 0.0, 0.0)
            if snap is not None and snap.is_uniform:
                assert snap.support == ()
                assert snap.support_size == 0
                assert snap.support_is_interval is True
                assert snap.support_rank_deficiency == 0


class TestRunSingleTrial:
    def test_tp_exponential_trial(self):
        x, y = make_moons(n_samples=50, rng=np.random.default_rng(42))
        result = run_single_trial(
            "moons",
            x,
            y,
            "tp_exponential",
            hidden_dim=6,
            epochs=20,
            lr=0.01,
            snapshot_interval=10,
            seed=42,
        )
        assert isinstance(result, EvolutionTrialResult)
        assert result.dataset_name == "moons"
        assert result.param_mode == "tp_exponential"
        assert result.hidden_dim == 6
        assert len(result.snapshots) > 0

    def test_negated_bidiagonal_trial(self):
        x, y = make_moons(n_samples=50, rng=np.random.default_rng(42))
        result = run_single_trial(
            "moons",
            x,
            y,
            "negated_bidiagonal",
            hidden_dim=6,
            epochs=20,
            lr=0.01,
            snapshot_interval=10,
            seed=6042,
        )
        assert isinstance(result, EvolutionTrialResult)
        assert result.param_mode == "negated_bidiagonal"
        assert len(result.snapshots) > 0

    def test_final_epoch_always_included(self):
        """Final epoch (epochs-1) should always be in the snapshots."""
        x, y = make_moons(n_samples=50, rng=np.random.default_rng(42))
        result = run_single_trial(
            "moons",
            x,
            y,
            "tp_exponential",
            hidden_dim=6,
            epochs=25,
            lr=0.01,
            snapshot_interval=10,
            seed=42,
        )
        snapshot_epochs = [s.epoch for s in result.snapshots]
        assert 24 in snapshot_epochs

    def test_snapshot_interval(self):
        """Snapshots should be at expected intervals."""
        x, y = make_moons(n_samples=50, rng=np.random.default_rng(42))
        result = run_single_trial(
            "moons",
            x,
            y,
            "tp_exponential",
            hidden_dim=6,
            epochs=30,
            lr=0.01,
            snapshot_interval=10,
            seed=42,
        )
        snapshot_epochs = [s.epoch for s in result.snapshots]
        # Should have epochs 0, 10, 20, 29
        assert 0 in snapshot_epochs
        assert 10 in snapshot_epochs
        assert 20 in snapshot_epochs
        assert 29 in snapshot_epochs


class TestRunEvolutionExperiment:
    def test_smoke(self):
        """Smoke test: 2 trials, 2 modes, moons, H=6, 20 epochs."""
        modes = ["tp_exponential", "negated_bidiagonal"]
        result = run_evolution_experiment(
            dataset_name="moons",
            hidden_dim=6,
            num_trials=2,
            param_modes=modes,
            n_samples=50,
            epochs=20,
            snapshot_interval=10,
            seed=42,
        )
        for mode in modes:
            assert len(result.trials_by_mode[mode]) == 2
            assert 0.0 <= result.always_positroid_rate(mode) <= 1.0
            assert 0.0 <= result.ever_nonuniform_rate(mode) <= 1.0
            assert 0.0 <= result.ever_noninterval_rate(mode) <= 1.0
            assert 0.0 <= result.mean_accuracy(mode) <= 1.0
