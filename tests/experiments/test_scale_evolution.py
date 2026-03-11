import numpy as np

from positroid.datasets.toy2d import make_moons
from positroid.experiments.scale_evolution import (
    ScaleSnapshot,
    ScaleTrialResult,
    analyze_scale_snapshot,
    contiguous_window_ranks,
    get_augmented_matrix,
    min_window_singular_value,
    random_subset_rank_test,
    rank_deficiency_support,
    run_scale_experiment,
    run_scale_trial,
)
from positroid.network.train import TrainConfig, train

# ---------------------------------------------------------------------------
# Proxy function tests
# ---------------------------------------------------------------------------


class TestContiguousWindowRanks:
    def test_uniform_full_rank(self):
        """Well-conditioned matrix: all windows should be full rank."""
        rng = np.random.default_rng(42)
        aug = rng.standard_normal((10, 4))
        k = 3
        results = contiguous_window_ranks(aug, k)
        assert len(results) == 10 - k + 1
        for start, rank in results:
            assert rank == k, f"Window at {start} has rank {rank}, expected {k}"

    def test_deficient_tail(self):
        """Collinear tail: last rows are copies, windows including them are rank-deficient."""
        rng = np.random.default_rng(42)
        aug = rng.standard_normal((8, 4))
        # Make rows 6 and 7 copies of row 5 (collinear tail)
        aug[6] = aug[5]
        aug[7] = aug[5]
        k = 3
        results = contiguous_window_ranks(aug, k)
        # Windows [5,6,7] and [6,7,...] include the collinear rows — deficient
        # Window [4,5,6] also includes rows 5,6 (identical) — may be deficient
        deficient_starts = {start for start, rank in results if rank < k}
        assert 5 in deficient_starts, "Window at 5 should be deficient"
        # Early windows should be full rank
        for start, rank in results:
            if start <= 3:
                assert rank == k, f"Window at {start} should be full rank"


class TestRandomSubsetRankTest:
    def test_returns_correct_shape(self):
        rng = np.random.default_rng(42)
        aug = rng.standard_normal((10, 4))
        k = 3
        n_def, n_tested = random_subset_rank_test(aug, k, 100, rng)
        assert n_tested == 100
        assert 0 <= n_def <= n_tested

    def test_uniform_matrix_no_deficient(self):
        rng = np.random.default_rng(42)
        aug = rng.standard_normal((10, 4))
        k = 3
        n_def, _ = random_subset_rank_test(aug, k, 200, rng)
        assert n_def == 0


class TestRankDeficiencySupport:
    def test_empty_for_uniform(self):
        rng = np.random.default_rng(42)
        aug = rng.standard_normal((10, 4))
        k = 3
        support = rank_deficiency_support(aug, k)
        assert support == frozenset()

    def test_tail_indices(self):
        rng = np.random.default_rng(42)
        aug = rng.standard_normal((8, 4))
        aug[6] = aug[5]
        aug[7] = aug[5]
        k = 3
        support = rank_deficiency_support(aug, k)
        # Should include indices 5, 6, 7 (the collinear tail)
        assert 5 in support
        assert 6 in support
        assert 7 in support
        # Should not include early indices
        assert 0 not in support


class TestMinWindowSingularValue:
    def test_positive_for_full_rank(self):
        rng = np.random.default_rng(42)
        aug = rng.standard_normal((10, 4))
        k = 3
        min_sv = min_window_singular_value(aug, k)
        assert min_sv > 0

    def test_near_zero_for_deficient(self):
        rng = np.random.default_rng(42)
        aug = rng.standard_normal((8, 4))
        aug[6] = aug[5]
        aug[7] = aug[5]
        k = 3
        min_sv = min_window_singular_value(aug, k)
        assert min_sv < 1e-10


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestAnalyzeScaleSnapshot:
    def test_exact_mode_small_h(self):
        """At small H, exact fields should be populated."""
        x, y = make_moons(n_samples=100, rng=np.random.default_rng(42))
        config = TrainConfig(
            hidden_dim=6,
            epochs=30,
            param_mode="tp_exponential",
            seed=42,
        )
        net, history = train(x, y, config)
        rng = np.random.default_rng(99)
        snap = analyze_scale_snapshot(
            net,
            29,
            history.losses[-1],
            history.accuracies[-1],
            n_random_samples=100,
            rng=rng,
        )
        assert snap is not None
        assert isinstance(snap, ScaleSnapshot)
        assert snap.exact_is_uniform is not None
        assert snap.exact_is_positroid is not None
        assert snap.exact_num_non_bases is not None
        assert snap.exact_support is not None

    def test_proxy_only_forced(self):
        """With max_exact_subsets=0, exact fields should be None."""
        x, y = make_moons(n_samples=100, rng=np.random.default_rng(42))
        config = TrainConfig(
            hidden_dim=6,
            epochs=30,
            param_mode="tp_exponential",
            seed=42,
        )
        net, history = train(x, y, config)
        rng = np.random.default_rng(99)
        snap = analyze_scale_snapshot(
            net,
            29,
            history.losses[-1],
            history.accuracies[-1],
            n_random_samples=100,
            rng=rng,
            max_exact_subsets=0,
        )
        assert snap is not None
        assert snap.exact_is_uniform is None
        assert snap.exact_is_positroid is None
        assert snap.exact_num_non_bases is None
        assert snap.exact_support is None
        # Proxy should still be computed
        assert snap.ground_set_size > 0
        assert snap.rank > 0

    def test_exact_and_proxy_relationship(self):
        """Proxy and exact should agree or proxy overestimates (false positive)."""
        x, y = make_moons(n_samples=100, rng=np.random.default_rng(42))
        config = TrainConfig(
            hidden_dim=6,
            epochs=50,
            param_mode="tp_exponential",
            seed=42,
        )
        net, history = train(x, y, config)
        rng = np.random.default_rng(99)
        snap = analyze_scale_snapshot(
            net,
            49,
            history.losses[-1],
            history.accuracies[-1],
            n_random_samples=100,
            rng=rng,
        )
        assert snap is not None
        # Both proxy and exact should exist at this small H
        assert snap.exact_is_uniform is not None
        # If exact says non-uniform, proxy should detect it too
        if snap.exact_support is not None and len(snap.exact_support) > 0:
            proxy_set = frozenset(snap.proxy_support)
            exact_set = frozenset(snap.exact_support)
            # Exact support should be superset of (or equal to) proxy support
            # Proxy may also have false positives, which is acceptable
            assert len(proxy_set) > 0 or len(exact_set) == 0


class TestRunScaleTrial:
    def test_smoke(self):
        x, y = make_moons(n_samples=50, rng=np.random.default_rng(42))
        result = run_scale_trial(
            "moons",
            x,
            y,
            "tp_exponential",
            hidden_dim=6,
            epochs=20,
            lr=0.01,
            snapshot_epochs=[0, 9, 19],
            seed=42,
            n_random_samples=50,
        )
        assert isinstance(result, ScaleTrialResult)
        assert result.dataset_name == "moons"
        assert result.param_mode == "tp_exponential"
        assert result.hidden_dim == 6
        assert len(result.snapshots) > 0
        assert isinstance(result.init_nonuniform, bool)
        assert isinstance(result.final_nonuniform, bool)
        assert isinstance(result.training_created_nonuniformity, bool)


class TestRunScaleExperiment:
    def test_smoke(self):
        """Smoke test: 2 trials, 2 modes, moons, H=6, 20 epochs."""
        modes = ["tp_exponential", "negated_bidiagonal"]
        result = run_scale_experiment(
            dataset_name="moons",
            hidden_dim=6,
            num_trials=2,
            param_modes=modes,
            n_samples=50,
            epochs=20,
            seed=42,
            n_random_samples=50,
        )
        for mode in modes:
            assert len(result.trials_by_mode[mode]) == 2
            assert 0.0 <= result.init_nonuniform_rate(mode) <= 1.0
            assert 0.0 <= result.final_nonuniform_rate(mode) <= 1.0
            assert 0.0 <= result.training_created_rate(mode) <= 1.0
            assert 0.0 <= result.training_eliminated_rate(mode) <= 1.0
            assert 0.0 <= result.always_contiguous_rate(mode) <= 1.0
            assert 0.0 <= result.mean_accuracy(mode) <= 1.0

    def test_get_augmented_matrix(self):
        """get_augmented_matrix returns correct shape and rank."""
        x, y = make_moons(n_samples=50, rng=np.random.default_rng(42))
        config = TrainConfig(
            hidden_dim=8,
            epochs=5,
            param_mode="tp_exponential",
            seed=42,
        )
        net, _ = train(x, y, config)
        aug, k = get_augmented_matrix(net)
        assert aug.shape == (8, 3)  # H=8, d+1=3
        assert 1 <= k <= 3
