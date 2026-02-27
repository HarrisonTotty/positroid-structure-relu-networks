import numpy as np

from positroid.experiments.activation_positroid import run_experiment, run_single_trial


class TestActivationPositroid:
    def test_single_trial_tp(self):
        rng = np.random.default_rng(42)
        result = run_single_trial(input_dim=2, hidden_dim=4, tp_weights=True, rng=rng)
        assert result.is_weight_tp
        assert result.linear_matroid_rank == 2
        assert result.linear_matroid_is_uniform  # TP with H > n -> U(n, H)
        assert result.linear_matroid_is_positroid  # U(n, H) is always a positroid

    def test_single_trial_random(self):
        rng = np.random.default_rng(42)
        result = run_single_trial(input_dim=2, hidden_dim=4, tp_weights=False, rng=rng)
        assert not result.is_weight_tp
        assert result.linear_matroid_rank == 2

    def test_experiment_tp(self):
        result = run_experiment(input_dim=2, hidden_dim=4, num_trials=10, tp_weights=True, seed=42)
        assert result.num_trials == 10
        assert len(result.trials) == 10
        # All linear matroids should be positroids (trivially: TP + H > n -> uniform)
        assert result.linear_positroid_count == 10

    def test_experiment_random(self):
        result = run_experiment(input_dim=2, hidden_dim=4, num_trials=10, tp_weights=False, seed=42)
        assert result.num_trials == 10

    def test_small_hidden_dim(self):
        """With H = n = 2, the matroid can be non-trivial."""
        result = run_experiment(input_dim=2, hidden_dim=2, num_trials=10, tp_weights=True, seed=42)
        assert len(result.trials) == 10
        # With H = n, the linear matroid has exactly 1 basis (the full set)
        for t in result.trials:
            assert t.linear_matroid_rank == 2
            assert t.linear_matroid_num_bases == 1

    def test_affine_matroid_computed(self):
        rng = np.random.default_rng(42)
        result = run_single_trial(input_dim=2, hidden_dim=4, tp_weights=True, rng=rng)
        assert result.affine_matroid_rank >= 2
        assert result.affine_matroid_rank <= 3  # max rank in R^{n+1} = R^3

    def test_necklace_computed_when_positroid(self):
        rng = np.random.default_rng(42)
        result = run_single_trial(input_dim=2, hidden_dim=4, tp_weights=True, rng=rng)
        if result.affine_matroid_is_positroid:
            assert result.affine_grassmann_necklace is not None
            assert result.affine_decorated_permutation is not None
