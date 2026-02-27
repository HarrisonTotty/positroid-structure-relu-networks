import numpy as np

from positroid.datasets.toy2d import make_moons
from positroid.experiments.trained_positroid import (
    TrainedTrialResult,
    analyze_network,
    run_experiment,
)
from positroid.network.train import TrainConfig, train


class TestAnalyzeNetwork:
    def test_returns_valid_result(self):
        x, y = make_moons(n_samples=100, rng=np.random.default_rng(42))
        config = TrainConfig(hidden_dim=6, epochs=30, seed=42)
        net, history = train(x, y, config)

        result = analyze_network(
            net,
            "moons",
            tp_constrained=False,
            final_loss=history.losses[-1],
            final_accuracy=history.accuracies[-1],
        )
        assert isinstance(result, TrainedTrialResult)
        assert result.dataset_name == "moons"
        assert result.affine_matroid_rank >= 1

    def test_tp_constrained_result(self):
        x, y = make_moons(n_samples=100, rng=np.random.default_rng(42))
        config = TrainConfig(
            hidden_dim=6,
            epochs=30,
            tp_constrained=True,
            seed=42,
        )
        net, history = train(x, y, config)

        result = analyze_network(
            net,
            "moons",
            tp_constrained=True,
            final_loss=history.losses[-1],
            final_accuracy=history.accuracies[-1],
        )
        assert result.is_weight_tp


class TestRunExperiment:
    def test_small_experiment(self):
        result = run_experiment(
            dataset_name="moons",
            hidden_dim=6,
            num_trials=3,
            n_samples=50,
            epochs=20,
            seed=42,
        )
        assert len(result.tp_trials) == 3
        assert len(result.unconstrained_trials) == 3
        assert 0.0 <= result.tp_positroid_rate <= 1.0
        assert 0.0 <= result.unconstrained_positroid_rate <= 1.0


class TestTrainedTPIsPositroid:
    """The revised (dynamical) conjecture: trained TP networks always produce
    positroid affine matroids. This is the claim behind '800 trials, zero
    counterexamples' in the blog posts."""

    def test_tp_moons_always_positroid(self):
        """TP-constrained networks trained on moons should produce positroid matroids."""
        result = run_experiment(
            dataset_name="moons",
            hidden_dim=8,
            num_trials=5,
            n_samples=100,
            epochs=100,
            seed=42,
        )
        for i, t in enumerate(result.tp_trials):
            assert t.affine_matroid_is_positroid, (
                f"Trained TP trial {i} produced non-positroid matroid "
                f"(rank={t.affine_matroid_rank}, "
                f"|bases|={t.affine_matroid_num_bases}, "
                f"uniform={t.affine_matroid_is_uniform})"
            )

    def test_tp_circles_always_positroid(self):
        """TP-constrained networks trained on circles should produce positroid matroids."""
        result = run_experiment(
            dataset_name="circles",
            hidden_dim=6,
            num_trials=5,
            n_samples=100,
            epochs=100,
            seed=42,
        )
        for i, t in enumerate(result.tp_trials):
            assert t.affine_matroid_is_positroid, (
                f"Trained TP trial {i} produced non-positroid matroid "
                f"(rank={t.affine_matroid_rank}, "
                f"|bases|={t.affine_matroid_num_bases}, "
                f"uniform={t.affine_matroid_is_uniform})"
            )

    def test_tp_cauchy_kernel_always_positroid(self):
        """TP-constrained networks with Cauchy kernel should produce positroid matroids."""
        result = run_experiment(
            dataset_name="moons",
            hidden_dim=6,
            num_trials=5,
            n_samples=100,
            epochs=100,
            seed=42,
            tp_kernel="cauchy",
        )
        for i, t in enumerate(result.tp_trials):
            assert t.affine_matroid_is_positroid, (
                f"Trained TP (Cauchy) trial {i} produced non-positroid matroid "
                f"(rank={t.affine_matroid_rank}, "
                f"|bases|={t.affine_matroid_num_bases}, "
                f"uniform={t.affine_matroid_is_uniform})"
            )


class TestRunExperimentDigits:
    def test_digits_experiment(self):
        result = run_experiment(
            dataset_name="digits_0v1_pca5",
            hidden_dim=8,
            num_trials=2,
            n_samples=50,
            epochs=10,
            seed=42,
        )
        assert len(result.tp_trials) == 2
        assert len(result.unconstrained_trials) == 2
        for t in result.tp_trials + result.unconstrained_trials:
            assert t.affine_matroid_rank <= 6
