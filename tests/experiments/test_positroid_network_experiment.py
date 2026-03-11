"""Smoke test for the positroid network experiment."""

import numpy as np

from positroid.experiments.positroid_network_experiment import run_trial


class TestPositroidNetworkExperiment:
    """Smoke tests for the experiment runner."""

    def test_positroid_fixed_trial(self) -> None:
        """Positroid fixed trial runs without error."""
        from positroid.datasets.toy2d import make_moons

        rng = np.random.default_rng(42)
        x, y = make_moons(n_samples=50, rng=rng)

        result = run_trial(
            dataset_name="moons",
            x=x,
            y=y,
            mode="positroid_fixed",
            k=2,
            n=4,
            epochs=10,
            learning_rate=0.01,
            seed=42,
        )
        assert result.mode == "positroid_fixed"
        assert 0 <= result.final_accuracy <= 1
        assert result.all_plucker_positive

    def test_relu_trial(self) -> None:
        """ReLU baseline trial runs without error."""
        from positroid.datasets.toy2d import make_moons

        rng = np.random.default_rng(42)
        x, y = make_moons(n_samples=50, rng=rng)

        result = run_trial(
            dataset_name="moons",
            x=x,
            y=y,
            mode="relu",
            k=2,
            n=4,
            epochs=10,
            learning_rate=0.01,
            seed=42,
        )
        assert result.mode == "relu"
        assert 0 <= result.final_accuracy <= 1

    def test_positroid_plucker_ratio_trial(self) -> None:
        """Positroid trial with plucker_ratio readout runs without error."""
        from positroid.datasets.toy2d import make_moons

        rng = np.random.default_rng(42)
        x, y = make_moons(n_samples=50, rng=rng)

        result = run_trial(
            dataset_name="moons",
            x=x,
            y=y,
            mode="positroid_fixed",
            k=2,
            n=4,
            epochs=10,
            learning_rate=0.01,
            seed=42,
            readout="plucker_ratio",
        )
        assert result.mode == "positroid_fixed"
        assert result.readout == "plucker_ratio"
        assert 0 <= result.final_accuracy <= 1
