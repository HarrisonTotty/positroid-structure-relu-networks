import numpy as np

from positroid.datasets.digits import (
    DIGIT_DATASETS,
    _load_and_project,
    _pca_project,
    make_digits,
    register_digits_dataset,
)


class TestPcaProject:
    def test_output_shape(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal((100, 20))
        x_proj, mean, components = _pca_project(x, n_components=5)
        assert x_proj.shape == (100, 5)
        assert mean.shape == (20,)
        assert components.shape == (5, 20)

    def test_centered(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal((100, 20)) + 10.0
        x_proj, _, _ = _pca_project(x, n_components=5)
        assert np.abs(x_proj.mean(axis=0)).max() < 1e-10

    def test_variance_ordering(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal((200, 20))
        x_proj, _, _ = _pca_project(x, n_components=5)
        variances = x_proj.var(axis=0)
        for i in range(len(variances) - 1):
            assert variances[i] >= variances[i + 1] - 1e-10


class TestLoadAndProject:
    def test_basic_load(self):
        x, y = _load_and_project(0, 1, pca_dim=5)
        assert x.shape[1] == 5
        assert set(np.unique(y)) == {0.0, 1.0}

    def test_correct_sample_count(self):
        x, _ = _load_and_project(0, 1, pca_dim=5)
        assert 300 < x.shape[0] < 400

    def test_caching(self):
        x1, y1 = _load_and_project(0, 1, pca_dim=5)
        x2, y2 = _load_and_project(0, 1, pca_dim=5)
        assert x1 is x2
        assert y1 is y2


class TestMakeDigits:
    def test_shape(self):
        x, y = make_digits(
            n_samples=100,
            rng=np.random.default_rng(42),
            digit_a=0,
            digit_b=1,
            pca_dim=5,
        )
        assert x.shape == (100, 5)
        assert y.shape == (100,)

    def test_labels_binary(self):
        _, y = make_digits(
            n_samples=100,
            rng=np.random.default_rng(42),
            digit_a=3,
            digit_b=8,
            pca_dim=10,
        )
        assert set(np.unique(y)).issubset({0.0, 1.0})

    def test_reproducibility(self):
        x1, y1 = make_digits(
            n_samples=100,
            rng=np.random.default_rng(42),
            digit_a=0,
            digit_b=1,
            pca_dim=5,
        )
        x2, y2 = make_digits(
            n_samples=100,
            rng=np.random.default_rng(42),
            digit_a=0,
            digit_b=1,
            pca_dim=5,
        )
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_differ(self):
        x1, _ = make_digits(
            n_samples=100,
            rng=np.random.default_rng(42),
            digit_a=0,
            digit_b=1,
            pca_dim=5,
        )
        x2, _ = make_digits(
            n_samples=100,
            rng=np.random.default_rng(99),
            digit_a=0,
            digit_b=1,
            pca_dim=5,
        )
        assert not np.array_equal(x1, x2)

    def test_oversampling(self):
        x, y = make_digits(
            n_samples=1000,
            rng=np.random.default_rng(42),
            digit_a=0,
            digit_b=1,
            pca_dim=5,
        )
        assert x.shape == (1000, 5)
        assert y.shape == (1000,)


class TestRegistry:
    def test_default_datasets_registered(self):
        for name in [
            "digits_0v1_pca5",
            "digits_0v1_pca10",
            "digits_3v8_pca5",
            "digits_3v8_pca10",
        ]:
            assert name in DIGIT_DATASETS

    def test_registry_functions_callable(self):
        for name, fn in DIGIT_DATASETS.items():
            x, y = fn(n_samples=50, rng=np.random.default_rng(0))
            assert x.ndim == 2, f"{name}: X should be 2D"
            assert x.shape[0] == 50, f"{name}: wrong n_samples"
            assert y.shape == (50,), f"{name}: wrong y shape"

    def test_register_custom(self):
        name = register_digits_dataset(4, 9, 7)
        assert name == "digits_4v9_pca7"
        assert name in DIGIT_DATASETS
        x, y = DIGIT_DATASETS[name](n_samples=30, rng=np.random.default_rng(0))
        assert x.shape == (30, 7)
        assert y.shape == (30,)


class TestTrainingIntegration:
    def test_train_on_digits(self):
        from positroid.network.train import TrainConfig, train

        x, y = make_digits(
            n_samples=100,
            rng=np.random.default_rng(42),
            digit_a=0,
            digit_b=1,
            pca_dim=5,
        )
        config = TrainConfig(hidden_dim=8, epochs=20, seed=42)
        net, history = train(x, y, config)
        assert net.input_dim == 5
        assert net.output_dim == 1
        assert len(history.losses) == 20
        assert history.losses[-1] < history.losses[0]

    def test_train_tp_on_digits(self):
        from positroid.network.train import TrainConfig, train

        x, y = make_digits(
            n_samples=100,
            rng=np.random.default_rng(42),
            digit_a=0,
            digit_b=1,
            pca_dim=5,
        )
        config = TrainConfig(
            hidden_dim=8,
            epochs=20,
            tp_constrained=True,
            seed=42,
        )
        net, history = train(x, y, config)
        assert net.input_dim == 5
        assert len(history.losses) == 20
