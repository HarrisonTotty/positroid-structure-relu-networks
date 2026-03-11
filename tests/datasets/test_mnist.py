"""Tests for multiclass digit and MNIST dataset loaders."""

import numpy as np

from positroid.datasets.digits import make_digits_multiclass


class TestDigitsMulticlass:
    """Tests for the multiclass digits loader."""

    def test_shape_default(self) -> None:
        x, y = make_digits_multiclass(n_samples=100, rng=np.random.default_rng(42))
        assert x.shape == (100, 20)  # default pca_dim=20
        assert y.shape == (100,)

    def test_labels_are_integer_classes(self) -> None:
        x, y = make_digits_multiclass(n_samples=500, rng=np.random.default_rng(42))
        unique = set(y.tolist())
        # All 10 classes should be present in 500 samples
        assert unique == {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

    def test_pca_dim_respected(self) -> None:
        x, _ = make_digits_multiclass(n_samples=50, rng=np.random.default_rng(42), pca_dim=10)
        assert x.shape[1] == 10

    def test_subset_digits(self) -> None:
        x, y = make_digits_multiclass(
            n_samples=100, rng=np.random.default_rng(42), digits=(0, 3, 7)
        )
        unique = set(y.tolist())
        # Labels should be remapped to 0, 1, 2
        assert unique <= {0, 1, 2}

    def test_standardized(self) -> None:
        x, _ = make_digits_multiclass(n_samples=1000, rng=np.random.default_rng(42))
        # Each feature should have reasonable scale (near unit variance)
        stds = x.std(axis=0)
        assert np.all(stds > 0.1), "Some features have near-zero variance"
        assert np.all(stds < 10.0), "Some features have very large variance"

    def test_reproducible(self) -> None:
        x1, y1 = make_digits_multiclass(n_samples=50, rng=np.random.default_rng(42))
        x2, y2 = make_digits_multiclass(n_samples=50, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)


class TestDigitsMulticlassRegistry:
    """Tests for registered multiclass digit datasets."""

    def test_registered_in_datasets(self) -> None:
        from positroid.datasets import DATASETS

        assert "digits_10class_pca10" in DATASETS
        assert "digits_10class_pca20" in DATASETS
        assert "digits_10class_pca50" in DATASETS

    def test_callable_from_registry(self) -> None:
        from positroid.datasets import DATASETS

        x, y = DATASETS["digits_10class_pca10"](n_samples=50, rng=np.random.default_rng(42))
        assert x.shape == (50, 10)
        assert y.shape == (50,)


class TestMNISTRegistry:
    """Tests for registered MNIST datasets (skip if download not available)."""

    def test_registered_in_datasets(self) -> None:
        from positroid.datasets import DATASETS

        assert "mnist_10class_pca20" in DATASETS
        assert "mnist_10class_pca50" in DATASETS
