import numpy as np

from positroid.datasets.toy2d import (
    DATASETS,
    make_circles,
    make_moons,
    make_spirals,
    make_xor,
)


class TestMakeMoons:
    def test_shape(self):
        pts, y = make_moons(n_samples=100, rng=np.random.default_rng(42))
        assert pts.shape == (100, 2)
        assert y.shape == (100,)

    def test_labels(self):
        _, y = make_moons(n_samples=100, rng=np.random.default_rng(42))
        assert set(np.unique(y)) == {0, 1}
        assert 30 <= np.sum(y == 0) <= 70

    def test_reproducibility(self):
        pts1, y1 = make_moons(rng=np.random.default_rng(42))
        pts2, y2 = make_moons(rng=np.random.default_rng(42))
        np.testing.assert_array_equal(pts1, pts2)
        np.testing.assert_array_equal(y1, y2)


class TestMakeCircles:
    def test_shape(self):
        pts, y = make_circles(n_samples=100, rng=np.random.default_rng(42))
        assert pts.shape == (100, 2)
        assert y.shape == (100,)

    def test_radial_separation(self):
        pts, y = make_circles(
            n_samples=200, noise=0.0, factor=0.3, rng=np.random.default_rng(42),
        )
        radii = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        assert np.mean(radii[y == 1]) < np.mean(radii[y == 0])


class TestMakeSpirals:
    def test_shape(self):
        pts, y = make_spirals(n_samples=100, rng=np.random.default_rng(42))
        assert pts.shape == (100, 2)
        assert y.shape == (100,)


class TestMakeXor:
    def test_shape(self):
        pts, y = make_xor(n_samples=100, rng=np.random.default_rng(42))
        assert pts.shape == (100, 2)
        assert y.shape == (100,)

    def test_quadrant_structure(self):
        pts, y = make_xor(n_samples=200, noise=0.01, rng=np.random.default_rng(42))
        class0 = pts[y == 0]
        assert np.mean(class0[:, 0] * class0[:, 1] > 0) > 0.9


class TestRegistry:
    def test_all_datasets_in_registry(self):
        for name in ["moons", "circles", "spirals", "xor"]:
            assert name in DATASETS

    def test_registry_functions_callable(self):
        for name, fn in DATASETS.items():
            pts, y = fn(n_samples=50, rng=np.random.default_rng(0))
            assert pts.shape == (50, 2), f"{name}: wrong pts shape"
            assert y.shape == (50,), f"{name}: wrong y shape"
