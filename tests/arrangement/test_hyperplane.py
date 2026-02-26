import numpy as np

from positroid.arrangement.hyperplane import Hyperplane, HyperplaneArrangement


class TestHyperplaneArrangement:
    def test_basic_construction(self):
        h1 = Hyperplane(normal=np.array([1.0, 0.0]), bias=0.0)
        h2 = Hyperplane(normal=np.array([0.0, 1.0]), bias=0.0)
        arr = HyperplaneArrangement([h1, h2])
        assert arr.ambient_dim == 2
        assert arr.num_hyperplanes == 2

    def test_normal_matrix(self):
        h1 = Hyperplane(normal=np.array([1.0, 0.0]), bias=0.5)
        h2 = Hyperplane(normal=np.array([0.0, 1.0]), bias=-0.3)
        arr = HyperplaneArrangement([h1, h2])
        normals = arr.normal_matrix()
        assert normals.shape == (2, 2)
        np.testing.assert_array_equal(normals[0], [1.0, 0.0])

    def test_augmented_matrix(self):
        h1 = Hyperplane(normal=np.array([1.0, 2.0]), bias=3.0)
        arr = HyperplaneArrangement([h1])
        aug = arr.augmented_matrix()
        assert aug.shape == (1, 3)
        np.testing.assert_array_equal(aug[0], [1.0, 2.0, 3.0])

    def test_linear_matroid_generic(self):
        """Generic hyperplanes in R^2 -> U(2, H)."""
        hyperplanes = [
            Hyperplane(normal=np.array([1.0, 0.5]), bias=0.1),
            Hyperplane(normal=np.array([0.3, 1.0]), bias=-0.2),
            Hyperplane(normal=np.array([1.0, -0.7]), bias=0.5),
            Hyperplane(normal=np.array([-0.2, 1.0]), bias=0.3),
        ]
        arr = HyperplaneArrangement(hyperplanes)
        m = arr.linear_matroid()
        assert m.rank == 2
        assert m.is_uniform()

    def test_affine_matroid(self):
        """Affine matroid has rank up to min(m, n+1)."""
        hyperplanes = [
            Hyperplane(normal=np.array([1.0, 0.0]), bias=1.0),
            Hyperplane(normal=np.array([0.0, 1.0]), bias=2.0),
            Hyperplane(normal=np.array([1.0, 1.0]), bias=0.5),  # not sum of above biases
        ]
        arr = HyperplaneArrangement(hyperplanes)
        m = arr.affine_matroid()
        assert m.rank == 3  # affinely independent in R^2

    def test_sign_vectors(self):
        h1 = Hyperplane(normal=np.array([1.0, 0.0]), bias=0.0)
        h2 = Hyperplane(normal=np.array([0.0, 1.0]), bias=0.0)
        arr = HyperplaneArrangement([h1, h2])

        points = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]])
        signs = arr.sign_vectors(points)
        np.testing.assert_array_equal(signs[0], [1, 1])
        np.testing.assert_array_equal(signs[1], [-1, 1])
        np.testing.assert_array_equal(signs[2], [-1, -1])

    def test_parallel_hyperplanes_matroid(self):
        """Parallel hyperplanes have rank-1 linear matroid."""
        hyperplanes = [
            Hyperplane(normal=np.array([1.0, 0.0]), bias=0.0),
            Hyperplane(normal=np.array([2.0, 0.0]), bias=1.0),
            Hyperplane(normal=np.array([3.0, 0.0]), bias=2.0),
        ]
        arr = HyperplaneArrangement(hyperplanes)
        m = arr.linear_matroid()
        assert m.rank == 1
