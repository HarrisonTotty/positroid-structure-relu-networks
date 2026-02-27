import numpy as np

from positroid.matroid.linear_matroid import linear_matroid_from_vectors


class TestLinearMatroid:
    def test_generic_vectors_give_uniform(self):
        """Generic vectors in R^2: 4 vectors -> U(2, 4)."""
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((4, 2))
        m = linear_matroid_from_vectors(vectors)
        assert m.rank == 2
        assert m.size == 4
        assert m.is_uniform()

    def test_parallel_vectors(self):
        """Two parallel vectors create a non-uniform matroid."""
        vectors = np.array(
            [
                [1.0, 0.0],
                [2.0, 0.0],  # parallel to first
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        m = linear_matroid_from_vectors(vectors)
        assert m.rank == 2
        assert not m.is_independent(frozenset({0, 1}))  # parallel
        assert m.is_independent(frozenset({0, 2}))

    def test_identity_matrix(self):
        """Rows of identity matrix form U(n, n) (only one basis)."""
        m = linear_matroid_from_vectors(np.eye(3))
        assert m.rank == 3
        assert len(m.bases) == 1
        assert frozenset({0, 1, 2}) in m.bases

    def test_single_vector(self):
        vectors = np.array([[1.0, 2.0]])
        m = linear_matroid_from_vectors(vectors)
        assert m.rank == 1
        assert m.size == 1

    def test_tp_matrix_gives_uniform(self):
        """A totally positive matrix with more rows than columns gives U(n, m)."""
        from positroid.linalg.totally_positive import random_totally_positive

        rng = np.random.default_rng(42)
        tp = random_totally_positive(4, 2, rng=rng)
        m = linear_matroid_from_vectors(tp)
        assert m.rank == 2
        assert m.is_uniform()
