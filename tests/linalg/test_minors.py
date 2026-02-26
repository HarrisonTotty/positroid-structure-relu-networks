import numpy as np
import pytest

from positroid.linalg.minors import all_maximal_minors, all_minors, minor


class TestMinor:
    def test_2x2_determinant(self):
        m = np.array([[1, 2], [3, 4]], dtype=float)
        assert minor(m, (0, 1), (0, 1)) == pytest.approx(-2.0)

    def test_identity_full_minor(self):
        m = np.eye(3)
        assert minor(m, (0, 1, 2), (0, 1, 2)) == pytest.approx(1.0)

    def test_identity_2x2_subminor(self):
        m = np.eye(3)
        # Rows {0,1}, cols {0,1} -> det([[1,0],[0,1]]) = 1
        assert minor(m, (0, 1), (0, 1)) == pytest.approx(1.0)
        # Rows {0,1}, cols {0,2} -> det([[1,0],[0,0]]) = 0
        assert minor(m, (0, 1), (0, 2)) == pytest.approx(0.0)

    def test_1x1_minor(self):
        m = np.array([[5.0, 3.0], [2.0, 7.0]])
        assert minor(m, (0,), (1,)) == pytest.approx(3.0)


class TestAllMaximalMinors:
    def test_2x3_matrix(self):
        m = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        result = all_maximal_minors(m)
        assert len(result) == 3  # C(3,2) = 3
        assert (0, 1) in result
        assert (0, 2) in result
        assert (1, 2) in result
        # det([[1,2],[4,5]]) = 5-8 = -3
        assert result[(0, 1)] == pytest.approx(-3.0)

    def test_square_matrix(self):
        m = np.eye(2)
        result = all_maximal_minors(m)
        assert len(result) == 1
        assert result[(0, 1)] == pytest.approx(1.0)

    def test_tall_matrix(self):
        m = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        result = all_maximal_minors(m)
        assert len(result) == 3  # C(3,2) = 3, keyed by row subsets


class TestAllMinors:
    def test_2x2_all_minors(self):
        m = np.array([[2, 3], [5, 7]], dtype=float)
        result = all_minors(m)
        # 1x1: 4 entries + 2x2: 1 entry = 5
        assert len(result) == 5
        assert result[((0,), (0,))] == pytest.approx(2.0)
        assert result[((0,), (1,))] == pytest.approx(3.0)
        assert result[((1,), (0,))] == pytest.approx(5.0)
        assert result[((1,), (1,))] == pytest.approx(7.0)
        assert result[((0, 1), (0, 1))] == pytest.approx(-1.0)
