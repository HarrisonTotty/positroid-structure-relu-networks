import numpy as np

from positroid.linalg.totally_positive import (
    is_totally_nonnegative,
    is_totally_positive,
    random_totally_positive,
    tp_from_cauchy_kernel,
    tp_from_exponential_kernel,
)


class TestTotallyPositive:
    def test_exponential_kernel_is_tp(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([0.5, 1.0])
        m = tp_from_exponential_kernel(a, b)
        assert m.shape == (3, 2)
        assert is_totally_positive(m)

    def test_random_tp_is_tp(self):
        rng = np.random.default_rng(42)
        for _ in range(10):
            m = random_totally_positive(4, 2, rng=rng)
            assert is_totally_positive(m), f"Failed TP check for matrix:\n{m}"

    def test_random_tp_various_shapes(self):
        rng = np.random.default_rng(123)
        for shape in [(2, 2), (3, 2), (4, 3), (2, 4), (5, 2)]:
            m = random_totally_positive(*shape, rng=rng)
            assert m.shape == shape
            assert is_totally_positive(m)

    def test_identity_is_tn_not_tp(self):
        m = np.eye(3)
        assert is_totally_nonnegative(m)
        assert not is_totally_positive(m)  # Off-diagonal minors are 0

    def test_random_matrix_usually_not_tp(self):
        rng = np.random.default_rng(99)
        m = rng.standard_normal((4, 3))
        assert not is_totally_positive(m)

    def test_cauchy_kernel_is_tp(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([0.5, 1.0])
        m = tp_from_cauchy_kernel(a, b)
        assert m.shape == (3, 2)
        assert is_totally_positive(m)

    def test_cauchy_kernel_various_shapes(self):
        """Cauchy kernel with well-separated params is TP."""
        for h, d in [(4, 2), (6, 3), (5, 2), (3, 3)]:
            a = np.linspace(1.0, 1.0 + 0.5 * h, h)
            b = np.linspace(1.0, 1.0 + 0.5 * d, d)
            m = tp_from_cauchy_kernel(a, b)
            assert m.shape == (h, d)
            assert is_totally_positive(m), f"Cauchy {h}x{d} failed TP"

    def test_vandermonde_is_tp(self):
        """Vandermonde matrix with increasing positive nodes is TP."""
        x = np.array([0.5, 1.0, 2.0])
        m = np.column_stack([x**0, x**1, x**2])
        assert is_totally_positive(m)
