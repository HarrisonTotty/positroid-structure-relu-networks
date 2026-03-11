"""Totally positive matrix generation and verification.

A matrix is totally positive (TP) if every minor (determinant of every
square submatrix) is strictly positive. A matrix is totally nonnegative (TN)
if every minor is nonnegative.
"""

import numpy as np

from positroid.linalg.minors import all_minors


def is_totally_positive(m: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if M is totally positive (all minors of all sizes > 0)."""
    return all(val > tol for val in all_minors(m).values())


def is_totally_nonnegative(m: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if M is totally nonnegative (all minors of all sizes >= 0)."""
    return all(val >= -tol for val in all_minors(m).values())


def tp_from_exponential_kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Generate TP matrix M_{ij} = exp(a_i * b_j) where a and b are strictly increasing.

    By the theory of totally positive kernels (Karlin), K(x,y) = exp(xy) is a
    TP kernel, so M_{ij} = K(a_i, b_j) is totally positive when a and b are
    strictly increasing sequences.

    Args:
        a: Strictly increasing 1D array of length m.
        b: Strictly increasing 1D array of length n.

    Returns:
        m x n totally positive matrix.
    """
    return np.exp(np.outer(a, b))


def tp_from_cauchy_kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Generate TP matrix M_{ij} = 1/(a_i + b_j) where a and b are strictly increasing.

    By the theory of totally positive kernels (Karlin), K(x,y) = 1/(x+y) is a
    TP kernel on (0, inf) x (0, inf), so M_{ij} = K(a_i, b_j) is totally positive
    when a and b are strictly increasing positive sequences.

    Args:
        a: Strictly increasing 1D array of positive values, length m.
        b: Strictly increasing 1D array of positive values, length n.

    Returns:
        m x n totally positive matrix.
    """
    result: np.ndarray = 1.0 / (a[:, np.newaxis] + b[np.newaxis, :])
    return result


def tp_from_loewner_whitney(
    diag: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    n_rows: int,
    n_cols: int,
) -> np.ndarray:
    """Build n_rows×n_cols TP matrix from Loewner-Whitney bidiagonal factorization.

    Uses three stages (all parameters strictly positive):

    1. Positive diagonal: n_cols params
    2. Upper bidiagonal (wiring diagram order): n_cols*(n_cols-1)/2 params
       Fills the upper triangle of the n_cols×n_cols block.
    3. Lower bidiagonal (Neville order): n_cols*(2*n_rows-n_cols-1)/2 params
       Fills the lower triangle and extends to n_rows rows.

    Total: n_rows * n_cols parameters.

    Args:
        diag: Positive 1D array of length n_cols.
        upper: Positive 1D array of length n_cols*(n_cols-1)//2.
        lower: Positive 1D array of length n_cols*(2*n_rows-n_cols-1)//2.
        n_rows: Number of rows (hidden dim).
        n_cols: Number of columns (input dim).

    Returns:
        n_rows×n_cols totally positive matrix.
    """
    a = np.zeros((n_rows, n_cols))
    # 1. Positive diagonal
    for i in range(n_cols):
        a[i, i] = diag[i]
    # 2. Upper bidiagonal (wiring diagram order):
    #    Level l (l=1..d-1): for k=d-l..d-1: a[k-1] += u * a[k]
    idx = 0
    for level in range(1, n_cols):
        for k in range(n_cols - level, n_cols):
            a[k - 1, :] += upper[idx] * a[k, :]
            idx += 1
    # 3. Lower bidiagonal (Neville order):
    #    For j=d-1..0: for i=j+1..H-1: a[i] += l * a[i-1]
    idx = 0
    for j in range(n_cols - 1, -1, -1):
        for i in range(j + 1, n_rows):
            a[i, :] += lower[idx] * a[i - 1, :]
            idx += 1
    return a


def random_totally_positive(m: int, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate a random totally positive m x n matrix.

    Uses the exponential kernel with random strictly increasing parameters.
    """
    if rng is None:
        rng = np.random.default_rng()

    a = np.sort(rng.uniform(0.5, 2.0, size=m))
    # Ensure strict increase
    for i in range(1, m):
        if a[i] <= a[i - 1]:
            a[i] = a[i - 1] + 0.01

    b = np.sort(rng.uniform(0.5, 2.0, size=n))
    for i in range(1, n):
        if b[i] <= b[i - 1]:
            b[i] = b[i - 1] + 0.01

    return tp_from_exponential_kernel(a, b)
