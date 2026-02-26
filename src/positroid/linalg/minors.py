"""Minor computation for matrices.

Computes minors (determinants of submatrices) needed for total positivity
checks, matroid construction, and positroid verification.
"""

from itertools import combinations

import numpy as np


def minor(m: np.ndarray, rows: tuple[int, ...], cols: tuple[int, ...]) -> float:
    """Compute the minor (determinant of submatrix) for given rows and columns."""
    submat = m[np.ix_(list(rows), list(cols))]
    return float(np.linalg.det(submat))


def all_maximal_minors(m: np.ndarray) -> dict[tuple[int, ...], float]:
    """Compute all maximal minors of an m x n matrix (m <= n).

    Returns dict mapping column-index tuples to minor values.
    For an m x n matrix with m <= n, these are the C(n, m) determinants
    obtained by choosing m columns.
    """
    rows, cols = m.shape
    k = min(rows, cols)
    result: dict[tuple[int, ...], float] = {}

    if rows <= cols:
        all_rows = tuple(range(rows))
        for col_subset in combinations(range(cols), k):
            result[col_subset] = minor(m, all_rows, col_subset)
    else:
        all_cols = tuple(range(cols))
        for row_subset in combinations(range(rows), k):
            result[row_subset] = minor(m, row_subset, all_cols)

    return result


def all_minors(m: np.ndarray) -> dict[tuple[tuple[int, ...], tuple[int, ...]], float]:
    """Compute ALL minors of a matrix (all sizes from 1x1 up to min(m,n) x min(m,n)).

    Returns dict mapping (row_indices, col_indices) to determinant values.
    """
    rows, cols = m.shape
    max_size = min(rows, cols)
    result: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = {}

    for size in range(1, max_size + 1):
        for row_subset in combinations(range(rows), size):
            for col_subset in combinations(range(cols), size):
                result[(row_subset, col_subset)] = minor(m, row_subset, col_subset)

    return result
