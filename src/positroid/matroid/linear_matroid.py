"""Linear matroid construction from vector configurations.

Given vectors v_1, ..., v_m in R^n (as rows of an m x n matrix),
the linear matroid has ground set [m] where a subset S is independent
iff {v_i : i in S} are linearly independent.
"""

from itertools import combinations

import numpy as np

from positroid.matroid.matroid import Matroid


def linear_matroid_from_vectors(vectors: np.ndarray, tol: float = 1e-8) -> Matroid:  # noqa: ARG001
    """Construct the linear matroid of row vectors.

    For numerical stability:
    1. Reduces the matrix via SVD: if A = U Sigma V^T has rank r, the row
       matroid of A equals the row matroid of U[:, :r]. Since U has orthonormal
       columns, all r x r minors lie in [0, 1].
    2. Determines bases via numerical rank of each r x r submatrix of U[:, :r],
       using numpy's standard SVD-based rank computation. Because U has
       orthonormal columns, the condition number issues of the original matrix
       are eliminated, and standard rank thresholds work reliably.

    Args:
        vectors: m x n matrix where rows are the vectors.
        tol: Kept for API compatibility; rank is determined by numpy's
             standard SVD threshold internally.

    Returns:
        Matroid on ground set {0, ..., m-1}.
    """
    m, n = vectors.shape

    # Compute full SVD
    u, sv, _ = np.linalg.svd(vectors, full_matrices=True)

    # Determine rank using standard numerical rank threshold
    rank_tol = max(m, n) * sv[0] * np.finfo(vectors.dtype).eps if sv[0] > 0 else 1e-15
    matrix_rank = int(np.sum(sv > rank_tol))

    if matrix_rank == 0:
        return Matroid(frozenset(range(m)), frozenset([frozenset()]))

    ground_set = frozenset(range(m))

    # Work with u[:, :r] — same row matroid, but orthonormal columns.
    # This eliminates the condition number of the original matrix: the
    # column norms are all 1, so rank decisions on submatrices are
    # governed only by the geometric arrangement of the rows.
    reduced = u[:, :matrix_rank]

    # A subset is a basis iff reduced[subset, :] has full rank.
    # numpy.linalg.matrix_rank uses SVD with threshold max(M,N)*σ_max*eps,
    # which is reliable here since reduced has orthonormal columns (σ_max ≤ 1
    # for any r x r submatrix).
    bases: set[frozenset[int]] = set()
    for subset in combinations(range(m), matrix_rank):
        submat = reduced[list(subset), :]
        if np.linalg.matrix_rank(submat) == matrix_rank:
            bases.add(frozenset(subset))

    if not bases:
        # Fallback: find subset with largest |det| and use it as sole basis
        best_det = -1.0
        best_subset = frozenset(range(matrix_rank))
        for subset in combinations(range(m), matrix_rank):
            submat = reduced[list(subset), :]
            det_val = abs(float(np.linalg.det(submat)))
            if det_val > best_det:
                best_det = det_val
                best_subset = frozenset(subset)
        bases = {best_subset}

    return Matroid(ground_set, frozenset(bases))
