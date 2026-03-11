"""Boundary measurement map for the top cell of Gr+(k,n).

Implements the Marsh-Rietsch parameterization: starting from the identity
matrix [I_k | 0], apply a sequence of column shears corresponding to a
reduced word for the top-cell permutation. Each shear has a positive weight
parameter (reparameterized via exp for unconstrained optimization).

The reduced word for the top cell of Gr+(k,n) consists of k phases:
  Phase p (p = 0, ..., k-1): s_{k-p}, s_{k-p+1}, ..., s_{n-1-p}
Each s_j adds a positive multiple of column j-1 to column j.

Total parameters: k * (n - k) = dim Gr(k,n).

References:
  Marsh-Rietsch, "Parametrizations of flag varieties" (2004)
  Postnikov, "Total positivity, Grassmannians, and networks" (2006)
"""

from __future__ import annotations

import numpy as np


def boundary_measurement_matrix(weights: np.ndarray, k: int, n: int) -> np.ndarray:
    """Build k x n boundary measurement matrix for the top cell of Gr+(k,n).

    Uses the Marsh-Rietsch parameterization with k*(n-k) strictly positive
    weights. All k x k minors of the resulting matrix are strictly positive.

    Args:
        weights: Strictly positive 1D array of length k*(n-k).
        k: Rank of the Grassmannian (number of rows).
        n: Size of the ground set (number of columns).

    Returns:
        k x n totally positive matrix (representative of a point in Gr+(k,n)).
    """
    m = k * (n - k)
    if len(weights) != m:
        raise ValueError(f"Expected {m} weights for Gr+({k},{n}), got {len(weights)}")

    mat = np.zeros((k, n))
    for i in range(k):
        mat[i, i] = 1.0

    idx = 0
    for phase in range(k):
        # Phase p: shears s_{k-p}, s_{k-p+1}, ..., s_{n-1-p}
        # In 0-indexed column terms: col[s] += weight * col[s-1]
        # where s ranges from (k - phase) to (n - 1 - phase)
        start_col = k - phase
        end_col = n - 1 - phase
        for col in range(start_col, end_col + 1):
            mat[:, col] += weights[idx] * mat[:, col - 1]
            idx += 1

    return mat


def boundary_measurement_backward(
    weights: np.ndarray,
    k: int,
    n: int,
    mat_final: np.ndarray,
    d_mat: np.ndarray,
) -> np.ndarray:
    """Compute gradient of weights given gradient of the boundary measurement matrix.

    Reverses the forward construction: undoes each column shear in reverse
    order to recover the pre-operation state, then accumulates gradients.

    Args:
        weights: The positive weights used in the forward pass.
        k: Rank of the Grassmannian.
        n: Size of the ground set.
        mat_final: The boundary measurement matrix from the forward pass.
        d_mat: Gradient of the loss with respect to mat_final.

    Returns:
        Gradient of the loss with respect to the weights.
    """
    m = k * (n - k)
    dweights = np.zeros(m)

    # Work with copies to avoid mutating inputs
    mat = mat_final.copy()
    d_mat = d_mat.copy()

    # Process in reverse order
    idx = m - 1
    for phase in range(k - 1, -1, -1):
        start_col = k - phase
        end_col = n - 1 - phase
        for col in range(end_col, start_col - 1, -1):
            # Undo the forward operation: mat[:, col] -= weights[idx] * mat[:, col-1]
            mat[:, col] -= weights[idx] * mat[:, col - 1]
            # Now mat[:, col-1] is the pre-operation value

            # Gradient of weight: dL/dw = dot(d_mat[:, col], mat_before[:, col-1])
            dweights[idx] = np.dot(d_mat[:, col], mat[:, col - 1])

            # Propagate gradient to mat[:, col-1]
            d_mat[:, col - 1] += weights[idx] * d_mat[:, col]

            idx -= 1

    return dweights


def plucker_coordinates(mat: np.ndarray) -> dict[tuple[int, ...], float]:
    """Compute all k x k minors (Plucker coordinates) of a k x n matrix.

    Args:
        mat: k x n matrix.

    Returns:
        Dictionary mapping index tuples (i_1, ..., i_k) to determinant values.
    """
    from itertools import combinations

    k, n = mat.shape
    coords: dict[tuple[int, ...], float] = {}
    for cols in combinations(range(n), k):
        submatrix = mat[:, list(cols)]
        coords[cols] = float(np.linalg.det(submatrix))
    return coords
