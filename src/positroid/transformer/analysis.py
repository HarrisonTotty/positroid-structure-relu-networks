"""Empirical analysis tools for positroid structure in trained networks.

Proposal E from the Cognihedron at Scale brainstorming document.

Provides tools to test whether trained transformer weights and attention
patterns exhibit positroid structure, using the existing AM matroid
and totally positive matrix analysis infrastructure.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from positroid.linalg.totally_positive import is_totally_nonnegative, is_totally_positive
from positroid.positroid_cell.boundary_map import (
    boundary_measurement_backward,
    boundary_measurement_matrix,
)


def weight_effective_rank(W: np.ndarray, threshold: float = 0.01) -> dict[str, Any]:
    """Compute effective rank and SVD statistics of a weight matrix.

    Args:
        W: 2D weight matrix.
        threshold: fraction of total singular value energy below which
            dimensions are considered negligible.

    Returns:
        Dict with singular_values, effective_rank, full_rank,
        condition_number, cumulative_energy.
    """
    sv = np.linalg.svd(W, compute_uv=False)
    total = sv.sum()
    cumulative = np.cumsum(sv) / total if total > 0 else np.ones_like(sv)
    eff_rank = int(np.searchsorted(cumulative, 1.0 - threshold) + 1)

    return {
        "singular_values": sv,
        "effective_rank": eff_rank,
        "full_rank": min(W.shape),
        "condition_number": float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"),
        "cumulative_energy": cumulative,
    }


def check_approximate_tp(
    W: np.ndarray, max_order: int | None = None, tol: float = 1e-6
) -> dict[str, Any]:
    """Check if a matrix is approximately totally positive.

    Computes all minors up to max_order and classifies them as
    positive, zero, or negative.

    Args:
        W: 2D matrix to check.
        max_order: Maximum minor order to check (default: min(rows, cols)).
        tol: Tolerance for positivity.

    Returns:
        Dict with is_tp, is_tn, min_minor, max_minor, fraction_positive,
        n_negative, n_zero, n_positive, total_minors.
    """
    from itertools import combinations

    from positroid.linalg.minors import minor

    rows, cols = W.shape
    if max_order is None:
        max_order = min(rows, cols)
    max_order = min(max_order, min(rows, cols))

    vals: list[float] = []
    for size in range(1, max_order + 1):
        for row_sub in combinations(range(rows), size):
            for col_sub in combinations(range(cols), size):
                vals.append(minor(W, row_sub, col_sub))

    n_neg = sum(1 for v in vals if v < -tol)
    n_zero = sum(1 for v in vals if abs(v) <= tol)
    n_pos = sum(1 for v in vals if v > tol)

    # Full TP/TN check only for small matrices
    is_tp = is_totally_positive(W, tol=tol) if max_order >= min(rows, cols) else None
    is_tn = is_totally_nonnegative(W, tol=tol) if max_order >= min(rows, cols) else None

    return {
        "is_tp": is_tp if is_tp is not None else (n_neg == 0 and n_zero == 0),
        "is_tn": is_tn if is_tn is not None else (n_neg == 0),
        "min_minor": float(min(vals)) if vals else 0.0,
        "max_minor": float(max(vals)) if vals else 0.0,
        "n_negative": n_neg,
        "n_zero": n_zero,
        "n_positive": n_pos,
        "total_minors": len(vals),
        "fraction_positive": n_pos / len(vals) if vals else 0.0,
    }


def check_attention_positroid(attn_weights: np.ndarray) -> dict[str, Any]:
    """Check if hard attention patterns form a positroid matroid.

    Extracts the hard attention pattern (argmax per query), builds
    the column matroid of the resulting binary pattern matrix, and
    checks if it's a positroid.

    Args:
        attn_weights: (T, T) attention weight matrix (after softmax).

    Returns:
        Dict with matroid properties (rank, n_bases, is_positroid, is_uniform).
    """
    from positroid.matroid.linear_matroid import linear_matroid_from_vectors
    from positroid.matroid.positroid import is_positroid as check_positroid

    T = attn_weights.shape[0]

    # Hard attention: binary pattern matrix
    hard = np.argmax(attn_weights, axis=-1)
    pattern = np.zeros((T, T))
    for i in range(T):
        pattern[i, hard[i]] = 1.0

    try:
        mat = linear_matroid_from_vectors(pattern.T)
        return {
            "is_positroid": check_positroid(mat),
            "rank": mat.rank,
            "n_bases": len(mat.bases),
            "is_uniform": mat.is_uniform(),
            "ground_set_size": mat.size,
        }
    except Exception as e:
        return {"error": str(e)}


def fit_boundary_measurement(
    target: np.ndarray,
    k: int,
    max_iter: int = 1000,
    lr: float = 0.01,
) -> dict[str, Any]:
    """Fit a boundary measurement matrix to approximate a target.

    Uses Adam to find face weights t such that
    boundary_measurement_matrix(exp(t), k, n) ≈ target.

    Args:
        target: (k, n) matrix to approximate.
        k: Grassmannian rank (should match target.shape[0]).
        max_iter: Maximum optimization iterations.
        lr: Learning rate.

    Returns:
        Dict with face_weights, approximation, error, relative_error.
    """
    n = target.shape[1]
    assert target.shape[0] == k
    n_face = k * (n - k)

    raw = np.zeros(n_face)
    # Adam state
    m_adam = np.zeros(n_face)
    v_adam = np.zeros(n_face)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for t in range(1, max_iter + 1):
        weights = np.exp(raw)
        approx = boundary_measurement_matrix(weights, k, n)
        diff = approx - target
        d_approx = 2.0 * diff
        d_weights = boundary_measurement_backward(weights, k, n, approx, d_approx)
        g = d_weights * weights  # chain through exp

        m_adam = beta1 * m_adam + (1 - beta1) * g
        v_adam = beta2 * v_adam + (1 - beta2) * g * g
        m_hat = m_adam / (1 - beta1**t)
        v_hat = v_adam / (1 - beta2**t)
        raw -= lr * m_hat / (np.sqrt(v_hat) + eps)

    weights = np.exp(raw)
    approx = boundary_measurement_matrix(weights, k, n)
    error = float(np.sum((approx - target) ** 2))
    target_norm = float(np.sum(target**2))

    return {
        "face_weights": weights,
        "face_weights_raw": raw,
        "approximation": approx,
        "error": error,
        "relative_error": error / target_norm if target_norm > 0 else float("inf"),
    }


def analyze_weight_matrix(W: np.ndarray) -> dict[str, Any]:
    """Full analysis of a weight matrix for positroid-relevant structure.

    Combines effective rank analysis, total positivity check, and
    boundary measurement fitting.

    Args:
        W: 2D weight matrix.

    Returns:
        Dict with rank_info, tp_info, and (for small matrices) fit_info.
    """
    result: dict[str, Any] = {}
    result["rank_info"] = weight_effective_rank(W)

    # Only check TP for small matrices (minors are exponential)
    min_dim = min(W.shape)
    if min_dim <= 10:
        result["tp_info"] = check_approximate_tp(W)
    else:
        # Check only order-2 minors for large matrices
        result["tp_info"] = check_approximate_tp(W, max_order=2)

    # Attempt boundary measurement fit for small matrices
    if min_dim <= 8:
        k = W.shape[0]
        n = W.shape[1]
        if k < n:
            result["fit_info"] = fit_boundary_measurement(W, k)

    return result
