"""PCA-reduced digit datasets for higher-dimensional positroid experiments.

Uses sklearn.datasets.load_digits (8x8 pixel images, 64 features) with
PCA via numpy SVD to produce binary classification datasets in d dimensions.
The affine matroid rank will be d+1, enabling richer matroid structure than
the rank-3 matroids from 2D toy datasets.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import numpy as np

# Module-level cache: (digit_a, digit_b, pca_dim) -> (X_projected, y_binary)
_cache: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}


def _pca_project(
    x: np.ndarray, n_components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center data and project onto top PCA components via SVD.

    Returns (x_projected, mean, components).
    """
    mean = x.mean(axis=0)
    x_centered = x - mean
    _, _, vt = np.linalg.svd(x_centered, full_matrices=False)
    components = vt[:n_components]
    x_projected: np.ndarray = x_centered @ components.T
    return x_projected, mean, components


def _load_and_project(
    digit_a: int, digit_b: int, pca_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load digits, filter to two classes, PCA-project, standardize, and cache.

    Returns the full projected dataset for subsampling by individual trials.
    y=0 for digit_a, y=1 for digit_b.
    """
    key = (digit_a, digit_b, pca_dim)
    if key in _cache:
        return _cache[key]

    from sklearn.datasets import load_digits

    data = load_digits()
    x_all, y_all = data.data, data.target

    mask = (y_all == digit_a) | (y_all == digit_b)
    x_pair = x_all[mask].astype(np.float64)
    y_pair = (y_all[mask] == digit_b).astype(np.float64)

    x_projected, _, _ = _pca_project(x_pair, pca_dim)

    # Standardize each component to unit variance for training stability
    std = x_projected.std(axis=0)
    std[std < 1e-10] = 1.0
    x_projected = x_projected / std

    _cache[key] = (x_projected, y_pair)
    return x_projected, y_pair


def make_digits(
    n_samples: int = 200,
    rng: np.random.Generator | None = None,
    *,
    digit_a: int = 0,
    digit_b: int = 1,
    pca_dim: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """PCA-reduced digit pair dataset.

    Loads the full digit pair, projects via PCA, then subsamples n_samples
    points (without replacement if enough data, with replacement otherwise).
    """
    if rng is None:
        rng = np.random.default_rng()

    x_full, y_full = _load_and_project(digit_a, digit_b, pca_dim)
    n_available = x_full.shape[0]

    replace = n_samples > n_available
    indices = rng.choice(n_available, size=n_samples, replace=replace)

    return x_full[indices].copy(), y_full[indices].copy()


def _dataset_name(digit_a: int, digit_b: int, pca_dim: int) -> str:
    return f"digits_{digit_a}v{digit_b}_pca{pca_dim}"


def register_digits_dataset(
    digit_a: int, digit_b: int, pca_dim: int,
) -> str:
    """Register a digit dataset configuration and return its registry name."""
    name = _dataset_name(digit_a, digit_b, pca_dim)
    DIGIT_DATASETS[name] = partial(
        make_digits, digit_a=digit_a, digit_b=digit_b, pca_dim=pca_dim,
    )
    return name


DIGIT_DATASETS: dict[str, Callable[..., tuple[np.ndarray, np.ndarray]]] = {}

_DEFAULT_CONFIGS: list[tuple[int, int, int]] = [
    (0, 1, 2),
    (0, 1, 3),
    (0, 1, 5),
    (0, 1, 10),
    (3, 8, 2),
    (3, 8, 3),
    (3, 8, 5),
    (3, 8, 10),
]

for _da, _db, _pd in _DEFAULT_CONFIGS:
    register_digits_dataset(_da, _db, _pd)
