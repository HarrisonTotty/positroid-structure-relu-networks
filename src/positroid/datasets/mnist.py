"""Full MNIST (28x28) dataset loader with PCA reduction.

Uses sklearn.datasets.fetch_openml to download MNIST-784 (lazy, cached).
Same PCA + standardize pipeline as digits.py but for the full 70k-sample dataset.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import numpy as np

from positroid.datasets.digits import _pca_project

# Cache: (pca_dim, digits_tuple) -> (X_projected, y_labels)
_cache: dict[tuple[int, tuple[int, ...]], tuple[np.ndarray, np.ndarray]] = {}


def _load_and_project_mnist(
    pca_dim: int,
    digits: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
) -> tuple[np.ndarray, np.ndarray]:
    """Load MNIST, filter to requested classes, PCA-project, standardize, cache."""
    key = (pca_dim, digits)
    if key in _cache:
        return _cache[key]

    from sklearn.datasets import fetch_openml

    mnist = fetch_openml("mnist_784", version=1, parser="auto", as_frame=False)
    x_all = mnist.data.astype(np.float64)
    y_all = mnist.target.astype(np.int64)

    mask = np.isin(y_all, digits)
    x_sel = x_all[mask]
    y_sel = y_all[mask]

    # Remap labels to contiguous 0..C-1
    label_map = {d: i for i, d in enumerate(sorted(digits))}
    y_mapped = np.array([label_map[int(yi)] for yi in y_sel], dtype=np.int64)

    x_projected, _, _ = _pca_project(x_sel, pca_dim)

    std = x_projected.std(axis=0)
    std[std < 1e-10] = 1.0
    x_projected = x_projected / std

    _cache[key] = (x_projected, y_mapped)
    return x_projected, y_mapped


def make_mnist(
    n_samples: int = 1000,
    rng: np.random.Generator | None = None,
    *,
    pca_dim: int = 50,
    digits: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
) -> tuple[np.ndarray, np.ndarray]:
    """PCA-reduced MNIST dataset.

    Returns (X, y) where y contains integer class labels 0..C-1.
    First call downloads MNIST via sklearn (cached thereafter).
    """
    if rng is None:
        rng = np.random.default_rng()

    x_full, y_full = _load_and_project_mnist(pca_dim, digits)
    n_available = x_full.shape[0]

    replace = n_samples > n_available
    indices = rng.choice(n_available, size=n_samples, replace=replace)

    return x_full[indices].copy(), y_full[indices].copy()


MNIST_DATASETS: dict[str, Callable[..., tuple[np.ndarray, np.ndarray]]] = {}

for _pca in [20, 50]:
    _name = f"mnist_10class_pca{_pca}"
    MNIST_DATASETS[_name] = partial(make_mnist, pca_dim=_pca)
