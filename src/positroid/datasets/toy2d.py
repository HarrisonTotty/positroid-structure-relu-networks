"""2D toy datasets for binary classification experiments.

All datasets return (X, y) where X has shape (n_samples, 2) and y has shape
(n_samples,) with values in {0, 1}.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def make_moons(
    n_samples: int = 200,
    noise: float = 0.1,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Two interleaving half-circles.

    Upper moon: centered at origin, radius 1, class 0.
    Lower moon: centered at (1, -0.3), radius 1, flipped, class 1.
    """
    if rng is None:
        rng = np.random.default_rng()

    n0 = n_samples // 2
    n1 = n_samples - n0

    theta0 = np.linspace(0, np.pi, n0)
    theta1 = np.linspace(0, np.pi, n1)

    x0 = np.column_stack([np.cos(theta0), np.sin(theta0)])
    x1 = np.column_stack([1 - np.cos(theta1), -0.3 - np.sin(theta1)])

    pts = np.vstack([x0, x1])
    pts += rng.normal(0, noise, pts.shape)
    y = np.concatenate([np.zeros(n0), np.ones(n1)])

    perm = rng.permutation(n_samples)
    return pts[perm], y[perm]


def make_circles(
    n_samples: int = 200,
    noise: float = 0.05,
    factor: float = 0.5,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Concentric circles: outer class 0, inner class 1."""
    if rng is None:
        rng = np.random.default_rng()

    n0 = n_samples // 2
    n1 = n_samples - n0

    theta0 = rng.uniform(0, 2 * np.pi, n0)
    theta1 = rng.uniform(0, 2 * np.pi, n1)

    x0 = np.column_stack([np.cos(theta0), np.sin(theta0)])
    x1 = factor * np.column_stack([np.cos(theta1), np.sin(theta1)])

    pts = np.vstack([x0, x1])
    pts += rng.normal(0, noise, pts.shape)
    y = np.concatenate([np.zeros(n0), np.ones(n1)])

    perm = rng.permutation(n_samples)
    return pts[perm], y[perm]


def make_spirals(
    n_samples: int = 200,
    noise: float = 0.2,
    n_turns: float = 1.5,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Two interleaving Archimedean spirals."""
    if rng is None:
        rng = np.random.default_rng()

    n0 = n_samples // 2
    n1 = n_samples - n0

    theta0 = np.linspace(0, n_turns * 2 * np.pi, n0)
    r0 = theta0 / (n_turns * 2 * np.pi)
    x0 = np.column_stack([r0 * np.cos(theta0), r0 * np.sin(theta0)])

    theta1 = np.linspace(0, n_turns * 2 * np.pi, n1)
    r1 = theta1 / (n_turns * 2 * np.pi)
    x1 = np.column_stack([r1 * np.cos(theta1 + np.pi), r1 * np.sin(theta1 + np.pi)])

    pts = np.vstack([x0, x1])
    pts += rng.normal(0, noise / (n_turns * 2 * np.pi), pts.shape)
    y = np.concatenate([np.zeros(n0), np.ones(n1)])

    perm = rng.permutation(n_samples)
    return pts[perm], y[perm]


def make_xor(
    n_samples: int = 200,
    noise: float = 0.1,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """XOR pattern: four Gaussian blobs at (+-1, +-1).

    Class 0: same-sign quadrants (+,+) and (-,-).
    Class 1: opposite-sign quadrants (+,-) and (-,+).
    """
    if rng is None:
        rng = np.random.default_rng()

    n_per = n_samples // 4
    remainder = n_samples - 4 * n_per

    centers = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]], dtype=float)
    labels = np.array([0, 0, 1, 1])
    counts = [n_per, n_per, n_per, n_per + remainder]

    xs, ys = [], []
    for center, label, count in zip(centers, labels, counts, strict=True):
        xs.append(center + rng.normal(0, noise, (count, 2)))
        ys.append(np.full(count, label))

    pts = np.vstack(xs)
    y = np.concatenate(ys)

    perm = rng.permutation(n_samples)
    return pts[perm], y[perm]


DATASETS: dict[str, Callable[..., tuple[np.ndarray, np.ndarray]]] = {
    "moons": make_moons,
    "circles": make_circles,
    "spirals": make_spirals,
    "xor": make_xor,
}
