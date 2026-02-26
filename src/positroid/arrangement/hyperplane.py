"""Hyperplane arrangements and their matroids.

For a ReLU network, each hidden neuron defines a hyperplane
{x : w^T x + b = 0} where the pre-activation changes sign.
"""

from dataclasses import dataclass

import numpy as np

from positroid.matroid.linear_matroid import linear_matroid_from_vectors
from positroid.matroid.matroid import Matroid


@dataclass(frozen=True)
class Hyperplane:
    """A hyperplane in R^n defined by normal . x + bias = 0."""

    normal: np.ndarray  # shape (n,)
    bias: float

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Hyperplane):
            return NotImplemented
        return bool(np.allclose(self.normal, other.normal) and np.isclose(self.bias, other.bias))

    def __hash__(self) -> int:
        return hash((tuple(self.normal.tolist()), self.bias))


class HyperplaneArrangement:
    """A collection of hyperplanes in R^n."""

    def __init__(self, hyperplanes: list[Hyperplane]) -> None:
        self._hyperplanes = hyperplanes

    @property
    def ambient_dim(self) -> int:
        if not self._hyperplanes:
            return 0
        return len(self._hyperplanes[0].normal)

    @property
    def num_hyperplanes(self) -> int:
        return len(self._hyperplanes)

    @property
    def hyperplanes(self) -> list[Hyperplane]:
        return self._hyperplanes

    def normal_matrix(self) -> np.ndarray:
        """m x n matrix of normal vectors (rows are normals)."""
        return np.array([h.normal for h in self._hyperplanes])

    def bias_vector(self) -> np.ndarray:
        """m-vector of biases."""
        return np.array([h.bias for h in self._hyperplanes])

    def augmented_matrix(self) -> np.ndarray:
        """m x (n+1) matrix [normals | biases] for affine matroid computation."""
        normals = self.normal_matrix()
        biases = self.bias_vector().reshape(-1, 1)
        return np.hstack([normals, biases])

    def linear_matroid(self, tol: float = 1e-10) -> Matroid:
        """Matroid of the arrangement from normal vectors only.

        This is the linear matroid of the rows of the normal matrix.
        For TP weight matrices with H > n, this is always U(n, H).
        """
        return linear_matroid_from_vectors(self.normal_matrix(), tol=tol)

    def affine_matroid(self, tol: float = 1e-10) -> Matroid:
        """Affine matroid: linear matroid of [normals | biases].

        This captures the full combinatorial structure of the arrangement
        including the positions of the hyperplanes, not just their orientations.
        """
        return linear_matroid_from_vectors(self.augmented_matrix(), tol=tol)

    def sign_vectors(self, points: np.ndarray) -> np.ndarray:
        """Compute sign vectors for a batch of points.

        Args:
            points: (num_points, n) array of input points.

        Returns:
            (num_points, m) array where entry [p, i] is
            +1 if normal_i . point_p + bias_i > 0, -1 if < 0, 0 if = 0.
        """
        normals = self.normal_matrix()
        biases = self.bias_vector()
        # (num_points, m) = (num_points, n) @ (n, m) + (m,)
        values = points @ normals.T + biases
        result: np.ndarray = np.sign(values).astype(int)
        return result
