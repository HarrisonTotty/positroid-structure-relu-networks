"""ReLU network with totally positive weight matrices."""

import numpy as np

from positroid.linalg.totally_positive import is_totally_positive, random_totally_positive
from positroid.network.relu_network import ReluLayer, ReluNetwork


class TotallyPositiveNetwork:
    """A ReLU network whose weight matrices are totally positive.

    Uses the exponential kernel to generate TP weight matrices.
    Bias vectors are drawn independently.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        if rng is None:
            rng = np.random.default_rng()

        self._layers: list[ReluLayer] = []
        dims = [input_dim, *hidden_dims, output_dim]

        for i in range(len(dims) - 1):
            w = random_totally_positive(dims[i + 1], dims[i], rng=rng)
            b = rng.uniform(-1.0, 1.0, size=dims[i + 1])
            self._layers.append(ReluLayer(weight=w, bias=b))

        self._input_dim = input_dim
        self._hidden_dims = hidden_dims
        self._output_dim = output_dim

    @property
    def layers(self) -> list[ReluLayer]:
        return self._layers

    def to_relu_network(self) -> ReluNetwork:
        """Convert to a standard ReluNetwork."""
        return ReluNetwork(self._layers)

    def verify_total_positivity(self) -> bool:
        """Verify that all weight matrices are totally positive."""
        return all(is_totally_positive(layer.weight) for layer in self._layers)

    def weight_matrices(self) -> list[np.ndarray]:
        """Return list of weight matrices."""
        return [layer.weight for layer in self._layers]
