"""Feedforward ReLU network construction and computation."""

from dataclasses import dataclass

import numpy as np

from positroid.arrangement.hyperplane import Hyperplane, HyperplaneArrangement


@dataclass
class ReluLayer:
    """A single layer: z = ReLU(W @ x + b) or z = W @ x + b (output)."""

    weight: np.ndarray  # shape (out_dim, in_dim)
    bias: np.ndarray  # shape (out_dim,)


class ReluNetwork:
    """A feedforward ReLU network.

    For a network with L layers:
    - Layers 0 to L-2 have ReLU activations.
    - Layer L-1 (output) is linear (no activation).
    """

    def __init__(self, layers: list[ReluLayer]) -> None:
        if not layers:
            raise ValueError("Network must have at least one layer")
        self._layers = layers

    @property
    def layers(self) -> list[ReluLayer]:
        return self._layers

    @property
    def input_dim(self) -> int:
        return int(self._layers[0].weight.shape[1])

    @property
    def output_dim(self) -> int:
        return int(self._layers[-1].weight.shape[0])

    @property
    def hidden_dims(self) -> list[int]:
        return [layer.weight.shape[0] for layer in self._layers[:-1]]

    @property
    def num_layers(self) -> int:
        return len(self._layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass. x shape: (batch, input_dim) or (input_dim,)."""
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)

        z = x
        for i, layer in enumerate(self._layers):
            z = z @ layer.weight.T + layer.bias
            if i < len(self._layers) - 1:
                z = np.maximum(z, 0)

        return z.squeeze(0) if single else z

    def pre_activations(self, x: np.ndarray) -> list[np.ndarray]:
        """Compute pre-activation values at each hidden layer."""
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)

        pre_acts = []
        z = x
        for i, layer in enumerate(self._layers):
            pre = z @ layer.weight.T + layer.bias
            pre_acts.append(pre.squeeze(0) if single else pre)
            z = np.maximum(pre, 0) if i < len(self._layers) - 1 else pre

        return pre_acts

    def activation_pattern(self, x: np.ndarray) -> list[np.ndarray]:
        """Compute binary activation pattern at each hidden layer.

        Returns list of boolean arrays (True = neuron active, i.e., pre-act > 0).
        Only includes hidden layers (not the output layer).
        """
        pre_acts = self.pre_activations(x)
        return [pa > 0 for pa in pre_acts[:-1]]

    def hyperplane_arrangement(self, layer_idx: int = 0) -> HyperplaneArrangement:
        """Extract the hyperplane arrangement for a given hidden layer.

        For the first hidden layer (layer_idx=0), the hyperplanes are:
            H_i = {x : W[i,:] . x + b[i] = 0}

        For deeper layers, the hyperplanes are 'bent' and depend on
        earlier layers. Currently only supports layer_idx=0.
        """
        if layer_idx != 0:
            raise NotImplementedError(
                "Bent hyperplanes for deeper layers not yet implemented"
            )

        layer = self._layers[0]
        hyperplanes = [
            Hyperplane(normal=layer.weight[i].copy(), bias=float(layer.bias[i]))
            for i in range(layer.weight.shape[0])
        ]
        return HyperplaneArrangement(hyperplanes)
