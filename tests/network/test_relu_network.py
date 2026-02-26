import numpy as np

from positroid.network.relu_network import ReluLayer, ReluNetwork


class TestReluNetwork:
    def _make_simple_network(self) -> ReluNetwork:
        """2 inputs, 3 hidden, 1 output."""
        layer1 = ReluLayer(
            weight=np.array([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]]),
            bias=np.array([0.0, 0.0, 0.0]),
        )
        layer2 = ReluLayer(
            weight=np.array([[1.0, 1.0, 1.0]]),
            bias=np.array([0.0]),
        )
        return ReluNetwork([layer1, layer2])

    def test_dimensions(self):
        net = self._make_simple_network()
        assert net.input_dim == 2
        assert net.output_dim == 1
        assert net.hidden_dims == [3]
        assert net.num_layers == 2

    def test_forward_single(self):
        net = self._make_simple_network()
        x = np.array([1.0, 2.0])
        y = net.forward(x)
        # Hidden: [1, 2, -1] -> ReLU -> [1, 2, 0]
        # Output: 1 + 2 + 0 = 3
        assert y.shape == (1,)
        np.testing.assert_allclose(y, [3.0])

    def test_forward_batch(self):
        net = self._make_simple_network()
        x = np.array([[1.0, 2.0], [-1.0, 3.0]])
        y = net.forward(x)
        assert y.shape == (2, 1)

    def test_pre_activations(self):
        net = self._make_simple_network()
        x = np.array([1.0, 2.0])
        pre = net.pre_activations(x)
        assert len(pre) == 2
        np.testing.assert_allclose(pre[0], [1.0, 2.0, -1.0])

    def test_activation_pattern(self):
        net = self._make_simple_network()
        x = np.array([1.0, 2.0])
        patterns = net.activation_pattern(x)
        assert len(patterns) == 1  # 1 hidden layer
        np.testing.assert_array_equal(patterns[0], [True, True, False])

    def test_hyperplane_arrangement(self):
        net = self._make_simple_network()
        arr = net.hyperplane_arrangement(0)
        assert arr.num_hyperplanes == 3
        assert arr.ambient_dim == 2


class TestTotallyPositiveNetwork:
    def test_creation_and_tp(self):
        from positroid.network.tp_network import TotallyPositiveNetwork

        tp_net = TotallyPositiveNetwork(
            input_dim=2, hidden_dims=[4], output_dim=1, rng=np.random.default_rng(42)
        )
        assert tp_net.verify_total_positivity()

    def test_conversion_to_relu(self):
        from positroid.network.tp_network import TotallyPositiveNetwork

        tp_net = TotallyPositiveNetwork(
            input_dim=2, hidden_dims=[4], output_dim=1, rng=np.random.default_rng(42)
        )
        net = tp_net.to_relu_network()
        assert net.input_dim == 2
        assert net.output_dim == 1
        assert net.hidden_dims == [4]

        x = np.array([1.0, 0.5])
        y = net.forward(x)
        assert y.shape == (1,)
