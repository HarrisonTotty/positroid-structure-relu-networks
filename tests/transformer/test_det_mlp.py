"""Tests for DetMLP ablation variants."""

import numpy as np
import pytest

from positroid.transformer.det_mlp import DetMLP


class TestDetMLP:
    """Tests for each matrix_mode of DetMLP."""

    @pytest.fixture(params=["positroid", "unconstrained", "fixed_random"])
    def mode(self, request):
        return request.param

    def test_forward_shapes(self, mode):
        mlp = DetMLP(d_in=16, d_out=8, n_cells=4, k=2, n=6, matrix_mode=mode)
        X = np.random.default_rng(0).standard_normal((10, 16))
        out, cache = mlp.forward(X)
        assert out.shape == (10, 8)

    def test_backward_shapes(self, mode):
        mlp = DetMLP(d_in=8, d_out=4, n_cells=3, k=2, n=5, matrix_mode=mode)
        X = np.random.default_rng(0).standard_normal((4, 8))
        out, cache = mlp.forward(X)
        d_X, grads = mlp.backward(np.ones_like(out), cache)
        assert d_X.shape == X.shape
        assert len(grads) == len(mlp.params())

    def test_param_count(self, mode):
        mlp = DetMLP(d_in=16, d_out=8, n_cells=4, k=2, n=6, matrix_mode=mode)
        count = mlp.param_count()
        assert count == sum(p.size for p in mlp.params())
        assert count > 0

    def test_gradient_finite_difference(self, mode):
        rng = np.random.default_rng(42)
        mlp = DetMLP(d_in=8, d_out=4, n_cells=3, k=2, n=5, matrix_mode=mode)
        X = rng.standard_normal((4, 8))

        def loss_fn():
            out, _ = mlp.forward(X)
            return np.sum(out**2)

        out, cache = mlp.forward(X)
        d_out = 2.0 * out
        _, grads = mlp.backward(d_out, cache)

        params = mlp.params()
        eps = 1e-5
        for p_idx, (p, g) in enumerate(zip(params, grads)):
            p_flat = p.ravel()
            g_flat = g.ravel()
            for j in range(min(3, len(p_flat))):
                old = p_flat[j]
                p_flat[j] = old + eps
                f_plus = loss_fn()
                p_flat[j] = old - eps
                f_minus = loss_fn()
                p_flat[j] = old
                fd = (f_plus - f_minus) / (2 * eps)
                assert abs(fd - g_flat[j]) < 1e-3 + 1e-3 * abs(fd), (
                    f"Mode {mode}, param {p_idx}[{j}]: fd={fd:.6f}, analytical={g_flat[j]:.6f}"
                )


class TestDetMLPPositroidMatchesTropical:
    """Verify positroid mode produces identical results to TropicalMLP."""

    def test_same_output(self):
        from positroid.transformer.tropical_mlp import TropicalMLP

        d_in, d_out, n_cells, k, n, seed = 8, 4, 3, 2, 5, 42
        trop = TropicalMLP(d_in, d_out, n_cells, k, n, seed)
        det = DetMLP(d_in, d_out, n_cells, k, n, "positroid", seed)

        X = np.random.default_rng(0).standard_normal((4, d_in))
        out_trop, _ = trop.forward(X)
        out_det, _ = det.forward(X)

        np.testing.assert_allclose(out_trop, out_det, atol=1e-12)


class TestDetMLPModes:
    """Test mode-specific properties."""

    def test_fixed_random_has_fewer_params(self):
        pos = DetMLP(d_in=16, d_out=16, n_cells=4, k=2, n=8, matrix_mode="positroid")
        unc = DetMLP(d_in=16, d_out=16, n_cells=4, k=2, n=8, matrix_mode="unconstrained")
        fix = DetMLP(d_in=16, d_out=16, n_cells=4, k=2, n=8, matrix_mode="fixed_random")
        assert fix.param_count() < pos.param_count()
        assert fix.param_count() < unc.param_count()

    def test_unconstrained_has_more_matrix_params(self):
        pos = DetMLP(d_in=16, d_out=16, n_cells=4, k=2, n=8, matrix_mode="positroid")
        unc = DetMLP(d_in=16, d_out=16, n_cells=4, k=2, n=8, matrix_mode="unconstrained")
        # positroid: 4 * 2*(8-2) = 48 matrix params
        # unconstrained: 4 * 2*8 = 64 matrix params
        assert unc.param_count() > pos.param_count()

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown matrix_mode"):
            DetMLP(d_in=8, d_out=4, matrix_mode="bogus")

    def test_nonzero_output(self):
        for mode in ("positroid", "unconstrained", "fixed_random"):
            mlp = DetMLP(d_in=8, d_out=4, n_cells=3, k=2, n=5, matrix_mode=mode)
            X = np.random.default_rng(0).standard_normal((4, 8))
            out, _ = mlp.forward(X)
            assert not np.allclose(out, 0.0), f"Mode {mode} produced all-zero output"
