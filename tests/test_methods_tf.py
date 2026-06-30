"""TF-dependent tests. Skipped automatically when TensorFlow is unavailable."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")


# ---- DeepONet --------------------------------------------------------------
def test_deeponet_operator_shapes():
    from sciml.methods.deeponet.operator import DeepONetOperator
    op = DeepONetOperator.build(n_sensors=20, n_branches=2, coord_dim=2,
                                width=8, hidden=[16, 16])
    out = op([tf.random.normal((3, 20)), tf.random.normal((3, 20))], tf.random.normal((7, 2)))
    assert out.shape == (3, 7)


def test_swe_ic_shortcut_exact_at_t0():
    from sciml.problems.swe.model import SWEDeepONet, warmup
    model = warmup(SWEDeepONet(n_sensors=20, width=8, hidden=(16, 16)), 20)
    n = 10
    xt0 = tf.constant(np.stack([np.linspace(0, 10, n), np.zeros(n)], 1).astype(np.float32))
    h0x = tf.constant((np.random.rand(4, n) + 0.5).astype(np.float32))
    h, hu = model(tf.zeros((4, 20)), tf.zeros((4, 20)), xt0, h0x, tf.zeros((4, n)))
    assert float(tf.reduce_max(tf.abs(h - h0x))) < 1e-5
    assert float(tf.reduce_max(tf.abs(hu))) < 1e-6


def test_grid_interp_matches_numpy():
    from sciml.tf_utils import grid_interp
    grid = np.linspace(0, 10, 50).astype(np.float32)[None, :]  # f(x)=x
    xq = np.array([0.0, 2.5, 7.3, 10.0], dtype=np.float32)
    out = grid_interp(tf.constant(grid), tf.constant(xq), tf.constant(10.0)).numpy()[0]
    assert np.allclose(out, xq, atol=1e-4)


# ---- PINN ------------------------------------------------------------------
def test_pinn_build_mlp_and_fourier():
    from sciml.methods.pinn.networks import build_mlp
    m = build_mlp(2, hidden=2, width=16, out_dim=1, fourier_freq=8)
    assert m(tf.zeros((5, 2))).shape == (5, 1)


def test_derivatives_2d_on_known_field():
    from sciml.methods.pinn.gradients import derivatives_2d
    # u = x^2 + 2 t^2  ->  u_xx = 2, u_tt = 4
    model = lambda xt, training=False: (xt[:, 0:1] ** 2 + 2.0 * xt[:, 1:2] ** 2)
    xt = tf.constant(np.random.rand(12, 2).astype(np.float32))
    d = derivatives_2d(model, xt)
    assert np.allclose(d["u_xx"].numpy(), 2.0, atol=1e-3)
    assert np.allclose(d["u_tt"].numpy(), 4.0, atol=1e-3)
