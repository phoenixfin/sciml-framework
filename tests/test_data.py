import numpy as np

from sciml.data.gp import GPSampler, PeriodicGPSampler
from sciml.data.interp import interp_many, interp_to_grid


def test_periodic_boundary_gap_small():
    x = np.linspace(0, 10, 100, dtype=np.float32)
    s = PeriodicGPSampler(period=10.0, length_scale=2.0, amplitude=0.4, mean=1.0)
    np.random.seed(0)
    out = s.sample(x, 200)
    assert out.shape == (200, 100) and out.dtype == np.float32
    assert np.abs(out[:, 0] - out[:, -1]).mean() < 0.01


def test_clipping_and_nonperiodic():
    x = np.linspace(0, 10, 30, dtype=np.float32)
    assert PeriodicGPSampler(period=10.0, amplitude=2.0, clip_min=0.0).sample(x, 20).min() >= 0.0
    assert GPSampler(length_scale=1.0).sample(x, 3).shape == (3, 30)


def test_interp():
    out = interp_to_grid(np.array([0.5, 1.5]), np.array([0., 1., 2.]), np.array([0., 10., 20.]))
    assert np.allclose(out, [5., 15.]) and out.dtype == np.float32
    assert interp_many(np.linspace(0, 1, 9), np.linspace(0, 1, 5),
                       np.random.rand(4, 5).astype(np.float32)).shape == (4, 9)
