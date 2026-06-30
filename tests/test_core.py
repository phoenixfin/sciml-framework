import numpy as np

from sciml.core.derivatives import savgol, savgol_derivative
from sciml.core.metrics import abs_error, rel_l2, rel_l2_batch, rmse


def test_rel_l2():
    a = np.array([1., 2., 3.])
    assert rel_l2(a, a) < 1e-9
    assert abs(rel_l2(np.zeros(2), np.array([3., 4.])) - 1.0) < 1e-6
    assert rel_l2_batch(np.random.rand(5, 10), np.random.rand(5, 10)).shape == (5,)
    assert np.allclose(abs_error(np.array([1., -1.]), np.zeros(2)), [1., 1.])
    assert rmse(np.array([1., 2.]), np.array([1., 4.])) > 0


def test_savgol_derivative_exact_on_polynomial():
    # SG with poly>=2 differentiates a quadratic exactly.
    t = np.linspace(0, 4, 200)
    y = 2.0 + 3.0 * t + 0.5 * t ** 2          # y' = 3 + t
    dy = savgol_derivative(y, t, window=11, poly=3)
    assert np.allclose(dy, 3.0 + t, atol=1e-6)


def test_savgol_smoothing_reduces_noise():
    t = np.linspace(0, 10, 400)
    clean = np.sin(t)
    np.random.seed(0)
    noisy = clean + 0.1 * np.random.randn(len(t))
    sm = savgol(noisy, window=21, poly=3)
    assert np.mean((sm - clean) ** 2) < np.mean((noisy - clean) ** 2)
