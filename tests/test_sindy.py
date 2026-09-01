"""Tests for the SINDy engine: sparse solvers, libraries and the model."""

import numpy as np

from sciml.methods.sindy.library import ConcatLibrary, FourierLibrary, PolynomialLibrary
from sciml.methods.sindy.model import SINDy, windowed_coefficients
from sciml.methods.sindy.sparse import ridge_regression, stridge


def test_stridge_recovers_sparse():
    """STRidge recovers the exact support of a sparse system."""
    rng = np.random.default_rng(0)
    Theta = rng.standard_normal((200, 6))
    true = np.array([0.0, 2.5, 0.0, 0.0, -1.3, 0.0])
    y = Theta @ true
    xi = stridge(Theta, y, threshold=0.1)
    assert np.allclose(xi, true, atol=1e-6)
    # spurious tiny coefficients are thresholded to exactly zero
    assert np.count_nonzero(xi) == 2


def test_ridge_equals_lstsq_at_zero_alpha():
    """Ridge with zero penalty equals least squares."""
    rng = np.random.default_rng(1)
    A = rng.standard_normal((50, 4)); y = rng.standard_normal(50)
    assert np.allclose(ridge_regression(A, y, 0.0), np.linalg.lstsq(A, y, rcond=None)[0])


def test_polynomial_library_shapes_and_names():
    """The polynomial library has the expected term count and readable names."""
    lib = PolynomialLibrary(degree=2, include_bias=True)
    X = np.random.rand(7, 2)
    Theta = lib.transform(X)
    names = lib.names(["a", "b"])
    # terms: 1, a, b, a^2, a*b, b^2
    assert Theta.shape == (7, 6) and len(names) == 6 and "a*b" in names


def test_fourier_and_concat():
    """Concatenating libraries sums their feature counts."""
    lib = ConcatLibrary([PolynomialLibrary(1), FourierLibrary(2, period=10.0)])
    X = np.linspace(0, 10, 20)[:, None]
    Theta = lib.transform(X)
    # poly(deg1): 1, x  -> 2 ; fourier(2 harmonics): 4 ; total 6
    assert Theta.shape == (20, 6)


def test_sindy_identifies_linear_system():
    """SINDy identifies the rate of a scalar decay and no bias."""
    # x' = -0.5 x  -> coefficient on x is -0.5, bias ~0.
    t = np.linspace(0, 10, 500)
    x = np.exp(-0.5 * t)[:, None]
    model = SINDy(PolynomialLibrary(degree=1), threshold=0.05).fit(x, t=t, input_names=["x"])
    coef = model.coef_.ravel()  # [bias, x]
    assert abs(coef[1] - (-0.5)) < 0.02 and abs(coef[0]) < 0.02


def test_windowed_coefficients():
    """Windowed fitting returns one coefficient set per window centre."""
    t = np.linspace(0, 10, 60)
    Theta = np.c_[np.ones_like(t), t]
    y = 2.0 * np.ones_like(t)
    centers, coeffs = windowed_coefficients(Theta, y, t, window=8, step=2, threshold=0.01)
    assert coeffs.shape[0] == len(centers) and coeffs.shape[1] == 2
