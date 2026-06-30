"""End-to-end checks that the numpy example pipelines actually recover dynamics."""

import numpy as np

from sciml.methods.dmd import DMD
from sciml.methods.sindy import PolynomialLibrary, SINDy
from sciml.solvers.dynamical import (harmonic_oscillator, lorenz,
                                      lotka_volterra, simulate)


def _coef(model, target_idx, feature):
    i = model.feature_names_.index(feature)
    return model.coef_[i, target_idx]


def test_sindy_recovers_lorenz():
    t = np.arange(0, 12, 0.002)
    X = simulate(lorenz(), [-8.0, 8.0, 27.0], t)
    m = SINDy(PolynomialLibrary(2), threshold=0.1).fit(X, t=t, input_names=["x", "y", "z"])
    assert abs(_coef(m, 0, "x") - (-10.0)) < 0.1   # x' = 10(y - x)
    assert abs(_coef(m, 0, "y") - 10.0) < 0.1
    assert abs(_coef(m, 1, "x") - 28.0) < 0.2      # y' = 28x - y - xz
    assert abs(_coef(m, 1, "x*z") - (-1.0)) < 0.05
    assert abs(_coef(m, 2, "z") - (-8.0 / 3.0)) < 0.05  # z' = xy - (8/3) z
    assert abs(_coef(m, 2, "x*y") - 1.0) < 0.05


def test_sindy_recovers_lotka_volterra():
    t = np.arange(0, 30, 0.01)
    X = simulate(lotka_volterra(), [10.0, 5.0], t)
    m = SINDy(PolynomialLibrary(2), threshold=0.05).fit(X, t=t, input_names=["x", "y"])
    assert abs(_coef(m, 0, "x") - 1.0) < 0.05
    assert abs(_coef(m, 0, "x*y") - (-0.1)) < 0.02
    assert abs(_coef(m, 1, "y") - (-1.5)) < 0.05
    assert abs(_coef(m, 1, "x*y") - 0.075) < 0.02


def test_dmd_recovers_oscillator_frequency():
    omega, dt = 2.0, 0.05
    t = np.arange(0, 12, dt)
    X = simulate(harmonic_oscillator(omega), [1.0, 0.0], t)
    dmd = DMD(rank=2).fit(X.T, dt=dt)
    assert abs(np.max(np.abs(dmd.omega.imag)) - omega) < 1e-2
