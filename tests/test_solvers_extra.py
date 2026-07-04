import numpy as np

from sciml.solvers.burgers import burgers_solution
from sciml.solvers.dynamical import (
    fitzhugh_nagumo,
    harmonic_oscillator,
    lorenz,
    lotka_volterra,
    simulate,
    van_der_pol,
)
from sciml.solvers.heat import heat_solution


def test_harmonic_energy_conserved():
    t = np.arange(0, 20, 0.01)
    X = simulate(harmonic_oscillator(1.5), [1.0, 0.0], t)
    energy = 1.5**2 * X[:, 0] ** 2 + X[:, 1] ** 2
    assert np.std(energy) / np.mean(energy) < 1e-3


def test_lotka_volterra_positive():
    t = np.arange(0, 30, 0.01)
    X = simulate(lotka_volterra(), [10.0, 5.0], t)
    assert X.min() > 0.0


def test_lorenz_bounded():
    t = np.arange(0, 40, 0.005)
    X = simulate(lorenz(), [1.0, 1.0, 1.0], t)
    assert np.isfinite(X).all() and np.abs(X).max() < 100


def test_heat_high_mode_decays():
    n = 128
    x = np.linspace(0, 1, n, endpoint=False)
    m, nu, t = 4, 0.01, 0.05
    u0 = np.sin(2 * np.pi * m * x)
    uT = heat_solution(u0, nu, t, length=1.0)
    expected = np.exp(-nu * (2 * np.pi * m) ** 2 * t)
    assert abs(np.max(np.abs(uT)) - expected) < 1e-3
    # constant (mean) mode is preserved
    u0c = u0 + 2.0
    assert abs(np.mean(heat_solution(u0c, nu, t)) - np.mean(u0c)) < 1e-6


def test_van_der_pol_limit_cycle_bounded():
    t = np.arange(0, 40, 0.01)
    X = simulate(van_der_pol(1.5), [2.0, 0.0], t)
    assert np.isfinite(X).all() and np.abs(X[:, 0]).max() < 5


def test_fitzhugh_nagumo_bounded():
    t = np.arange(0, 200, 0.05)
    X = simulate(fitzhugh_nagumo(), [-1.0, 1.0], t)
    assert np.isfinite(X).all() and np.abs(X).max() < 5


def test_burgers_conserves_mean_dissipates_energy():
    n = 128
    x = np.linspace(0, 1, n, endpoint=False)
    u0 = np.sin(2 * np.pi * x) + 0.3
    uT = burgers_solution(u0, nu=0.02, t_final=0.3, length=1.0, nt=1000)
    assert np.isfinite(uT).all()
    assert abs(np.mean(uT) - np.mean(u0)) < 1e-3          # mean (mass) conserved
    assert np.mean(uT ** 2) < np.mean(u0 ** 2)            # viscous energy decay
