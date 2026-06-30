import numpy as np

from sciml.solvers.darcy import solve_darcy_2d
from sciml.solvers.kuramoto_sivashinsky import kuramoto_sivashinsky
from sciml.solvers.transport import advection_diffusion_solution
from sciml.solvers.wave1d import wave1d_dalembert


def test_advection_full_period_returns_ic():
    n = 128
    x = np.linspace(0, 1, n, endpoint=False)
    u0 = np.sin(2 * np.pi * x) + 0.5 * np.cos(6 * np.pi * x)
    # pure advection (nu=0), one full period (c*t = L) -> back to IC
    uT = advection_diffusion_solution(u0, c=1.0, nu=0.0, t=1.0, length=1.0)
    assert np.allclose(uT, u0, atol=1e-8)
    # energy preserved under pure advection
    assert abs(np.linalg.norm(uT) - np.linalg.norm(u0)) < 1e-8


def test_wave1d_dalembert_initial_condition():
    f0 = lambda x: np.sin(2 * np.pi * x)
    x = np.linspace(0, 1, 50)
    assert np.allclose(wave1d_dalembert(f0, x, t=0.0, c=1.0, length=1.0), f0(x))
    # bounded by the IC amplitude
    uT = wave1d_dalembert(f0, x, t=0.3, c=1.0, length=1.0)
    assert np.abs(uT).max() <= 1.0 + 1e-9


def test_kuramoto_sivashinsky_bounded():
    data = kuramoto_sivashinsky(n=128, length=22.0, t_final=40.0, dt=0.25, n_save=80)
    u = data["u"]
    assert u.shape[1] == 128 and np.isfinite(u).all()
    assert np.abs(u).max() < 10            # KS stays O(1), no blow-up


def test_darcy_manufactured_solution():
    # a = 1 -> -laplacian(u) = f. Use u = sin(pi x) sin(pi y), f = 2 pi^2 u.
    m = 33
    xy = np.linspace(0, 1, m)
    X, Y = np.meshgrid(xy, xy, indexing="ij")
    u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    f = 2 * np.pi**2 * u_exact
    u = solve_darcy_2d(np.ones((m, m)), f)
    rel = np.linalg.norm(u - u_exact) / np.linalg.norm(u_exact)
    assert rel < 0.02                      # second-order FD accuracy
    assert u.min() > -1e-9                 # positivity (maximum principle)
