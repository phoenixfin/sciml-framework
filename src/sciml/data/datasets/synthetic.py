"""Synthetic built-in datasets (pure numpy): sanity checks and usage examples.

- ``lti_demo``: a small stable linear system with control inputs -- the
  ground-truth-known test case for the :mod:`sciml.tasks.sysid` pipeline.
- ``advection_pairs``: GP initial conditions mapped through the periodic
  advection-diffusion solution operator -- a :class:`FunctionPairData`
  example for operator-learning methods.
"""

from __future__ import annotations

import numpy as np

from ...solvers.transport import advection_diffusion_dataset
from ..gp import PeriodicGPSampler
from .base import FunctionPairData, TimeSeriesData

#: Ground-truth dynamics of ``lti_demo``: ``x' = A x + B u``.
A_TRUE = np.array([[-0.15, 0.05], [0.10, -0.20]])
B_TRUE = np.array([[0.10, 0.00], [0.00, -0.15]])


def _smooth_inputs(n: int, dt: float, rng: np.random.Generator) -> np.ndarray:
    """Two smooth, band-limited random input signals.

    Parameters
    ----------
    n : int
        Number of samples.
    dt : float
        Sample spacing in hours.
    rng : np.random.Generator
        Source of randomness.

    Returns
    -------
    np.ndarray
        Input matrix of shape ``(n, 2)``.
    """
    t = np.arange(n) * dt
    U = np.zeros((n, 2))
    for j in range(2):
        for _ in range(4):
            period = rng.uniform(12.0, 120.0)              # half-day .. 5 days
            U[:, j] += rng.normal(0, 0.5) * np.sin(2 * np.pi * t / period + rng.uniform(0, 6.28))
    return U


def load_lti_demo(n_segments: int = 3, seg_len: int = 400, dt_hours: float = 1.0,
                  noise: float = 0.01, seed: int = 0) -> TimeSeriesData:
    """Stable 2-state linear system with 2 control inputs (ground truth known).

    Simulates ``x' = A x + B u`` (``A``, ``B`` in the module constants
    ``A_TRUE``/``B_TRUE``, also stored in ``meta``) with smooth random
    inputs, sub-stepped RK4 integration and optional measurement noise.
    Channels: ``x1, x2`` (states) and ``u1, u2`` (inputs).

    Parameters
    ----------
    n_segments : int
        Number of independent segments (fresh IC and inputs each).
    seg_len : int
        Samples per segment.
    dt_hours : float
        Sample spacing in hours.
    noise : float
        Standard deviation of additive measurement noise on the states.
    seed : int
        Random seed.

    Returns
    -------
    TimeSeriesData
        The simulated dataset with channels ``["x1", "x2", "u1", "u2"]``.
    """
    rng = np.random.default_rng(seed)
    sub = 10                                                # RK4 sub-steps per sample
    h = dt_hours / sub
    segments = []
    for _ in range(n_segments):
        U = _smooth_inputs(seg_len, dt_hours, rng)
        t = np.arange(seg_len) * dt_hours
        u_of = lambda tt: np.array([np.interp(tt, t, U[:, j]) for j in range(2)])
        x = rng.normal(0, 0.5, size=2)
        X = np.zeros((seg_len, 2))
        X[0] = x
        f = lambda tt, xx: A_TRUE @ xx + B_TRUE @ u_of(tt)
        for k in range(seg_len - 1):
            for m in range(sub):
                tt = t[k] + m * h
                k1 = f(tt, x); k2 = f(tt + h / 2, x + h / 2 * k1)
                k3 = f(tt + h / 2, x + h / 2 * k2); k4 = f(tt + h, x + h * k3)
                x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            X[k + 1] = x
        X += rng.normal(0, noise, size=X.shape)
        segments.append(np.hstack([X, U]))
    return TimeSeriesData(
        segments=segments, channels=["x1", "x2", "u1", "u2"], dt_hours=dt_hours,
        meta={"A": A_TRUE.tolist(), "B": B_TRUE.tolist(), "noise": noise, "seed": seed})


def load_advection_pairs(n_samples: int = 256, grid: int = 128, c: float = 1.0,
                         nu: float = 0.005, t_final: float = 0.5,
                         length_scale: float = 1.5, seed: int = 0) -> FunctionPairData:
    """GP initial conditions through the periodic advection-diffusion operator.

    Input functions are periodic-GP samples on ``[0, 1]``; outputs are the
    exact spectral solution ``u(., t_final)`` -- a clean operator-learning
    dataset (DeepONet/FNO-shaped).

    Parameters
    ----------
    n_samples : int
        Number of (input, output) function pairs.
    grid : int
        Number of spatial grid points.
    c : float
        Advection speed.
    nu : float
        Diffusion coefficient.
    t_final : float
        Time at which the output function is evaluated.
    length_scale : float
        GP kernel length scale of the input functions.
    seed : int
        Random seed.

    Returns
    -------
    FunctionPairData
        The paired dataset on a shared periodic grid.
    """
    x = np.linspace(0.0, 1.0, grid, endpoint=False)
    sampler = PeriodicGPSampler(period=1.0, length_scale=length_scale)
    u0 = sampler.sample(x, n_samples, rng=np.random.default_rng(seed)).astype(float)
    s = advection_diffusion_dataset(u0, c=c, nu=nu, t=t_final, length=1.0)
    return FunctionPairData(u=u0, s=s, u_grid=x, s_grid=x,
                            meta={"c": c, "nu": nu, "t_final": t_final, "seed": seed})
