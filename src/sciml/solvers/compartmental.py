"""Compartmental epidemic models (SI / SIR / SIRS) with time-varying beta(t).

Integrated with a fixed-step RK4 (pure numpy, no SciPy) so the SINDy example
runs without extra dependencies. The right-hand sides match the dengue notebook.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np


def rk4_integrate(rhs: Callable[[float, np.ndarray], np.ndarray],
                  y0: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Classic RK4 on the time grid ``t``. Returns ``(len(t), len(y0))``.

    Parameters
    ----------
    rhs : Callable[[float, np.ndarray], np.ndarray]
        Right-hand side ``rhs(t, y)`` of the ODE system.
    y0 : np.ndarray
        Initial state vector.
    t : np.ndarray
        Time grid over which to integrate.

    Returns
    -------
    np.ndarray
        Integrated trajectory of shape ``(len(t), len(y0))``.
    """
    y0 = np.asarray(y0, dtype=float)
    ys = np.empty((len(t), len(y0)), dtype=float)
    ys[0] = y0
    y = y0.copy()
    for k in range(len(t) - 1):
        h = t[k + 1] - t[k]
        tk = t[k]
        k1 = np.asarray(rhs(tk, y))
        k2 = np.asarray(rhs(tk + 0.5 * h, y + 0.5 * h * k1))
        k3 = np.asarray(rhs(tk + 0.5 * h, y + 0.5 * h * k2))
        k4 = np.asarray(rhs(tk + h, y + h * k3))
        y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        ys[k + 1] = y
    return ys


def _rhs(model: str, N: float, beta_fn, gamma: float, mu: float, omega: float):
    def rhs(t, y):
        if model == "SI":
            S, I = y
            b = beta_fn(t)
            return np.array([-b * S * I / N, b * S * I / N])
        if model == "SIR":
            S, I, R = y
            b = beta_fn(t)
            return np.array([-b * S * I / N,
                             b * S * I / N - gamma * I,
                             gamma * I])
        # SIRS (births/deaths + waning immunity)
        S, I, R = y
        b = beta_fn(t)
        return np.array([mu * N - b * S * I / N - mu * S + omega * R,
                         b * S * I / N - gamma * I - mu * I,
                         gamma * I - mu * R - omega * R])
    return rhs


def simulate_compartmental(model: str, N: float, I0: float, n_weeks: int,
                           beta_fn: Callable[[float], float], *,
                           gamma: float = 0.0, mu: float = 0.0, omega: float = 0.0,
                           noise_std: float = 0.0,
                           rng: np.random.Generator | None = None) -> Dict[str, np.ndarray]:
    """Simulate SI/SIR/SIRS with time-varying ``beta(t)``.

    Returns a dict with ``t, S, I, R, N, beta_true, model``.

    Parameters
    ----------
    model : str
        Compartmental model to simulate, one of ``"SI"``, ``"SIR"`` or ``"SIRS"``.
    N : float
        Total population size.
    I0 : float
        Initial number of infected individuals.
    n_weeks : int
        Number of weekly time steps to simulate.
    beta_fn : Callable[[float], float]
        Time-varying transmission rate ``beta(t)``.
    gamma : float
        Recovery rate.
    mu : float
        Birth/death rate (used by SIRS).
    omega : float
        Immunity-waning rate (used by SIRS).
    noise_std : float
        Standard deviation (as a fraction of ``N``) of observation noise added
        to the infected series; no noise is added when ``0``.
    rng : np.random.Generator | None
        Random generator for the noise draw; falls back to ``np.random`` when
        ``None``.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with keys ``t``, ``S``, ``I``, ``R``, ``N``, ``beta_true``
        and ``model``.
    """
    t = np.arange(n_weeks, dtype=float)
    y0 = ([N - I0, I0] if model == "SI" else [N - I0, I0, 0.0])
    ys = rk4_integrate(_rhs(model, N, beta_fn, gamma, mu, omega), np.array(y0), t)
    S, I = ys[:, 0], ys[:, 1]
    R = ys[:, 2] if model != "SI" else np.zeros_like(I)

    if noise_std > 0:
        draw = (np.random.normal if rng is None else rng.normal)
        I = np.clip(I + draw(0, noise_std * N, len(I)), 0, N)
        S = np.clip(N - I - R, 0, N)

    return {"t": t, "S": S, "I": I, "R": R, "N": N,
            "beta_true": np.array([beta_fn(tt) for tt in t]), "model": model}
