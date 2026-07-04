"""Canonical dynamical systems (ODE right-hand sides) + a simulate helper.

Pure numpy. Used by the SINDy / DMD / Neural-ODE examples.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .compartmental import rk4_integrate


def linear_decay(k: float = 0.5) -> Callable:
    """x' = -k x  (state dim 1).

    Parameters
    ----------
    k : float
        Decay rate coefficient.

    Returns
    -------
    Callable
        Right-hand side ``rhs(t, y)`` for the linear decay ODE.
    """
    return lambda t, y: np.array([-k * y[0]])


def harmonic_oscillator(omega: float = 1.0) -> Callable:
    """Undamped oscillator x'' = -omega^2 x, as a 1st-order system [x, v].

    Parameters
    ----------
    omega : float
        Angular frequency of the oscillator.

    Returns
    -------
    Callable
        Right-hand side ``rhs(t, y)`` for the oscillator system ``[x, v]``.
    """
    return lambda t, y: np.array([y[1], -omega**2 * y[0]])


def lotka_volterra(a: float = 1.0, b: float = 0.1, c: float = 1.5, d: float = 0.075) -> Callable:
    """Predator-prey: x'=a x - b x y,  y'=-c y + d x y  (state [prey, predator]).

    Parameters
    ----------
    a : float
        Prey growth rate.
    b : float
        Predation rate coefficient.
    c : float
        Predator death rate.
    d : float
        Predator growth rate per prey consumed.

    Returns
    -------
    Callable
        Right-hand side ``rhs(t, y)`` for the Lotka-Volterra system.
    """
    return lambda t, y: np.array([a * y[0] - b * y[0] * y[1],
                                  -c * y[1] + d * y[0] * y[1]])


def lorenz(sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0) -> Callable:
    """Lorenz system (chaotic) with the classic parameters.

    Parameters
    ----------
    sigma : float
        Prandtl number parameter.
    rho : float
        Rayleigh number parameter.
    beta : float
        Geometric aspect-ratio parameter.

    Returns
    -------
    Callable
        Right-hand side ``rhs(t, y)`` for the Lorenz system.
    """
    return lambda t, y: np.array([sigma * (y[1] - y[0]),
                                  y[0] * (rho - y[2]) - y[1],
                                  y[0] * y[1] - beta * y[2]])


def van_der_pol(mu: float = 1.5) -> Callable:
    """Van der Pol oscillator (nonlinear limit cycle): x'=y, y'=mu(1-x^2)y - x.

    Parameters
    ----------
    mu : float
        Nonlinearity / damping strength parameter.

    Returns
    -------
    Callable
        Right-hand side ``rhs(t, y)`` for the Van der Pol oscillator.
    """
    return lambda t, y: np.array([y[1], mu * (1.0 - y[0] ** 2) * y[1] - y[0]])


def fitzhugh_nagumo(a: float = 0.7, b: float = 0.8, eps: float = 0.08,
                    current: float = 0.5) -> Callable:
    """FitzHugh-Nagumo excitable neuron model (state [v, w]).

    Parameters
    ----------
    a : float
        Recovery variable offset parameter.
    b : float
        Recovery variable coupling parameter.
    eps : float
        Time-scale separation parameter.
    current : float
        External stimulus current.

    Returns
    -------
    Callable
        Right-hand side ``rhs(t, y)`` for the FitzHugh-Nagumo model.
    """
    return lambda t, y: np.array([y[0] - y[0] ** 3 / 3.0 - y[1] + current,
                                  eps * (y[0] + a - b * y[1])])


def simulate(rhs: Callable, y0: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Integrate ``rhs`` from ``y0`` over times ``t`` (RK4). Returns ``(len(t), dim)``.

    Parameters
    ----------
    rhs : Callable
        Right-hand side ``rhs(t, y)`` of the ODE system.
    y0 : np.ndarray
        Initial state vector.
    t : np.ndarray
        Time grid over which to integrate.

    Returns
    -------
    np.ndarray
        Integrated trajectory of shape ``(len(t), dim)``.
    """
    return rk4_integrate(rhs, np.asarray(y0, dtype=float), np.asarray(t, dtype=float))
