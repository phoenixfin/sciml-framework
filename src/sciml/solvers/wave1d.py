"""1D wave equation reference (d'Alembert) on a periodic domain (pure numpy).

For ``u_tt = c^2 u_xx`` with ``u(x,0)=f(x)`` and ``u_t(x,0)=0`` the solution is
``u(x,t) = 0.5 (f(x-ct) + f(x+ct))`` (periodic wrap). Used as the reference for
the 1D-wave PINN example.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def wave1d_dalembert(f0: Callable[[np.ndarray], np.ndarray], x: np.ndarray,
                     t: float, c: float, length: float) -> np.ndarray:
    """Reference displacement at coords ``x`` and time ``t``.

    Parameters
    ----------
    f0 : Callable[[np.ndarray], np.ndarray]
        Initial displacement profile ``f(x)``, evaluated on arrays.
    x : np.ndarray
        Spatial coordinates at which to evaluate the solution.
    t : float
        Time at which to evaluate the solution.
    c : float
        Wave propagation speed.
    length : float
        Length of the periodic spatial domain.

    Returns
    -------
    np.ndarray
        Displacement ``u(x, t)`` at the given coordinates.
    """
    x = np.asarray(x, dtype=float)
    xm = np.mod(x - c * t, length)
    xp = np.mod(x + c * t, length)
    return 0.5 * (np.asarray(f0(xm)) + np.asarray(f0(xp)))
