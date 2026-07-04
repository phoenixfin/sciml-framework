"""Finite-difference reference for the moving-boundary (obstacle) wave problem.

The string vibrates on the moving domain ``s(tau) < xbar < 1``. Internally the
wave is integrated on a fixed reference grid ``xi in [0, 1]`` (explicit
second-order leapfrog), then mapped to physical coordinates
``xbar = s(tau) + xi * (1 - s(tau))`` and added to the stationary solution.
Pure numpy.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np


def wave_moving_boundary_fdm(
    s_analytic: Callable[[np.ndarray], np.ndarray],
    u_stationary: Callable[[np.ndarray], np.ndarray],
    s_y: float,
    *,
    delta: float,
    t_final: float,
    nx: int = 300,
    cfl: float = 0.45,
    n_snaps: int = 150,
) -> Dict[str, List]:
    """Integrate the reference wave and return snapshots.

    Returns a dict with keys ``t`` (times), ``s`` (contact point),
    ``xbar`` (list of physical grids) and ``u`` (list of displacement fields).

    Parameters
    ----------
    s_analytic : Callable[[np.ndarray], np.ndarray]
        Analytic contact-point trajectory ``s(tau)``, evaluated on arrays.
    u_stationary : Callable[[np.ndarray], np.ndarray]
        Stationary displacement solution ``u(xbar)``, evaluated on arrays.
    s_y : float
        Slope parameter of the moving boundary used in the effective wave speed.
    delta : float
        Amplitude of the initial displacement bump.
    t_final : float
        Final integration time.
    nx : int
        Number of spatial intervals on the reference grid.
    cfl : float
        Courant-Friedrichs-Lewy number controlling the time step.
    n_snaps : int
        Maximum number of snapshots to store.

    Returns
    -------
    Dict[str, List]
        Dictionary with keys ``t`` (times), ``s`` (contact point), ``xbar``
        (physical grids) and ``u`` (displacement fields).
    """
    c_eff = 1.0 / (1.0 - s_y)
    dx = 1.0 / nx
    dt = cfl * dx / c_eff
    nt = int(t_final / dt) + 1
    nu2 = (dt * c_eff / dx) ** 2

    xi = np.linspace(0, 1, nx + 1, dtype=np.float32)
    vc = (delta * np.sin(np.pi * xi)).astype(np.float32)
    vc[0] = vc[-1] = 0.0
    vp = vc.copy()

    out: Dict[str, List] = {"t": [], "s": [], "xbar": [], "u": []}
    stride = max(1, (nt - 1) // n_snaps)

    for n in range(1, nt):
        vn = np.zeros(nx + 1, dtype=np.float32)
        vn[1:-1] = (2 * vc[1:-1] - vp[1:-1]
                    + nu2 * (vc[2:] - 2 * vc[1:-1] + vc[:-2]))
        vn[0] = vn[-1] = 0.0
        vp[:] = vc
        vc[:] = vn
        if n % stride == 0 and len(out["t"]) < n_snaps:
            tau_n = n * dt
            sv = float(np.asarray(s_analytic(np.array([tau_n])))[0])
            xbar_n = sv + xi * (1.0 - sv)
            out["t"].append(tau_n)
            out["s"].append(sv)
            out["xbar"].append(xbar_n.copy())
            out["u"].append((vc + u_stationary(xbar_n)).copy())
    return out
