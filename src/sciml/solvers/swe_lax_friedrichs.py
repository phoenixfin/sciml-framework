"""Lax-Friedrichs reference solver for the 1D Shallow Water Equations.

Conservative form (depth ``h``, momentum ``hu``)::

    h_t  + (hu)_x = 0
    (hu)_t + (hu^2 + 1/2 g h^2)_x = -g h b_x

on a periodic domain with bathymetry ``b(x)``. Pure numpy.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


def lax_friedrichs_swe(
    h0_fn: Callable[[np.ndarray], np.ndarray],
    b_fn: Callable[[np.ndarray], np.ndarray],
    *,
    length: float = 10.0,
    t_final: float = 1.0,
    gravity: float = 9.81,
    nx: int = 400,
    nt: int = 4000,
    t_out: Optional[Sequence[float]] = None,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, List]]:
    """Integrate the 1D SWE and capture one snapshot per time in ``t_out``.

    Returns ``(x, snaps)`` with ``snaps`` keyed by ``"t"``, ``"h"``, ``"hu"``.
    The ``captured`` set prevents duplicate snapshots within the ``1.5*dt``
    capture window.

    Parameters
    ----------
    h0_fn : Callable[[np.ndarray], np.ndarray]
        Initial water depth profile ``h0(x)``, evaluated on arrays.
    b_fn : Callable[[np.ndarray], np.ndarray]
        Bathymetry profile ``b(x)``, evaluated on arrays.
    length : float
        Length of the periodic spatial domain.
    t_final : float
        Final integration time.
    gravity : float
        Gravitational acceleration.
    nx : int
        Number of spatial cells.
    nt : int
        Number of time steps.
    t_out : Optional[Sequence[float]]
        Times at which to capture snapshots; defaults to quarter fractions of
        ``t_final`` when ``None``.
    eps : float
        Small floor added to depth to avoid division by zero.

    Returns
    -------
    Tuple[np.ndarray, Dict[str, List]]
        The cell-center grid ``x`` and a snapshot dictionary keyed by ``"t"``,
        ``"h"`` and ``"hu"``.
    """
    if t_out is None:
        t_out = [0.25 * t_final, 0.5 * t_final, 0.75 * t_final, t_final]
    dx = length / nx
    dt = t_final / nt
    x = np.linspace(dx / 2, length - dx / 2, nx)
    h = h0_fn(x).astype(np.float64)
    hu = np.zeros(nx, dtype=np.float64)
    b = b_fn(x).astype(np.float64)
    db = np.gradient(b, dx)

    snaps: Dict[str, List] = {"t": [], "h": [], "hu": []}
    captured: set = set()

    for step in range(nt):
        u = hu / (h + eps)
        F1 = hu
        F2 = hu * u + 0.5 * gravity * h**2
        h = np.maximum(
            0.5 * (np.roll(h, -1) + np.roll(h, 1))
            - dt / (2 * dx) * (np.roll(F1, -1) - np.roll(F1, 1)), eps)
        hu = (0.5 * (np.roll(hu, -1) + np.roll(hu, 1))
              - dt / (2 * dx) * (np.roll(F2, -1) - np.roll(F2, 1))
              - dt * gravity * h * db)
        t_cur = (step + 1) * dt
        for t_want in t_out:
            if t_want not in captured and abs(t_cur - t_want) < 1.5 * dt:
                snaps["t"].append(t_want)
                snaps["h"].append(h.astype(np.float32))
                snaps["hu"].append(hu.astype(np.float32))
                captured.add(t_want)
    return x.astype(np.float32), snaps
