"""Well-balanced shallow-water solvers on a periodic 1D domain.

Two schemes live here, and the point of the module is the contrast between them:

``lxf_paper``
    The Lax-Friedrichs scheme of the original manuscript (Eq. 6), kept verbatim
    so its CFL number and numerical viscosity can be audited rather than
    assumed.
``swe_solve``
    An Audusse hydrostatic-reconstruction HLL solver, first order (Euler) or
    second order (minmod-MUSCL + SSP-RK2). It is well-balanced: the
    lake-at-rest state is preserved to machine precision.

Both integrate the 1D shallow-water equations with a bed source term,

.. math::

    \\partial_t h + \\partial_x (hu) = 0, \\qquad
    \\partial_t (hu) + \\partial_x \\left( \\frac{(hu)^2}{h}
        + \\tfrac{1}{2} g h^2 \\right) = -g h\\, \\partial_x b,

for depth ``h``, discharge ``hu`` and bathymetry ``b``. The conserved state is
stacked as ``q = np.stack([h, hu])`` with shape ``(2, nx)``.

See ``RESULTS.md`` §1.5 for the error budget that motivated the second solver.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

#: Gravitational acceleration [m/s^2].
G = 9.81
#: Floor used to desingularise divisions by the depth.
_EPS = 1e-12


# ----------------------------------------------------------------------
# the manuscript's scheme, kept verbatim for the audit
# ----------------------------------------------------------------------
def _flux(q: np.ndarray) -> np.ndarray:
    """Physical flux of the shallow-water system.

    Parameters
    ----------
    q : np.ndarray
        Conserved state ``(2, nx)``: depth ``h`` and discharge ``hu``.

    Returns
    -------
    np.ndarray
        Flux ``(2, nx)``: ``[hu, (hu)^2 / h + g h^2 / 2]``, with the depth
        floored at ``_EPS`` in the denominator.
    """
    h, hu = q[0], q[1]
    hs = np.maximum(h, _EPS)
    return np.stack([hu, hu * hu / hs + 0.5 * G * h * h])


def lxf_paper(
    x: np.ndarray, h0: np.ndarray, b: np.ndarray, T: float, nt: int
) -> Tuple[np.ndarray, float, float, float]:
    """Integrate with Lax-Friedrichs exactly as in Eq. (6) of the manuscript.

    The timestep is *fixed* at ``T / nt`` rather than chosen from a CFL
    condition, which is what makes the audit necessary: the realised CFL number
    is returned so the scheme's numerical viscosity can be evaluated with
    ``lxf_viscosity`` instead of guessed.

    Parameters
    ----------
    x : np.ndarray
        Cell centres ``(nx,)``, uniformly spaced.
    h0 : np.ndarray
        Initial depth ``(nx,)``. The initial discharge is zero.
    b : np.ndarray
        Bathymetry ``(nx,)`` on the same grid.
    T : float
        Final time [s].
    nt : int
        Number of fixed timesteps.

    Returns
    -------
    tuple of (np.ndarray, float, float, float)
        ``(q, cfl_max, dx, dt)``: the final state ``(2, nx)``, the largest CFL
        number realised over the run, the grid spacing and the timestep.
    """
    dx = x[1] - x[0]
    dt = T / nt
    q = np.stack([h0.copy(), np.zeros_like(h0)])
    dbdx = (np.roll(b, -1) - np.roll(b, 1)) / (2 * dx)
    cfl_max = 0.0
    for _ in range(nt):
        qp, qm = np.roll(q, -1, axis=-1), np.roll(q, 1, axis=-1)
        S = np.stack([np.zeros_like(q[0]), -G * q[0] * dbdx])
        q = 0.5 * (qp + qm) - dt / (2 * dx) * (_flux(qp) - _flux(qm)) - dt * S
        c = np.sqrt(G * np.maximum(q[0], _EPS))
        cfl_max = max(cfl_max, float(np.max(np.abs(q[1] / np.maximum(q[0], _EPS)) + c) * dt / dx))
    return q, cfl_max, dx, dt


def lxf_viscosity(dx: float, dt: float, cfl: float) -> float:
    """Modified-equation diffusion coefficient of Lax-Friedrichs.

    The scheme behaves like the exact equations plus a diffusion term of this
    strength. Note it *grows* as ``dt`` falls at fixed ``dx``, so a
    conservatively small timestep makes the smearing worse, not better.

    Parameters
    ----------
    dx : float
        Grid spacing [m].
    dt : float
        Timestep [s].
    cfl : float
        Realised CFL number (e.g. ``cfl_max`` from :func:`lxf_paper`).

    Returns
    -------
    float
        Diffusion coefficient ``dx^2 / (2 dt) (1 - cfl^2)`` [m^2/s].
    """
    return dx ** 2 / (2.0 * dt) * (1.0 - cfl ** 2)


# ----------------------------------------------------------------------
# well-balanced HLL solver
# ----------------------------------------------------------------------
def _vel(h: np.ndarray, hu: np.ndarray) -> np.ndarray:
    """Desingularised velocity (Kurganov-Petrova).

    Parameters
    ----------
    h : np.ndarray
        Depth.
    hu : np.ndarray
        Discharge, same shape as ``h``.

    Returns
    -------
    np.ndarray
        ``hu / h`` where the cell is wet, and exactly zero where it is dry
        (``h <= 1e-8``), so dry cells cannot produce spurious velocities.
    """
    return np.where(h > 1e-8, hu / np.maximum(h, _EPS), 0.0)


def _hll(
    hL: np.ndarray, huL: np.ndarray, hR: np.ndarray, huR: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """HLL approximate Riemann solver for the shallow-water system.

    Parameters
    ----------
    hL : np.ndarray
        Depth of the left state at each interface.
    huL : np.ndarray
        Discharge of the left state.
    hR : np.ndarray
        Depth of the right state.
    huR : np.ndarray
        Discharge of the right state.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        ``(F, smax)``: the numerical flux ``(2, nx)`` at each interface and the
        largest absolute wave speed there, used for the adaptive timestep.
    """
    uL, uR = _vel(hL, huL), _vel(hR, huR)
    cL, cR = np.sqrt(G * hL), np.sqrt(G * hR)
    sL = np.minimum(uL - cL, uR - cR)
    sR = np.maximum(uL + cL, uR + cR)
    FL = _flux(np.stack([hL, huL]))
    FR = _flux(np.stack([hR, huR]))
    den = np.where(np.abs(sR - sL) < _EPS, 1.0, sR - sL)
    Fhll = (sR * FL - sL * FR + sL * sR * (np.stack([hR, huR]) - np.stack([hL, huL]))) / den
    F = np.where(sL >= 0.0, FL, np.where(sR <= 0.0, FR, Fhll))
    return F, np.maximum(np.abs(sL), np.abs(sR))


def _minmod(a: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Minmod slope limiter.

    Parameters
    ----------
    a : np.ndarray
        Backward difference.
    c : np.ndarray
        Forward difference, same shape as ``a``.

    Returns
    -------
    np.ndarray
        The smaller of the two in magnitude when they share a sign, else zero.
    """
    return np.where(a * c <= 0.0, 0.0, np.sign(a) * np.minimum(np.abs(a), np.abs(c)))


def _rhs(q: np.ndarray, b: np.ndarray, dx: float, order: int) -> Tuple[np.ndarray, float]:
    """Semi-discrete right-hand side of the Audusse well-balanced HLL scheme.

    The bed is reconstructed together with the free surface ``eta = h + b``
    (rather than with the depth), and the interface bed level is raised to
    ``max`` of the two sides; those two choices are what keep the lake-at-rest
    state exactly stationary.

    Parameters
    ----------
    q : np.ndarray
        Conserved state ``(2, nx)``, periodic in ``x``.
    b : np.ndarray
        Bathymetry ``(nx,)``.
    dx : float
        Grid spacing [m].
    order : int
        ``1`` for piecewise-constant states, ``2`` for minmod-limited MUSCL
        reconstruction plus the centred interior bed-slope correction.

    Returns
    -------
    tuple of (np.ndarray, float)
        ``(dq/dt, smax)``: the state derivative ``(2, nx)`` and the largest
        wave speed over all interfaces.
    """
    h, hu = q[0], q[1]
    eta = h + b

    if order == 1:
        etaL = etaR = eta
        huL = huR = hu
        bL = bR = b
    else:
        # minmod-limited slopes on (eta, hu, b); reconstructing eta (not h)
        # together with b is what preserves the lake-at-rest state.
        # limited cell slope of a field, from its backward/forward differences
        def slope(f: np.ndarray) -> np.ndarray:
            return _minmod(f - np.roll(f, 1, axis=-1), np.roll(f, -1, axis=-1) - f)

        se, sm, sb = slope(eta), slope(hu), slope(b)
        etaL, etaR = eta - 0.5 * se, eta + 0.5 * se      # cell's left / right face value
        huL, huR = hu - 0.5 * sm, hu + 0.5 * sm
        bL, bR = b - 0.5 * sb, b + 0.5 * sb

    hLf = np.maximum(etaR - bR, 0.0)          # right face of cell i   -> left  state of i+1/2
    hRf = np.maximum(etaL - bL, 0.0)          # left  face of cell i   -> right state of i-1/2

    # interface i+1/2 : left state from cell i (right face), right state from cell i+1 (left face)
    hminus, huminus, bminus = hLf, huR, bR
    hplus = np.roll(hRf, -1, axis=-1)
    huplus = np.roll(huL, -1, axis=-1)
    bplus = np.roll(bL, -1, axis=-1)

    z = np.maximum(bminus, bplus)                       # Audusse interface bed level
    hsL = np.maximum(hminus + bminus - z, 0.0)
    hsR = np.maximum(hplus + bplus - z, 0.0)
    husL = hsL * _vel(hminus, huminus)
    husR = hsR * _vel(hplus, huplus)

    F, smax = _hll(hsL, husL, hsR, husR)                # F at i+1/2

    # source corrections (Audusse et al. 2004): at i+1/2, cell i and cell i+1 side
    Sminus = np.stack([np.zeros_like(z), 0.5 * G * (hminus ** 2 - hsL ** 2)])
    Splus = np.stack([np.zeros_like(z), 0.5 * G * (hplus ** 2 - hsR ** 2)])

    Fright = F + Sminus                    # flux leaving cell i through i+1/2
    Fleft = np.roll(F + Splus, 1, axis=-1)  # flux entering cell i through i-1/2

    # centred well-balanced correction for the interior bed slope (2nd order only)
    if order == 1:
        interior = 0.0
    else:
        interior = np.stack([np.zeros_like(z),
                             -0.5 * G * (hLf + hRf) * (bR - bL)])
    return -(Fright - Fleft) / dx + interior / dx, np.max(smax)


def swe_solve(
    x: np.ndarray,
    h0: np.ndarray,
    b: np.ndarray,
    T: float,
    cfl: float = 0.45,
    order: int = 2,
    snapshots: Optional[List[float]] = None,
    max_steps: int = 2_000_000,
) -> Tuple[np.ndarray, Dict[float, Tuple[np.ndarray, np.ndarray]]]:
    """Integrate the shallow-water equations with the well-balanced HLL scheme.

    The timestep adapts to the fastest wave, and is shortened when needed to
    land exactly on a requested snapshot time. The depth is clipped at zero
    after every stage so dry cells stay dry.

    Parameters
    ----------
    x : np.ndarray
        Cell centres ``(nx,)``, uniformly spaced.
    h0 : np.ndarray
        Initial depth ``(nx,)``. The initial discharge is zero.
    b : np.ndarray
        Bathymetry, broadcastable to the shape of ``h0``.
    T : float
        Final time [s].
    cfl : float
        Courant number used to size the adaptive timestep.
    order : int
        ``1`` for first-order Euler, ``2`` for MUSCL + SSP-RK2.
    snapshots : Optional[List[float]]
        Times at which to record the state. Need not be sorted; ``T`` itself is
        only recorded if it appears in the list.
    max_steps : int
        Safety cap on the number of timesteps.

    Returns
    -------
    tuple of (np.ndarray, dict)
        ``(q, out)``: the final state ``(2, nx)``, and a mapping from each
        requested snapshot time to its ``(h, hu)`` pair of ``(nx,)`` arrays.
    """
    dx = x[1] - x[0]
    h0 = np.atleast_1d(np.asarray(h0, dtype=float))
    q = np.stack([h0.copy(), np.zeros_like(h0)])
    b = np.broadcast_to(np.asarray(b, dtype=float), h0.shape).copy()
    targets = sorted(snapshots) if snapshots else []
    out, t, k = {}, 0.0, 0

    # advance the state by dt: Euler (order 1) or SSP-RK2 (order 2)
    def step(qq: np.ndarray, dt: float) -> np.ndarray:
        k1, _ = _rhs(qq, b, dx, order)
        if order == 1:
            return qq + dt * k1
        q1 = qq + dt * k1
        q1[0] = np.maximum(q1[0], 0.0)
        k2, _ = _rhs(q1, b, dx, order)
        return 0.5 * (qq + q1 + dt * k2)

    while t < T - 1e-13 and k < max_steps:
        _, smax = _rhs(q, b, dx, order)
        dt = cfl * dx / max(smax, _EPS)
        if targets and t + dt > targets[0]:
            dt = targets[0] - t
        dt = min(dt, T - t)
        q = step(q, dt)
        q[0] = np.maximum(q[0], 0.0)
        t += dt
        k += 1
        while targets and t >= targets[0] - 1e-12:
            out[targets.pop(0)] = (q[0].copy(), q[1].copy())
    return q, out


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def cell_centers(L: float, nx: int) -> np.ndarray:
    """Cell centres of a uniform finite-volume grid on ``[0, L]``.

    Parameters
    ----------
    L : float
        Domain length [m].
    nx : int
        Number of cells.

    Returns
    -------
    np.ndarray
        Centres ``(nx,)``, offset by half a cell from the edges.
    """
    dx = L / nx
    return (np.arange(nx) + 0.5) * dx


def rel_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    """Relative L2 error against the total magnitude of the reference.

    Parameters
    ----------
    pred : np.ndarray
        Predicted field.
    ref : np.ndarray
        Reference field, same shape as ``pred``.

    Returns
    -------
    float
        ``||pred - ref|| / ||ref||``. For a depth field this is flattered by
        the constant background depth — prefer :func:`rel_l2_anomaly`.
    """
    return float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))


def rel_l2_anomaly(pred: np.ndarray, ref: np.ndarray, rest: Optional[float] = None) -> float:
    """Relative L2 error normalised by the *anomaly* of the reference.

    Parameters
    ----------
    pred : np.ndarray
        Predicted field.
    ref : np.ndarray
        Reference field, same shape as ``pred``.
    rest : Optional[float]
        Rest state to subtract from the reference before taking its norm.
        Defaults to the mean of ``ref``.

    Returns
    -------
    float
        ``||pred - ref|| / ||ref - rest||``, i.e. the error measured against
        the part of the signal that actually varies.
    """
    r = ref.mean() if rest is None else rest
    return float(np.linalg.norm(pred - ref) / np.linalg.norm(ref - r))
