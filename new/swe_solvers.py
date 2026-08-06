"""
Well-balanced shallow-water solvers, periodic BCs, 1D.

  lxf_paper(...)  : the Lax-Friedrichs scheme of the manuscript (Eq. 6), for the CFL audit
  swe_solve(...)  : Audusse hydrostatic-reconstruction HLL solver,
                    order=1 (Euler) or order=2 (MUSCL + SSP-RK2)
"""
import numpy as np

G = 9.81
_EPS = 1e-12


# ----------------------------------------------------------------------
# the manuscript's scheme, kept verbatim for the audit
# ----------------------------------------------------------------------
def _flux(q):
    h, hu = q[0], q[1]
    hs = np.maximum(h, _EPS)
    return np.stack([hu, hu * hu / hs + 0.5 * G * h * h])


def lxf_paper(x, h0, b, T, nt):
    """Lax-Friedrichs exactly as Eq. (6): fixed dt = T/nt, periodic."""
    nx = x.size
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


def lxf_viscosity(dx, dt, cfl):
    """Modified-equation diffusion coefficient of Lax-Friedrichs [m^2/s]."""
    return dx ** 2 / (2.0 * dt) * (1.0 - cfl ** 2)


# ----------------------------------------------------------------------
# well-balanced HLL solver
# ----------------------------------------------------------------------
def _vel(h, hu):
    """Desingularised velocity (Kurganov-Petrova)."""
    return np.where(h > 1e-8, hu / np.maximum(h, _EPS), 0.0)


def _hll(hL, huL, hR, huR):
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


def _minmod(a, c):
    return np.where(a * c <= 0.0, 0.0, np.sign(a) * np.minimum(np.abs(a), np.abs(c)))


def _rhs(q, b, dx, order):
    """dq/dt for the Audusse well-balanced HLL scheme. q=(h,hu), periodic."""
    h, hu = q[0], q[1]
    eta = h + b

    if order == 1:
        etaL = etaR = eta
        huL = huR = hu
        bL = bR = b
    else:
        # minmod-limited slopes on (eta, hu, b); reconstructing eta (not h)
        # together with b is what preserves the lake-at-rest state.
        def slope(f):
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

    # source corrections (Audusse et al. 2004)
    Sminus = np.stack([np.zeros_like(z), 0.5 * G * (hminus ** 2 - hsL ** 2)])   # at i+1/2, cell i side
    Splus = np.stack([np.zeros_like(z), 0.5 * G * (hplus ** 2 - hsR ** 2)])     # at i+1/2, cell i+1 side

    Fright = F + Sminus                    # flux leaving cell i through i+1/2
    Fleft = np.roll(F + Splus, 1, axis=-1)  # flux entering cell i through i-1/2

    # centred well-balanced correction for the interior bed slope (2nd order only)
    if order == 1:
        interior = 0.0
    else:
        interior = np.stack([np.zeros_like(z),
                             -0.5 * G * (hLf + hRf) * (bR - bL)])
    return -(Fright - Fleft) / dx + interior / dx, np.max(smax)


def swe_solve(x, h0, b, T, cfl=0.45, order=2, snapshots=None, max_steps=2_000_000):
    """
    Well-balanced HLL solver with adaptive dt.

    snapshots : sorted list of output times (T is always included implicitly if listed)
    returns   : (q_final, out_dict) where out_dict maps requested time -> (h, hu)
    """
    dx = x[1] - x[0]
    h0 = np.atleast_1d(np.asarray(h0, dtype=float))
    q = np.stack([h0.copy(), np.zeros_like(h0)])
    b = np.broadcast_to(np.asarray(b, dtype=float), h0.shape).copy()
    targets = sorted(snapshots) if snapshots else []
    out, t, k = {}, 0.0, 0

    def step(qq, dt):
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
def cell_centers(L, nx):
    dx = L / nx
    return (np.arange(nx) + 0.5) * dx


def rel_l2(pred, ref):
    return float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))


def rel_l2_anomaly(pred, ref, rest=None):
    """Relative L2 normalised by the *anomaly* of the reference, not its total magnitude."""
    r = ref.mean() if rest is None else rest
    return float(np.linalg.norm(pred - ref) / np.linalg.norm(ref - r))
