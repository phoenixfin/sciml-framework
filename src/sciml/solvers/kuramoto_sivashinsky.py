"""Kuramoto-Sivashinsky equation (pure numpy, ETDRK4 spectral).

``u_t = -u u_x - u_xx - u_xxxx`` on a periodic domain of length ``L``. For large
``L`` (e.g. 22+) the dynamics are spatiotemporally chaotic -- a standard hard
test bed. Integrated with the exponential time-differencing RK4 scheme of
Kassam & Trefethen (2005). Returns space-time snapshots.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def kuramoto_sivashinsky(n: int = 128, length: float = 22.0, t_final: float = 150.0,
                         dt: float = 0.25, n_save: int = 300,
                         u0: Optional[np.ndarray] = None, seed: int = 0) -> Dict:
    """Integrate KS and return ``{x, t, u}`` with ``u`` of shape ``(n_save, n)``.

    Parameters
    ----------
    n : int
        Number of spatial grid points.
    length : float
        Length of the periodic spatial domain.
    t_final : float
        Final integration time.
    dt : float
        Time step for the ETDRK4 integrator.
    n_save : int
        Target number of snapshots to save.
    u0 : Optional[np.ndarray]
        Initial condition; if ``None`` a default noisy profile is generated.
    seed : int
        Seed for the random noise used when ``u0`` is ``None``.

    Returns
    -------
    Dict
        Dictionary with keys ``x`` (grid), ``t`` (times) and ``u`` (snapshots
        of shape ``(n_save, n)``).
    """
    x = length * np.arange(n) / n
    if u0 is None:
        rng = np.random.default_rng(seed)
        u0 = np.cos(2 * np.pi * x / length) * (1 + np.sin(2 * np.pi * x / length)) \
            + 0.01 * rng.standard_normal(n)
    v = np.fft.fft(u0)

    # Wavenumbers (Nyquist zeroed via the central 0, per Trefethen).
    k = (2 * np.pi / length) * np.concatenate(
        [np.arange(0, n // 2), [0], np.arange(-n // 2 + 1, 0)])
    L = k**2 - k**4                                    # linear operator
    E = np.exp(dt * L)
    E2 = np.exp(dt * L / 2)

    # ETDRK4 coefficients via contour integral.
    M = 16
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * L[:, None] + r[None, :]
    Q = dt * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
    f1 = dt * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, axis=1))
    f2 = dt * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=1))
    f3 = dt * np.real(np.mean((-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=1))
    g = -0.5j * k

    def nonlinear(v):
        return g * np.fft.fft(np.real(np.fft.ifft(v)) ** 2)

    nt = int(round(t_final / dt))
    save_every = max(1, nt // n_save)
    snaps, times = [np.real(np.fft.ifft(v)).copy()], [0.0]
    for step in range(1, nt + 1):
        Nv = nonlinear(v)
        a = E2 * v + Q * Nv
        Na = nonlinear(a)
        b = E2 * v + Q * Na
        Nb = nonlinear(b)
        c = E2 * a + Q * (2 * Nb - Nv)
        Nc = nonlinear(c)
        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        if step % save_every == 0:
            snaps.append(np.real(np.fft.ifft(v)).copy())
            times.append(step * dt)
    return {"x": x, "t": np.array(times), "u": np.array(snaps)}
