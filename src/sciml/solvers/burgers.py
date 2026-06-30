"""1D viscous Burgers' equation on a periodic domain (pure numpy, pseudo-spectral).

``u_t + u u_x = nu u_xx`` on ``[0, L)``. Integrated with an explicit RK4 step and
2/3-rule dealiasing. Burgers develops steep gradients (near-shocks) for small
``nu``, making the operator ``u(.,0) -> u(.,T)`` a non-trivial benchmark for
operator learning (FNO / DeepONet).
"""

from __future__ import annotations

import numpy as np


def burgers_solution(u0: np.ndarray, nu: float = 0.01, t_final: float = 1.0,
                     length: float = 1.0, nt: int = 2000) -> np.ndarray:
    """Integrate periodic Burgers from IC ``u0`` to ``t_final``. Returns the field."""
    u0 = np.asarray(u0, dtype=float)
    n = len(u0)
    k = 2 * np.pi * np.fft.fftfreq(n, d=length / n)
    k2 = k**2
    kmax = np.max(np.abs(k))
    dealias = (np.abs(k) < (2.0 / 3.0) * kmax).astype(float)
    dt = t_final / nt

    def rhs(u):
        uh = np.fft.fft(u)
        ux = np.real(np.fft.ifft(1j * k * uh))
        uxx = np.real(np.fft.ifft(-k2 * uh))
        return -u * ux + nu * uxx

    u = u0.copy()
    for _ in range(nt):
        k1 = rhs(u)
        k2_ = rhs(u + 0.5 * dt * k1)
        k3 = rhs(u + 0.5 * dt * k2_)
        k4 = rhs(u + dt * k3)
        u = u + (dt / 6.0) * (k1 + 2 * k2_ + 2 * k3 + k4)
        u = np.real(np.fft.ifft(np.fft.fft(u) * dealias))  # 2/3 dealiasing
    return u


def burgers_dataset(u0_batch: np.ndarray, nu: float = 0.01, t_final: float = 1.0,
                    length: float = 1.0, nt: int = 2000) -> np.ndarray:
    """Apply :func:`burgers_solution` to a batch of ICs ``(n, grid) -> (n, grid)``."""
    return np.stack([burgers_solution(u0_batch[i], nu, t_final, length, nt)
                     for i in range(len(u0_batch))]).astype(np.float32)
