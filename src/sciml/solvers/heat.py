"""1D heat / diffusion equation on a periodic domain (pure numpy, spectral).

``u_t = nu u_xx`` on ``[0, L)`` with periodic BCs has the exact Fourier solution
``u_hat(k, t) = u_hat(k, 0) exp(-nu k^2 t)``. This makes a clean, cheap operator
``u(.,0) -> u(.,T)`` for operator-learning examples (FNO / DeepONet).
"""

from __future__ import annotations

import numpy as np


def heat_solution(u0: np.ndarray, nu: float, t: float, length: float = 1.0) -> np.ndarray:
    """Solve the periodic heat equation from IC ``u0`` to time ``t``."""
    u0 = np.asarray(u0, dtype=float)
    n = len(u0)
    k = 2 * np.pi * np.fft.fftfreq(n, d=length / n)
    u_hat = np.fft.fft(u0) * np.exp(-nu * k**2 * t)
    return np.real(np.fft.ifft(u_hat))


def heat_dataset(u0_batch: np.ndarray, nu: float, t: float,
                 length: float = 1.0) -> np.ndarray:
    """Apply :func:`heat_solution` to a batch of ICs ``(n, grid) -> (n, grid)``."""
    return np.stack([heat_solution(u0_batch[i], nu, t, length)
                     for i in range(len(u0_batch))]).astype(np.float32)
