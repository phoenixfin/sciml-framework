"""1D linear advection-diffusion on a periodic domain (pure numpy, spectral).

``u_t + c u_x = nu u_xx`` has the exact Fourier solution
``u_hat(k, t) = u_hat(k, 0) exp((-i c k - nu k^2) t)``. A clean, cheap operator
``u(.,0) -> u(.,T)`` (a shifted, diffused copy of the IC) for operator learning.
"""

from __future__ import annotations

import numpy as np


def advection_diffusion_solution(u0: np.ndarray, c: float, nu: float, t: float,
                                 length: float = 1.0) -> np.ndarray:
    """Solve periodic advection-diffusion from IC ``u0`` to time ``t``."""
    u0 = np.asarray(u0, dtype=float)
    n = len(u0)
    k = 2 * np.pi * np.fft.fftfreq(n, d=length / n)
    u_hat = np.fft.fft(u0) * np.exp((-1j * c * k - nu * k**2) * t)
    return np.real(np.fft.ifft(u_hat))


def advection_diffusion_dataset(u0_batch: np.ndarray, c: float, nu: float, t: float,
                                length: float = 1.0) -> np.ndarray:
    """Apply the solution operator to a batch ``(n, grid) -> (n, grid)``."""
    return np.stack([advection_diffusion_solution(u0_batch[i], c, nu, t, length)
                     for i in range(len(u0_batch))]).astype(np.float32)
