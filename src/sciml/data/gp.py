"""Gaussian-process function samplers (pure numpy).

Used to draw random input functions (initial conditions, coefficient fields,
forcing) for operator-learning datasets. The periodic squared-exponential
kernel suits problems with periodic boundaries; a non-periodic variant is also
provided.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _cholesky_jitter(K: np.ndarray, jitter: float = 1e-5) -> np.ndarray:
    return np.linalg.cholesky(K + jitter * np.eye(K.shape[0]))


@dataclass
class PeriodicGPSampler:
    """Periodic squared-exponential GP sampler on ``[0, period]``.

    Kernel ``amp**2 * exp(-2 sin^2(pi |x-x'| / period) / ls^2)`` -- smooth and
    periodic, so the boundary gap ``|f(0) - f(L)|`` is ~0 by construction.
    """

    period: float
    length_scale: float = 2.0
    amplitude: float = 0.4
    mean: float = 0.0
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None
    jitter: float = 1e-5

    def kernel(self, x: np.ndarray) -> np.ndarray:
        diff = x[:, None] - x[None, :]
        return self.amplitude**2 * np.exp(
            -2.0 * np.sin(np.pi * diff / self.period) ** 2 / self.length_scale**2)

    def sample(self, x: np.ndarray, n: int,
               rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Draw ``n`` functions at points ``x`` -> ``(n, len(x))`` float32."""
        x = np.asarray(x, dtype=np.float64)
        L = _cholesky_jitter(self.kernel(x), self.jitter)
        z = (np.random.randn(len(x), n) if rng is None
             else rng.standard_normal((len(x), n)))
        samples = self.mean + (L @ z).T
        if self.clip_min is not None or self.clip_max is not None:
            samples = np.clip(samples, self.clip_min, self.clip_max)
        return samples.astype(np.float32)


@dataclass
class GPSampler:
    """Non-periodic squared-exponential GP sampler.

    Kernel ``amp**2 * exp(-(x - x')^2 / (2 ls^2))``.
    """

    length_scale: float = 1.0
    amplitude: float = 1.0
    mean: float = 0.0
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None
    jitter: float = 1e-5

    def kernel(self, x: np.ndarray) -> np.ndarray:
        diff = x[:, None] - x[None, :]
        return self.amplitude**2 * np.exp(-(diff**2) / (2.0 * self.length_scale**2))

    def sample(self, x: np.ndarray, n: int,
               rng: Optional[np.random.Generator] = None) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        L = _cholesky_jitter(self.kernel(x), self.jitter)
        z = (np.random.randn(len(x), n) if rng is None
             else rng.standard_normal((len(x), n)))
        samples = self.mean + (L @ z).T
        if self.clip_min is not None or self.clip_max is not None:
            samples = np.clip(samples, self.clip_min, self.clip_max)
        return samples.astype(np.float32)
