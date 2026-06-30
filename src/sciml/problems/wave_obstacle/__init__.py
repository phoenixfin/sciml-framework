"""PINN example: a moving-boundary (obstacle) wave problem.

The string vibrates on the moving domain ``s(tau) < xbar < 1``. Two networks are
trained jointly: ``Nu(xbar, tau)`` (displacement) and ``Ns(tau)`` (free
boundary / contact point). ``config`` is pure-python; ``problem``/``runners``
require TensorFlow (and SciPy for the L-BFGS phase) and import lazily.
"""

from __future__ import annotations

from .config import WaveObstacleConfig

__all__ = ["WaveObstacleConfig", "WaveObstacleProblem"]


def __getattr__(name: str):
    if name == "WaveObstacleProblem":
        from .problem import WaveObstacleProblem
        return WaveObstacleProblem
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
