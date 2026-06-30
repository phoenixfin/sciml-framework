"""DeepONet example: the 1D Shallow Water Equations.

``cases`` is pure numpy; ``model``, ``problem`` and ``runners`` require
TensorFlow and are imported lazily.
"""

from __future__ import annotations

from . import cases
from .config import SWEConfig

__all__ = ["cases", "SWEConfig", "SWEProblem", "SWEDeepONet"]


def __getattr__(name: str):
    if name == "SWEProblem":
        from .problem import SWEProblem
        return SWEProblem
    if name == "SWEDeepONet":
        from .model import SWEDeepONet
        return SWEDeepONet
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
