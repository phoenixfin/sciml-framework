"""SINDy example: time-varying transmission rate beta(t) for dengue.

Identify beta(t) of an SI/SIR/SIRS model from a weekly case series. The default
config *simulates* data so the example runs with no external files. Everything
here is pure numpy/pandas; scikit-learn is optional (used only by LASSO-based
global regression).
"""

from __future__ import annotations

from .config import EpiConfig

__all__ = ["EpiConfig", "EpiProblem"]


def __getattr__(name: str):
    if name == "EpiProblem":
        from .problem import EpiProblem
        return EpiProblem
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
