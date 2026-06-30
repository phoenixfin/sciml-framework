"""Worked examples wiring a problem definition to a method engine.

Each subpackage couples a problem (its math: reference solver, sampling,
geometry) to one method and exposes runnable ``runners``:

* :mod:`sciml.problems.swe`           -- DeepONet on the 1D Shallow Water Equations
* :mod:`sciml.problems.wave_obstacle` -- PINN on a moving-boundary wave problem
* :mod:`sciml.problems.epidemiology`  -- SINDy on dengue beta(t) identification
"""

from .base import Problem

__all__ = ["Problem"]
