"""Data layer: function samplers and interpolation (pure numpy)."""

from .gp import GPSampler, PeriodicGPSampler
from .interp import interp_many, interp_to_grid

__all__ = ["PeriodicGPSampler", "GPSampler", "interp_to_grid", "interp_many"]
