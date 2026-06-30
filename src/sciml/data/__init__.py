"""Data layer: function samplers and interpolation (pure numpy)."""

from .gp import PeriodicGPSampler, GPSampler
from .interp import interp_to_grid, interp_many

__all__ = ["PeriodicGPSampler", "GPSampler", "interp_to_grid", "interp_many"]
