"""Numerical reference solvers (pure numpy) used to generate ground truth."""

from .swe_lax_friedrichs import lax_friedrichs_swe
from .wave_fdm import wave_moving_boundary_fdm
from .compartmental import simulate_compartmental, rk4_integrate

__all__ = [
    "lax_friedrichs_swe",
    "wave_moving_boundary_fdm",
    "simulate_compartmental",
    "rk4_integrate",
]
