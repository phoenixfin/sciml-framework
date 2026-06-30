"""Numerical reference solvers (pure numpy) used to generate ground truth."""

from .swe_lax_friedrichs import lax_friedrichs_swe
from .wave_fdm import wave_moving_boundary_fdm
from .compartmental import simulate_compartmental, rk4_integrate
from .dynamical import (linear_decay, harmonic_oscillator, lotka_volterra,
                        lorenz, simulate)
from .heat import heat_solution, heat_dataset
from .burgers import burgers_solution, burgers_dataset

__all__ = [
    "lax_friedrichs_swe",
    "wave_moving_boundary_fdm",
    "simulate_compartmental", "rk4_integrate",
    "linear_decay", "harmonic_oscillator", "lotka_volterra", "lorenz", "simulate",
    "heat_solution", "heat_dataset",
    "burgers_solution", "burgers_dataset",
]
