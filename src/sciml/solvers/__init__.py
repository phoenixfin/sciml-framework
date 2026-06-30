"""Numerical reference solvers (pure numpy) used to generate ground truth."""

from .swe_lax_friedrichs import lax_friedrichs_swe
from .wave_fdm import wave_moving_boundary_fdm
from .compartmental import simulate_compartmental, rk4_integrate
from .dynamical import (linear_decay, harmonic_oscillator, lotka_volterra,
                        lorenz, van_der_pol, fitzhugh_nagumo, simulate)
from .heat import heat_solution, heat_dataset
from .burgers import burgers_solution, burgers_dataset
from .transport import advection_diffusion_solution, advection_diffusion_dataset
from .wave1d import wave1d_dalembert
from .kuramoto_sivashinsky import kuramoto_sivashinsky
from .darcy import solve_darcy_2d, sample_permeability, darcy_dataset

__all__ = [
    "lax_friedrichs_swe",
    "wave_moving_boundary_fdm",
    "simulate_compartmental", "rk4_integrate",
    "linear_decay", "harmonic_oscillator", "lotka_volterra", "lorenz",
    "van_der_pol", "fitzhugh_nagumo", "simulate",
    "heat_solution", "heat_dataset",
    "burgers_solution", "burgers_dataset",
    "advection_diffusion_solution", "advection_diffusion_dataset",
    "wave1d_dalembert", "kuramoto_sivashinsky",
    "solve_darcy_2d", "sample_permeability", "darcy_dataset",
]
