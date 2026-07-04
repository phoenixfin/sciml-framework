"""Numerical reference solvers (pure numpy) used to generate ground truth."""

from .burgers import burgers_dataset, burgers_solution
from .compartmental import rk4_integrate, simulate_compartmental
from .darcy import darcy_dataset, sample_permeability, solve_darcy_2d
from .dynamical import (
                        fitzhugh_nagumo,
                        harmonic_oscillator,
                        linear_decay,
                        lorenz,
                        lotka_volterra,
                        simulate,
                        van_der_pol,
)
from .heat import heat_dataset, heat_solution
from .kuramoto_sivashinsky import kuramoto_sivashinsky
from .swe_lax_friedrichs import lax_friedrichs_swe
from .transport import advection_diffusion_dataset, advection_diffusion_solution
from .wave1d import wave1d_dalembert
from .wave_fdm import wave_moving_boundary_fdm

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
