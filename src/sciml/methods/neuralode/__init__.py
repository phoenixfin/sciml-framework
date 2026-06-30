"""Neural ODE engine (TensorFlow).

Learn continuous-time dynamics ``dy/dt = f_theta(t, y)`` with a neural network
``f`` and integrate it over time with a fixed-step solver. Useful for learning
ODE systems (e.g. compartmental epidemic models) from trajectories.

* :func:`build_odefunc` -- an MLP for the dynamics function ``f``.
* :func:`odeint`        -- fixed-step (Euler/RK4) integration.
* :class:`NeuralODE`    -- a model wrapping ``f`` with a trajectory fit helper.
"""

from .integrators import odeint, euler_step, rk4_step
from .model import NeuralODE, build_odefunc

__all__ = ["odeint", "euler_step", "rk4_step", "NeuralODE", "build_odefunc"]
