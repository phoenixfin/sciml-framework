"""Fixed-step ODE integrators (TensorFlow, differentiable)."""

from __future__ import annotations

from typing import Callable

import numpy as np
import tensorflow as tf


def euler_step(func: Callable, t: float, y: tf.Tensor, dt: float) -> tf.Tensor:
    return y + dt * func(t, y)


def rk4_step(func: Callable, t: float, y: tf.Tensor, dt: float) -> tf.Tensor:
    k1 = func(t, y)
    k2 = func(t + dt / 2, y + dt / 2 * k1)
    k3 = func(t + dt / 2, y + dt / 2 * k2)
    k4 = func(t + dt, y + dt * k3)
    return y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def odeint(func: Callable, y0: tf.Tensor, t, method: str = "rk4") -> tf.Tensor:
    """Integrate ``dy/dt = func(t, y)`` over the time points ``t``.

    ``func(t, y)`` takes a scalar time and a state ``(B, d)`` and returns
    ``(B, d)``. ``t`` is a 1D array of (static length) increasing times. Returns
    the trajectory stacked along a new leading axis: ``(len(t), B, d)``.
    """
    ts = np.asarray(t, dtype=np.float32)
    step = rk4_step if method == "rk4" else euler_step
    y = y0
    ys = [y0]
    for i in range(len(ts) - 1):
        y = step(func, float(ts[i]), y, float(ts[i + 1] - ts[i]))
        ys.append(y)
    return tf.stack(ys, axis=0)
