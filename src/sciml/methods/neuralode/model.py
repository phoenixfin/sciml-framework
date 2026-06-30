"""Neural ODE model with a trajectory-fitting helper."""

from __future__ import annotations

from typing import Sequence

import tensorflow as tf
from tensorflow import keras

from .integrators import odeint


def build_odefunc(state_dim: int, hidden: Sequence[int] = (64, 64),
                  activation: str = "tanh", time_dependent: bool = False,
                  name: str = "odefunc") -> keras.Model:
    """MLP for the dynamics ``f(t, y) -> dy``.

    If ``time_dependent``, ``t`` is concatenated to the state as an extra input.
    """
    in_dim = state_dim + (1 if time_dependent else 0)
    inp = keras.Input((in_dim,))
    h = inp
    for u in hidden:
        h = keras.layers.Dense(u, activation)(h)
    return keras.Model(inp, keras.layers.Dense(state_dim)(h), name=name)


class NeuralODE(keras.Model):
    """Wrap a dynamics network ``f`` and integrate it over time."""

    def __init__(self, func: keras.Model, method: str = "rk4",
                 time_dependent: bool = False, name: str = "neural_ode"):
        super().__init__(name=name)
        self.func_net = func
        self.method = method
        self.time_dependent = time_dependent

    def _f(self, t, y):
        if self.time_dependent:
            tcol = tf.fill((tf.shape(y)[0], 1), tf.cast(t, tf.float32))
            return self.func_net(tf.concat([y, tcol], axis=1))
        return self.func_net(y)

    def call(self, y0, t):
        """Integrate from ``y0`` ``(B, d)`` over times ``t`` -> ``(len(t), B, d)``."""
        return odeint(self._f, y0, t, self.method)

    def fit_trajectory(self, y0, t, target, *, steps: int = 500, lr: float = 1e-2,
                       log_every: int = 100, verbose: bool = True):
        """Fit ``f`` so the integrated trajectory matches ``target`` ``(len(t), B, d)``.

        Returns the loss history.
        """
        opt = keras.optimizers.Adam(lr)
        target = tf.convert_to_tensor(target, dtype=tf.float32)
        history = []
        for s in range(1, steps + 1):
            with tf.GradientTape() as tape:
                pred = self.call(y0, t)
                loss = tf.reduce_mean((pred - target) ** 2)
            grads = tape.gradient(loss, self.func_net.trainable_variables)
            opt.apply_gradients(zip(grads, self.func_net.trainable_variables))
            history.append(float(loss))
            if verbose and (s % log_every == 0 or s == 1):
                print(f"  [{s:5d}/{steps}] loss={float(loss):.3e}")
        return history
