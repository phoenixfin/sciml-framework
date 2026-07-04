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

    Parameters
    ----------
    state_dim : int
        Dimension of the ODE state.
    hidden : Sequence[int]
        Widths of the hidden layers.
    activation : str
        Activation function used in the hidden layers.
    time_dependent : bool
        If True, concatenate ``t`` to the state as an extra input.
    name : str
        Name assigned to the constructed Keras model.

    Returns
    -------
    keras.Model
        The constructed dynamics network.
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
        """Wrap a dynamics network and configure the integrator.

        Parameters
        ----------
        func : keras.Model
            The dynamics network ``f(t, y) -> dy``.
        method : str
            Integration method, ``"rk4"`` (default) or ``"euler"``.
        time_dependent : bool
            Whether ``func`` expects the time concatenated to the state.
        name : str
            Name assigned to the Keras model.
        """
        super().__init__(name=name)
        self.func_net = func
        self.method = method
        self.time_dependent = time_dependent

    def _f(self, t: float, y: tf.Tensor) -> tf.Tensor:
        """Evaluate the dynamics network, optionally injecting the time input.

        Parameters
        ----------
        t : float
            Current time.
        y : tf.Tensor
            Current state of shape ``(B, d)``.

        Returns
        -------
        tf.Tensor
            The time derivative of the state.
        """
        if self.time_dependent:
            tcol = tf.fill((tf.shape(y)[0], 1), tf.cast(t, tf.float32))
            return self.func_net(tf.concat([y, tcol], axis=1))
        return self.func_net(y)

    def call(self, y0: tf.Tensor, t: np.ndarray) -> tf.Tensor:
        """Integrate from ``y0`` ``(B, d)`` over times ``t`` -> ``(len(t), B, d)``.

        Parameters
        ----------
        y0 : tf.Tensor
            Initial state of shape ``(B, d)``.
        t : np.ndarray
            1D array of increasing times to integrate over.

        Returns
        -------
        tf.Tensor
            The integrated trajectory of shape ``(len(t), B, d)``.
        """
        return odeint(self._f, y0, t, self.method)

    def fit_trajectory(self, y0: tf.Tensor, t: np.ndarray, target: tf.Tensor, *,
                       steps: int = 500, lr: float = 1e-2,
                       log_every: int = 100, verbose: bool = True) -> list:
        """Fit ``f`` so the integrated trajectory matches ``target`` ``(len(t), B, d)``.

        Parameters
        ----------
        y0 : tf.Tensor
            Initial state of shape ``(B, d)``.
        t : np.ndarray
            1D array of increasing times to integrate over.
        target : tf.Tensor
            Target trajectory of shape ``(len(t), B, d)`` to match.
        steps : int
            Number of gradient-descent steps.
        lr : float
            Learning rate for the Adam optimizer.
        log_every : int
            Print the loss every this many steps.
        verbose : bool
            Whether to print progress messages.

        Returns
        -------
        list
            The recorded per-step loss history.
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
