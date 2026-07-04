"""MLP factory for DeepONet branch/trunk networks."""

from __future__ import annotations

from typing import Sequence

import tensorflow as tf


def make_mlp(in_dim: int, hidden: Sequence[int], out_dim: int, name: str,
             activation: str = "tanh", out_std: float = 0.1) -> tf.keras.Model:
    """Fully-connected MLP with a small-variance linear output layer.

    A small output init keeps the operator near zero at the start of training,
    which stabilizes physics-informed setups.

    Parameters
    ----------
    in_dim : int
        Input dimension of the network.
    hidden : Sequence[int]
        Widths of the hidden layers.
    out_dim : int
        Output dimension of the network.
    name : str
        Name assigned to the constructed Keras model.
    activation : str
        Activation function used in the hidden layers.
    out_std : float
        Standard deviation of the truncated-normal output-layer initializer.

    Returns
    -------
    tf.keras.Model
        The constructed fully-connected MLP.
    """
    inp = tf.keras.Input(shape=(in_dim,))
    x = inp
    for h in hidden:
        x = tf.keras.layers.Dense(h, activation=activation,
                                  kernel_initializer="glorot_uniform")(x)
    out = tf.keras.layers.Dense(
        out_dim,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=out_std),
        bias_initializer="zeros")(x)
    return tf.keras.Model(inputs=inp, outputs=out, name=name)
