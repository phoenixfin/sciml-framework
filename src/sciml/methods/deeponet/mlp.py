"""MLP factory for DeepONet branch/trunk networks."""

from __future__ import annotations

from typing import Sequence

import tensorflow as tf


def make_mlp(in_dim: int, hidden: Sequence[int], out_dim: int, name: str,
             activation: str = "tanh", out_std: float = 0.1) -> tf.keras.Model:
    """Fully-connected MLP with a small-variance linear output layer.

    A small output init keeps the operator near zero at the start of training,
    which stabilizes physics-informed setups.
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
