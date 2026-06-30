"""FNO block and 1D FNO model builder."""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras

from .spectral import SpectralConv1D


class FNOBlock(keras.layers.Layer):
    """One FNO layer: ``activation(SpectralConv1D(v) + Conv1D_1x1(v))``."""

    def __init__(self, width: int, modes: int, activation=tf.nn.gelu, **kw):
        super().__init__(**kw)
        self.width = width
        self.modes = modes
        self.activation = activation
        self.spec = SpectralConv1D(width, modes)
        self.pointwise = keras.layers.Conv1D(width, 1)

    def call(self, x):
        return self.activation(self.spec(x) + self.pointwise(x))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"width": self.width, "modes": self.modes})
        return cfg


def build_fno1d(modes: int = 16, width: int = 64, n_layers: int = 4,
                in_channels: int = 2, out_channels: int = 1,
                name: str = "fno1d") -> keras.Model:
    """A 1D Fourier Neural Operator.

    Maps an input field of shape ``(batch, n_x, in_channels)`` to
    ``(batch, n_x, out_channels)``. ``in_channels`` typically stacks the input
    function's values with the spatial coordinate. ``modes`` is the number of
    retained Fourier modes (``<= n_x // 2 + 1``).
    """
    inp = keras.Input((None, in_channels))
    v = keras.layers.Dense(width)(inp)                    # lifting
    for i in range(n_layers):
        v = FNOBlock(width, modes, name=f"fno_block_{i}")(v)
    v = keras.layers.Dense(128, activation=tf.nn.gelu)(v)  # projection
    out = keras.layers.Dense(out_channels)(v)
    return keras.Model(inp, out, name=name)
