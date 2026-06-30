"""Network builders for PINNs."""

from __future__ import annotations

from typing import Optional

import tensorflow as tf
from tensorflow import keras

from .layers import FourierEmbedding


def build_mlp(in_dim: int = 2, hidden: int = 5, width: int = 128, out_dim: int = 1,
              *, fourier_freq: Optional[int] = None, fourier_sigma: float = 1.0,
              activation: str = "tanh", name: str = "mlp") -> keras.Model:
    """Fully-connected PINN network with an optional Fourier feature front-end.

    Parameters
    ----------
    in_dim : input dimension (e.g. 2 for ``(x, t)``).
    hidden : number of hidden layers.
    width : hidden-layer width.
    out_dim : output dimension.
    fourier_freq : if given, prepend a :class:`FourierEmbedding` with this many
        frequencies (recommended for high-frequency solutions).
    """
    inp = keras.Input((in_dim,))
    h = FourierEmbedding(fourier_freq, fourier_sigma, name=f"{name}_fourier")(inp) \
        if fourier_freq else inp
    for _ in range(hidden):
        h = keras.layers.Dense(width, activation, kernel_initializer="glorot_normal")(h)
    return keras.Model(inp, keras.layers.Dense(out_dim)(h), name=name)
