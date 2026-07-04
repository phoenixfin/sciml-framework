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
    in_dim : int
        Input dimension (e.g. 2 for ``(x, t)``).
    hidden : int
        Number of hidden layers.
    width : int
        Hidden-layer width.
    out_dim : int
        Output dimension.
    fourier_freq : Optional[int]
        If given, prepend a :class:`FourierEmbedding` with this many
        frequencies (recommended for high-frequency solutions).
    fourier_sigma : float
        Standard deviation of the Fourier embedding's random projection.
    activation : str
        Activation function used in the hidden layers.
    name : str
        Name assigned to the constructed Keras model.

    Returns
    -------
    keras.Model
        The constructed PINN network.
    """
    inp = keras.Input((in_dim,))
    h = FourierEmbedding(fourier_freq, fourier_sigma, name=f"{name}_fourier")(inp) \
        if fourier_freq else inp
    for _ in range(hidden):
        h = keras.layers.Dense(width, activation, kernel_initializer="glorot_normal")(h)
    return keras.Model(inp, keras.layers.Dense(out_dim)(h), name=name)
