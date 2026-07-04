"""FNO block and 1D FNO model builder."""

from __future__ import annotations

from typing import Callable

import tensorflow as tf
from tensorflow import keras

from .spectral import SpectralConv1D, SpectralConv2D


class FNOBlock(keras.layers.Layer):
    """One FNO layer: ``activation(SpectralConv1D(v) + Conv1D_1x1(v))``."""

    def __init__(self, width: int, modes: int, activation: Callable = tf.nn.gelu, **kw):
        """Build the spectral and pointwise sub-layers.

        Parameters
        ----------
        width : int
            Number of channels / output width of the block.
        modes : int
            Number of retained Fourier modes in the spectral convolution.
        activation : Callable
            Activation applied to the summed spectral and pointwise paths.
        **kw
            Extra keyword arguments forwarded to :class:`keras.layers.Layer`.
        """
        super().__init__(**kw)
        self.width = width
        self.modes = modes
        self.activation = activation
        self.spec = SpectralConv1D(width, modes)
        self.pointwise = keras.layers.Conv1D(width, 1)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Spectral-conv path plus a pointwise residual path, then activation.

        Parameters
        ----------
        x : tf.Tensor
            Input field of shape ``(batch, n_x, width)``.

        Returns
        -------
        tf.Tensor
            The activated sum of the spectral and pointwise paths.
        """
        return self.activation(self.spec(x) + self.pointwise(x))

    def get_config(self) -> dict:
        """Return the serializable layer configuration.

        Returns
        -------
        dict
            The layer configuration including ``width`` and ``modes``.
        """
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

    Parameters
    ----------
    modes : int
        Number of retained Fourier modes per block.
    width : int
        Channel width of the lifted representation.
    n_layers : int
        Number of stacked FNO blocks.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    name : str
        Name assigned to the constructed Keras model.

    Returns
    -------
    keras.Model
        The constructed 1D Fourier Neural Operator.
    """
    inp = keras.Input((None, in_channels))
    v = keras.layers.Dense(width)(inp)                    # lifting
    for i in range(n_layers):
        v = FNOBlock(width, modes, name=f"fno_block_{i}")(v)
    v = keras.layers.Dense(128, activation=tf.nn.gelu)(v)  # projection
    out = keras.layers.Dense(out_channels)(v)
    return keras.Model(inp, out, name=name)


class FNOBlock2D(keras.layers.Layer):
    """One 2D FNO layer: ``activation(SpectralConv2D(v) + Conv2D_1x1(v))``."""

    def __init__(self, width: int, modes1: int, modes2: int,
                 activation: Callable = tf.nn.gelu, **kw):
        """Build the 2D spectral and pointwise sub-layers.

        Parameters
        ----------
        width : int
            Number of channels / output width of the block.
        modes1 : int
            Retained Fourier modes in the first spatial dimension.
        modes2 : int
            Retained Fourier modes in the second spatial dimension.
        activation : Callable
            Activation applied to the summed spectral and pointwise paths.
        **kw
            Extra keyword arguments forwarded to :class:`keras.layers.Layer`.
        """
        super().__init__(**kw)
        self.width, self.modes1, self.modes2 = width, modes1, modes2
        self.activation = activation
        self.spec = SpectralConv2D(width, modes1, modes2)
        self.pointwise = keras.layers.Conv2D(width, 1)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """2D spectral-conv path plus a pointwise residual path, then activation.

        Parameters
        ----------
        x : tf.Tensor
            Input field of shape ``(batch, H, W, width)``.

        Returns
        -------
        tf.Tensor
            The activated sum of the spectral and pointwise paths.
        """
        return self.activation(self.spec(x) + self.pointwise(x))


def build_fno2d(grid: int, modes: int = 12, width: int = 32, n_layers: int = 4,
                in_channels: int = 3, out_channels: int = 1,
                name: str = "fno2d") -> keras.Model:
    """A 2D Fourier Neural Operator on a fixed ``grid x grid`` mesh.

    Maps ``(batch, grid, grid, in_channels)`` to ``(batch, grid, grid, out_channels)``.
    ``in_channels`` typically stacks the input field with the ``(x, y)`` coords.
    ``modes`` must satisfy ``modes <= grid // 2``.

    Parameters
    ----------
    grid : int
        Side length of the fixed square mesh.
    modes : int
        Number of retained Fourier modes per spatial dimension.
    width : int
        Channel width of the lifted representation.
    n_layers : int
        Number of stacked 2D FNO blocks.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    name : str
        Name assigned to the constructed Keras model.

    Returns
    -------
    keras.Model
        The constructed 2D Fourier Neural Operator.
    """
    inp = keras.Input((grid, grid, in_channels))
    v = keras.layers.Dense(width)(inp)
    for i in range(n_layers):
        v = FNOBlock2D(width, modes, modes, name=f"fno2d_block_{i}")(v)
    v = keras.layers.Dense(128, activation=tf.nn.gelu)(v)
    out = keras.layers.Dense(out_channels)(v)
    return keras.Model(inp, out, name=name)
