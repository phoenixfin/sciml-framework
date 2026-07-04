"""Custom Keras layers for PINNs."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras


class FourierEmbedding(keras.layers.Layer):
    """Random Fourier Features -- mitigates spectral bias.

    Maps an input ``x`` to ``[sin(2 pi B^T x), cos(2 pi B^T x)]`` with a fixed
    (non-trainable) random projection ``B ~ N(0, sigma^2)``.
    """

    def __init__(self, n_freq: int = 64, sigma: float = 1.0, **kw):
        """Store the number of frequencies and the projection scale.

        Parameters
        ----------
        n_freq : int
            Number of random Fourier frequencies.
        sigma : float
            Standard deviation of the random projection ``B``.
        **kw
            Extra keyword arguments forwarded to :class:`keras.layers.Layer`.
        """
        super().__init__(**kw)
        self.n_freq = n_freq
        self.sigma = sigma

    def build(self, input_shape: tf.TensorShape) -> None:
        """Create the fixed random projection matrix ``B``.

        Parameters
        ----------
        input_shape : tf.TensorShape
            Shape of the input tensor; its last dim sets the projection rows.

        Returns
        -------
        None
        """
        self.B = self.add_weight(
            name="B", shape=(int(input_shape[-1]), self.n_freq),
            initializer=tf.initializers.RandomNormal(0.0, self.sigma),
            trainable=False)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Return the sine/cosine Fourier features of ``x``.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor whose last dimension is projected.

        Returns
        -------
        tf.Tensor
            The concatenated sine and cosine Fourier features.
        """
        proj = 2.0 * np.pi * tf.matmul(x, self.B)
        return tf.concat([tf.sin(proj), tf.cos(proj)], axis=-1)

    def get_config(self) -> dict:
        """Return the serializable layer configuration.

        Returns
        -------
        dict
            The layer configuration including ``n_freq`` and ``sigma``.
        """
        cfg = super().get_config()
        cfg.update({"n_freq": self.n_freq, "sigma": self.sigma})
        return cfg


class ScaledSigmoid(keras.layers.Layer):
    """Sigmoid rescaled to ``[s_lo, s_hi]`` (GPU-safe, no Lambda)."""

    def __init__(self, s_lo: float, s_hi: float, **kw):
        """Store the low/high bounds of the rescaled sigmoid.

        Parameters
        ----------
        s_lo : float
            Lower bound of the output range.
        s_hi : float
            Upper bound of the output range.
        **kw
            Extra keyword arguments forwarded to :class:`keras.layers.Layer`.
        """
        super().__init__(**kw)
        self.s_lo = float(s_lo)
        self.s_hi = float(s_hi)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Map ``x`` through a sigmoid rescaled to ``[s_lo, s_hi]``.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor.

        Returns
        -------
        tf.Tensor
            The sigmoid of ``x`` rescaled to ``[s_lo, s_hi]``.
        """
        return self.s_lo + (self.s_hi - self.s_lo) * tf.sigmoid(x)

    def get_config(self) -> dict:
        """Return the serializable layer configuration.

        Returns
        -------
        dict
            The layer configuration including ``s_lo`` and ``s_hi``.
        """
        cfg = super().get_config()
        cfg.update({"s_lo": self.s_lo, "s_hi": self.s_hi})
        return cfg
