"""Spectral-convolution layer for the 1D Fourier Neural Operator."""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras


class SpectralConv1D(keras.layers.Layer):
    """1D spectral convolution: a learned complex linear map on low Fourier modes.

    Input/output shape ``(batch, n_x, channels)``. Only the first ``modes``
    rfft frequencies are kept and mixed across channels by a learned complex
    weight; higher frequencies are zeroed. ``modes`` must be ``<= n_x // 2 + 1``.
    """

    def __init__(self, out_channels: int, modes: int, **kw):
        super().__init__(**kw)
        self.out_channels = out_channels
        self.modes = modes

    def build(self, input_shape):
        """Create the learned complex weights (real/imag parts) for each mode."""
        in_ch = int(input_shape[-1])
        scale = 1.0 / (in_ch * self.out_channels)
        init = keras.initializers.RandomUniform(-scale, scale)
        self.w_real = self.add_weight(
            name="w_real", shape=(in_ch, self.out_channels, self.modes), initializer=init)
        self.w_imag = self.add_weight(
            name="w_imag", shape=(in_ch, self.out_channels, self.modes), initializer=init)

    def call(self, x):
        """Apply the spectral convolution to ``x`` ``(B, N, C)`` -> ``(B, N, O)``."""
        n = tf.shape(x)[1]
        x_t = tf.transpose(x, [0, 2, 1])                  # (B, in_ch, N)
        x_ft = tf.signal.rfft(x_t)                        # (B, in_ch, N//2+1) complex
        x_ft_m = x_ft[..., :self.modes]                   # (B, in_ch, modes)
        w = tf.complex(self.w_real, self.w_imag)          # (in_ch, out_ch, modes)
        out_m = tf.einsum("bim,iom->bom", x_ft_m, w)      # (B, out_ch, modes)
        n_freq = n // 2 + 1
        out_ft = tf.pad(out_m, [[0, 0], [0, 0], [0, n_freq - self.modes]])
        out = tf.signal.irfft(out_ft, fft_length=tf.reshape(n, [1]))  # (B, out_ch, N)
        return tf.transpose(out, [0, 2, 1])               # (B, N, out_ch)

    def get_config(self):
        """Return the serializable layer configuration."""
        cfg = super().get_config()
        cfg.update({"out_channels": self.out_channels, "modes": self.modes})
        return cfg


class SpectralConv2D(keras.layers.Layer):
    """2D spectral convolution on a fixed-size grid.

    Input/output shape ``(batch, H, W, channels)``. Keeps ``modes1`` low
    frequencies in the first spatial dim (both low and high via the two corner
    blocks) and ``modes2`` in the second (rfft) dim. The spatial size is read at
    build time (``modes1 <= H//2``, ``modes2 <= W//2 + 1``).
    """

    def __init__(self, out_channels: int, modes1: int, modes2: int, **kw):
        super().__init__(**kw)
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

    def build(self, input_shape):
        """Read the fixed grid size and create the two corner-block weight sets."""
        in_ch = int(input_shape[-1])
        self.H = int(input_shape[1])
        self.W = int(input_shape[2])
        self.Wf = self.W // 2 + 1
        scale = 1.0 / (in_ch * self.out_channels)
        init = keras.initializers.RandomUniform(-scale, scale)
        shape = (in_ch, self.out_channels, self.modes1, self.modes2)
        self.w1_real = self.add_weight(name="w1_real", shape=shape, initializer=init)
        self.w1_imag = self.add_weight(name="w1_imag", shape=shape, initializer=init)
        self.w2_real = self.add_weight(name="w2_real", shape=shape, initializer=init)
        self.w2_imag = self.add_weight(name="w2_imag", shape=shape, initializer=init)

    def call(self, x):
        """Apply the 2D spectral convolution to ``x`` ``(B, H, W, C)`` -> ``(B, H, W, O)``."""
        m1, m2 = self.modes1, self.modes2
        x_t = tf.transpose(x, [0, 3, 1, 2])               # (B, C, H, W)
        x_ft = tf.signal.rfft2d(x_t)                      # (B, C, H, Wf) complex
        w1 = tf.complex(self.w1_real, self.w1_imag)       # (C, O, m1, m2)
        w2 = tf.complex(self.w2_real, self.w2_imag)
        ll = tf.einsum("bcij,coij->boij", x_ft[:, :, :m1, :m2], w1)         # low-low
        hl = tf.einsum("bcij,coij->boij", x_ft[:, :, self.H - m1:, :m2], w2)  # high-low
        padW = self.Wf - m2
        ll = tf.pad(ll, [[0, 0], [0, 0], [0, 0], [0, padW]])
        hl = tf.pad(hl, [[0, 0], [0, 0], [0, 0], [0, padW]])
        mid = tf.zeros([tf.shape(x)[0], self.out_channels, self.H - 2 * m1, self.Wf],
                       dtype=tf.complex64)
        out_ft = tf.concat([ll, mid, hl], axis=2)         # (B, O, H, Wf)
        out = tf.signal.irfft2d(out_ft, fft_length=[self.H, self.W])  # (B, O, H, W)
        return tf.transpose(out, [0, 2, 3, 1])            # (B, H, W, O)

    def get_config(self):
        """Return the serializable layer configuration."""
        cfg = super().get_config()
        cfg.update({"out_channels": self.out_channels,
                    "modes1": self.modes1, "modes2": self.modes2})
        return cfg
