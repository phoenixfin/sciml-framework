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
        in_ch = int(input_shape[-1])
        scale = 1.0 / (in_ch * self.out_channels)
        init = keras.initializers.RandomUniform(-scale, scale)
        self.w_real = self.add_weight(
            name="w_real", shape=(in_ch, self.out_channels, self.modes), initializer=init)
        self.w_imag = self.add_weight(
            name="w_imag", shape=(in_ch, self.out_channels, self.modes), initializer=init)

    def call(self, x):
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
        cfg = super().get_config()
        cfg.update({"out_channels": self.out_channels, "modes": self.modes})
        return cfg
