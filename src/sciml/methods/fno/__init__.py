"""Fourier Neural Operator (FNO) engine (TensorFlow).

An FNO learns an operator by composing spectral-convolution layers: each layer
transforms the input to Fourier space, keeps the lowest ``modes`` frequencies,
applies a learned complex linear map per mode, and transforms back -- plus a
pointwise (1x1 conv) residual path. This makes it resolution-flexible and
efficient for smooth PDE solution operators.

* :class:`SpectralConv1D` -- the spectral-convolution layer.
* :class:`FNOBlock`        -- spectral conv + pointwise conv + activation.
* :func:`build_fno1d`      -- a 1D FNO model.
"""

from .spectral import SpectralConv1D
from .model import FNOBlock, build_fno1d

__all__ = ["SpectralConv1D", "FNOBlock", "build_fno1d"]
