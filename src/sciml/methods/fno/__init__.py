"""Fourier Neural Operator (FNO) engine (TensorFlow).

An FNO learns an operator by composing spectral-convolution layers: each layer
transforms the input to Fourier space, keeps the lowest ``modes`` frequencies,
applies a learned complex linear map per mode, and transforms back -- plus a
pointwise (1x1 conv) residual path. This makes it resolution-flexible and
efficient for smooth PDE solution operators.

1D:  :class:`SpectralConv1D`, :class:`FNOBlock`, :func:`build_fno1d`
2D:  :class:`SpectralConv2D`, :class:`FNOBlock2D`, :func:`build_fno2d`
"""

from .spectral import SpectralConv1D, SpectralConv2D
from .model import FNOBlock, build_fno1d, FNOBlock2D, build_fno2d

__all__ = ["SpectralConv1D", "FNOBlock", "build_fno1d",
           "SpectralConv2D", "FNOBlock2D", "build_fno2d"]
