"""Numpy interpolation helpers.

The differentiable TensorFlow grid interpolation used inside neural training
steps lives in :func:`sciml.tf_utils.grid_interp`.
"""

from __future__ import annotations

import numpy as np


def interp_to_grid(x_query: np.ndarray, x_known: np.ndarray,
                   f_known: np.ndarray) -> np.ndarray:
    """1D linear interpolation, returning float32."""
    return np.interp(x_query, x_known, f_known).astype(np.float32)


def interp_many(x_query: np.ndarray, x_known: np.ndarray,
                F_known: np.ndarray) -> np.ndarray:
    """Interpolate a batch ``(n, len(x_known)) -> (n, len(x_query))``."""
    return np.stack(
        [np.interp(x_query, x_known, F_known[i]) for i in range(len(F_known))]
    ).astype(np.float32)
