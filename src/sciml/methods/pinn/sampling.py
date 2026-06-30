"""Collocation-point sampling helpers for PINNs."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def beta_max_sampling(n: int, high: float):
    """Sample ``n`` values on ``[0, high]`` biased toward ``high``.

    Draws ``max(U1, U2)`` (a Beta(2,1) shape), useful for concentrating
    boundary/collocation points at later times where the solution is harder.
    Returns a TF tensor of shape ``(n, 1)``.
    """
    import tensorflow as tf
    u1 = tf.random.uniform([n, 1]); u2 = tf.random.uniform([n, 1])
    return tf.maximum(u1, u2) * high


def replace_high_residual(buffer_x: np.ndarray, buffer_t: np.ndarray,
                          cand_x: np.ndarray, cand_t: np.ndarray, residuals: np.ndarray,
                          *, frac: float = 0.2, percentile: float = 70.0,
                          jitter_x: float = 0.02, jitter_t: float = 0.05,
                          x_bounds: Tuple[float, float] = (0.0, 1.0),
                          t_bounds: Tuple[float, float] = (0.0, 1.0),
                          rng: np.random.Generator | None = None
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Residual-based adaptive refinement (RAR).

    Replace a ``frac`` fraction of buffer points with (jittered) candidates whose
    residual exceeds the ``percentile`` threshold. Returns the updated
    ``(buffer_x, buffer_t)`` arrays (both shape ``(n, 1)``).
    """
    draw = rng if rng is not None else np.random
    thresh = np.percentile(residuals, percentile)
    keep = residuals >= thresh
    if keep.sum() <= 10:
        return buffer_x, buffer_t
    n_replace = max(1, int(len(buffer_x) * frac))
    cand_x = cand_x[keep]; cand_t = cand_t[keep]
    pick = draw.choice(len(cand_x), n_replace, replace=True)
    new_x = np.clip(cand_x[pick] + draw.normal(0, jitter_x, n_replace), *x_bounds)
    new_t = np.clip(cand_t[pick] + draw.normal(0, jitter_t, n_replace), *t_bounds)
    slots = draw.choice(len(buffer_x), n_replace, replace=False)
    buffer_x = buffer_x.copy(); buffer_t = buffer_t.copy()
    buffer_x[slots, 0] = new_x
    buffer_t[slots, 0] = new_t
    return buffer_x, buffer_t
