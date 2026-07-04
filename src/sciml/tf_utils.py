"""TensorFlow-specific helpers (importing this module requires TensorFlow)."""

from __future__ import annotations

from typing import List

import tensorflow as tf


@tf.function(reduce_retracing=True)
def grid_interp(grid_data: tf.Tensor, x_query: tf.Tensor, length: tf.Tensor) -> tf.Tensor:
    """Differentiable linear interpolation on a uniform grid.

    The differentiable counterpart of
    :func:`sciml.data.interp.interp_to_grid`, safe (and fast) inside a
    ``@tf.function`` training step.

    Parameters
    ----------
    grid_data : tf.Tensor
        Values of shape ``(batch, G)`` sampled on a uniform grid over
        ``[0, length]``.
    x_query : tf.Tensor
        Query coordinates of shape ``(N,)``.
    length : tf.Tensor
        Length of the domain the grid spans.

    Returns
    -------
    tf.Tensor
        Interpolated values of shape ``(batch, N)``.
    """
    g = tf.shape(grid_data)[1]
    idx_f = x_query / length * tf.cast(g - 1, tf.float32)
    idx_lo = tf.clip_by_value(tf.cast(tf.floor(idx_f), tf.int32), 0, g - 2)
    alpha = idx_f - tf.cast(idx_lo, tf.float32)
    return (tf.gather(grid_data, idx_lo, axis=1) * (1.0 - alpha)
            + tf.gather(grid_data, idx_lo + 1, axis=1) * alpha)


def global_grad_norm(grads: List[tf.Tensor]) -> float:
    """Global L2 norm of a gradient list (ignoring ``None`` entries).

    Parameters
    ----------
    grads : List[tf.Tensor]
        Gradients, some of which may be ``None``.

    Returns
    -------
    float
        The global L2 norm over the non-``None`` gradients.
    """
    return float(tf.linalg.global_norm([g for g in grads if g is not None]))
