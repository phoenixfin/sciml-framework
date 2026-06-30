"""Optimizer / learning-rate-schedule factory."""

from __future__ import annotations

import tensorflow as tf


def make_optimizer(lr: float = 1e-3, decay_steps: int = 10000,
                   decay_rate: float = 0.5) -> tf.keras.optimizers.Optimizer:
    """Adam with an exponential-decay (staircase) learning-rate schedule."""
    schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        lr, decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)
    return tf.keras.optimizers.Adam(schedule)
