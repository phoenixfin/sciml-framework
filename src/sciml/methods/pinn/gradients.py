"""Automatic-differentiation helpers for PINN residuals."""

from __future__ import annotations

from typing import Dict

import tensorflow as tf


def derivatives_2d(model, xt: tf.Tensor, training: bool = True) -> Dict[str, tf.Tensor]:
    """First and second derivatives of a scalar field over 2D inputs ``(x, t)``.

    Returns a dict with ``u, u_x, u_t, u_xx, u_tt`` (each ``(N, 1)``), computed
    with nested gradient tapes. Suitable for second-order PDEs such as the wave
    equation ``u_xx = u_tt``.
    """
    with tf.GradientTape(persistent=True) as t2:
        t2.watch(xt)
        with tf.GradientTape(persistent=True) as t1:
            t1.watch(xt)
            u = model(xt, training=training)
        g = t1.gradient(u, xt)
        u_x = g[:, 0:1]
        u_t = g[:, 1:2]
    u_xx = t2.gradient(u_x, xt)[:, 0:1]
    u_tt = t2.gradient(u_t, xt)[:, 1:2]
    del t1, t2
    return {"u": u, "u_x": u_x, "u_t": u_t, "u_xx": u_xx, "u_tt": u_tt}


def first_derivatives(model, xt: tf.Tensor, training: bool = True) -> Dict[str, tf.Tensor]:
    """First derivatives ``u, u_x, u_t`` of a scalar field over ``(x, t)``."""
    with tf.GradientTape() as tape:
        tape.watch(xt)
        u = model(xt, training=training)
    g = tape.gradient(u, xt)
    return {"u": u, "u_x": g[:, 0:1], "u_t": g[:, 1:2]}
