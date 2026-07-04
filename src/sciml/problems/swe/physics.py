"""Finite-difference PDE residual and a purely physics-informed SWE step.

Used by the ``F=0`` attractor experiment: a DeepONet trained only on the SWE
residual + BC (no data) collapses to the trivial ``F=0`` state.
"""

from __future__ import annotations

from typing import Callable

import tensorflow as tf

from ...tf_utils import grid_interp


def pde_residual_fd(model: tf.keras.Model, h0b: tf.Tensor, bb: tf.Tensor, h0g: tf.Tensor,
                    bg: tf.Tensor, xc: tf.Tensor, tc: tf.Tensor, L: float,
                    eps_fd: float = 1e-3) -> tf.Tensor:
    """Mean-squared SWE residual via central finite differences (4 extra fwd passes).

    Parameters
    ----------
    model : tf.keras.Model
        The SWE DeepONet model.
    h0b : tf.Tensor
        Initial-depth sensor values for the batch.
    bb : tf.Tensor
        Bathymetry sensor values for the batch.
    h0g : tf.Tensor
        Initial depth on the interpolation grid.
    bg : tf.Tensor
        Bathymetry on the interpolation grid.
    xc : tf.Tensor
        Collocation x-coordinates.
    tc : tf.Tensor
        Collocation t-coordinates.
    L : float
        Domain length (period) for grid interpolation.
    eps_fd : float
        Finite-difference step size.

    Returns
    -------
    tf.Tensor
        The scalar mean-squared PDE residual.
    """
    eps = tf.constant(eps_fd, tf.float32)

    def fwd(xq, tq):
        xt = tf.stack([xq, tq], 1)
        return model(h0b, bb, xt, grid_interp(h0g, xq, L), grid_interp(bg, xq, L))

    h_tp, hu_tp = fwd(xc, tc + eps); h_tm, hu_tm = fwd(xc, tc - eps)
    h_xp, hu_xp = fwd(xc + eps, tc); h_xm, hu_xm = fwd(xc - eps, tc)
    dh_dt = (h_tp - h_tm) / (2 * eps)
    dhu_dt = (hu_tp - hu_tm) / (2 * eps)
    dhu_dx = (hu_xp - hu_xm) / (2 * eps)
    return tf.reduce_mean((dh_dt + dhu_dx) ** 2 + dhu_dt ** 2)


def make_pi_step(model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer, L: float,
                 lam_pde: float = 1.0, lam_bc: float = 5.0,
                 grad_clip: float = 1.0) -> Callable:
    """Physics-informed step: FD-PDE residual + periodic BC, no data loss.

    Parameters
    ----------
    model : tf.keras.Model
        The SWE DeepONet model to train.
    optimizer : tf.keras.optimizers.Optimizer
        Optimizer applying the gradient updates.
    L : float
        Domain length (period) for grid interpolation and boundary conditions.
    lam_pde : float
        Weight on the PDE-residual loss term.
    lam_bc : float
        Weight on the periodic boundary-condition loss term.
    grad_clip : float
        Global-norm gradient-clipping threshold.

    Returns
    -------
    Callable
        A ``step`` function performing one physics-informed optimization step.
    """
    lp = tf.constant(lam_pde, tf.float32)
    lbc = tf.constant(lam_bc, tf.float32)
    L = tf.constant(float(L), tf.float32)

    def step(h0_b, b_b, h0_g, b_g, t_bc, xc, tc):
        x0 = tf.zeros_like(t_bc); xL = tf.fill(tf.shape(t_bc), L)
        h0_bcl = grid_interp(h0_g, x0, L); h0_bcr = grid_interp(h0_g, xL, L)
        b_bcl = grid_interp(b_g, x0, L); b_bcr = grid_interp(b_g, xL, L)
        with tf.GradientTape() as tape:
            Lpde = pde_residual_fd(model, h0_b, b_b, h0_g, b_g, xc, tc, L)
            hl, hul = model(h0_b, b_b, tf.stack([x0, t_bc], 1), h0_bcl, b_bcl)
            hr, hur = model(h0_b, b_b, tf.stack([xL, t_bc], 1), h0_bcr, b_bcr)
            Lbc = tf.reduce_mean((hl - hr) ** 2 + (hul - hur) ** 2)
            loss = lp * Lpde + lbc * Lbc
        gs = tape.gradient(loss, model.trainable_variables)
        gs, gn = tf.clip_by_global_norm(gs, grad_clip)
        optimizer.apply_gradients(zip(gs, model.trainable_variables))
        return loss, Lpde, Lbc, gn

    return step
