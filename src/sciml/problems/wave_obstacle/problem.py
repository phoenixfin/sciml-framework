""":class:`WaveObstacleProblem` -- the moving-boundary wave PINN wired to the engine.

Holds the physical parameters, the two networks (``Nu`` displacement, ``Ns``
free boundary), the masked physical-PDE loss (9 weighted terms with causal
weighting), the RAR residual evaluator, and the FDM-reference evaluation.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from ...solvers.wave_fdm import wave_moving_boundary_fdm
from ..base import Problem
from ...methods.pinn.layers import ScaledSigmoid
from ...methods.pinn.networks import build_mlp
from .config import WaveObstacleConfig


class _NsWithHint(keras.Model):
    """Ns(tau) with a frequency hint ``[tau, cos(w tau), sin(w tau)]`` and a
    sigmoid output rescaled to ``[s_lo, s_hi]``."""

    def __init__(self, omega, s_lo, s_hi, hidden=5, width=96, **kw):
        super().__init__(**kw)
        self.omega = float(omega)
        self.dense_layers = [keras.layers.Dense(width, "tanh",
                             kernel_initializer="glorot_normal") for _ in range(hidden)]
        self.out_dense = keras.layers.Dense(1)
        self.scaler = ScaledSigmoid(s_lo, s_hi)

    def call(self, tau, training=False):
        h = tf.concat([tau, tf.cos(self.omega * tau), tf.sin(self.omega * tau)], axis=1)
        for layer in self.dense_layers:
            h = layer(h)
        return self.scaler(self.out_dense(h))


class WaveObstacleProblem(Problem):
    """Moving-boundary wave PINN: networks, derived parameters, loss, RAR and evaluation."""

    name = "wave_obstacle"

    def __init__(self, config: Optional[WaveObstacleConfig] = None):
        super().__init__(config or WaveObstacleConfig())
        p = self.config.params
        self.b, self.a_bc, self.eps, self.delta, self.T = p.b, p.a_bc, p.eps, p.delta, p.t_final

        # Derived analytic quantities.
        self.s_y = 1.0 - math.sqrt(1.0 + self.a_bc - self.b)
        self.c_slope = self.b - 2.0 * self.s_y
        self.omega = math.pi / (1.0 - self.s_y)
        c_param = (self.b - 2.0 * self.s_y) / (self.b - 2.0)
        self.A_sv = math.pi / ((2.0 - self.b) * (c_param - 1.0)) * (self.delta / self.eps)
        self.amp_s = abs(self.eps * self.A_sv)

        # Networks.
        net = self.config.net
        self.Nu = build_mlp(2, net.nu_hidden, net.nu_width, 1,
                            fourier_freq=net.nu_freq,
                            fourier_sigma=float(self.omega / (2 * math.pi)), name="Nu")
        self.Ns = _NsWithHint(self.omega, net.s_lo, float(self.b) - net.s_hi_margin,
                              hidden=net.ns_hidden, width=net.ns_width, name="Ns")
        self.Ns(tf.zeros([1, 1]))  # build

        self.eps_causal = tf.Variable(5.0, trainable=False, dtype=tf.float32, name="eps_causal")
        lc = self.config.loss
        self.mask_eps = tf.constant(lc.mask_eps, tf.float32)
        self._xbar_buf = tf.Variable(tf.zeros([lc.n_pde, 1]), trainable=False)
        self._tau_buf = tf.Variable(tf.zeros([lc.n_pde, 1]), trainable=False)
        self._tau_sv_buf = tf.Variable(tf.zeros([lc.n_sv, 1]), trainable=False)

    # -- analytic / reference --------------------------------------------
    def s_analytic(self, tau: np.ndarray) -> np.ndarray:
        """Analytic free-boundary position ``s(tau)``."""
        tau = np.asarray(tau, dtype=np.float32)
        return (self.s_y + self.eps * self.A_sv * np.cos(self.omega * tau)).astype(np.float32)

    def u_stationary(self, xbar: np.ndarray) -> np.ndarray:
        """Stationary (equilibrium) displacement profile ``u_bar(xbar)``."""
        return self.a_bc + (np.asarray(xbar) - 1.0) * self.c_slope

    def reference(self, n_snaps: int = 150) -> Dict:
        """FDM reference solution (space-time snapshots) for evaluation."""
        return wave_moving_boundary_fdm(
            self.s_analytic, self.u_stationary, self.s_y, delta=self.delta,
            t_final=self.T, nx=self.config.train.fdm_nx, n_snaps=n_snaps)

    @property
    def trainable_variables(self):
        """Combined trainable variables of the ``Nu`` and ``Ns`` networks."""
        return self.Nu.trainable_variables + self.Ns.trainable_variables

    def domain_mask(self, xbar, s_tau):
        """Soft sigmoid mask (~1 inside the domain, ~0 inside the obstacle region)."""
        return tf.sigmoid((xbar - s_tau) / self.mask_eps)

    # -- loss -------------------------------------------------------------
    @property
    def loss_component_names(self):
        """Names of the nine loss components returned by the loss function."""
        return ["pde", "excl", "bcd", "bcn", "bcr", "icu", "icv", "s0", "sv"]

    def make_loss(self):
        """Return a ``loss_components()`` tf.function returning (total, *components)."""
        cfg = self.config
        lc = cfg.loss
        w = lc.weights
        N_pde, N_bc, N_bcr, N_ic, N_sv = lc.n_pde, lc.n_bc, lc.n_bcr, lc.n_ic, lc.n_sv
        T = tf.constant(self.T, tf.float32)
        a_bc = tf.constant(self.a_bc, tf.float32)
        c_slope = tf.constant(self.c_slope, tf.float32)
        s_y = tf.constant(self.s_y, tf.float32)
        b = tf.constant(self.b, tf.float32)
        delta = tf.constant(self.delta, tf.float32)
        Nu, Ns = self.Nu, self.Ns

        @tf.function
        def loss_components(training=True):
            # PDE residual in physical coords (xbar, tau), masked + causal.
            self._xbar_buf.assign(tf.random.uniform([N_pde, 1], 0.0, 1.0))
            self._tau_buf.assign(tf.random.uniform([N_pde, 1], 0.0, self.T))
            xt = tf.concat([self._xbar_buf, self._tau_buf], axis=1)
            with tf.GradientTape(persistent=True) as t2:
                t2.watch(xt)
                with tf.GradientTape(persistent=True) as t1:
                    t1.watch(xt); u = Nu(xt, training=training)
                ux = t1.gradient(u, xt)[:, 0:1]
                ut = t1.gradient(u, xt)[:, 1:2]
            uxx = t2.gradient(ux, xt)[:, 0:1]
            utt = t2.gradient(ut, xt)[:, 1:2]
            del t1, t2
            s_pde = Ns(self._tau_buf, training=training)
            mask_in = self.domain_mask(self._xbar_buf, s_pde)
            w_c = tf.exp(-self.eps_causal * self._tau_buf / T)
            L_pde = tf.reduce_mean(w_c * mask_in * tf.square(uxx - utt))
            u_stat = a_bc + (self._xbar_buf - 1.0) * c_slope
            L_excl = tf.reduce_mean((1.0 - mask_in) * tf.square(u - u_stat))

            # BC at xbar = Ns(tau), Beta(2,1) time sampling.
            tau_bc = tf.maximum(tf.random.uniform([N_bc, 1]), tf.random.uniform([N_bc, 1])) * T
            s_bc = Ns(tau_bc, training=training)
            xt_bc = tf.concat([s_bc, tau_bc], axis=1)
            with tf.GradientTape() as tbc:
                tbc.watch(xt_bc); u_bc = Nu(xt_bc, training=training)
            ux_bc = tbc.gradient(u_bc, xt_bc)[:, 0:1]
            L_bcd = tf.reduce_mean(tf.square(u_bc - s_bc * (b - s_bc)))
            L_bcn = tf.reduce_mean(tf.square(ux_bc - (b - 2.0 * s_bc)))

            # Fixed end xbar = 1.
            tau_r = tf.random.uniform([N_bcr, 1], 0.0, self.T)
            u_r = Nu(tf.concat([tf.ones([N_bcr, 1]), tau_r], 1), training=training)
            L_bcr = tf.reduce_mean(tf.square(u_r - a_bc))

            # Initial condition (displacement + velocity).
            x_ic = tf.random.uniform([N_ic, 1], float(self.s_y), 1.0)
            xt_ic = tf.concat([x_ic, tf.zeros([N_ic, 1])], 1)
            with tf.GradientTape() as tic:
                tic.watch(xt_ic); u_ic = Nu(xt_ic, training=training)
            ut_ic = tic.gradient(u_ic, xt_ic)[:, 1:2]
            u_ic_true = (a_bc + (x_ic - 1.0) * c_slope
                         + delta * tf.sin(np.pi * (x_ic - s_y) / (1.0 - s_y)))
            L_icu = tf.reduce_mean(tf.square(u_ic - u_ic_true))
            L_icv = tf.reduce_mean(tf.square(ut_ic))

            # Anchor Ns(0) = s_y.
            L_s0 = tf.reduce_mean(tf.square(Ns(tf.zeros([1, 1]), training=training) - s_y))

            # Boundary-velocity consistency: u_t (1-s) - u_x s' = 0 at xbar=s.
            self._tau_sv_buf.assign(
                tf.maximum(tf.random.uniform([N_sv, 1]), tf.random.uniform([N_sv, 1])) * T)
            with tf.GradientTape() as tsv_g:
                tsv_g.watch(self._tau_sv_buf)
                s_sv = Ns(self._tau_sv_buf, training=training)
            sd_sv = tsv_g.gradient(s_sv, self._tau_sv_buf)
            xt_sv = tf.concat([s_sv, self._tau_sv_buf], axis=1)
            with tf.GradientTape() as tsv_u:
                tsv_u.watch(xt_sv); u_sv = Nu(xt_sv, training=training)
            g_sv = tsv_u.gradient(u_sv, xt_sv)
            L_sv = tf.reduce_mean(tf.square(g_sv[:, 1:2] * (1.0 - s_sv) - g_sv[:, 0:1] * sd_sv))

            total = (w["pde"] * L_pde + w["excl"] * L_excl + w["bcd"] * L_bcd
                     + w["bcn"] * L_bcn + w["bcr"] * L_bcr + w["icu"] * L_icu
                     + w["icv"] * L_icv + w["s0"] * L_s0 + w["sv"] * L_sv)
            return total, L_pde, L_excl, L_bcd, L_bcn, L_bcr, L_icu, L_icv, L_s0, L_sv

        return loss_components

    # -- RAR --------------------------------------------------------------
    def pde_residuals(self, n_sample: int = 3000):
        """Evaluate PDE residuals on points in the valid domain (for RAR)."""
        tau_s = np.random.uniform(0, self.T, (n_sample, 1)).astype(np.float32)
        s_est = self.Ns(tau_s).numpy()
        xi_s = np.random.uniform(0, 1, (n_sample, 1)).astype(np.float32)
        xbar_s = s_est + xi_s * (1.0 - s_est)
        xt = tf.constant(np.concatenate([xbar_s, tau_s], axis=1))
        with tf.GradientTape(persistent=True) as t2:
            t2.watch(xt)
            with tf.GradientTape(persistent=True) as t1:
                t1.watch(xt); u = self.Nu(xt, training=False)
            ux = t1.gradient(u, xt)[:, 0:1]
            ut = t1.gradient(u, xt)[:, 1:2]
        uxx = t2.gradient(ux, xt)[:, 0:1]
        utt = t2.gradient(ut, xt)[:, 1:2]
        del t1, t2
        res = tf.square(uxx - utt).numpy().flatten()
        return xbar_s.flatten(), tau_s.flatten(), res

    def assign_collocation(self, xbar: np.ndarray, tau: np.ndarray):
        """Overwrite the PDE collocation buffers (used by RAR resampling)."""
        self._xbar_buf.assign(xbar.reshape(-1, 1).astype(np.float32))
        self._tau_buf.assign(tau.reshape(-1, 1).astype(np.float32))

    def current_collocation(self):
        """Return the current PDE collocation points ``(xbar, tau)`` as numpy arrays."""
        return self._xbar_buf.numpy(), self._tau_buf.numpy()

    # -- evaluation -------------------------------------------------------
    def evaluate(self, n_eval: int = 600) -> Dict[str, float]:
        """Free-boundary error e_s and amplitude/frequency recovery."""
        tau = np.linspace(0, self.T, n_eval, dtype=np.float32)
        s_pinn = self.Ns(tau[:, None]).numpy().flatten()
        s_ref = self.s_analytic(tau)
        e_s = float(np.linalg.norm(s_pinn - s_ref) / np.linalg.norm(s_ref) * 100)
        amp_pinn = float((s_pinn.max() - s_pinn.min()) / 2.0)
        S = np.fft.rfft(s_pinn - s_pinn.mean())
        fs = np.fft.rfftfreq(len(s_pinn), d=tau[1] - tau[0])
        freq_pk = float(fs[np.argmax(np.abs(S[1:])) + 1] * 2 * np.pi)
        return {"e_s_pct": e_s, "amp_pinn": amp_pinn, "amp_ref": self.amp_s,
                "amp_ratio": amp_pinn / self.amp_s, "freq_pinn": freq_pk,
                "freq_ref": self.omega}
