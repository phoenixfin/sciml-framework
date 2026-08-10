"""Port of the manuscript's v6 pipeline as a module.

Ported from ``pi_deeponet_swe_v6.ipynb``, which is no longer in the tree; retrieve
it with ``git show ff45b6b^:notebooks/pi_deeponet_swe/pi_deeponet_swe_v6.ipynb``
if you need to check this port against the original.

This is the *paper's own* architecture, loss and training loop — not the minimal
reimplementation in ``deeponet_tf.py``. It exists so the 40k-step run can be
repeated on the regenerated well-balanced data, and so the PDE gradient norm can
be re-measured inside the code that produced the number in §3.7.3.

Deliberate differences from the notebook, all of them additive:

* the training data arrives in a :class:`Bundle` instead of module globals, so the
  same pipeline can be pointed at either dataset;
* :class:`PIDeepONetSWE` gains an ``ic_mode`` switch — ``"paper"`` reproduces
  Eq. (12) exactly (ELU, ``+EPS`` outside), ``"exp"`` is the exact-and-positive
  replacement proposed in the revision;
* three PDE residuals are provided side by side rather than one:

  ==================  =========================================================
  ``time_only``       v6 verbatim: ``R1 = h_t + hu_x``, ``R2 = hu_t``.
                      **The momentum flux divergence and the bed source term are
                      absent**, so this is not the SWE momentum residual. Its
                      global minimum is ``hu == 0``, ``h == h0`` — i.e. exactly
                      the F=0 state, which is why the v6 experiment finds an
                      attractor at h0.
  ``full``            the same central differences plus
                      ``d/dx(hu^2/h + g h^2/2) + g h db/dx``.
  ``pde_residual_ad`` autodiff, full momentum. Requires a single trajectory: v6's
                      ``call`` shares one query set across the batch, so a taped
                      gradient w.r.t. the query coordinates would sum over the
                      batch rather than give per-sample derivatives.
  ==================  =========================================================

Requires TensorFlow >= 2.10.
"""

import time

import numpy as np
import tensorflow as tf

# ----------------------------------------------------------------------
# constants (v6 §2), kept at the manuscript's values
# ----------------------------------------------------------------------
L_np, T_np = 10.0, 1.0
L = tf.constant(L_np, dtype=tf.float32)
G = tf.constant(9.81, dtype=tf.float32)
EPS = tf.constant(1e-4, dtype=tf.float32)
EPS_DIV = tf.constant(0.01, dtype=tf.float32)
H_MIN = tf.constant(0.05, dtype=tf.float32)

M = 100          # sensor points
P = 64           # branch / trunk width
GRID = 500       # spatial grid the supervised data lives on
BATCH = 8        # trajectories per BC mini-batch
N_BC = 200       # boundary collocation times per step
N_COLL = 500     # PDE collocation points per step
LAM_HU = tf.constant(5.0, dtype=tf.float32)
T_SNAPS = [0.25, 0.5, 0.75, 1.0]

x_sensors_np = np.linspace(0, L_np, M, dtype=np.float32)
x_grid_np = np.linspace(0, L_np, GRID, dtype=np.float32)

# benchmark profiles (v6 §3)
h0_c1 = lambda x: (1.0 + 0.5 * np.exp(-2 * (x - 5) ** 2)).astype(np.float32)
h0_c3 = lambda x: (1.5 - 0.5 * np.tanh(5 * (x - 5))).astype(np.float32)
b_flat = lambda x: np.zeros_like(x, dtype=np.float32)
b_bump = lambda x: (0.2 * np.exp(-(x - 5) ** 2)).astype(np.float32)


def periodic_fn(x_src, f):
    """Wrap a sampled profile as a callable, interpolating periodically."""
    return lambda x: np.interp(x, x_src, f, period=L_np).astype(np.float32)


# ----------------------------------------------------------------------
# model (v6 §6)
# ----------------------------------------------------------------------
def _elu_plus_one(z):
    """``elu(z) + 1``, evaluated without the cancellation.

    Writing it literally as ``tf.nn.elu(z) + 1`` forms ``exp(z) - 1`` and then adds
    one back; in float32 that rounds to exactly 0 for z below about -16, which pins
    the depth to the floor *and* zeroes the gradient there. Taking the branches
    directly keeps it strictly positive and differentiable. The lower clip stops
    exp underflowing at large negative z.
    """
    z = tf.clip_by_value(z, -20.0, 20.0)
    return tf.where(z >= 0.0, z + 1.0, tf.exp(tf.minimum(z, 0.0)))


def make_mlp(in_d, hidden, out_d, name, out_std=0.1):
    """Fully connected MLP with tanh activations and small output init."""
    inp = tf.keras.Input(shape=(in_d,))
    x = inp
    for h in hidden:
        x = tf.keras.layers.Dense(h, activation='tanh',
                                  kernel_initializer='glorot_uniform')(x)
    out = tf.keras.layers.Dense(
        out_d,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=out_std),
        bias_initializer='zeros')(x)
    return tf.keras.Model(inputs=inp, outputs=out, name=name)


class PIDeepONetSWE(tf.keras.Model):
    """Four-branch DeepONet for the 1D SWE, exactly as in v6 §6.

        beta_h  = B1h(h0_s)  + B2h(b_s)
        beta_hu = B1hu(h0_s) + B2hu(b_s)
        F_h  = beta_h  @ Th(x,t)^T
        F_hu = beta_hu @ Thu(x,t)^T

    ``ic_mode="paper"`` is Eq. (12):

        h  = elu(h0 + t F_h - b - H_MIN) + b + H_MIN + EPS
        hu = t F_hu

    ``ic_mode="exp"`` replaces it with ``b + H_MIN + (h0-b-H_MIN) exp(t F_h)``,
    which is exact at t=0 and cannot fall below the floor.

    ``ic_mode="elu_scaled"`` is ``b + H_MIN + (h0-b-H_MIN) (elu(t F_h) + 1)``. Since
    ``elu(z) + 1`` is ``exp(z)`` for z < 0 and ``z + 1`` for z >= 0, it is exact at
    t=0 and strictly above the floor exactly as ``exp`` is, but it grows *linearly*
    rather than exponentially in the correction field. That matters: ``exp`` costs a
    factor ~2.4 on unseen-pair generalisation, and exponential amplification of F is
    the obvious suspect.
    """

    def __init__(self, m=M, p=P, ic_mode="paper"):
        super().__init__()
        hid = [128, 128, 128]
        self.m, self.p, self.ic_mode = m, p, ic_mode
        self.b1h = make_mlp(m, hid, p, 'b1h')
        self.b2h = make_mlp(m, hid, p, 'b2h')
        self.b1hu = make_mlp(m, hid, p, 'b1hu')
        self.b2hu = make_mlp(m, hid, p, 'b2hu')
        self.th = make_mlp(2, hid, p, 'th', out_std=0.1)
        self.thu = make_mlp(2, hid, p, 'thu', out_std=0.1)

    def call(self, h0s, bs, xt, h0_at_x, b_at_x):
        """h0s, bs: (B, M) sensors; xt: (N, 2); h0_at_x, b_at_x: (B, N)."""
        t = tf.transpose(xt[:, 1:2])                                    # (1, N)
        beta_h = self.b1h(h0s) + self.b2h(bs)                           # (B, P)
        beta_hu = self.b1hu(h0s) + self.b2hu(bs)
        F_h = tf.linalg.matmul(beta_h, tf.transpose(self.th(xt)))       # (B, N)
        F_hu = tf.linalg.matmul(beta_hu, tf.transpose(self.thu(xt)))
        if self.ic_mode == "paper":
            h_pred = (tf.nn.elu(h0_at_x + t * F_h - (b_at_x + H_MIN))
                      + (b_at_x + H_MIN) + EPS)
        elif self.ic_mode == "shifted":
            # Eq. (12) with EPS moved inside the ELU: exact at t=0, same additive
            # form and same (absent) floor guarantee as the published shortcut
            h_pred = (tf.nn.elu(h0_at_x + t * F_h - (b_at_x + H_MIN) - EPS)
                      + (b_at_x + H_MIN) + EPS)
        elif self.ic_mode == "exp":
            d = tf.maximum(h0_at_x - b_at_x - H_MIN, 1e-6)
            h_pred = b_at_x + H_MIN + d * tf.exp(tf.clip_by_value(t * F_h, -20.0, 20.0))
        elif self.ic_mode == "elu_scaled":
            d = tf.maximum(h0_at_x - b_at_x - H_MIN, 1e-6)
            h_pred = b_at_x + H_MIN + d * _elu_plus_one(t * F_h)
        else:
            raise ValueError(self.ic_mode)
        return h_pred, t * F_hu

    def build_once(self):
        """Warm-up call so ``trainable_variables`` is populated."""
        d = tf.zeros((2, self.m))
        self(d, d, tf.zeros((4, 2)), tf.ones((2, 4)), tf.zeros((2, 4)))
        return self


class SharedBranchDeepONet(tf.keras.Model):
    """A1: one shared beta feeding both outputs (v6 §18). Couples the h and hu
    heads, which is what triggers the BC collapse the ablation is meant to show."""

    def __init__(self, m=M, p=P):
        super().__init__()
        hid = [128, 128, 128]
        self.m, self.p = m, p
        self.b1 = make_mlp(m, hid, p, "b1_shared")
        self.b2 = make_mlp(m, hid, p, "b2_shared")
        self.th = make_mlp(2, hid, p, "th_shared", out_std=0.1)
        self.thu = make_mlp(2, hid, p, "thu_shared", out_std=0.1)

    def call(self, h0s, bs, xt, h0_at_x, b_at_x):
        t = tf.transpose(xt[:, 1:2])
        beta = self.b1(h0s) + self.b2(bs)                      # shared
        F_h = tf.linalg.matmul(beta, tf.transpose(self.th(xt)))
        F_hu = tf.linalg.matmul(beta, tf.transpose(self.thu(xt)))
        h_pred = (tf.nn.elu(h0_at_x + t * F_h - (b_at_x + H_MIN))
                  + (b_at_x + H_MIN) + EPS)
        return h_pred, t * F_hu

    def build_once(self):
        d = tf.zeros((2, self.m))
        self(d, d, tf.zeros((4, 2)), tf.ones((2, 4)), tf.zeros((2, 4)))
        return self


class NoICShortcutDeepONet(tf.keras.Model):
    """A2: separate branches, no analytic IC shortcut (v6 §18). The initial
    condition is imposed by a loss term instead, so the outputs are raw."""

    def __init__(self, m=M, p=P):
        super().__init__()
        hid = [128, 128, 128]
        self.m, self.p = m, p
        self.b1h = make_mlp(m, hid, p, "b1h_noic")
        self.b2h = make_mlp(m, hid, p, "b2h_noic")
        self.b1hu = make_mlp(m, hid, p, "b1hu_noic")
        self.b2hu = make_mlp(m, hid, p, "b2hu_noic")
        self.th = make_mlp(2, hid, p, "th_noic", out_std=0.1)
        self.thu = make_mlp(2, hid, p, "thu_noic", out_std=0.1)

    def call(self, h0s, bs, xt, h0_at_x, b_at_x):
        bh = self.b1h(h0s) + self.b2h(bs)
        bhu = self.b1hu(h0s) + self.b2hu(bs)
        return (tf.linalg.matmul(bh, tf.transpose(self.th(xt))),
                tf.linalg.matmul(bhu, tf.transpose(self.thu(xt))))

    def build_once(self):
        d = tf.zeros((2, self.m))
        self(d, d, tf.zeros((4, 2)), tf.ones((2, 4)), tf.zeros((2, 4)))
        return self


# ----------------------------------------------------------------------
# training utilities (v6 §7)
# ----------------------------------------------------------------------
@tf.function(reduce_retracing=True)
def grid_interp(grid_data, x_query):
    """Linear interpolation: (batch, GRID) x (N,) -> (batch, N)."""
    idx_f = x_query / L * tf.cast(GRID - 1, tf.float32)
    idx_lo = tf.clip_by_value(tf.cast(tf.floor(idx_f), tf.int32), 0, GRID - 2)
    alpha = idx_f - tf.cast(idx_lo, tf.float32)
    return (tf.gather(grid_data, idx_lo, axis=1) * (1.0 - alpha)
            + tf.gather(grid_data, idx_lo + 1, axis=1) * alpha)


def make_optimizer():
    lr = tf.keras.optimizers.schedules.ExponentialDecay(
        1e-3, decay_steps=10000, decay_rate=0.5, staircase=True)
    return tf.keras.optimizers.Adam(lr)


class Bundle:
    """The tensors v6's training step needs, on v6's sensor/grid layout."""

    def __init__(self, H0_s, B_s, H0_grid, B_grid, D_h, D_hu, sup_idx, t_snaps):
        self.H0_s, self.B_s = H0_s, B_s
        self.H0_grid, self.B_grid = H0_grid, B_grid
        self.n_pool = len(H0_s)
        self.n_sup = len(sup_idx)
        self.t_snaps = list(t_snaps)
        self.n_snaps = len(self.t_snaps)
        self.xt_snaps = [
            tf.constant(np.stack([x_grid_np, np.full(GRID, t)], 1).astype(np.float32))
            for t in self.t_snaps]
        self._D_h, self._D_hu = list(D_h), list(D_hu)     # kept for .subset()
        self._sup_idx = np.asarray(sup_idx)
        self.D_h_tf = [tf.constant(a) for a in D_h]
        self.D_hu_tf = [tf.constant(a) for a in D_hu]
        self.D_H0s = tf.constant(H0_s[sup_idx])
        self.D_Bs = tf.constant(B_s[sup_idx])
        self.D_H0g = tf.constant(H0_grid[sup_idx])
        self.D_Bg = tf.constant(B_grid[sup_idx])


    def subset(self, n):
        """A bundle supervised on only the first n trajectories (pool unchanged)."""
        return Bundle(self.H0_s, self.B_s, self.H0_grid, self.B_grid,
                      [a[:n] for a in self._D_h], [a[:n] for a in self._D_hu],
                      self._sup_idx[:n], self.t_snaps)


def build_bundle(x_src, h0_all, b_all, h_snaps, hu_snaps, t_snaps, n_sup=None):
    """Map a periodic dataset on an arbitrary grid onto v6's layout.

    Parameters
    ----------
    x_src : (nx,) source grid (cell centres are fine — interpolation is periodic).
    h0_all, b_all : (N, nx) initial depth and bathymetry for the whole pool.
    h_snaps, hu_snaps : (n_sup, n_t, nx) supervised snapshots.
    t_snaps : the snapshot times.
    n_sup : how many leading rows carry supervision (defaults to len(h_snaps)).
    """
    n_sup = len(h_snaps) if n_sup is None else n_sup

    def to(xt, arr):
        return np.stack([np.interp(xt, x_src, a, period=L_np) for a in arr]).astype(np.float32)

    H0_s, B_s = to(x_sensors_np, h0_all), to(x_sensors_np, b_all)
    H0_grid, B_grid = to(x_grid_np, h0_all), to(x_grid_np, b_all)
    D_h = [to(x_grid_np, h_snaps[:n_sup, s]) for s in range(len(t_snaps))]
    D_hu = [to(x_grid_np, hu_snaps[:n_sup, s]) for s in range(len(t_snaps))]
    return Bundle(H0_s, B_s, H0_grid, B_grid, D_h, D_hu,
                  np.arange(n_sup), t_snaps)


def make_train_step(mdl, opt, bd, lam_bc_val=5.0):
    """v6's compiled supervised step: full-batch data loss + periodic BC."""
    lam_bc = tf.constant(lam_bc_val, dtype=tf.float32)

    @tf.function(reduce_retracing=True)
    def step(h0_b, b_b, h0_g, b_g, t_bc):
        x0 = tf.zeros_like(t_bc)
        xL = tf.fill(tf.shape(t_bc), L)
        h0_bcl = grid_interp(h0_g, x0); h0_bcr = grid_interp(h0_g, xL)
        b_bcl = grid_interp(b_g, x0); b_bcr = grid_interp(b_g, xL)
        with tf.GradientTape() as tape:
            Ld = tf.constant(0.0)
            for s in range(bd.n_snaps):
                hp, hup = mdl(bd.D_H0s, bd.D_Bs, bd.xt_snaps[s], bd.D_H0g, bd.D_Bg)
                Ld = Ld + (tf.reduce_mean((hp - bd.D_h_tf[s]) ** 2)
                           + LAM_HU * tf.reduce_mean((hup - bd.D_hu_tf[s]) ** 2))
            Ld = Ld / tf.cast(bd.n_snaps, tf.float32)
            hl, hul = mdl(h0_b, b_b, tf.stack([x0, t_bc], 1), h0_bcl, b_bcl)
            hr, hur = mdl(h0_b, b_b, tf.stack([xL, t_bc], 1), h0_bcr, b_bcr)
            Lb = tf.reduce_mean((hl - hr) ** 2 + (hul - hur) ** 2)
            loss = Ld + lam_bc * Lb
        gs = tape.gradient(loss, mdl.trainable_variables)
        gs, gn = tf.clip_by_global_norm(gs, 1.0)
        opt.apply_gradients(zip(gs, mdl.trainable_variables))
        return loss, Ld, Lb, gn

    return step


def train_model(mdl, bd, n_iter, seed=42, log_every=1000, verbose=True):
    """v6's full supervised training loop. Returns the history dict."""
    rng = np.random.default_rng(seed)
    opt = make_optimizer()
    step = make_train_step(mdl, opt, bd)

    idx = np.arange(min(BATCH, bd.n_pool))
    tbc = rng.uniform(0, T_np, N_BC).astype(np.float32)
    step(tf.constant(bd.H0_s[idx]), tf.constant(bd.B_s[idx]),
         tf.constant(bd.H0_grid[idx]), tf.constant(bd.B_grid[idx]), tf.constant(tbc))

    t0 = time.time()
    for _ in range(5):
        idx = rng.choice(bd.n_pool, BATCH, replace=False)
        tbc = rng.uniform(0, T_np, N_BC).astype(np.float32)
        step(tf.constant(bd.H0_s[idx]), tf.constant(bd.B_s[idx]),
             tf.constant(bd.H0_grid[idx]), tf.constant(bd.B_grid[idx]), tf.constant(tbc))
    ms = (time.time() - t0) / 5 * 1000
    if verbose:
        print(f"  step: {ms:.0f} ms | ETA {ms * n_iter / 60000:.0f} min")

    history = {"iter": [], "Ld": [], "Lb": [], "total": [], "gnorm": []}
    t_start = time.time()
    for it in range(1, n_iter + 1):
        idx = rng.choice(bd.n_pool, BATCH, replace=False)
        tbc = rng.uniform(0, T_np, N_BC).astype(np.float32)
        lo, Ld, Lb, gn = step(
            tf.constant(bd.H0_s[idx]), tf.constant(bd.B_s[idx]),
            tf.constant(bd.H0_grid[idx]), tf.constant(bd.B_grid[idx]),
            tf.constant(tbc))
        if it % log_every == 0 or it == n_iter:
            history["iter"].append(it)
            history["Ld"].append(float(Ld))
            history["Lb"].append(float(Lb))
            history["total"].append(float(lo))
            history["gnorm"].append(float(gn))
            if verbose:
                el = time.time() - t_start
                print(f"  {it:6d}/{n_iter} Ld={float(Ld):.3e} Lb={float(Lb):.4f} "
                      f"gn={float(gn):.3f} {el / 60:.1f}/{el / it * n_iter / 60:.0f} min")
    if verbose:
        print(f"  training done in {(time.time() - t_start) / 60:.1f} min")
    return history


def make_ablation_step(mdl, opt, bd, lam_bc=5.0, lam_ic=10.0, use_ic_shortcut=True):
    """v6 §18's step: the supervised step plus an optional explicit IC loss.

    A2 has no analytic shortcut, so the initial condition has to be imposed by a
    penalty. ``lam_ic`` and the t=0 query grid follow v6.
    """
    lbc = tf.constant(lam_bc, tf.float32)
    lic = tf.constant(lam_ic, tf.float32)
    xt_ic = tf.constant(np.stack([x_grid_np, np.zeros(GRID)], 1).astype(np.float32))

    @tf.function(reduce_retracing=True)
    def step(h0_b, b_b, h0_g, b_g, t_bc, h0_ic_s, b_ic_s, h0_ic_g, b_ic_g):
        x0 = tf.zeros_like(t_bc)
        xL = tf.fill(tf.shape(t_bc), L)
        h0_bcl = grid_interp(h0_g, x0); h0_bcr = grid_interp(h0_g, xL)
        b_bcl = grid_interp(b_g, x0); b_bcr = grid_interp(b_g, xL)
        with tf.GradientTape() as tape:
            Ld = tf.constant(0.0)
            for s in range(bd.n_snaps):
                hp, hup = mdl(bd.D_H0s, bd.D_Bs, bd.xt_snaps[s], bd.D_H0g, bd.D_Bg)
                Ld = Ld + (tf.reduce_mean((hp - bd.D_h_tf[s]) ** 2)
                           + LAM_HU * tf.reduce_mean((hup - bd.D_hu_tf[s]) ** 2))
            Ld = Ld / tf.cast(bd.n_snaps, tf.float32)
            hl, hul = mdl(h0_b, b_b, tf.stack([x0, t_bc], 1), h0_bcl, b_bcl)
            hr, hur = mdl(h0_b, b_b, tf.stack([xL, t_bc], 1), h0_bcr, b_bcr)
            Lb = tf.reduce_mean((hl - hr) ** 2 + (hul - hur) ** 2)
            loss = Ld + lbc * Lb
            if not use_ic_shortcut:
                hp_ic, hup_ic = mdl(h0_ic_s, b_ic_s, xt_ic, h0_ic_g, b_ic_g)
                loss = loss + lic * tf.reduce_mean((hp_ic - h0_ic_g) ** 2 + hup_ic ** 2)
        gs = tape.gradient(loss, mdl.trainable_variables)
        gs, gn = tf.clip_by_global_norm(gs, 1.0)
        opt.apply_gradients(zip(gs, mdl.trainable_variables))
        return loss, Ld, Lb, gn

    return step


def train_arch_variant(mdl, bd, n_iter, use_ic_shortcut=True, seed=0,
                       log_every=2000, verbose=True):
    """Training loop for the A1 / A2 / A3 architecture ablation."""
    rng = np.random.default_rng(seed)
    opt = make_optimizer()
    step = make_ablation_step(mdl, opt, bd, use_ic_shortcut=use_ic_shortcut)
    n_ic = min(BATCH, bd.n_sup)
    hist = {"iter": [], "Ld": [], "Lb": []}
    t0 = time.time()
    for it in range(1, n_iter + 1):
        idx = rng.choice(bd.n_pool, BATCH, replace=False)
        ic = rng.choice(bd.n_sup, n_ic, replace=False)
        _, Ld, Lb, _ = step(
            tf.constant(bd.H0_s[idx]), tf.constant(bd.B_s[idx]),
            tf.constant(bd.H0_grid[idx]), tf.constant(bd.B_grid[idx]),
            tf.constant(rng.uniform(0, T_np, N_BC).astype(np.float32)),
            tf.gather(bd.D_H0s, ic), tf.gather(bd.D_Bs, ic),
            tf.gather(bd.D_H0g, ic), tf.gather(bd.D_Bg, ic))
        if it % log_every == 0 or it == n_iter:
            hist["iter"].append(it)
            hist["Ld"].append(float(Ld))
            hist["Lb"].append(float(Lb))
            if verbose:
                print(f"    {it:6d}  Ld={float(Ld):.3e}  Lb={float(Lb):.3e}"
                      f"  ({time.time() - t0:.0f}s)")
    return hist


# ----------------------------------------------------------------------
# PDE residuals
# ----------------------------------------------------------------------
def pde_residual_fd(mdl, h0b, bb, h0g, bg, xc, tc, momentum="time_only", eps=1e-3):
    """Central-difference SWE residual, mean of ``R1**2 + R2**2``.

    ``momentum="time_only"`` is v6 verbatim (``R2 = hu_t``); ``"full"`` adds the
    flux divergence and the bed source term that make it the actual momentum
    equation.
    """
    e = tf.constant(eps, tf.float32)

    def fwd(xq, tq):
        xt = tf.stack([xq, tq], 1)
        return mdl(h0b, bb, xt, grid_interp(h0g, xq), grid_interp(bg, xq))

    h_tp, hu_tp = fwd(xc, tc + e)
    h_tm, hu_tm = fwd(xc, tc - e)
    h_xp, hu_xp = fwd(xc + e, tc)
    h_xm, hu_xm = fwd(xc - e, tc)

    dh_dt = (h_tp - h_tm) / (2 * e)
    dhu_dt = (hu_tp - hu_tm) / (2 * e)
    dhu_dx = (hu_xp - hu_xm) / (2 * e)

    R1 = dh_dt + dhu_dx
    if momentum == "time_only":
        R2 = dhu_dt
    elif momentum == "full":
        def flux(h, hu):
            return hu * hu / tf.maximum(h, EPS_DIV) + 0.5 * G * h * h
        h_c, _ = fwd(xc, tc)
        db_dx = (grid_interp(bg, xc + e) - grid_interp(bg, xc - e)) / (2 * e)
        R2 = dhu_dt + (flux(h_xp, hu_xp) - flux(h_xm, hu_xm)) / (2 * e) + G * h_c * db_dx
    else:
        raise ValueError(momentum)
    return tf.reduce_mean(R1 ** 2 + R2 ** 2)


def pde_residual_ad(mdl, h0b, bb, h0g, bg, xc, tc):
    """Autodiff SWE residual with the full momentum equation.

    Requires a single trajectory (``h0b.shape[0] == 1``): v6's ``call`` shares one
    query set across the batch, so ``tape.gradient`` w.r.t. the query coordinates
    would sum over the batch instead of giving per-sample derivatives.
    """
    if int(h0b.shape[0]) != 1:
        raise ValueError("pde_residual_ad needs a single trajectory (B = 1); "
                         f"got B = {int(h0b.shape[0])}")
    with tf.GradientTape(persistent=True) as g:
        g.watch([xc, tc])
        xt = tf.stack([xc, tc], 1)
        b_q = grid_interp(bg, xc)
        h, hu = mdl(h0b, bb, xt, grid_interp(h0g, xc), b_q)
        flux = hu * hu / tf.maximum(h, EPS_DIV) + 0.5 * G * h * h

    def d(y, v):
        gr = g.gradient(y, v)
        return tf.zeros_like(v) if gr is None else gr

    R1 = d(h, tc) + d(hu, xc)
    R2 = d(hu, tc) + d(flux, xc) + G * tf.reshape(h, [-1]) * d(b_q, xc)
    return tf.reduce_mean(R1 ** 2 + R2 ** 2)


def make_pi_step(mdl, opt, momentum="time_only", lam_pde=1.0, lam_bc=5.0, use_bc=True):
    """Physics-only training step (v6 §17), with the BC term switchable off."""
    lp = tf.constant(lam_pde, tf.float32)
    lbc = tf.constant(lam_bc, tf.float32)

    @tf.function(reduce_retracing=True)
    def step(h0_b, b_b, h0_g, b_g, t_bc, xc, tc):
        with tf.GradientTape() as tape:
            Lpde = pde_residual_fd(mdl, h0_b, b_b, h0_g, b_g, xc, tc, momentum=momentum)
            if use_bc:
                x0 = tf.zeros_like(t_bc); xL = tf.fill(tf.shape(t_bc), L)
                hl, hul = mdl(h0_b, b_b, tf.stack([x0, t_bc], 1),
                              grid_interp(h0_g, x0), grid_interp(b_g, x0))
                hr, hur = mdl(h0_b, b_b, tf.stack([xL, t_bc], 1),
                              grid_interp(h0_g, xL), grid_interp(b_g, xL))
                Lbc = tf.reduce_mean((hl - hr) ** 2 + (hul - hur) ** 2)
            else:
                Lbc = tf.constant(0.0)
            loss = lp * Lpde + lbc * Lbc
        gs = tape.gradient(loss, mdl.trainable_variables)
        gs, gn = tf.clip_by_global_norm(gs, 1.0)
        opt.apply_gradients(zip(gs, mdl.trainable_variables))
        return loss, Lpde, Lbc, gn

    return step


# ----------------------------------------------------------------------
# inference helpers (v6 §10)
# ----------------------------------------------------------------------
def rel_l2(pred, ref):
    return float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-10))


def predict_at(mdl, h0_s, b_s, x, t, h0_at_x, b_at_x):
    """Predict at one time on a shared x-grid for a batch of (h0, b) pairs.

    h0_s, b_s : (B, M) sensors; x : (N,); h0_at_x, b_at_x : (B, N).
    Returns (h, hu), both (B, N).
    """
    xt = tf.constant(np.stack([np.asarray(x, np.float32),
                               np.full(len(x), t, np.float32)], 1))
    h, hu = mdl(tf.constant(np.asarray(h0_s, np.float32)),
                tf.constant(np.asarray(b_s, np.float32)), xt,
                tf.constant(np.asarray(h0_at_x, np.float32)),
                tf.constant(np.asarray(b_at_x, np.float32)))
    return h.numpy(), hu.numpy()


def predict_grid(mdl, h0_fn, b_fn, nx=200, nt=100):
    """Predict on an (nx x nt) space-time grid. Returns (xs, ts, h, hu)."""
    h0_s = h0_fn(x_sensors_np).reshape(1, -1).astype(np.float32)
    b_s = b_fn(x_sensors_np).reshape(1, -1).astype(np.float32)
    xs = np.linspace(0, L_np, nx, dtype=np.float32)
    ts = np.linspace(0, T_np, nt, dtype=np.float32)
    XX, TT = np.meshgrid(xs, ts)
    xt = tf.constant(np.stack([XX.ravel(), TT.ravel()], 1).astype(np.float32))
    h0_at = tf.constant(h0_fn(XX.ravel()).reshape(1, -1).astype(np.float32))
    b_at = tf.constant(b_fn(XX.ravel()).reshape(1, -1).astype(np.float32))
    hp, hup = mdl(tf.constant(h0_s), tf.constant(b_s), xt, h0_at, b_at)
    return xs, ts, hp.numpy()[0].reshape(nt, nx), hup.numpy()[0].reshape(nt, nx)
