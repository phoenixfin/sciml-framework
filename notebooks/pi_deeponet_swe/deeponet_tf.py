"""
Minimal re-implementation of the manuscript's four-branch DeepONet, instrumented
for the diagnostics required by the revision.

Two things differ from the paper on purpose, both switchable:

  fusion   : 'add'      beta = B1(h0) + B2(b)                (paper; additively separable)
             'concat'   beta = B([h0, b])                    (can represent h0-b coupling)
             'bilinear' beta = B1(h0) + B2(b) + B1(h0)*B2(b)

  ic_mode  : 'paper'    elu(h0 + tF - b - hmin) + b + hmin + eps
                        -> NOT exact at t=0 (off by eps) and NOT positivity-guaranteed
             'shifted'  same, with eps moved inside  -> exact at t=0
             'exp'      b + hmin + (h0 - b - hmin) * exp(tF)  -> exact AND strictly > b+hmin
             'softplus' b + hmin + softplus(tF + softplus_inv(h0-b-hmin))  (stable softplus)
             'elu_scaled'  b + hmin + (h0 - b - hmin) * (elu(tF) + 1)
                        -> exact AND floored like 'exp', but linear in F rather than
                           exponential

Requires TensorFlow >= 2.10.
"""
import numpy as np
import tensorflow as tf

G = 9.81
DTYPE = tf.float32


# ----------------------------------------------------------------------
def mlp(in_dim, width=128, depth=3, p=64, out_std=0.01, name=None):
    layers = [tf.keras.Input(shape=(in_dim,))]
    xx = layers[0]
    for i in range(depth):
        xx = tf.keras.layers.Dense(
            width, activation="tanh",
            kernel_initializer=tf.keras.initializers.GlorotUniform())(xx)
    xx = tf.keras.layers.Dense(
        p, activation=None,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=out_std),
        bias_initializer="zeros")(xx)
    return tf.keras.Model(layers[0], xx, name=name)


def _softplus_inv(y):
    # stable log(exp(y) - 1)
    return y + tf.math.log(-tf.math.expm1(-y))


def _softplus(z):
    return tf.nn.softplus(z)


def _elu_plus_one(z):
    """``elu(z) + 1`` without the cancellation.

    ``tf.nn.elu(z) + 1`` forms ``exp(z) - 1`` and adds one back, which rounds to
    exactly 0 in float32 below z ~ -16 — pinning the depth to the floor and zeroing
    the gradient. Taking the branches directly avoids both.
    """
    z = tf.clip_by_value(z, -20.0, 20.0)
    return tf.where(z >= 0.0, z + 1.0, tf.exp(tf.minimum(z, 0.0)))


class FourBranchDeepONet(tf.keras.Model):
    def __init__(self, m=100, p=64, width=128, depth=3,
                 fusion="add", ic_mode="paper", hmin=0.05, eps=1e-4, **kw):
        super().__init__(**kw)
        self.m, self.p = m, p
        self.fusion, self.ic_mode = fusion, ic_mode
        self.hmin, self.eps = hmin, eps
        bin_dim = 2 * m if fusion == "concat" else m
        n_branch = 1 if fusion == "concat" else 2
        self.branch = {}
        for out in ("h", "hu"):
            self.branch[out] = [mlp(bin_dim, width, depth, p, name=f"B{i}_{out}")
                                for i in range(n_branch)]
        self.trunk = {out: mlp(2, width, depth, p, name=f"T_{out}") for out in ("h", "hu")}

    # -- coefficients -------------------------------------------------
    def beta(self, out, h0s, bs):
        B = self.branch[out]
        if self.fusion == "concat":
            return B[0](tf.concat([h0s, bs], axis=-1))
        b1, b2 = B[0](h0s), B[1](bs)
        if self.fusion == "bilinear":
            return b1 + b2 + b1 * b2
        return b1 + b2                                  # 'add' = paper

    def correction(self, out, h0s, bs, x, t):
        """F_out(x,t): (Nf, p) . (Nq, p) -> (Nf, Nq) if broadcast, here elementwise."""
        be = self.beta(out, h0s, bs)                    # (N, p)
        ta = self.trunk[out](tf.concat([x, t], axis=-1))  # (N, p)
        return tf.reduce_sum(be * ta, axis=-1, keepdims=True)

    # -- IC shortcut --------------------------------------------------
    def shortcut_h(self, h0q, bq, F, t):
        hmin, eps = self.hmin, self.eps
        z = h0q + t * F - bq - hmin
        if self.ic_mode == "paper":
            return tf.nn.elu(z) + bq + hmin + eps
        if self.ic_mode == "shifted":
            return tf.nn.elu(z - eps) + bq + hmin + eps
        d = tf.maximum(h0q - bq - hmin, 1e-6)           # depth above the floor at t=0
        if self.ic_mode == "exp":
            return bq + hmin + d * tf.exp(tf.clip_by_value(t * F, -20.0, 20.0))
        if self.ic_mode == "elu_scaled":
            # elu(z)+1 is exp(z) below 0 and z+1 above: exact at t=0 and strictly
            # above the floor like 'exp', but linear rather than exponential in F
            return bq + hmin + d * _elu_plus_one(t * F)
        if self.ic_mode == "softplus":
            return bq + hmin + _softplus(t * F + _softplus_inv(d))
        raise ValueError(self.ic_mode)

    def call(self, inputs, training=False):
        h0s, bs, h0q, bq, x, t = inputs
        Fh = self.correction("h", h0s, bs, x, t)
        Fhu = self.correction("hu", h0s, bs, x, t)
        return self.shortcut_h(h0q, bq, Fh, t), t * Fhu

    # -- parameter groups for the gradient-split diagnostic -----------
    def trunk_vars(self):
        return [v for out in ("h", "hu") for v in self.trunk[out].trainable_variables]

    def branch_vars(self):
        return [v for out in ("h", "hu") for net in self.branch[out]
                for v in net.trainable_variables]


# ----------------------------------------------------------------------
def swe_residual(model, h0s, bs, h0_fn, b_fn, x, t):
    """
    SWE mass/momentum residuals via autodiff.
    h0_fn, b_fn must be TF-differentiable callables of x (use analytic benchmark
    profiles here so d/dx of the shortcut feed-through is exact).
    """
    with tf.GradientTape(persistent=True) as g:
        g.watch([x, t])
        h0q, bq = h0_fn(x), b_fn(x)
        h, hu = model([h0s, bs, h0q, bq, x, t])
        mom_flux = hu * hu / tf.maximum(h, 1e-6) + 0.5 * G * h * h
    def _d(y, v):
        gr = g.gradient(y, v)
        return tf.zeros_like(v) if gr is None else gr
    h_t = _d(h, t)
    hu_x = _d(hu, x)
    hu_t = _d(hu, t)
    f_x = _d(mom_flux, x)
    b_x = _d(bq, x)          # zeros for a flat bed (tf.zeros_like is not tape-connected)
    r_mass = h_t + hu_x
    r_mom = hu_t + f_x + G * h * b_x
    return r_mass, r_mom


def gradient_split_at_F0(model, h0s, bs, h0_fn, b_fn, x, t):
    """
    Core diagnostic for the corrected Proposition 1.

    Forces F == 0 by zeroing every branch output layer (so beta == 0), then
    reports ||grad L_PDE|| separately for trunk and branch parameters, and the
    mass / momentum residual norms.

    Prediction of the corrected statement:
        trunk gradient  == 0 exactly
        mass residual   == 0 exactly
        branch gradient != 0 unless h0 - b = const (lake at rest)
    """
    saved = []
    for out in ("h", "hu"):
        for net in model.branch[out]:
            w, bvec = net.layers[-1].get_weights()
            saved.append((net, w.copy(), bvec.copy()))
            net.layers[-1].set_weights([np.zeros_like(w), np.zeros_like(bvec)])

    with tf.GradientTape(persistent=True) as tape:
        r_mass, r_mom = swe_residual(model, h0s, bs, h0_fn, b_fn, x, t)
        L_mass = tf.reduce_mean(tf.square(r_mass))
        L_mom = tf.reduce_mean(tf.square(r_mom))
        L = L_mass + L_mom
    tv, bv = model.trunk_vars(), model.branch_vars()
    gt = tape.gradient(L, tv)
    gb = tape.gradient(L, bv)
    del tape

    def gnorm(gs):
        gs = [g for g in gs if g is not None]
        return 0.0 if not gs else float(tf.linalg.global_norm(gs))

    for net, w, bvec in saved:                      # restore
        net.layers[-1].set_weights([w, bvec])

    return dict(
        trunk_grad_norm=gnorm(gt),
        branch_grad_norm=gnorm(gb),
        mass_residual_rms=float(tf.sqrt(L_mass)),
        momentum_residual_rms=float(tf.sqrt(L_mom)),
    )
