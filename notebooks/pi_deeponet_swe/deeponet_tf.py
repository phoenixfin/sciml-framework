"""Minimal four-branch DeepONet for the SWE, instrumented for the revision.

This is a deliberately small re-implementation of the manuscript's architecture:
two branch nets per output field (depth ``h`` and discharge ``hu``) reading the
initial depth ``h0`` and the bathymetry ``b`` at ``m`` sensors, one trunk net per
output reading the query point ``(x, t)``, and an initial-condition shortcut that
makes the prediction reduce to ``h0`` at ``t = 0``.

Two things differ from the paper on purpose, both switchable, because both are
under test in the audit:

``fusion``
    How the two branch outputs combine into the coefficient vector ``beta``.

    - ``'add'`` — ``beta = B1(h0) + B2(b)``; the paper's choice, additively
      separable, so it cannot represent an ``h0``-``b`` interaction.
    - ``'concat'`` — ``beta = B([h0, b])`` with a single branch; can.
    - ``'bilinear'`` — ``beta = B1(h0) + B2(b) + B1(h0) * B2(b)``.

``ic_mode``
    The initial-condition shortcut. ``F`` below is the network correction and
    ``d = h0 - b - hmin`` the depth above the floor at ``t = 0``.

    - ``'paper'`` — ``elu(h0 + tF - b - hmin) + b + hmin + eps``: off by ``eps``
      at ``t = 0``, and *not* positivity-guaranteed (see RESULTS.md §2.5).
    - ``'shifted'`` — the same with ``eps`` moved inside the ELU: exact at
      ``t = 0``.
    - ``'exp'`` — ``b + hmin + d * exp(tF)``: exact and strictly above the floor.
    - ``'elu_scaled'`` — ``b + hmin + d * (elu(tF) + 1)``: exact and floored like
      ``'exp'``, but linear in ``F`` rather than exponential.
    - ``'softplus'`` — ``b + hmin + softplus(tF + softplus_inv(d))``.

Requires TensorFlow >= 2.10.
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

#: Gravitational acceleration [m/s^2].
G = 9.81
#: Floating-point type used throughout the model.
DTYPE = tf.float32


# ----------------------------------------------------------------------
def mlp(
    in_dim: int,
    width: int = 128,
    depth: int = 3,
    p: int = 64,
    out_std: float = 0.01,
    name: Optional[str] = None,
) -> "tf.keras.Model":
    """Build a tanh MLP with a small-variance linear head.

    Parameters
    ----------
    in_dim : int
        Input dimension.
    width : int
        Units per hidden layer.
    depth : int
        Number of hidden layers.
    p : int
        Output dimension (the DeepONet latent width).
    out_std : float
        Standard deviation of the truncated-normal initialiser on the output
        layer. Small values start the correction ``F`` near zero.
    name : Optional[str]
        Keras model name.

    Returns
    -------
    tf.keras.Model
        Functional model mapping ``(batch, in_dim)`` to ``(batch, p)``.
    """
    layers = [tf.keras.Input(shape=(in_dim,))]
    xx = layers[0]
    for _ in range(depth):
        xx = tf.keras.layers.Dense(
            width, activation="tanh",
            kernel_initializer=tf.keras.initializers.GlorotUniform())(xx)
    xx = tf.keras.layers.Dense(
        p, activation=None,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=out_std),
        bias_initializer="zeros")(xx)
    return tf.keras.Model(layers[0], xx, name=name)


def _softplus_inv(y: "tf.Tensor") -> "tf.Tensor":
    """Inverse softplus, computed stably as ``y + log(-expm1(-y))``.

    Parameters
    ----------
    y : tf.Tensor
        Strictly positive tensor.

    Returns
    -------
    tf.Tensor
        ``log(exp(y) - 1)``, without the overflow of the naive form.
    """
    return y + tf.math.log(-tf.math.expm1(-y))


def _softplus(z: "tf.Tensor") -> "tf.Tensor":
    """Softplus activation.

    Parameters
    ----------
    z : tf.Tensor
        Input tensor.

    Returns
    -------
    tf.Tensor
        ``log(1 + exp(z))``.
    """
    return tf.nn.softplus(z)


def _elu_plus_one(z: "tf.Tensor") -> "tf.Tensor":
    """``elu(z) + 1`` without the cancellation.

    ``tf.nn.elu(z) + 1`` forms ``exp(z) - 1`` and adds one back, which rounds to
    exactly 0 in float32 below z ~ -16 — pinning the depth to the floor and zeroing
    the gradient. Taking the branches directly avoids both.

    Parameters
    ----------
    z : tf.Tensor
        Input tensor, clipped to ``[-20, 20]`` before the exponential.

    Returns
    -------
    tf.Tensor
        ``z + 1`` where ``z >= 0`` and ``exp(z)`` below, i.e. a strictly
        positive, once-differentiable multiplier.
    """
    z = tf.clip_by_value(z, -20.0, 20.0)
    return tf.where(z >= 0.0, z + 1.0, tf.exp(tf.minimum(z, 0.0)))


class FourBranchDeepONet(tf.keras.Model):
    """Four-branch DeepONet with a positivity-aware initial-condition shortcut.

    The model holds two branch groups and two trunks, one of each per output
    field (``'h'`` and ``'hu'``). ``fusion`` selects how the branch outputs
    combine and ``ic_mode`` selects the shortcut; see the module docstring for
    what each option means and why it is switchable.

    Parameters
    ----------
    m : int
        Number of sensor points at which ``h0`` and ``b`` are sampled.
    p : int
        Latent width shared by branches and trunks.
    width : int
        Units per hidden layer in every sub-network.
    depth : int
        Hidden layers per sub-network.
    fusion : str
        One of ``'add'``, ``'concat'``, ``'bilinear'``.
    ic_mode : str
        One of ``'paper'``, ``'shifted'``, ``'exp'``, ``'elu_scaled'``,
        ``'softplus'``.
    hmin : float
        Minimum depth above the bed enforced by the shortcut [m].
    eps : float
        Offset used by the ``'paper'`` and ``'shifted'`` shortcuts [m].
    **kw
        Forwarded to ``tf.keras.Model``.
    """

    def __init__(self, m: int = 100, p: int = 64, width: int = 128, depth: int = 3,
                 fusion: str = "add", ic_mode: str = "paper",
                 hmin: float = 0.05, eps: float = 1e-4, **kw):
        super().__init__(**kw)
        self.m, self.p = m, p
        self.fusion, self.ic_mode = fusion, ic_mode
        self.hmin, self.eps = hmin, eps
        bin_dim = 2 * m if fusion == "concat" else m
        n_branch = 1 if fusion == "concat" else 2
        self.branch: Dict[str, List["tf.keras.Model"]] = {}
        for out in ("h", "hu"):
            self.branch[out] = [mlp(bin_dim, width, depth, p, name=f"B{i}_{out}")
                                for i in range(n_branch)]
        self.trunk = {out: mlp(2, width, depth, p, name=f"T_{out}") for out in ("h", "hu")}

    # -- coefficients -------------------------------------------------
    def beta(self, out: str, h0s: "tf.Tensor", bs: "tf.Tensor") -> "tf.Tensor":
        """Branch coefficients for one output field.

        Parameters
        ----------
        out : str
            Output field, ``'h'`` or ``'hu'``.
        h0s : tf.Tensor
            Initial depth at the sensors, ``(N, m)``.
        bs : tf.Tensor
            Bathymetry at the sensors, ``(N, m)``.

        Returns
        -------
        tf.Tensor
            Coefficient vector ``(N, p)``, combined according to ``fusion``.
        """
        B = self.branch[out]
        if self.fusion == "concat":
            return B[0](tf.concat([h0s, bs], axis=-1))
        b1, b2 = B[0](h0s), B[1](bs)
        if self.fusion == "bilinear":
            return b1 + b2 + b1 * b2
        return b1 + b2                                  # 'add' = paper

    def correction(self, out: str, h0s: "tf.Tensor", bs: "tf.Tensor",
                   x: "tf.Tensor", t: "tf.Tensor") -> "tf.Tensor":
        """Network correction ``F_out(x, t)`` for one output field.

        Parameters
        ----------
        out : str
            Output field, ``'h'`` or ``'hu'``.
        h0s : tf.Tensor
            Initial depth at the sensors, ``(N, m)``.
        bs : tf.Tensor
            Bathymetry at the sensors, ``(N, m)``.
        x : tf.Tensor
            Query positions, ``(N, 1)``.
        t : tf.Tensor
            Query times, ``(N, 1)``.

        Returns
        -------
        tf.Tensor
            The branch-trunk contraction ``(N, 1)``, taken elementwise over the
            batch (one query per function, not an outer product).
        """
        be = self.beta(out, h0s, bs)                    # (N, p)
        ta = self.trunk[out](tf.concat([x, t], axis=-1))  # (N, p)
        return tf.reduce_sum(be * ta, axis=-1, keepdims=True)

    # -- IC shortcut --------------------------------------------------
    def shortcut_h(self, h0q: "tf.Tensor", bq: "tf.Tensor",
                   F: "tf.Tensor", t: "tf.Tensor") -> "tf.Tensor":
        """Map the raw correction to a depth that respects the initial condition.

        Parameters
        ----------
        h0q : tf.Tensor
            Initial depth at the query points, ``(N, 1)``.
        bq : tf.Tensor
            Bathymetry at the query points, ``(N, 1)``.
        F : tf.Tensor
            Network correction for ``h`` at the query points, ``(N, 1)``.
        t : tf.Tensor
            Query times, ``(N, 1)``.

        Returns
        -------
        tf.Tensor
            Predicted depth ``(N, 1)``, built by the variant named in
            ``ic_mode``.

        Raises
        ------
        ValueError
            If ``ic_mode`` is not one of the five supported variants.
        """
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

    def call(self, inputs: Sequence["tf.Tensor"],
             training: bool = False) -> Tuple["tf.Tensor", "tf.Tensor"]:
        """Predict depth and discharge at the query points.

        Parameters
        ----------
        inputs : Sequence[tf.Tensor]
            ``(h0s, bs, h0q, bq, x, t)``: sensor profiles ``(N, m)`` for the
            initial depth and bathymetry, their values at the query points
            ``(N, 1)``, and the query coordinates ``(N, 1)``.
        training : bool
            Keras training flag; unused, the model has no train-time-only
            layers.

        Returns
        -------
        tuple of (tf.Tensor, tf.Tensor)
            ``(h, hu)``, each ``(N, 1)``. The discharge is ``t * F_hu``, so it
            vanishes at ``t = 0`` by construction.
        """
        h0s, bs, h0q, bq, x, t = inputs
        Fh = self.correction("h", h0s, bs, x, t)
        Fhu = self.correction("hu", h0s, bs, x, t)
        return self.shortcut_h(h0q, bq, Fh, t), t * Fhu

    # -- parameter groups for the gradient-split diagnostic -----------
    def trunk_vars(self) -> List["tf.Variable"]:
        """Trainable variables of both trunks.

        Returns
        -------
        List[tf.Variable]
            Trunk parameters, in ``('h', 'hu')`` order.
        """
        return [v for out in ("h", "hu") for v in self.trunk[out].trainable_variables]

    def branch_vars(self) -> List["tf.Variable"]:
        """Trainable variables of every branch.

        Returns
        -------
        List[tf.Variable]
            Branch parameters, in ``('h', 'hu')`` order.
        """
        return [v for out in ("h", "hu") for net in self.branch[out]
                for v in net.trainable_variables]


# ----------------------------------------------------------------------
def swe_residual(
    model: FourBranchDeepONet,
    h0s: "tf.Tensor",
    bs: "tf.Tensor",
    h0_fn: Callable[["tf.Tensor"], "tf.Tensor"],
    b_fn: Callable[["tf.Tensor"], "tf.Tensor"],
    x: "tf.Tensor",
    t: "tf.Tensor",
) -> Tuple["tf.Tensor", "tf.Tensor"]:
    """Full shallow-water mass and momentum residuals, by autodiff.

    Unlike the manuscript's residual — which is ``d(hu)/dt`` alone, and is
    therefore minimised exactly by ``F = 0`` (RESULTS.md §4.2) — this includes
    the flux divergence and the bed source term.

    Parameters
    ----------
    model : FourBranchDeepONet
        The operator being differentiated.
    h0s : tf.Tensor
        Initial depth at the sensors, ``(N, m)``.
    bs : tf.Tensor
        Bathymetry at the sensors, ``(N, m)``.
    h0_fn : Callable[[tf.Tensor], tf.Tensor]
        TF-differentiable initial-depth profile evaluated at ``x``. Use the
        analytic benchmark profile so the shortcut's feed-through derivative is
        exact rather than interpolated.
    b_fn : Callable[[tf.Tensor], tf.Tensor]
        TF-differentiable bathymetry profile evaluated at ``x``.
    x : tf.Tensor
        Collocation positions ``(N, 1)``, watched by the tape.
    t : tf.Tensor
        Collocation times ``(N, 1)``, watched by the tape.

    Returns
    -------
    tuple of (tf.Tensor, tf.Tensor)
        ``(r_mass, r_mom)``, each ``(N, 1)``: ``h_t + (hu)_x`` and
        ``(hu)_t + [(hu)^2/h + g h^2/2]_x + g h b_x``.
    """
    with tf.GradientTape(persistent=True) as g:
        g.watch([x, t])
        h0q, bq = h0_fn(x), b_fn(x)
        h, hu = model([h0s, bs, h0q, bq, x, t])
        mom_flux = hu * hu / tf.maximum(h, 1e-6) + 0.5 * G * h * h

    # a disconnected variable (e.g. a flat bed) yields None, not a zero gradient
    def _d(y: "tf.Tensor", v: "tf.Tensor") -> "tf.Tensor":
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


def gradient_split_at_F0(
    model: FourBranchDeepONet,
    h0s: "tf.Tensor",
    bs: "tf.Tensor",
    h0_fn: Callable[["tf.Tensor"], "tf.Tensor"],
    b_fn: Callable[["tf.Tensor"], "tf.Tensor"],
    x: "tf.Tensor",
    t: "tf.Tensor",
) -> Dict[str, float]:
    """Core diagnostic for the corrected Proposition 1.

    Forces ``F == 0`` by zeroing every branch output layer (so ``beta == 0``),
    then reports ``||grad L_PDE||`` separately for trunk and branch parameters,
    and the mass / momentum residual norms. The original weights are restored
    before returning.

    The corrected statement predicts: the trunk gradient and the mass residual
    vanish exactly, while the branch gradient does not unless ``h0 - b`` is
    constant (lake at rest).

    Parameters
    ----------
    model : FourBranchDeepONet
        The operator to probe. Its branch output layers are temporarily zeroed.
    h0s : tf.Tensor
        Initial depth at the sensors, ``(N, m)``.
    bs : tf.Tensor
        Bathymetry at the sensors, ``(N, m)``.
    h0_fn : Callable[[tf.Tensor], tf.Tensor]
        TF-differentiable initial-depth profile evaluated at ``x``.
    b_fn : Callable[[tf.Tensor], tf.Tensor]
        TF-differentiable bathymetry profile evaluated at ``x``.
    x : tf.Tensor
        Collocation positions ``(N, 1)``.
    t : tf.Tensor
        Collocation times ``(N, 1)``.

    Returns
    -------
    Dict[str, float]
        ``trunk_grad_norm``, ``branch_grad_norm``, ``mass_residual_rms`` and
        ``momentum_residual_rms``.
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

    # global norm over a parameter group, tolerating disconnected variables
    def gnorm(gs: Sequence[Optional["tf.Tensor"]]) -> float:
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
