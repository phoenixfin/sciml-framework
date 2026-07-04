"""DeepONet architectures for the 1D SWE (TensorFlow).

The flagship :class:`SWEDeepONet` uses one operator per output field (depth
``h`` and momentum ``hu``), each fed by separate branch pairs ``(B1(h0), B2(b))``
and its own trunk, with an analytical IC shortcut:

    h(x,t)  = elu(h0(x) + t*F_h - (b+H_MIN)) + (b+H_MIN) + EPS   (positive)
    hu(x,t) = t * F_hu                                            (zero at t=0)

ELU (not softplus) avoids float32 underflow to exact 0. The two ablation
variants remove one architectural choice each.
"""

from __future__ import annotations

from typing import Sequence

import tensorflow as tf

from ...methods.deeponet.mlp import make_mlp
from ...methods.deeponet.operator import DeepONetOperator


class SWEDeepONet(tf.keras.Model):
    """Full model: separate branch pairs + analytical IC shortcut."""

    def __init__(self, *, n_sensors: int = 100, width: int = 64,
                 hidden: Sequence[int] = (128, 128, 128), out_std: float = 0.1,
                 h_min: float = 0.05, eps: float = 1e-4, name: str = "swe_deeponet"):
        super().__init__(name=name)
        hid = list(hidden)
        self.h_op = DeepONetOperator(
            [make_mlp(n_sensors, hid, width, "b1h"),
             make_mlp(n_sensors, hid, width, "b2h")],
            make_mlp(2, hid, width, "th", out_std=out_std), name="h_op")
        self.hu_op = DeepONetOperator(
            [make_mlp(n_sensors, hid, width, "b1hu"),
             make_mlp(n_sensors, hid, width, "b2hu")],
            make_mlp(2, hid, width, "thu", out_std=out_std), name="hu_op")
        self.h_min = tf.constant(h_min, tf.float32)
        self.eps = tf.constant(eps, tf.float32)

    def call(self, h0s, bs, xt, h0_at_x, b_at_x):
        """Predict ``(h, hu)`` at query points ``xt`` given sensor and pointwise inputs."""
        t = tf.transpose(xt[:, 1:2])
        F_h = self.h_op([h0s, bs], xt)
        F_hu = self.hu_op([h0s, bs], xt)
        base = b_at_x + self.h_min
        h_pred = tf.nn.elu(h0_at_x + t * F_h - base) + base + self.eps
        hu_pred = t * F_hu
        return h_pred, hu_pred


class SharedBranchSWEDeepONet(tf.keras.Model):
    """Ablation A1: a single shared ``beta`` feeds both trunks (BC collapse)."""

    def __init__(self, *, n_sensors: int = 100, width: int = 64,
                 hidden: Sequence[int] = (128, 128, 128), out_std: float = 0.1,
                 h_min: float = 0.05, eps: float = 1e-4, name: str = "swe_shared"):
        super().__init__(name=name)
        hid = list(hidden)
        self.b1 = make_mlp(n_sensors, hid, width, "b1_shared")
        self.b2 = make_mlp(n_sensors, hid, width, "b2_shared")
        self.th = make_mlp(2, hid, width, "th_shared", out_std=out_std)
        self.thu = make_mlp(2, hid, width, "thu_shared", out_std=out_std)
        self.h_min = tf.constant(h_min, tf.float32)
        self.eps = tf.constant(eps, tf.float32)

    def call(self, h0s, bs, xt, h0_at_x, b_at_x):
        """Predict ``(h, hu)`` using a single shared branch coefficient for both fields."""
        t = tf.transpose(xt[:, 1:2])
        beta = self.b1(h0s) + self.b2(bs)
        F_h = tf.linalg.matmul(beta, self.th(xt), transpose_b=True)
        F_hu = tf.linalg.matmul(beta, self.thu(xt), transpose_b=True)
        base = b_at_x + self.h_min
        h_pred = tf.nn.elu(h0_at_x + t * F_h - base) + base + self.eps
        hu_pred = t * F_hu
        return h_pred, hu_pred


class NoICShortcutSWEDeepONet(tf.keras.Model):
    """Ablation A2: separate branch pairs but no analytical IC shortcut."""

    def __init__(self, *, n_sensors: int = 100, width: int = 64,
                 hidden: Sequence[int] = (128, 128, 128), out_std: float = 0.1,
                 name: str = "swe_noic", **_ignored):
        super().__init__(name=name)
        hid = list(hidden)
        self.h_op = DeepONetOperator(
            [make_mlp(n_sensors, hid, width, "b1h_noic"),
             make_mlp(n_sensors, hid, width, "b2h_noic")],
            make_mlp(2, hid, width, "th_noic", out_std=out_std), name="h_op_noic")
        self.hu_op = DeepONetOperator(
            [make_mlp(n_sensors, hid, width, "b1hu_noic"),
             make_mlp(n_sensors, hid, width, "b2hu_noic")],
            make_mlp(2, hid, width, "thu_noic", out_std=out_std), name="hu_op_noic")

    def call(self, h0s, bs, xt, h0_at_x, b_at_x):
        """Predict ``(h, hu)`` as raw operator outputs (no IC shortcut applied)."""
        return self.h_op([h0s, bs], xt), self.hu_op([h0s, bs], xt)


VARIANTS = {
    "full": SWEDeepONet,
    "shared_branch": SharedBranchSWEDeepONet,
    "no_ic_shortcut": NoICShortcutSWEDeepONet,
}


def warmup(model: tf.keras.Model, n_sensors: int) -> tf.keras.Model:
    """Build the model's weights with one dummy forward pass and return it."""
    d = tf.zeros((2, n_sensors))
    model(d, d, tf.zeros((4, 2)), tf.ones((2, 4)), tf.zeros((2, 4)))
    return model
