"""Generic DeepONet operators.

A DeepONet approximates ``G(a)(y) ~= sum_k beta_k(a) * phi_k(y) (+ bias)`` where
the branch net(s) map sensor values -> coefficients ``beta`` and the trunk net
maps query coordinates -> basis ``phi``. :class:`DeepONetOperator` supports
several summed branch nets (to feed multiple input functions into one operator).
"""

from __future__ import annotations

from typing import List, Sequence

import tensorflow as tf

from .mlp import make_mlp


class DeepONetOperator(tf.keras.Model):
    """Inner-product DeepONet with one or more (summed) branch networks."""

    def __init__(self, branch_nets: Sequence[tf.keras.Model],
                 trunk_net: tf.keras.Model, use_bias: bool = False,
                 name: str = "deeponet_operator"):
        super().__init__(name=name)
        self.branch_nets: List[tf.keras.Model] = list(branch_nets)
        self.trunk_net = trunk_net
        self.use_bias = use_bias
        if use_bias:
            self.bias = self.add_weight(name="bias", shape=(), initializer="zeros")

    def coefficients(self, branch_inputs: Sequence[tf.Tensor]) -> tf.Tensor:
        """Summed branch coefficients ``beta`` ``(B, P)``."""
        if len(branch_inputs) != len(self.branch_nets):
            raise ValueError(f"Expected {len(self.branch_nets)} branch inputs, "
                             f"got {len(branch_inputs)}")
        beta = self.branch_nets[0](branch_inputs[0])
        for net, inp in zip(self.branch_nets[1:], branch_inputs[1:]):
            beta = beta + net(inp)
        return beta

    def call(self, branch_inputs: Sequence[tf.Tensor], coords: tf.Tensor) -> tf.Tensor:
        """``branch_inputs``: list of ``(B, m_i)``; ``coords``: ``(N, d)`` -> ``(B, N)``."""
        beta = self.coefficients(branch_inputs)               # (B, P)
        basis = self.trunk_net(coords)                        # (N, P)
        out = tf.linalg.matmul(beta, basis, transpose_b=True)  # (B, N)
        if self.use_bias:
            out = out + self.bias
        return out

    @classmethod
    def build(cls, *, n_sensors: int, n_branches: int, coord_dim: int,
              width: int, hidden: Sequence[int], out_std: float = 0.1,
              use_bias: bool = False, name: str = "deeponet_operator"
              ) -> "DeepONetOperator":
        branches = [make_mlp(n_sensors, hidden, width, f"{name}_branch{i}", out_std=out_std)
                    for i in range(n_branches)]
        trunk = make_mlp(coord_dim, hidden, width, f"{name}_trunk", out_std=out_std)
        return cls(branches, trunk, use_bias=use_bias, name=name)


class DeepONet(DeepONetOperator):
    """Standard single-input-function DeepONet (one branch, one trunk)."""

    @classmethod
    def build(cls, *, n_sensors: int, coord_dim: int, width: int,
              hidden: Sequence[int], out_std: float = 0.1, use_bias: bool = True,
              name: str = "deeponet") -> "DeepONet":
        branch = make_mlp(n_sensors, hidden, width, f"{name}_branch", out_std=out_std)
        trunk = make_mlp(coord_dim, hidden, width, f"{name}_trunk", out_std=out_std)
        return cls([branch], trunk, use_bias=use_bias, name=name)

    def call(self, branch_input, coords):  # type: ignore[override]
        if isinstance(branch_input, (list, tuple)):
            return super().call(branch_input, coords)
        return super().call([branch_input], coords)
