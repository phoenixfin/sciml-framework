"""Example 09 (PDE): solve the 1D wave equation with a PINN.

A from-scratch PINN for ``u_tt = c^2 u_xx`` on a periodic domain with a smooth
initial pulse and zero initial velocity. Compared against the exact d'Alembert
solution. Requires TensorFlow. (Contrast with the heavier moving-boundary wave
in ``problems/wave_obstacle``.)
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from sciml.core.metrics import rel_l2
from sciml.core.plotting import set_paper_style
from sciml.solvers.wave1d import wave1d_dalembert

OUT = "outputs/examples"
L, C, T = 1.0, 1.0, 0.5


def f0_np(x):
    return np.exp(-((np.mod(x, L) - 0.5) ** 2) / (2 * 0.06 ** 2)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    args = ap.parse_args()

    import tensorflow as tf
    from sciml.methods.pinn import build_mlp, PINNTrainer
    from sciml.methods.pinn.gradients import derivatives_2d

    net = build_mlp(2, hidden=4, width=64, out_dim=1, fourier_freq=16,
                    fourier_sigma=2.0, name="u")

    def f0_tf(x):
        return tf.exp(-((tf.math.floormod(x, L) - 0.5) ** 2) / (2 * 0.06 ** 2))

    def loss_fn():
        # PDE residual u_tt - c^2 u_xx = 0
        xt = tf.concat([tf.random.uniform([2000, 1], 0.0, L),
                        tf.random.uniform([2000, 1], 0.0, T)], 1)
        d = derivatives_2d(net, xt)
        L_pde = tf.reduce_mean((d["u_tt"] - C**2 * d["u_xx"]) ** 2)
        # IC: u(x,0)=f0,  u_t(x,0)=0
        x_ic = tf.random.uniform([400, 1], 0.0, L)
        xt0 = tf.concat([x_ic, tf.zeros_like(x_ic)], 1)
        d0 = derivatives_2d(net, xt0)
        L_icu = tf.reduce_mean((d0["u"] - f0_tf(x_ic)) ** 2)
        L_icv = tf.reduce_mean(d0["u_t"] ** 2)
        # Periodic BC u(0,t)=u(L,t)
        tb = tf.random.uniform([400, 1], 0.0, T)
        uL = net(tf.concat([tf.zeros_like(tb), tb], 1))
        uR = net(tf.concat([tf.fill(tf.shape(tb), L), tb], 1))
        L_bc = tf.reduce_mean((uL - uR) ** 2)
        return L_pde + 20.0 * L_icu + 5.0 * L_icv + 10.0 * L_bc

    trainer = PINNTrainer(net.trainable_variables, loss_fn)
    print("Training 1D-wave PINN...")
    trainer.run_adam(args.steps, 1e-3, print_every=max(args.steps // 8, 1))

    # Evaluate vs d'Alembert at t = T.
    xs = np.linspace(0, L, 200, dtype=np.float32)
    u_ref = wave1d_dalembert(f0_np, xs, T, C, L)
    u_pinn = net(np.stack([xs, np.full_like(xs, T)], 1)).numpy().ravel()
    err = rel_l2(u_pinn, u_ref)
    print(f"1D-wave PINN: relative L2 vs d'Alembert at t={T} = {err:.3e}")

    set_paper_style(font_size=10)
    import matplotlib.pyplot as plt
    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(xs, f0_np(xs), "k:", lw=1.2, label="u(.,0)")
    ax.plot(xs, u_ref, "b-", lw=2, label="d'Alembert u(.,T)")
    ax.plot(xs, u_pinn, "r--", lw=1.4, label="PINN")
    ax.set(xlabel="x", title=f"Example 09: 1D-wave PINN (rel L2={err:.1e})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "09_wave1d_pinn.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}/09_wave1d_pinn.png")


if __name__ == "__main__":
    main()
