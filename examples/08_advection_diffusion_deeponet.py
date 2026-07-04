"""Example 08 (PDE): learn the advection-diffusion operator with a DeepONet.

Operator learning with a DeepONet: map an initial condition ``u(.,0)`` to the
solution ``u(.,T)`` of periodic advection-diffusion. Requires TensorFlow.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from sciml.core.metrics import rel_l2
from sciml.core.plotting import set_paper_style
from sciml.data.gp import PeriodicGPSampler
from sciml.solvers.transport import advection_diffusion_dataset

OUT = "outputs/examples"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--c", type=float, default=1.0)
    ap.add_argument("--nu", type=float, default=0.005)
    ap.add_argument("--t-final", type=float, default=0.2)
    args = ap.parse_args()

    import tensorflow as tf

    from sciml.methods.deeponet import DeepONet

    rng = np.random.default_rng(0)
    xs = np.linspace(0, 1, args.grid, endpoint=False, dtype=np.float32)
    sampler = PeriodicGPSampler(period=1.0, length_scale=0.12, amplitude=1.0, mean=0.0)
    u0 = sampler.sample(xs, args.n, rng=rng)
    uT = advection_diffusion_dataset(u0, args.c, args.nu, args.t_final)

    ntr = int(0.8 * args.n)
    coords = tf.constant(xs[:, None])                      # (grid, 1)
    U0tr, Ytr = tf.constant(u0[:ntr]), tf.constant(uT[:ntr])
    U0te, Yte = u0[ntr:], uT[ntr:]

    model = DeepONet.create(n_sensors=args.grid, coord_dim=1, width=64,
                           hidden=[128, 128, 128])
    opt = tf.keras.optimizers.Adam(1e-3)

    @tf.function
    def step(u0b, yb):
        with tf.GradientTape() as tape:
            pred = model(u0b, coords)
            loss = tf.reduce_mean((pred - yb) ** 2)
        opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables),
                                model.trainable_variables))
        return loss

    bs = 32
    for ep in range(1, args.epochs + 1):
        idx = rng.permutation(ntr)
        losses = [float(step(tf.gather(U0tr, idx[i:i + bs]), tf.gather(Ytr, idx[i:i + bs])))
                  for i in range(0, ntr, bs)]
        if ep % 10 == 0 or ep == 1:
            print(f"  epoch {ep:3d}/{args.epochs}  loss={np.mean(losses):.3e}")

    pred = model(tf.constant(U0te), coords).numpy()
    err = np.mean([rel_l2(pred[i], Yte[i]) for i in range(len(Yte))])
    print(f"DeepONet advection-diffusion operator: mean relative L2 = {err:.3e}")

    set_paper_style(font_size=10)
    import matplotlib.pyplot as plt
    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(xs, U0te[0], "k:", lw=1.2, label="u(.,0)")
    ax.plot(xs, Yte[0], "b-", lw=2, label="u(.,T) true")
    ax.plot(xs, pred[0], "r--", lw=1.4, label="DeepONet")
    ax.set(xlabel="x", title=f"Example 08: DeepONet advection-diffusion (rel L2={err:.1e})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "08_advection_diffusion.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}/08_advection_diffusion.png")


if __name__ == "__main__":
    main()
