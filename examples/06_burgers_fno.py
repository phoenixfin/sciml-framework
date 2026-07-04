"""Example 06: learn the Burgers' equation operator with an FNO.

Harder than the heat operator: viscous Burgers develops steep gradients
(near-shocks), so the operator ``u(.,0) -> u(.,T)`` is genuinely nonlinear.
Requires TensorFlow (`pip install -e ".[all]"`).
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from sciml.core.metrics import rel_l2
from sciml.core.plotting import set_paper_style
from sciml.data.gp import PeriodicGPSampler
from sciml.methods.fno import build_fno1d
from sciml.solvers.burgers import burgers_dataset

OUT = "outputs/examples"


def make_dataset(n, grid, nu, t_final, nt, length=1.0, seed=0):
    rng = np.random.default_rng(seed)
    xs = np.linspace(0, length, grid, endpoint=False, dtype=np.float32)
    sampler = PeriodicGPSampler(period=length, length_scale=0.15, amplitude=1.0, mean=0.0)
    u0 = sampler.sample(xs, n, rng=rng)
    uT = burgers_dataset(u0, nu, t_final, length, nt)
    X = np.stack([u0, np.broadcast_to(xs, u0.shape)], axis=-1).astype(np.float32)
    Y = uT[..., None].astype(np.float32)
    return xs, X, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--nu", type=float, default=0.02)
    ap.add_argument("--t-final", type=float, default=0.3)
    ap.add_argument("--nt", type=int, default=1000)
    args = ap.parse_args()

    from tensorflow import keras

    print("Generating Burgers dataset (this runs the spectral solver)...")
    xs, X, Y = make_dataset(args.n, args.grid, args.nu, args.t_final, args.nt)
    ntr = int(0.8 * len(X))
    Xtr, Ytr, Xte, Yte = X[:ntr], Y[:ntr], X[ntr:], Y[ntr:]

    model = build_fno1d(modes=20, width=48, n_layers=4, in_channels=2, out_channels=1)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    model.fit(Xtr, Ytr, epochs=args.epochs, batch_size=16, verbose=2)

    pred = model.predict(Xte, verbose=0)
    err = np.mean([rel_l2(pred[i, :, 0], Yte[i, :, 0]) for i in range(len(Xte))])
    print(f"FNO Burgers operator: mean relative L2 error = {err:.3e}")

    set_paper_style(font_size=10)
    import matplotlib.pyplot as plt
    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(xs, Xte[0, :, 0], "k:", lw=1.2, label="u(.,0)")
    ax.plot(xs, Yte[0, :, 0], "b-", lw=2, label="u(.,T) true")
    ax.plot(xs, pred[0, :, 0], "r--", lw=1.4, label="FNO")
    ax.set(xlabel="x", title=f"Example 06: FNO Burgers operator (rel L2={err:.1e})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "06_burgers_fno.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}/06_burgers_fno.png")


if __name__ == "__main__":
    main()
