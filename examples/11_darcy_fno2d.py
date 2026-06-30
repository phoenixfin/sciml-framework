"""Example 11 (PDE, 2D): learn the Darcy-flow operator with a 2D FNO.

The canonical 2D operator-learning benchmark: map a permeability field
``a(x,y)`` to the pressure ``u(x,y)`` solving ``-div(a grad u) = 1`` with zero
boundary conditions. Requires TensorFlow.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from sciml.core.metrics import rel_l2
from sciml.core.plotting import set_paper_style
from sciml.solvers.darcy import darcy_dataset, sample_permeability

OUT = "outputs/examples"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--grid", type=int, default=29)
    ap.add_argument("--epochs", type=int, default=80)
    args = ap.parse_args()

    import tensorflow as tf
    from tensorflow import keras
    from sciml.methods.fno import build_fno2d

    m = args.grid
    rng = np.random.default_rng(0)
    print("Generating Darcy dataset (FD solves)...")
    a = sample_permeability(m, args.n, rng, length_scale=0.12)
    u = darcy_dataset(a, f_const=1.0)

    xy = np.linspace(0, 1, m, dtype=np.float32)
    XX, YY = np.meshgrid(xy, xy, indexing="ij")
    coords = np.broadcast_to(np.stack([XX, YY], -1), (args.n, m, m, 2))
    X = np.concatenate([a[..., None], coords], axis=-1).astype(np.float32)  # (n,m,m,3)
    Y = u[..., None].astype(np.float32)

    ntr = int(0.8 * args.n)
    model = build_fno2d(grid=m, modes=12, width=32, n_layers=4,
                        in_channels=3, out_channels=1)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    model.fit(X[:ntr], Y[:ntr], epochs=args.epochs, batch_size=16, verbose=2)

    pred = model.predict(X[ntr:], verbose=0)
    err = np.mean([rel_l2(pred[i, ..., 0], Y[ntr + i, ..., 0]) for i in range(len(pred))])
    print(f"FNO2d Darcy operator: mean relative L2 = {err:.3e}")

    set_paper_style(font_size=10)
    import matplotlib.pyplot as plt
    os.makedirs(OUT, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    im0 = axes[0].imshow(a[ntr], origin="lower", cmap="viridis")
    axes[0].set_title("permeability a(x,y)"); plt.colorbar(im0, ax=axes[0], shrink=0.8)
    im1 = axes[1].imshow(Y[ntr, ..., 0], origin="lower", cmap="magma")
    axes[1].set_title("pressure u (true)"); plt.colorbar(im1, ax=axes[1], shrink=0.8)
    im2 = axes[2].imshow(pred[0, ..., 0], origin="lower", cmap="magma")
    axes[2].set_title(f"FNO2d (rel L2={err:.1e})"); plt.colorbar(im2, ax=axes[2], shrink=0.8)
    fig.suptitle("Example 11: 2D FNO learns the Darcy-flow operator")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "11_darcy_fno2d.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}/11_darcy_fno2d.png")


if __name__ == "__main__":
    main()
