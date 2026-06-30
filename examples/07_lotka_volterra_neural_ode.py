"""Example 07: learn predator-prey dynamics from a trajectory with a Neural ODE.

Unlike SINDy (which returns symbolic equations from a fixed library), a Neural
ODE learns a *black-box* ``dy/dt = f_theta(y)`` that reproduces the trajectory.
Requires TensorFlow (`pip install -e ".[all]"`).
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from sciml.core.plotting import set_paper_style
from sciml.solvers.dynamical import lotka_volterra, simulate

OUT = "outputs/examples"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-2)
    args = ap.parse_args()

    import tensorflow as tf
    from sciml.methods.neuralode import NeuralODE, build_odefunc

    # One trajectory, lightly subsampled to keep integration cheap.
    t = np.linspace(0, 15, 150, dtype=np.float32)
    X = simulate(lotka_volterra(), [10.0, 5.0], t).astype(np.float32)   # (n_t, 2)
    # Normalize for stable training; learn dynamics in the scaled space.
    scale = X.std(0)
    Xs = X / scale
    y0 = tf.constant(Xs[0][None, :])                                    # (1, 2)
    target = Xs[:, None, :]                                             # (n_t, 1, 2)

    node = NeuralODE(build_odefunc(2, hidden=(64, 64)))
    print("Training Neural ODE on the trajectory...")
    hist = node.fit_trajectory(y0, t, target, steps=args.steps, lr=args.lr, log_every=100)
    print(f"Final trajectory MSE: {hist[-1]:.3e}")

    pred = node(y0, t).numpy()[:, 0, :] * scale                        # (n_t, 2)

    set_paper_style(font_size=10)
    import matplotlib.pyplot as plt
    os.makedirs(OUT, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].plot(t, X[:, 0], "k-", lw=2, label="prey (true)")
    axes[0].plot(t, pred[:, 0], "r--", lw=1.3, label="prey (NODE)")
    axes[0].plot(t, X[:, 1], "b-", lw=2, label="predator (true)")
    axes[0].plot(t, pred[:, 1], "m--", lw=1.3, label="predator (NODE)")
    axes[0].set(xlabel="t", title="Time series"); axes[0].legend(fontsize=7)
    axes[1].plot(X[:, 0], X[:, 1], "k-", lw=2, label="true")
    axes[1].plot(pred[:, 0], pred[:, 1], "r--", lw=1.3, label="NODE")
    axes[1].set(xlabel="prey", ylabel="predator", title="Phase portrait"); axes[1].legend()
    fig.suptitle("Example 07: Neural ODE learns Lotka-Volterra")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "07_neural_ode.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}/07_neural_ode.png")


if __name__ == "__main__":
    main()
