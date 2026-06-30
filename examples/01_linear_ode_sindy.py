"""Example 01 (simplest): identify x' = -k x from data with SINDy.

The "hello world" of system identification: simulate exponential decay, then
recover the governing equation (and the rate k) from the trajectory alone.
"""

from __future__ import annotations

import os

import numpy as np

from sciml.core.plotting import set_paper_style
from sciml.methods.sindy import PolynomialLibrary, SINDy
from sciml.solvers.dynamical import linear_decay, simulate

OUT = "outputs/examples"


def main():
    k_true = 0.7
    t = np.arange(0, 8, 0.05)
    X = simulate(linear_decay(k_true), [2.0], t)            # (n_t, 1)

    model = SINDy(PolynomialLibrary(degree=1), threshold=0.05).fit(
        X, t=t, input_names=["x"])
    print("Identified:", model.equations(["dx/dt"])[0], f"   (true: dx/dt = -{k_true} x)")

    set_paper_style(font_size=10)
    import matplotlib.pyplot as plt
    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(t, X[:, 0], "k-", lw=2, label="data")
    ax.plot(t, 2.0 * np.exp(-k_true * t), "r--", lw=1.2, label="$2e^{-kt}$")
    ax.set(xlabel="t", ylabel="x(t)", title="Example 01: SINDy on $x'=-kx$")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "01_linear_ode.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}/01_linear_ode.png")


if __name__ == "__main__":
    main()
