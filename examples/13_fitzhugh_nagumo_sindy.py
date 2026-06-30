"""Example 13 (ODE): identify the FitzHugh-Nagumo neuron model with SINDy.

A two-variable excitable system with a cubic nonlinearity (``v^3``) and slow
recovery variable -- a step up in stiffness/time-scale separation.
"""

from __future__ import annotations

import os

import numpy as np

from sciml.core.plotting import set_paper_style
from sciml.methods.sindy import PolynomialLibrary, SINDy
from sciml.solvers.dynamical import fitzhugh_nagumo, simulate

OUT = "outputs/examples"


def main():
    t = np.arange(0, 200, 0.05)
    X = simulate(fitzhugh_nagumo(), [-1.0, 1.0], t)

    model = SINDy(PolynomialLibrary(degree=3), threshold=0.02).fit(
        X, t=t, input_names=["v", "w"])
    print("Identified (true: v'=v - v^3/3 - w + 0.5,  w'=0.08(v + 0.7 - 0.8w)):")
    for eq in model.equations(["v'", "w'"]):
        print("  ", eq)

    set_paper_style(font_size=10)
    import matplotlib.pyplot as plt
    os.makedirs(OUT, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].plot(t, X[:, 0], label="v (membrane)")
    axes[0].plot(t, X[:, 1], label="w (recovery)")
    axes[0].set(xlabel="t", title="Spiking dynamics"); axes[0].legend(fontsize=8)
    axes[1].plot(X[:, 0], X[:, 1], "k-", lw=1)
    axes[1].set(xlabel="v", ylabel="w", title="Phase portrait")
    fig.suptitle("Example 13: SINDy on FitzHugh-Nagumo")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "13_fitzhugh_nagumo.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}/13_fitzhugh_nagumo.png")


if __name__ == "__main__":
    main()
