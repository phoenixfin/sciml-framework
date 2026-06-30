"""Example 03: identify the nonlinear predator-prey (Lotka-Volterra) ODE.

Now the dynamics are nonlinear: SINDy must pick the quadratic interaction terms
``x*y`` out of a degree-2 candidate library while rejecting the rest.
"""

from __future__ import annotations

import os

import numpy as np

from sciml.core.plotting import set_paper_style
from sciml.methods.sindy import PolynomialLibrary, SINDy
from sciml.solvers.dynamical import lotka_volterra, simulate

OUT = "outputs/examples"


def main():
    t = np.arange(0, 30, 0.01)
    X = simulate(lotka_volterra(), [10.0, 5.0], t)         # (n_t, 2)

    model = SINDy(PolynomialLibrary(degree=2), threshold=0.05).fit(
        X, t=t, input_names=["x", "y"])
    print("Identified system   (true: x'=x-0.1xy,  y'=-1.5y+0.075xy):")
    for eq in model.equations(["x'", "y'"]):
        print("  ", eq)

    set_paper_style(font_size=10)
    import matplotlib.pyplot as plt
    os.makedirs(OUT, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].plot(t, X[:, 0], label="prey x")
    axes[0].plot(t, X[:, 1], label="predator y")
    axes[0].set(xlabel="t", ylabel="population", title="Time series"); axes[0].legend()
    axes[1].plot(X[:, 0], X[:, 1], "k-", lw=1)
    axes[1].set(xlabel="prey x", ylabel="predator y", title="Phase portrait (limit cycle)")
    fig.suptitle("Example 03: SINDy on Lotka-Volterra")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "03_lotka_volterra.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}/03_lotka_volterra.png")


if __name__ == "__main__":
    main()
