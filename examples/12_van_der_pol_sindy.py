"""Example 12 (ODE): identify the Van der Pol oscillator with SINDy.

A nonlinear limit-cycle oscillator whose dynamics contain a cubic ``x^2 y``
term, so SINDy needs a degree-3 candidate library.
"""

from __future__ import annotations

import os

import numpy as np

from sciml.core.plotting import set_paper_style
from sciml.methods.sindy import PolynomialLibrary, SINDy
from sciml.solvers.dynamical import simulate, van_der_pol

OUT = "outputs/examples"


def main():
    mu = 1.5
    t = np.arange(0, 30, 0.01)
    X = simulate(van_der_pol(mu), [2.0, 0.0], t)

    model = SINDy(PolynomialLibrary(degree=3), threshold=0.05).fit(
        X, t=t, input_names=["x", "y"])
    print(f"Identified (true: x'=y,  y'={mu}y - x - {mu}x^2 y):")
    for eq in model.equations(["x'", "y'"]):
        print("  ", eq)

    set_paper_style(font_size=10)
    import matplotlib.pyplot as plt
    os.makedirs(OUT, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].plot(t, X[:, 0], lw=1)
    axes[0].set(xlabel="t", ylabel="x", title="Time series")
    axes[1].plot(X[:, 0], X[:, 1], "k-", lw=1)
    axes[1].set(xlabel="x", ylabel="y", title="Limit cycle")
    fig.suptitle("Example 12: SINDy on the Van der Pol oscillator")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "12_van_der_pol.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}/12_van_der_pol.png")


if __name__ == "__main__":
    main()
