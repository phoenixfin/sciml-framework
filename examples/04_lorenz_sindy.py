"""Example 04: identify the chaotic Lorenz system with SINDy.

The classic SINDy showcase. From a single chaotic trajectory, recover all three
governing equations -- including the quadratic couplings ``x*z`` and ``x*y`` --
then re-simulate the identified model and overlay it on the true attractor.
"""

from __future__ import annotations

import os

import numpy as np

from sciml.core.plotting import set_paper_style
from sciml.methods.sindy import PolynomialLibrary, SINDy
from sciml.solvers.dynamical import lorenz, simulate

OUT = "outputs/examples"


def main():
    t = np.arange(0, 12, 0.002)
    X = simulate(lorenz(), [-8.0, 8.0, 27.0], t)           # (n_t, 3)

    model = SINDy(PolynomialLibrary(degree=2), threshold=0.1).fit(
        X, t=t, input_names=["x", "y", "z"])
    print("Identified Lorenz system   (true: x'=10(y-x), y'=28x-y-xz, z'=xy-2.667z):")
    for eq in model.equations(["x'", "y'", "z'"]):
        print("  ", eq)

    # Re-simulate the identified model from the same IC.
    rhs = lambda tt, y: model.predict(y[None, :])[0]
    X_id = simulate(rhs, X[0], t)

    set_paper_style(font_size=10)
    import matplotlib.pyplot as plt
    os.makedirs(OUT, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].plot(X[:, 0], X[:, 2], "k-", lw=0.4, alpha=0.8)
    axes[0].set(xlabel="x", ylabel="z", title="True attractor")
    axes[1].plot(X_id[:, 0], X_id[:, 2], "r-", lw=0.4, alpha=0.8)
    axes[1].set(xlabel="x", ylabel="z", title="Identified model")
    fig.suptitle("Example 04: SINDy recovers the Lorenz system")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "04_lorenz.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}/04_lorenz.png")


if __name__ == "__main__":
    main()
