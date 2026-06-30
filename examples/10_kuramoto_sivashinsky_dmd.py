"""Example 10 (PDE): Kuramoto-Sivashinsky + Dynamic Mode Decomposition.

KS is a spatiotemporally chaotic PDE. DMD fits a best linear (Koopman)
approximation to the snapshot data: it cannot fully reconstruct chaos, but it
extracts the dominant coherent spatial modes and their growth/frequency
spectrum. Pure numpy -- runs without a deep-learning backend.
"""

from __future__ import annotations

import os

import numpy as np

from sciml.core.plotting import set_paper_style
from sciml.methods.dmd import DMD
from sciml.solvers.kuramoto_sivashinsky import kuramoto_sivashinsky

OUT = "outputs/examples"


def main():
    data = kuramoto_sivashinsky(n=128, length=22.0, t_final=150.0, dt=0.25, n_save=300)
    x, t, U = data["x"], data["t"], data["u"]            # U: (n_time, n_space)

    # Discard the initial transient, then fit DMD on the (space x time) window.
    burn = 60
    snapshots = U[burn:].T                               # (n_space, n_time)
    dt_save = float(t[1] - t[0])
    r = 30
    dmd = DMD(rank=r).fit(snapshots, dt=dt_save)
    recon = dmd.reconstruct(snapshots.shape[1])
    err = np.linalg.norm(recon - snapshots) / np.linalg.norm(snapshots)
    print(f"KS: {U.shape[0]} snapshots on {U.shape[1]} grid points")
    print(f"DMD rank {r}: {np.sum(np.abs(dmd.eigenvalues) > 0.99)} near-neutral modes, "
          f"reconstruction rel. error = {err:.2f} (chaos is not linearly reducible)")

    set_paper_style(font_size=10)
    import matplotlib.pyplot as plt
    os.makedirs(OUT, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    axes[0].imshow(U.T, aspect="auto", origin="lower",
                   extent=[t[0], t[-1], x[0], x[-1]], cmap="twilight")
    axes[0].set(xlabel="t", ylabel="x", title="(a) KS spacetime field")

    th = np.linspace(0, 2 * np.pi, 200)
    axes[1].plot(np.cos(th), np.sin(th), "k:", lw=0.8)
    axes[1].scatter(dmd.eigenvalues.real, dmd.eigenvalues.imag, s=18, c="crimson")
    axes[1].set(xlabel="Re", ylabel="Im", title="(b) DMD eigenvalues")
    axes[1].set_aspect("equal")

    order = np.argsort(-np.abs(dmd.amplitudes))
    for idx in order[:3]:
        axes[2].plot(x, dmd.modes[:, idx].real, lw=1.2)
    axes[2].set(xlabel="x", title="(c) Leading DMD modes")
    fig.suptitle("Example 10: DMD modal analysis of Kuramoto-Sivashinsky")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "10_kuramoto_sivashinsky.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}/10_kuramoto_sivashinsky.png")


if __name__ == "__main__":
    main()
