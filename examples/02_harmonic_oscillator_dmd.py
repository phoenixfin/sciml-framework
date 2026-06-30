"""Example 02: recover an undamped oscillator with Dynamic Mode Decomposition.

DMD finds the linear operator behind snapshots of the state ``[x, v]``. For
``x'' = -omega^2 x`` it should return a conjugate eigenvalue pair on the unit
circle whose angle gives ``omega``, and reconstruct the trajectory exactly.
"""

from __future__ import annotations

import os

import numpy as np

from sciml.core.plotting import set_paper_style
from sciml.methods.dmd import DMD
from sciml.solvers.dynamical import harmonic_oscillator, simulate

OUT = "outputs/examples"


def main():
    omega_true = 2.0
    dt = 0.05
    t = np.arange(0, 12, dt)
    X = simulate(harmonic_oscillator(omega_true), [1.0, 0.0], t)   # (n_t, 2)
    snapshots = X.T                                                # (2, n_t)

    dmd = DMD(rank=2).fit(snapshots, dt=dt)
    omega_dmd = float(np.max(np.abs(dmd.omega.imag)))
    print(f"DMD eigenvalues: {np.round(dmd.eigenvalues, 4)}")
    print(f"Recovered omega = {omega_dmd:.4f}   (true: {omega_true})")
    recon = dmd.reconstruct(len(t))
    err = np.linalg.norm(recon - snapshots) / np.linalg.norm(snapshots)
    print(f"Reconstruction relative error: {err:.2e}")

    set_paper_style(font_size=10)
    import matplotlib.pyplot as plt
    os.makedirs(OUT, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].plot(t, X[:, 0], "k-", lw=2, label="x(t) data")
    axes[0].plot(t, recon[0], "r--", lw=1.2, label="DMD")
    axes[0].set(xlabel="t", ylabel="x", title="Trajectory"); axes[0].legend()
    axes[1].plot(X[:, 0], X[:, 1], "k-", lw=2, label="data")
    axes[1].plot(recon[0], recon[1], "r--", lw=1.2, label="DMD")
    axes[1].set(xlabel="x", ylabel="v", title="Phase portrait"); axes[1].legend()
    fig.suptitle(f"Example 02: DMD on a harmonic oscillator (omega={omega_dmd:.2f})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "02_harmonic_dmd.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}/02_harmonic_dmd.png")


if __name__ == "__main__":
    main()
