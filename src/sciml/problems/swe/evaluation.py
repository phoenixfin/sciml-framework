"""SWE-specific evaluation/plotting helpers."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from ...core.metrics import rel_l2


def case_errors(xs_p: np.ndarray, h_p: np.ndarray, hu_p: np.ndarray, ts_p: np.ndarray,
                x_ref: np.ndarray, snaps: Dict, t_eval: float = 1.0) -> Tuple[float, float]:
    """Relative L2 errors of ``h`` and ``hu`` at time ``t_eval`` against the reference.

    Parameters
    ----------
    xs_p : np.ndarray
        Prediction spatial grid.
    h_p : np.ndarray
        Predicted depth field (time x space).
    hu_p : np.ndarray
        Predicted momentum field (time x space).
    ts_p : np.ndarray
        Prediction time grid.
    x_ref : np.ndarray
        Reference spatial grid.
    snaps : Dict
        Reference snapshots with keys ``t``, ``h`` and ``hu``.
    t_eval : float
        Evaluation time at which to compute the errors.

    Returns
    -------
    Tuple[float, float]
        The relative L2 errors of ``h`` and ``hu`` at ``t_eval``.
    """
    t_arr = np.array(snaps["t"])
    ri = int(np.argmin(np.abs(t_arr - t_eval)))
    pi = int(np.argmin(np.abs(ts_p - t_eval)))
    return (rel_l2(h_p[pi], np.interp(xs_p, x_ref, snaps["h"][ri])),
            rel_l2(hu_p[pi], np.interp(xs_p, x_ref, snaps["hu"][ri])))


def plot_case_with_error(xs_p: np.ndarray, ts_p: np.ndarray, h_p: np.ndarray,
                         hu_p: np.ndarray, x_ref: np.ndarray, snaps: Dict, title: str,
                         fname: str, color: str = "b") -> Tuple[float, float]:
    """2x3 panel (h, hu at t=0.5/1.0 + pointwise error). Returns (eps_h, eps_hu) at t=1.

    Parameters
    ----------
    xs_p : np.ndarray
        Prediction spatial grid.
    ts_p : np.ndarray
        Prediction time grid.
    h_p : np.ndarray
        Predicted depth field (time x space).
    hu_p : np.ndarray
        Predicted momentum field (time x space).
    x_ref : np.ndarray
        Reference spatial grid.
    snaps : Dict
        Reference snapshots with keys ``t``, ``h`` and ``hu``.
    title : str
        Figure super-title.
    fname : str
        Output path for the saved figure.
    color : str
        Matplotlib color code for the prediction curves.

    Returns
    -------
    Tuple[float, float]
        The relative L2 errors of ``h`` and ``hu`` at ``t=1``.
    """
    import matplotlib.pyplot as plt
    t_arr = np.array(snaps["t"])
    i05, i10 = int(np.argmin(np.abs(t_arr - 0.5))), int(np.argmin(np.abs(t_arr - 1.0)))
    p05, p10 = int(np.argmin(np.abs(ts_p - 0.5))), int(np.argmin(np.abs(ts_p - 1.0)))
    hr05 = np.interp(xs_p, x_ref, snaps["h"][i05]); hr10 = np.interp(xs_p, x_ref, snaps["h"][i10])
    qr05 = np.interp(xs_p, x_ref, snaps["hu"][i05]); qr10 = np.interp(xs_p, x_ref, snaps["hu"][i10])

    fig, axes = plt.subplots(2, 3, figsize=(9, 4), gridspec_kw={"width_ratios": [2, 2, 1.8]})
    for ci, (ri, pi, tv) in enumerate([(i05, p05, 0.5), (i10, p10, 1.0)]):
        axes[0, ci].plot(x_ref, snaps["h"][ri], "k-", lw=1.4, label="ref")
        axes[0, ci].plot(xs_p, h_p[pi], color + "--", lw=1.1, label="DeepONet")
        axes[0, ci].set_title(f"$t={tv}$ s", fontsize=8)
        if ci == 0:
            axes[0, ci].set_ylabel("$h$ [m]"); axes[0, ci].legend(fontsize=7)
        axes[0, ci].set_xticklabels([])
        axes[1, ci].plot(x_ref, snaps["hu"][ri], "k-", lw=1.4)
        axes[1, ci].plot(xs_p, hu_p[pi], color + "--", lw=1.1)
        axes[1, ci].set_xlabel("$x$ [m]")
        if ci == 0:
            axes[1, ci].set_ylabel("$hu$ [m$^2$/s]")
    axes[0, 2].semilogy(xs_p, np.abs(h_p[p05] - hr05), "b-", lw=1.0, label="$t=0.5$")
    axes[0, 2].semilogy(xs_p, np.abs(h_p[p10] - hr10), "b--", lw=1.0, label="$t=1.0$")
    axes[0, 2].set_ylabel(r"$|\hat h - h|$"); axes[0, 2].set_title("Pointwise error", fontsize=8)
    axes[0, 2].legend(fontsize=7); axes[0, 2].set_xticklabels([])
    axes[1, 2].semilogy(xs_p, np.abs(hu_p[p05] - qr05), "r-", lw=1.0)
    axes[1, 2].semilogy(xs_p, np.abs(hu_p[p10] - qr10), "r--", lw=1.0)
    axes[1, 2].set_ylabel(r"$|\widehat{hu} - hu|$"); axes[1, 2].set_xlabel("$x$ [m]")
    fig.suptitle(title, fontsize=9)
    plt.tight_layout(); plt.savefig(fname, dpi=200, bbox_inches="tight"); plt.close(fig)
    return rel_l2(h_p[p10], hr10), rel_l2(hu_p[p10], qr10)
