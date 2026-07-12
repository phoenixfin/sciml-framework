"""B3: hyperparameter sensitivity of the WNTS SINDYc results.

Two sweeps (within-year 2019, causal protocol):

1. **threshold x alpha grid** (dt = 3 h, clip = 3): heatmaps of 72 h NRMSE
   and divergence for the headline SINDYc-2, plus the divergence of the
   naive 5-state SINDYc across the same grid -- does *any* tuning rescue
   the naive model?
2. **dt x clip table** (threshold/alpha at defaults) for SINDYc-2.

Usage::

    python -m experiments.wnts.sweep_hyper
"""

from __future__ import annotations

import json
import os

import numpy as np

from sciml.core.plotting import set_paper_style

from .run import (
    P_NAMES,
    Q_NAMES,
    ModelSpec,
    build_parser,
    prepare_data,
    rollout_metrics,
    seg_data,
)

THRESHOLDS = [0.01, 0.02, 0.05, 0.1]
ALPHAS = [1e-3, 1e-2, 1e-1, 1.0]
DTS = [1, 3, 6]
CLIPS = [0.0, 2.0, 3.0, 5.0]


def base_args(**overrides):
    args = build_parser().parse_args([])
    args.years = [2019]
    args.ic_stride = 24
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def eval_one(data, args, form: str, threshold: float, alpha: float, clip: float) -> dict:
    state_names = P_NAMES if form == "sindyc" else ["p_up", "p_orf"]
    spec = ModelSpec(form, form, state_names, Q_NAMES, clip=clip)
    spec.fit(
        seg_data(data["train_z"], form),
        args.fit,
        data["dt"],
        args.savgol_window,
        threshold,
        alpha,
        args.degree,
    )
    d_te = seg_data(data["test_z"], form)
    r2_te = spec.r2_pressures(d_te, args.fit, data["dt"], args.savgol_window)
    roll = rollout_metrics(
        spec,
        d_te,
        data["dt"],
        data["integrator"],
        args.ic_stride,
        args.blowup,
        data["warmup_h"],
    )
    hz = roll["horizons"]
    return {
        "r2_test": float(np.mean(r2_te)),
        "nrmse_24h": hz[24]["nrmse"],
        "nrmse_72h": hz[72]["nrmse"],
        "diverged": hz[72]["diverged_frac"],
    }


def main() -> None:
    out = "outputs/wnts_B3"
    os.makedirs(out, exist_ok=True)

    # ---- sweep 1: threshold x alpha ---------------------------------------
    args = base_args()
    data = prepare_data(args)
    red_nrmse = np.full((len(THRESHOLDS), len(ALPHAS)), np.nan)
    red_div = np.full_like(red_nrmse, np.nan)
    naive_div = np.full_like(red_nrmse, np.nan)
    grid = {}
    for i, th in enumerate(THRESHOLDS):
        for j, al in enumerate(ALPHAS):
            r = eval_one(data, args, "sindyc_red", th, al, args.clip)
            n = eval_one(data, args, "sindyc", th, al, args.clip)
            red_nrmse[i, j] = r["nrmse_72h"]
            red_div[i, j] = r["diverged"]
            naive_div[i, j] = n["diverged"]
            grid[f"th={th},al={al}"] = {"sindyc_red": r, "sindyc_5state": n}
            print(
                f"th={th:<5} al={al:<5} | red: NRMSE72 "
                f"{r['nrmse_72h']:.3f} div {r['diverged']:.0%} | "
                f"naive 5-state div {n['diverged']:.0%}"
            )
    print(
        f"\nnaive 5-state minimum divergence over grid: {np.nanmin(naive_div):.0%} "
        f"(the naive model fails everywhere)"
        if np.nanmin(naive_div) > 0.5
        else f"\nnaive 5-state minimum divergence over grid: {np.nanmin(naive_div):.0%}"
    )

    # ---- sweep 2: dt x clip ------------------------------------------------
    dtclip = {}
    for dt_h in DTS:
        args2 = base_args(dt_hours=dt_h)
        data2 = prepare_data(args2)
        for clip in CLIPS:
            r = eval_one(data2, args2, "sindyc_red", args2.threshold, args2.alpha, clip)
            dtclip[f"dt={dt_h},clip={clip}"] = r
            print(
                f"dt={dt_h}h clip={clip:<3} | NRMSE 24h {r['nrmse_24h']:.3f} "
                f"72h {r['nrmse_72h']:.3f} div {r['diverged']:.0%}"
            )

    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as fh:
        json.dump({"grid_threshold_alpha": grid, "grid_dt_clip": dtclip}, fh, indent=2)

    # ---- figures -----------------------------------------------------------
    set_paper_style(font_size=10)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    panels = [
        (red_nrmse, "SINDYc-2: 72 h NRMSE", "viridis", None),
        (red_div, "SINDYc-2: diverged frac", "Reds", (0, 1)),
        (naive_div, "naive 5-state: diverged frac", "Reds", (0, 1)),
    ]
    for ax, (M, title, cmap, lim) in zip(axes, panels):
        vmin, vmax = lim if lim else (np.nanmin(M), np.nanmax(M))
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(ALPHAS)))
        ax.set_xticklabels(ALPHAS, fontsize=7)
        ax.set_yticks(range(len(THRESHOLDS)))
        ax.set_yticklabels(THRESHOLDS, fontsize=7)
        ax.set_xlabel("ridge alpha")
        ax.set_ylabel("STRidge threshold")
        ax.set_title(title, fontsize=9)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.045)
    plt.tight_layout()
    fig.savefig(os.path.join(out, "fig_heatmaps.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved results.json and fig_heatmaps.png to {out}")


if __name__ == "__main__":
    main()
