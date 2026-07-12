"""A3: state-space ablation -- where between 5 and 1 states does
rollout stability appear, and why?

All variants are SINDYc (flows as inputs); only the pressure state
definition changes:

- ``P5``: the 5 raw node pressures (the naive choice).
- ``P3``: first two principal components of the 4 source-pressure
  fluctuations + the ORF pressure.
- ``P2``: upstream pool mean + ORF pressure (the physical reduction).
- ``P1``: ORF pressure only.

For each variant we report the one-step-fit R^2, rollout divergence and
NRMSE, and the **spectral radius of the identified linear one-step map**
``I + dt A`` (A = linear state-block of the coefficients): the
quantitative mechanism behind "collinearity makes the full-state model
explode".

Usage::

    python -m experiments.wnts.ablation_states --years 2019
"""

from __future__ import annotations

import json
import os
from typing import List

import numpy as np

from sciml.core.plotting import set_paper_style

from .data import Scaler
from .run import (
    Q_NAMES,
    ModelSpec,
    SegData,
    baseline_metrics,
    build_parser,
    prepare_data,
    rollout_metrics,
)


def spectral_radius(spec: ModelSpec, dt: float) -> float:
    """Max |eigenvalue| of the identified linear one-step map I + dt*A."""
    names = spec.model.feature_names_
    coef = spec.model.coef_
    d = len(spec.state_names)
    A = np.zeros((d, d))
    for j in range(d):  # equation for state j
        for i, nm in enumerate(names):
            if nm in spec.state_names:
                A[j, spec.state_names.index(nm)] = coef[i, j]
    return float(np.max(np.abs(np.linalg.eigvals(np.eye(d) + dt * A))))


def main() -> None:
    ap = build_parser()
    ap.set_defaults(out="outputs/wnts_A3")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    data = prepare_data(args)
    dt = data["dt"]

    # ---- extra state constructions from the raw arrays -------------------
    # P3: PCA of the 4 source-pressure fluctuations (train only) + p_orf.
    S = np.vstack([r["P"][:, :4] - r["CP"][:, :4] for r in data["train_raw"]])
    _, _, Vt = np.linalg.svd(S - S.mean(axis=0), full_matrices=False)
    W = Vt[:2].T  # (4, 2)
    evr = np.linalg.svd(S - S.mean(0), compute_uv=False) ** 2
    print(f"source-pressure PCA explained variance: {evr[:3] / evr.sum()}")

    def p3_arrays(raws: List[dict]) -> List[dict]:
        out = []
        for r in raws:
            fl = np.column_stack(
                [(r["P"][:, :4] - r["CP"][:, :4]) @ W, r["P"][:, 4] - r["CP"][:, 4]]
            )
            ab = np.column_stack([r["P"][:, :4] @ W, r["P"][:, 4]])
            out.append({"fl": fl, "ab": ab})
        return out

    def p1_arrays(raws: List[dict]) -> List[dict]:
        return [
            {"fl": (r["P"][:, 4] - r["CP"][:, 4])[:, None], "ab": r["P"][:, 4][:, None]}
            for r in raws
        ]

    def z_pairs(make, q_key: str):
        tr_r, te_r = make(data["train_raw"]), make(data["test_raw"])
        sc = Scaler().fit(np.vstack([d["fl"] for d in tr_r]))

        def z(items, zsegs) -> List[SegData]:
            return [
                (sc.transform(d["fl"]), zs[q_key], sc.transform(d["ab"]))
                for d, zs in zip(items, zsegs)
            ]

        return z(tr_r, data["train_z"]), z(te_r, data["test_z"])

    p3_tr, p3_te = z_pairs(p3_arrays, "q")
    p1_tr, p1_te = z_pairs(p1_arrays, "q")

    from .run import seg_data as std_seg_data

    variants = [
        (
            "P5",
            "5 node pressures",
            [f"p{i}" for i in range(5)],
            std_seg_data(data["train_z"], "sindyc"),
            std_seg_data(data["test_z"], "sindyc"),
        ),
        ("P3", "2 source PCs + p_orf", ["p_pc1", "p_pc2", "p_orf"], p3_tr, p3_te),
        (
            "P2",
            "p_up + p_orf",
            ["p_up", "p_orf"],
            std_seg_data(data["train_z"], "sindyc_red"),
            std_seg_data(data["test_z"], "sindyc_red"),
        ),
        ("P1", "p_orf only", ["p_orf"], p1_tr, p1_te),
    ]

    base = baseline_metrics(
        [(z["P2"], z["P2abs"]) for z in data["test_z"]], dt, args.ic_stride, data["warmup_h"]
    )
    results = {}
    for key, label, state_names, d_tr, d_te in variants:
        spec = ModelSpec(key, label, state_names, Q_NAMES, clip=args.clip)
        spec.fit(d_tr, args.fit, dt, args.savgol_window, args.threshold, args.alpha, args.degree)
        r2_te = spec.r2_pressures(d_te, args.fit, dt, args.savgol_window)
        roll = rollout_metrics(
            spec, d_te, dt, data["integrator"], args.ic_stride, args.blowup, data["warmup_h"]
        )
        rho = spectral_radius(spec, dt)
        hz = roll["horizons"]
        results[key] = {
            "label": label,
            "n_states": len(state_names),
            "spectral_radius": rho,
            "r2_test_mean": float(np.mean(r2_te)),
            "diverged_frac_72h": hz[72]["diverged_frac"],
            "nrmse_24h": hz[24]["nrmse"],
            "nrmse_72h": hz[72]["nrmse"],
        }
        print(
            f"[{key}] dim={len(state_names)}  rho(I+dtA)={rho:.3f}  "
            f"R2te {np.mean(r2_te):+.3f}  div72 {hz[72]['diverged_frac']:.0%}  "
            f"NRMSE 24h {hz[24]['nrmse']:.3f} 72h {hz[72]['nrmse']:.3f}"
        )
    print(
        f"best trivial baseline: 24h {min(v[24] for v in base.values()):.3f}, "
        f"72h {min(v[72] for v in base.values()):.3f}"
    )

    with open(os.path.join(args.out, "states_ablation.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "results": results,
                "baselines": {b: {str(h): v for h, v in m.items()} for b, m in base.items()},
            },
            fh,
            indent=2,
        )

    # ---- figure -----------------------------------------------------------
    set_paper_style(font_size=10)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = ["P1", "P2", "P3", "P5"]
    dims = [results[k]["n_states"] for k in keys]
    rho = [results[k]["spectral_radius"] for k in keys]
    div = [100 * results[k]["diverged_frac_72h"] for k in keys]
    nr = [results[k]["nrmse_72h"] for k in keys]
    fig, ax1 = plt.subplots(figsize=(6.5, 3.6))
    ax1.plot(dims, rho, "C3o-", lw=1.4, label=r"spectral radius of $I + \Delta t\,A$")
    ax1.axhline(1.0, color="C3", lw=0.7, ls=":")
    ax1.set_xlabel("number of pressure states")
    ax1.set_ylabel(r"$\rho(I + \Delta t\,A)$", color="C3")
    ax1.set_xticks(dims)
    ax2 = ax1.twinx()
    nr_plot = [v if np.isfinite(v) else np.nan for v in nr]
    ax2.plot(dims, nr_plot, "C0s-", lw=1.4, label="72 h NRMSE")
    ax2.set_ylabel("median 72 h NRMSE", color="C0")
    for x, d_ in zip(dims, div):
        ax1.annotate(
            f"{d_:.0f}% div",
            (x, rho[dims.index(x)]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=7,
        )
    fig.suptitle("A3: rollout stability vs pressure-state dimension", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_states.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved states_ablation.json and fig_states.png to {args.out}")


if __name__ == "__main__":
    main()
