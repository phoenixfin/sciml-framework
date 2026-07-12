"""B4: benchmark against DMDc -- what does sparse regression add?

B1 showed the identified SINDYc-2 model is linear, so the natural null
model is DMDc (linear one-step map with control,
``x_{k+1} = A x_k + B u_k``). Fitting an unthresholded, unregularized
linear least-squares model inside the same discrete-SINDy machinery is
mathematically identical to DMDc in companion form
(``x_{k+1} = x_k + dt (A x + B u)``), which keeps data handling and the
rollout protocol exactly the same -- only the estimator changes.

Variants (configs: within-2019 and pooled 2016-2019 -> 2020):

- ``dmdc_P5``  : DMDc on the naive 5-pressure state (does *any* linear
  regression survive the collinear state, or is instability inherent?).
- ``dmdc_P2``  : DMDc on the reduced state (p_up, p_orf).
- ``ridge_P2`` : + ridge regularization, no sparsity (isolates the ridge
  contribution).
- ``sindyc2``  : the headline SINDYc-2 (ridge + STRidge sparsification).

Usage::

    python -m experiments.wnts.benchmark_dmdc
"""

from __future__ import annotations

import json
import os

import numpy as np

from .run import (
    P_NAMES,
    Q_NAMES,
    ModelSpec,
    build_parser,
    prepare_data,
    rollout_metrics,
    seg_data,
)

CONFIGS = [
    ("2019", dict(years=[2019], test_years=None)),
    ("pool_2020", dict(years=[2016, 2017, 2018, 2019], test_years=[2020])),
]

# key, label, state form, degree, threshold, alpha
VARIANTS = [
    ("dmdc_P5", "DMDc, 5 pressures", "sindyc", 1, 0.0, 0.0),
    ("dmdc_P2", "DMDc, reduced", "sindyc_red", 1, 0.0, 0.0),
    ("ridge_P2", "ridge linear, reduced", "sindyc_red", 1, 0.0, 1e-2),
    ("sindyc2", "SINDYc-2 (ridge + STRidge)", "sindyc_red", 2, 0.02, 1e-2),
]


def config_args(overrides: dict):
    args = build_parser().parse_args([])
    args.ic_stride = 24
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def main() -> None:
    out = "outputs/wnts_B4"
    os.makedirs(out, exist_ok=True)
    results, rows, eq_lines = {}, [], []
    for cfg_name, overrides in CONFIGS:
        args = config_args(overrides)
        data = prepare_data(args)
        for key, label, form, degree, threshold, alpha in VARIANTS:
            state_names = P_NAMES if form == "sindyc" else ["p_up", "p_orf"]
            d_tr = seg_data(data["train_z"], form)
            d_te = seg_data(data["test_z"], form)
            spec = ModelSpec(f"{cfg_name}:{key}", label, state_names, Q_NAMES, clip=args.clip)
            spec.fit(d_tr, args.fit, data["dt"], args.savgol_window, threshold, alpha, degree)
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
            row = {
                "config": cfg_name,
                "model": key,
                "n_active": int(np.count_nonzero(spec.model.coef_)),
                "r2_test": round(float(np.mean(r2_te)), 3),
                "nrmse_24h": round(hz[24]["nrmse"], 3),
                "nrmse_72h": round(hz[72]["nrmse"], 3),
                "diverged": round(hz[72]["diverged_frac"], 2),
            }
            rows.append(row)
            results[f"{cfg_name}:{key}"] = row
            if form == "sindyc_red":
                eq_lines.append(f"### {cfg_name} / {label}")
                eq_lines += spec.model.equations([f"d/dt {n}" for n in state_names])
                eq_lines.append("")
            print(
                f"[{cfg_name}:{key:9s}] active {row['n_active']:3d}  "
                f"R2te {np.mean(r2_te):+.3f}  div {hz[72]['diverged_frac']:.0%}  "
                f"NRMSE 24h {hz[24]['nrmse']:.3f} 72h {hz[72]['nrmse']:.3f}"
            )

    cols = ["config", "model", "n_active", "r2_test", "nrmse_24h", "nrmse_72h", "diverged"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    with open(os.path.join(out, "table.md"), "w", encoding="utf-8") as fh:
        fh.write("# B4 DMDc benchmark\n\n" + "\n".join(lines) + "\n")
    with open(os.path.join(out, "equations.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(eq_lines))
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved table.md, equations.txt, results.json to {out}")


if __name__ == "__main__":
    main()
