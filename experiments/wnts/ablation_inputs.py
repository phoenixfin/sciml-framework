"""B2: input ablation -- which boundary flows actually drive the pressures?

All variants use the stable reduced state (p_up, p_orf) and the default
quadratic library (SINDYc-2); only the control-input set changes:

- ``all5``  : all five flows (the default so far).
- ``qorf``  : ORF offtake only (the demand signal).
- ``qsrc``  : the four source flows only (the nomination signals).
- ``imb``   : a single input, the net flow imbalance sum(q_src) - q_orf.
- ``none``  : no inputs (autonomous 2-state SINDy -- lower anchor).

Evaluated on within-year 2019 and pooled 2016-2019 -> 2020.

Usage::

    python -m experiments.wnts.ablation_inputs
"""

from __future__ import annotations

import json
import os
from typing import List

import numpy as np

from .data import Scaler
from .run import (
    Q_NAMES,
    ModelSpec,
    SegData,
    build_parser,
    prepare_data,
    rollout_metrics,
)

STATE_NAMES = ["p_up", "p_orf"]

CONFIGS = [
    ("2019", dict(years=[2019], test_years=None)),
    ("pool_2020", dict(years=[2016, 2017, 2018, 2019], test_years=[2020])),
]


def config_args(overrides: dict):
    args = build_parser().parse_args([])
    args.ic_stride = 24
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def imbalance_z(data: dict):
    """Single-column z-scored net-imbalance input per segment."""

    def fluct(r):
        f = r["q"] - r["Cq"]
        return (f[:, :4].sum(axis=1) - f[:, 4])[:, None]

    sc = Scaler().fit(np.vstack([fluct(r) for r in data["train_raw"]]))
    tr = [sc.transform(fluct(r)) for r in data["train_raw"]]
    te = [sc.transform(fluct(r)) for r in data["test_raw"]]
    return tr, te


def main() -> None:
    out = "outputs/wnts_B2"
    os.makedirs(out, exist_ok=True)
    results, eq_lines, rows = {}, [], []
    for cfg_name, overrides in CONFIGS:
        args = config_args(overrides)
        data = prepare_data(args)
        imb_tr, imb_te = imbalance_z(data)

        def sets(zsegs, imb) -> dict:
            return {
                "all5": ([(z["P2"], z["q"], z["P2abs"]) for z in zsegs], Q_NAMES),
                "qorf": ([(z["P2"], z["q"][:, [4]], z["P2abs"]) for z in zsegs], ["q_orf"]),
                "qsrc": ([(z["P2"], z["q"][:, :4], z["P2abs"]) for z in zsegs], Q_NAMES[:4]),
                "imb": (
                    [(z["P2"], i, z["P2abs"]) for z, i in zip(zsegs, imb)],
                    ["q_imb"],
                ),
                "none": ([(z["P2"], None, z["P2abs"]) for z in zsegs], []),
            }

        tr_sets = sets(data["train_z"], imb_tr)
        te_sets = sets(data["test_z"], imb_te)
        for key in ("all5", "qorf", "qsrc", "imb", "none"):
            d_tr: List[SegData]
            d_tr, input_names = tr_sets[key]
            d_te, _ = te_sets[key]
            spec = ModelSpec(f"{cfg_name}:{key}", key, STATE_NAMES, input_names, clip=args.clip)
            spec.fit(
                d_tr,
                args.fit,
                data["dt"],
                args.savgol_window,
                args.threshold,
                args.alpha,
                args.degree,
            )
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
                "inputs": key,
                "n_inputs": len(input_names),
                "n_active": int(np.count_nonzero(spec.model.coef_)),
                "r2_test": round(float(np.mean(r2_te)), 3),
                "nrmse_24h": round(hz[24]["nrmse"], 3),
                "nrmse_72h": round(hz[72]["nrmse"], 3),
                "diverged": round(hz[72]["diverged_frac"], 2),
            }
            rows.append(row)
            results[f"{cfg_name}:{key}"] = row
            eq_lines.append(f"### {cfg_name} / inputs={key}")
            eq_lines += spec.model.equations([f"d/dt {n}" for n in STATE_NAMES])
            eq_lines.append("")
            print(
                f"[{cfg_name}:{key:5s}] inputs {len(input_names)}  "
                f"R2te {np.mean(r2_te):+.3f}  div {hz[72]['diverged_frac']:.0%}  "
                f"NRMSE 24h {hz[24]['nrmse']:.3f} 72h {hz[72]['nrmse']:.3f}"
            )

    cols = [
        "config",
        "inputs",
        "n_inputs",
        "n_active",
        "r2_test",
        "nrmse_24h",
        "nrmse_72h",
        "diverged",
    ]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    with open(os.path.join(out, "table.md"), "w", encoding="utf-8") as fh:
        fh.write("# B2 input ablation (SINDYc-2)\n\n" + "\n".join(lines) + "\n")
    with open(os.path.join(out, "equations.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(eq_lines))
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved table.md, equations.txt, results.json to {out}")


if __name__ == "__main__":
    main()
