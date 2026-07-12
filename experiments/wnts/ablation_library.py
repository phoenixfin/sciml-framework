"""B1: library ablation -- does a physics-informed library beat a generic
polynomial, and which physics terms survive sparsification?

All variants use the stable reduced state (p_up, p_orf) with all flows as
inputs (SINDYc-2); only the candidate-term library changes:

- ``lin``       : linear terms only (a linear state-space model).
- ``quad``      : full degree-2 polynomial (the default so far).
- ``phys``      : linear terms + physics terms: the Weymouth driving term
  ``p_up^2 - p_orf^2`` (q^2 across a pipe is proportional to the squared
  pressure drop), turbulent-friction ``q_orf|q_orf|``, and
  pressure-dependent draw ``p*q_orf`` crosses.
- ``quad_phys`` : degree-2 polynomial + ``q_orf|q_orf|`` (the only physics
  term a polynomial cannot express).

Evaluated on two configurations: within-year 2019 and pooled 2016-2019
trained, tested on 2020 (the strongest A1 configuration).

Usage::

    python -m experiments.wnts.ablation_library
"""

from __future__ import annotations

import json
import os

import numpy as np

from sciml.methods.sindy import CustomLibrary, PolynomialLibrary

from .run import (
    Q_NAMES,
    ModelSpec,
    build_parser,
    prepare_data,
    rollout_metrics,
    seg_data,
)

STATE_NAMES = ["p_up", "p_orf"]
# Column order of the augmented matrix: [p_up, p_orf, q_anoa, q_kakap,
# q_hangtuah, q_gajahbaru, q_orf].
I_UP, I_ORF, I_QORF = 0, 1, 6

PHYS_TERMS = CustomLibrary(
    [
        lambda X: X[:, I_UP] ** 2 - X[:, I_ORF] ** 2,
        lambda X: X[:, I_QORF] * np.abs(X[:, I_QORF]),
        lambda X: X[:, I_UP] * X[:, I_QORF],
        lambda X: X[:, I_ORF] * X[:, I_QORF],
    ],
    ["p_up^2-p_orf^2", "q_orf|q_orf|", "p_up*q_orf", "p_orf*q_orf"],
)
FRICTION_ONLY = CustomLibrary([lambda X: X[:, I_QORF] * np.abs(X[:, I_QORF])], ["q_orf|q_orf|"])

VARIANTS = [
    ("lin", "linear", lambda: PolynomialLibrary(degree=1)),
    ("quad", "quadratic", lambda: PolynomialLibrary(degree=2)),
    ("phys", "linear + physics", lambda: PolynomialLibrary(degree=1) + PHYS_TERMS),
    ("quad_phys", "quadratic + friction", lambda: PolynomialLibrary(degree=2) + FRICTION_ONLY),
]

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


def main() -> None:
    out = "outputs/wnts_B1"
    os.makedirs(out, exist_ok=True)
    results, eq_lines, rows = {}, [], []
    for cfg_name, overrides in CONFIGS:
        args = config_args(overrides)
        data = prepare_data(args)
        d_tr = seg_data(data["train_z"], "sindyc_red")
        d_te = seg_data(data["test_z"], "sindyc_red")
        for key, label, make_lib in VARIANTS:
            spec = ModelSpec(f"{cfg_name}:{key}", label, STATE_NAMES, Q_NAMES, clip=args.clip)
            spec.fit(
                d_tr,
                args.fit,
                data["dt"],
                args.savgol_window,
                args.threshold,
                args.alpha,
                args.degree,
                library=make_lib(),
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
            n_lib = spec.model.coef_.shape[0]
            n_active = int(np.count_nonzero(spec.model.coef_))
            phys_active = sorted(
                {
                    t
                    for eq in spec.coef_dict().values()
                    for t in eq
                    if ("|" in t or "-" in t) and t != "1"
                }
            )
            row = {
                "config": cfg_name,
                "library": key,
                "n_library_terms": n_lib,
                "n_active": n_active,
                "r2_test": round(float(np.mean(r2_te)), 3),
                "nrmse_24h": round(hz[24]["nrmse"], 3),
                "nrmse_72h": round(hz[72]["nrmse"], 3),
                "diverged": round(hz[72]["diverged_frac"], 2),
                "physics_terms_active": phys_active,
            }
            rows.append(row)
            results[f"{cfg_name}:{key}"] = row
            eq_lines.append(f"### {cfg_name} / {label}")
            eq_lines += spec.model.equations([f"d/dt {n}" for n in STATE_NAMES])
            eq_lines.append("")
            print(
                f"[{cfg_name}:{key:9s}] terms {n_lib:2d} active {n_active:2d}  "
                f"R2te {np.mean(r2_te):+.3f}  div {hz[72]['diverged_frac']:.0%}  "
                f"NRMSE 24h {hz[24]['nrmse']:.3f} 72h {hz[72]['nrmse']:.3f}  "
                f"phys: {phys_active}"
            )

    cols = [
        "config",
        "library",
        "n_library_terms",
        "n_active",
        "r2_test",
        "nrmse_24h",
        "nrmse_72h",
        "diverged",
    ]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    with open(os.path.join(out, "table.md"), "w", encoding="utf-8") as fh:
        fh.write("# B1 library ablation (SINDYc-2)\n\n" + "\n".join(lines) + "\n")
    with open(os.path.join(out, "equations.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(eq_lines))
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved table.md, equations.txt, results.json to {out}")


if __name__ == "__main__":
    main()
