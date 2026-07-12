"""A1: multi-year robustness matrix for the WNTS SINDYc experiments.

Runs the six-model ladder for every contract year (within-year
chronological split) and for consecutive-year transfer (train on year Y,
test on all clean segments of year Y+1), then aggregates:

- a results table (CSV + printed markdown) focused on ``sindyc_red``;
- a coefficient-stability figure: identified coefficients of the reduced
  model across years -- is it the same physics every year?

Usage::

    python -m experiments.wnts.multi_year --years 2014 2015 ... 2021
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional

import numpy as np

from sciml.core.plotting import set_paper_style

from .run import build_parser, run_experiment

KEY = "sindyc_red"


def one_run(year: int, test_year: Optional[int], out_dir: str, base_args) -> Optional[dict]:
    """Run one experiment (within-year or transfer); None on failure."""
    args = build_parser().parse_args([])
    for k, v in vars(base_args).items():
        if hasattr(args, k) and v is not None:
            setattr(args, k, v)
    args.years = [year]
    args.test_years = [test_year] if test_year else None
    tag = f"{year}" if test_year is None else f"{year}_to_{test_year}"
    args.out = os.path.join(out_dir, tag)
    args.no_figures = True
    try:
        return run_experiment(args)
    except Exception as exc:  # noqa: BLE001 - a bad year should not kill the sweep
        print(f"!! {tag}: {exc}")
        return None


def row(tag: str, res: dict) -> dict:
    m = res["models"][KEY]
    b = res["baselines"]
    best24 = min(v["24"] for v in b.values())
    best72 = min(v["72"] for v in b.values())
    return {
        "run": tag,
        "n_rollouts": m["n_rollouts"],
        "r2_test": round(m["r2_dPdt_test_mean"], 3),
        "nrmse_24h": round(m["nrmse_24h"], 3),
        "nrmse_72h": round(m["nrmse_72h"], 3),
        "nrmse_168h": round(m["horizons"]["168"]["nrmse"], 3),
        "diverged": round(m["diverged_frac"], 2),
        "best_baseline_24h": round(best24, 3),
        "best_baseline_72h": round(best72, 3),
        "skill_24h": round(1 - m["nrmse_24h"] / best24, 2),
        "skill_72h": round(1 - m["nrmse_72h"] / best72, 2),
    }


def coeff_matrix(results: dict, state: str):
    """(years, terms, matrix) of identified coefficients for one equation."""
    years = sorted(results)
    terms: List[str] = []
    for y in years:
        for t in results[y]["models"][KEY]["coefficients"][state]:
            if t not in terms:
                terms.append(t)
    M = np.zeros((len(years), len(terms)))
    for i, y in enumerate(years):
        c = results[y]["models"][KEY]["coefficients"][state]
        for j, t in enumerate(terms):
            M[i, j] = c.get(t, 0.0)
    return years, terms, M


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=list(range(2016, 2022)),
        help="contract years (2014-2015 excluded by default: "
        "source telemetry frozen 20-63%% of the time)",
    )
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--center", default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument(
        "--ic-stride", type=int, default=24, help="denser IC grid than the single-run default (48)"
    )
    ap.add_argument("--no-transfer", action="store_true")
    ap.add_argument("--out", default="outputs/wnts_A1")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    within, transfer = {}, {}
    for y in args.years:
        print(f"\n===== within-year {y} =====")
        res = one_run(y, None, args.out, args)
        if res:
            within[y] = res
    if not args.no_transfer:
        for y in args.years[:-1]:
            y2 = y + 1
            if y2 not in args.years:
                continue
            print(f"\n===== transfer {y} -> {y2} =====")
            res = one_run(y, y2, args.out, args)
            if res:
                transfer[y] = res

    # ------- tables -------
    rows = [row(str(y), r) for y, r in within.items()]
    rows += [row(f"{y}->{y + 1}", r) for y, r in transfer.items()]
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    table = "\n".join(lines)
    print("\n" + table)
    with open(os.path.join(args.out, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(f"# A1 multi-year robustness ({KEY})\n\n" + table + "\n")
    with open(os.path.join(args.out, "table.csv"), "w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        fh.writelines(",".join(str(r[c]) for c in cols) + "\n" for r in rows)
    with open(os.path.join(args.out, "all_results.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "within": {str(k): v for k, v in within.items()},
                "transfer": {str(k): v for k, v in transfer.items()},
            },
            fh,
            indent=2,
        )

    # ------- coefficient stability figure -------
    set_paper_style(font_size=10)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(9, 6))
    for ax, state in zip(axes, ("p_up", "p_orf")):
        years, terms, M = coeff_matrix(within, state)
        vmax = max(1e-9, np.abs(M).max())
        im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_yticks(range(len(years)))
        ax.set_yticklabels(years, fontsize=7)
        ax.set_xticks(range(len(terms)))
        ax.set_xticklabels(terms, rotation=60, ha="right", fontsize=6)
        ax.set_title(f"d/dt {state}: identified coefficients per contract year", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.025)
    plt.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_coeff_stability.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved table.md/csv, all_results.json and fig_coeff_stability.png to {args.out}")


if __name__ == "__main__":
    main()
