"""High-level runner for the dengue beta(t) SINDy example."""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np

from ...core.io import save_json
from ...core.logging import get_logger
from ...core.metrics import rmse
from ...core.plotting import set_paper_style
from ...core.seeding import seed_everything
from .config import EpiConfig
from .problem import EpiProblem

_log = get_logger(__name__)


def run(cfg: Optional[EpiConfig] = None, *, out_dir: Optional[str] = None,
        verbose: bool = True) -> Dict:
    """Run the full pipeline (load/simulate -> reconstruct -> estimate -> plot/save)."""
    cfg = cfg or EpiConfig()
    seed_everything(cfg.seed, tensorflow=False)
    out_dir = out_dir or cfg.output_dir
    prob = EpiProblem(cfg)
    prob.load_or_simulate()
    prob.reconstruct()
    results = prob.estimate()

    summary: Dict = {"model": cfg.model.model, "s_recon": cfg.estim.s_recon, "rmse": {}}
    beta_true = prob.reference()
    if results["global"] and results["global"]["terms"] and verbose:
        _log.info("Global terms: %s",
                  ", ".join(f"{c:+.4f}*{n}" for n, c in results["global"]["terms"][:6]))

    # RMSE vs true beta (simulated data only).
    if beta_true is not None:
        t = prob.data["t"]
        for name, res in results["local"].items():
            bt = np.interp(res["t"], t, beta_true)
            valid = ~np.isnan(res["beta"])
            summary["rmse"][name] = rmse(res["beta"][valid], bt[valid])
        if results["global"]:
            summary["rmse"]["global"] = rmse(results["global"]["beta"], beta_true)
        if verbose:
            _log.info("RMSE vs true beta: %s",
                      {k: round(v, 4) for k, v in summary["rmse"].items()})

    _plot(prob, results, beta_true, out_dir)
    save_json(summary, os.path.join(out_dir, "summary.json"))
    if verbose:
        _log.info("Done. Figures + summary in %s/", out_dir)
    return {"summary": summary, "results": results}


def _plot(prob: EpiProblem, results: Dict, beta_true, out_dir: str) -> None:
    import matplotlib.pyplot as plt
    set_paper_style(font_size=10)
    os.makedirs(out_dir, exist_ok=True)
    t = prob.data["t"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 6))
    axes[0].bar(prob.raw["t"], prob.raw["I_raw"], width=1, alpha=0.4,
                color="steelblue", label="reported")
    axes[0].plot(t, prob.data["I"], color="navy", lw=1.3, label="I (smoothed)")
    axes[0].set(xlabel="Week", ylabel="Cases", title="Input series")
    axes[0].legend(fontsize=8)

    if beta_true is not None:
        axes[1].plot(t, beta_true, "k--", lw=2.2, label=r"$\beta$ true")
    colors = {"direct": "tab:blue", "windowed": "tab:orange", "ekf": "tab:red"}
    for name, res in results["local"].items():
        valid = ~np.isnan(res["beta"])
        axes[1].plot(res["t"][valid], res["beta"][valid], color=colors.get(name),
                     alpha=0.8, lw=1.2, label=name)
    if results["global"]:
        axes[1].plot(results["global"]["t"], results["global"]["beta"], "g-", lw=2,
                     label="global (time basis)")
    axes[1].set(xlabel="Week", ylabel=r"$\beta(t)$", title=r"Estimated $\beta(t)$")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_beta_estimates.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
