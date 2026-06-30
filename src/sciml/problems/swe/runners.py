"""High-level runners for the SWE / DeepONet example."""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np

from ...core.io import save_json
from ...core.logging import get_logger
from ...core.metrics import rel_l2
from ...core.plotting import set_paper_style
from ...core.seeding import seed_everything
from ...methods.deeponet.optim import make_optimizer
from ...methods.deeponet.trainer import Trainer
from . import cases as swe_cases
from .config import SWEConfig
from .evaluation import plot_case_with_error
from .problem import SWEProblem

_log = get_logger(__name__)


def prepare_problem(cfg: SWEConfig, *, verbose: bool = True) -> SWEProblem:
    prob = SWEProblem(cfg)
    prob.prepare(seed=cfg.train.seed)
    if verbose:
        _log.info("GP pool: %d (boundary gap %.4f)", prob.n_total, prob.boundary_gap())
    prob.generate_dataset(verbose=verbose)
    return prob


def _optimizer(cfg: SWEConfig):
    return make_optimizer(cfg.train.lr, cfg.train.lr_decay_steps, cfg.train.lr_decay_rate)


def train(cfg: SWEConfig, prob: Optional[SWEProblem] = None, *, variant: str = "full",
          ckpt_dir: Optional[str] = None, weights_path: Optional[str] = None,
          verbose: bool = True) -> Tuple[object, object, SWEProblem]:
    seed_everything(cfg.train.seed)
    prob = prob or prepare_problem(cfg, verbose=verbose)
    model = prob.build_model(variant)
    if verbose:
        n = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
        _log.info("Variant %s: %d parameters", variant, n)
    opt = _optimizer(cfg)
    step = prob.make_step(model, opt, variant=variant)
    history = Trainer(model, opt, step, prob.component_names).fit(
        prob.sample_batch, cfg.train.n_iter, log_every=cfg.train.log_every,
        ckpt_dir=ckpt_dir, ckpt_every=cfg.train.ckpt_every, verbose=verbose)
    if weights_path:
        os.makedirs(os.path.dirname(weights_path) or ".", exist_ok=True)
        model.save_weights(weights_path)
        if verbose:
            _log.info("Weights saved: %s", weights_path)
    return model, history, prob


def evaluate_cases(prob: SWEProblem, model, out_dir: str = "outputs/swe",
                   verbose: bool = True) -> Dict[str, Dict[str, float]]:
    set_paper_style()
    os.makedirs(out_dir, exist_ok=True)
    results: Dict[str, Dict[str, float]] = {}
    for case in (swe_cases.C1, swe_cases.C2, swe_cases.C3):
        xs, ts, h, hu = prob.predict_grid(model, case.h0, case.bath)
        x_ref, snaps = prob.reference(case.h0, case.bath)
        fname = os.path.join(out_dir, f"fig_{case.name.lower()}.png")
        eh, ehu = plot_case_with_error(xs, ts, h, hu, x_ref, snaps,
                                       f"{case.name}: {case.description}", fname, case.color)
        results[case.name] = {"eps_h": eh, "eps_hu": ehu, "description": case.description}
        if verbose:
            _log.info("%s: eps_h=%.3e eps_hu=%.3e", case.name, eh, ehu)
    return results


def generalization(prob: SWEProblem, model, n_test: Optional[int] = None,
                   out_dir: str = "outputs/swe", seed: int = 0,
                   verbose: bool = True) -> Dict[str, float]:
    n_test = n_test if n_test is not None else prob.config.data.n_test
    np.random.seed(seed)
    H0_test = prob.h0_sampler.sample(prob.x_sensors, n_test)
    B_test = prob.bath_sampler.sample(prob.x_sensors, n_test)
    errs_h, errs_hu, xq = [], [], np.linspace(0, prob.L, 100)
    for k in range(n_test):
        h0k = lambda x, k=k: np.interp(x, prob.x_sensors, H0_test[k])
        bk = lambda x, k=k: np.interp(x, prob.x_sensors, B_test[k])
        _, _, h_pk, hu_pk = prob.predict_grid(model, h0k, bk, nx=100, nt=50)
        xrk, snk = prob.reference(h0k, bk, nx=300, nt=3000)
        errs_h.append(rel_l2(h_pk[-1], np.interp(xq, xrk, snk["h"][-1])))
        errs_hu.append(rel_l2(hu_pk[-1], np.interp(xq, xrk, snk["hu"][-1])))
    errs_h, errs_hu = np.array(errs_h), np.array(errs_hu)
    summary = {
        "mean_eps_h": float(errs_h.mean()), "mean_eps_hu": float(errs_hu.mean()),
        "median_eps_h": float(np.median(errs_h)), "median_eps_hu": float(np.median(errs_hu)),
        "p90_eps_h": float(np.percentile(errs_h, 90)), "p90_eps_hu": float(np.percentile(errs_hu, 90)),
    }
    _generalization_fig(errs_h, errs_hu, out_dir)
    if verbose:
        _log.info("Generalization mean eps_h=%.3e eps_hu=%.3e (%d pairs)",
                  summary["mean_eps_h"], summary["mean_eps_hu"], n_test)
    return summary


def _generalization_fig(errs_h, errs_hu, out_dir):
    import matplotlib.pyplot as plt
    set_paper_style(); os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(6, 2.8))
    for ax, errs, lbl in zip(axes, [errs_h, errs_hu],
                             [r"$\varepsilon_h$", r"$\varepsilon_{hu}$"]):
        ax.hist(errs, bins=15, color="steelblue", edgecolor="white", alpha=0.85)
        ax.axvline(errs.mean(), color="k", lw=1.2, ls="--", label=f"Mean={errs.mean():.3e}")
        ax.set(xlabel=lbl, ylabel="Count"); ax.legend(fontsize=7)
    fig.suptitle("Operator generalization: unseen periodic pairs", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_generalization.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_results(results: Dict, path: str) -> None:
    save_json(results, path)
