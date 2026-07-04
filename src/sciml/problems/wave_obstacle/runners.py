"""High-level runners for the moving-boundary wave PINN example."""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np

from ...core.io import save_json
from ...core.logging import get_logger
from ...core.plotting import set_paper_style
from ...core.seeding import seed_everything
from ...methods.pinn.sampling import replace_high_residual
from ...methods.pinn.training import PINNTrainer
from .config import WaveObstacleConfig
from .problem import WaveObstacleProblem

_log = get_logger(__name__)


def _run_phase(trainer: PINNTrainer, prob: WaveObstacleProblem, phase, *, verbose=True):
    schedule = np.linspace(phase.eps_c_start, phase.eps_c_end, max(phase.steps, 1))

    def on_step(ep: int):
        prob.eps_causal.assign(schedule[min(ep, phase.steps) - 1])
        if phase.rar_every and ep % phase.rar_every == 0 and float(prob.eps_causal) < 0.5:
            xbar_r, tau_r, res = prob.pde_residuals(3000)
            bx, bt = prob.current_collocation()
            bx, bt = replace_high_residual(bx, bt, xbar_r, tau_r, res,
                                           x_bounds=(0.0, 1.0), t_bounds=(0.0, prob.T))
            prob.assign_collocation(bx, bt)

    if verbose:
        _log.info("Phase %s: Adam lr=%.0e eps_c %.1f->%.1f (%d steps)%s",
                  phase.name, phase.lr, phase.eps_c_start, phase.eps_c_end, phase.steps,
                  "  +RAR" if phase.rar_every else "")
    trainer.run_adam(phase.steps, phase.lr, on_step=on_step, verbose=verbose)


def train(cfg: Optional[WaveObstacleConfig] = None, *, lbfgs: bool = True,
          out_dir: Optional[str] = None, verbose: bool = True
          ) -> Tuple[WaveObstacleProblem, PINNTrainer]:
    """Train the PINN through the Adam phases (+ optional L-BFGS). Returns (problem, trainer)."""
    cfg = cfg or WaveObstacleConfig()
    seed_everything(cfg.train.seed)
    prob = WaveObstacleProblem(cfg)
    if verbose:
        _log.info("Derived: s_y=%.4f omega=%.4f amp_s=%.5f Nu params=%d",
                  prob.s_y, prob.omega, prob.amp_s, prob.Nu.count_params())
    loss_components = prob.make_loss()
    trainer = PINNTrainer(prob.trainable_variables, lambda: loss_components(True)[0])

    for phase in cfg.train.phases:
        _run_phase(trainer, prob, phase, verbose=verbose)

    if lbfgs:
        prob.eps_causal.assign(0.0)
        if verbose:
            _log.info("Phase L-BFGS (<= %d iters)", cfg.train.lbfgs_maxiter)
        trainer.run_lbfgs(cfg.train.lbfgs_maxiter, verbose=verbose)
        if cfg.train.lbfgs_restart_maxiter:
            if verbose:
                _log.info("Phase L-BFGS restart (<= %d iters)", cfg.train.lbfgs_restart_maxiter)
            trainer.run_lbfgs(cfg.train.lbfgs_restart_maxiter, restart_from_best=True,
                              verbose=verbose)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        prob.Nu.save_weights(os.path.join(out_dir, "Nu.weights.h5"))
        np.save(os.path.join(out_dir, "Ns_weights.npy"),
                np.array([v.numpy() for v in prob.Ns.trainable_variables], dtype=object))
    return prob, trainer


def field_error(prob: WaveObstacleProblem, nh: int = 80) -> Tuple[float, np.ndarray, np.ndarray]:
    """Relative L2 field error e_u against the FDM reference (and the two fields)."""
    import tensorflow as tf
    ref = prob.reference(n_snaps=100)
    n_t = len(ref["t"])
    U_pinn = np.full((nh, n_t), np.nan)
    U_ref = np.full((nh, n_t), np.nan)
    for j in range(n_t):
        tau_j = float(ref["t"][j])
        s_pinn = float(prob.Ns(np.array([[tau_j]], np.float32)).numpy()[0, 0])
        sp = max(s_pinn, 1e-3)
        xq = np.linspace(sp, 1.0, nh, dtype=np.float32)
        U_pinn[:, j] = prob.Nu(np.stack([xq, np.full(nh, tau_j, np.float32)], 1)).numpy().flatten()
        sv = ref["s"][j]
        U_ref[:, j] = np.interp(np.linspace(sv, 1.0, nh), ref["xbar"][j], ref["u"][j])
    diff = np.abs(U_pinn - U_ref)
    e_u = float(np.linalg.norm(diff[~np.isnan(diff)])
                / np.linalg.norm(U_ref[~np.isnan(U_ref)]) * 100)
    return e_u, U_pinn, U_ref


def evaluate(prob: WaveObstacleProblem, trainer: Optional[PINNTrainer] = None,
             out_dir: str = "outputs/wave_obstacle", verbose: bool = True) -> Dict:
    """Compute e_s/e_u and amplitude/frequency metrics, save an overview figure."""
    set_paper_style()
    os.makedirs(out_dir, exist_ok=True)
    metrics = prob.evaluate()
    e_u, U_pinn, U_ref = field_error(prob)
    metrics["e_u_pct"] = e_u
    if verbose:
        _log.info("e_s=%.2f%%  e_u=%.2f%%  amp_ratio=%.3f",
                  metrics["e_s_pct"], e_u, metrics["amp_ratio"])
    _overview_fig(prob, trainer, U_pinn, U_ref, metrics, out_dir)
    save_json(metrics, os.path.join(out_dir, "metrics.json"))
    return metrics


def _overview_fig(prob, trainer, U_pinn, U_ref, metrics, out_dir):
    import matplotlib.pyplot as plt
    tau = np.linspace(0, prob.T, 600, dtype=np.float32)
    s_pinn = prob.Ns(tau[:, None]).numpy().flatten()
    s_ref = prob.s_analytic(tau)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    if trainer is not None and trainer.history:
        axes[0].semilogy(trainer.history, "b-", lw=0.8)
    axes[0].set(xlabel="Iteration", ylabel="Loss", title="(a) Training loss")
    axes[0].grid(alpha=0.3)
    axes[1].plot(tau, s_ref, "b-", lw=2, label="analytic")
    axes[1].plot(tau, s_pinn, "r--", lw=1.6, label=f"PINN ($e_s$={metrics['e_s_pct']:.1f}%)")
    axes[1].axhline(prob.s_y, color="gray", ls=":", lw=1)
    axes[1].set(xlabel=r"$\tau$", ylabel=r"$s(\tau)$", title="(b) Free boundary")
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
    diff = np.abs(U_pinn - U_ref)
    vmax = np.nanpercentile(diff, 98) if not np.all(np.isnan(diff)) else 1e-3
    im = axes[2].imshow(diff, aspect="auto", origin="lower", extent=[0, prob.T, 0, 1],
                        cmap="Oranges", vmin=0, vmax=max(vmax, 1e-5))
    axes[2].set(xlabel=r"$\tau$", ylabel=r"$\bar x$",
                title=f"(c) Field error ($e_u$={metrics['e_u_pct']:.1f}%)")
    plt.colorbar(im, ax=axes[2], shrink=0.85)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_overview.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
