"""Command-line interface: one subcommand per worked example.

    sciml swe     [--quick] [--config C] [--out D] [--timing]   # DeepONet / SWE
    sciml wave    [--quick] [--config C] [--out D] [--no-lbfgs] # PINN / wave-obstacle
    sciml dengue  [--quick] [--config C] [--out D]              # SINDy / dengue beta(t)

Run ``sciml <cmd> -h`` for options.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Sequence


# ----------------------------------------------------------------------- SWE
def _cmd_swe(args: argparse.Namespace) -> None:
    """Run the SWE / DeepONet example from parsed CLI arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments for the ``swe`` subcommand.

    Returns
    -------
    None
    """
    from .problems.swe.config import SWEConfig
    from .problems.swe import runners
    cfg = SWEConfig.load(args.config) if args.config else SWEConfig()
    if args.quick:
        cfg.data.n_train, cfg.data.n_data_gp, cfg.data.n_test = 40, 12, 8
        cfg.data.solver_nx, cfg.data.solver_nt = 200, 1500
        cfg.train.n_iter, cfg.train.log_every, cfg.train.ckpt_every = 300, 100, 1000
    out = args.out or cfg.output_dir
    os.makedirs(out, exist_ok=True)
    model, history, prob = runners.train(
        cfg, ckpt_dir=os.path.join(out, "ckpt"),
        weights_path=os.path.join(out, "model.weights.h5"))
    results = {"history": history.to_dict(),
               "cases": runners.evaluate_cases(prob, model, out_dir=out),
               "generalization": runners.generalization(prob, model, out_dir=out)}
    runners.save_results(results, os.path.join(out, "results.json"))


# ---------------------------------------------------------------------- wave
def _cmd_wave(args: argparse.Namespace) -> None:
    """Run the wave-obstacle / PINN example from parsed CLI arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments for the ``wave`` subcommand.

    Returns
    -------
    None
    """
    from .problems.wave_obstacle.config import WaveObstacleConfig
    from .problems.wave_obstacle import runners
    cfg = WaveObstacleConfig.load(args.config) if args.config else WaveObstacleConfig()
    if args.quick:
        for p in cfg.train.phases:
            p.steps = max(200, p.steps // 20)
        cfg.train.fdm_nx = 120
        args.no_lbfgs = True
    out = args.out or cfg.output_dir
    prob, trainer = runners.train(cfg, lbfgs=not args.no_lbfgs, out_dir=out)
    runners.evaluate(prob, trainer, out_dir=out)


# -------------------------------------------------------------------- dengue
def _cmd_dengue(args: argparse.Namespace) -> None:
    """Run the dengue / SINDy example from parsed CLI arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments for the ``dengue`` subcommand.

    Returns
    -------
    None
    """
    from .problems.epidemiology.config import EpiConfig
    from .problems.epidemiology import runners
    cfg = EpiConfig.load(args.config) if args.config else EpiConfig()
    if args.quick:
        cfg.data.sim_weeks = 156
    out = args.out or cfg.output_dir
    runners.run(cfg, out_dir=out)


def build_parser() -> argparse.ArgumentParser:
    """Build the ``sciml`` argument parser (one subcommand per example).

    Returns
    -------
    argparse.ArgumentParser
        The configured top-level argument parser.
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="Path to a .yaml/.json config")
    common.add_argument("--out", help="Output directory (defaults to the config's)")
    common.add_argument("--quick", action="store_true", help="Tiny smoke-test run")

    p = argparse.ArgumentParser(prog="sciml", description="Scientific ML examples")
    sub = p.add_subparsers(dest="command", required=True)

    ps = sub.add_parser("swe", parents=[common], help="DeepONet: 1D Shallow Water Equations")
    ps.add_argument("--timing", action="store_true")
    ps.set_defaults(func=_cmd_swe)

    pw = sub.add_parser("wave", parents=[common], help="PINN: moving-boundary wave")
    pw.add_argument("--no-lbfgs", action="store_true", help="Skip the L-BFGS phase")
    pw.set_defaults(func=_cmd_wave)

    pd = sub.add_parser("dengue", parents=[common], help="SINDy: dengue beta(t)")
    pd.set_defaults(func=_cmd_dengue)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point: parse ``argv`` and dispatch to the chosen subcommand.

    Parameters
    ----------
    argv : Optional[Sequence[str]]
        Argument vector to parse; uses ``sys.argv`` when ``None``.

    Returns
    -------
    None
    """
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
