"""Command-line interface: worked examples plus generic dataset tasks.

    sciml swe      [--quick] [--config C] [--out D] [--timing]   # DeepONet / SWE
    sciml wave     [--quick] [--config C] [--out D] [--no-lbfgs] # PINN / wave-obstacle
    sciml dengue   [--quick] [--config C] [--out D]              # SINDy / dengue beta(t)
    sciml datasets                                               # list registered datasets
    sciml sysid --data NAME --states S1 S2 [--inputs all|U1 ...] # system identification
               [--data-arg k=v ...] [--method sindyc|sindy|dmdc] [--out results.json]

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
    from .problems.swe import runners
    from .problems.swe.config import SWEConfig
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
    from .problems.wave_obstacle import runners
    from .problems.wave_obstacle.config import WaveObstacleConfig
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
    from .problems.epidemiology import runners
    from .problems.epidemiology.config import EpiConfig
    cfg = EpiConfig.load(args.config) if args.config else EpiConfig()
    if args.quick:
        cfg.data.sim_weeks = 156
    out = args.out or cfg.output_dir
    runners.run(cfg, out_dir=out)


def _parse_kv(pairs: Optional[Sequence[str]]) -> dict:
    """Parse repeated ``key=value`` options with Python-literal values.

    Parameters
    ----------
    pairs : Optional[Sequence[str]]
        Strings of the form ``key=value``; values are parsed with
        ``ast.literal_eval`` and fall back to the raw string.

    Returns
    -------
    dict
        The parsed keyword arguments.
    """
    import ast
    out = {}
    for p in pairs or []:
        k, _, v = p.partition("=")
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def _cmd_datasets(args: argparse.Namespace) -> None:
    """List all registered datasets with a one-line description.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments (unused).

    Returns
    -------
    None
    """
    from .data.datasets import list_datasets
    for name, desc in list_datasets().items():
        print(f"{name:18s} {desc}")


def _cmd_sysid(args: argparse.Namespace) -> None:
    """Run the system-identification task on a registered dataset.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments for the ``sysid`` subcommand.

    Returns
    -------
    None
    """
    from .data.datasets import load
    from .tasks import sysid
    data = load(args.data, **_parse_kv(args.data_arg))
    inputs = args.inputs or []
    if inputs == ["all"]:
        inputs = [c for c in data.channels if c not in args.states]
    res = sysid.run(data, states=args.states, inputs=inputs, method=args.method,
                    center=args.center, op_window_h=args.op_window,
                    threshold=args.threshold, alpha=args.alpha, degree=args.degree)
    print(res.summary())
    if args.out:
        import json
        payload = {"states": res.states, "inputs": res.inputs,
                   "equations": res.equations, "coefficients": res.coefficients,
                   "r2_train": res.r2_train, "r2_test": res.r2_test,
                   "metrics": res.metrics,
                   "baselines": {b: {str(h): v for h, v in m.items()}
                                 for b, m in res.baselines.items()},
                   "details": res.details}
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Saved {args.out}")


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

    pl = sub.add_parser("datasets", help="List registered datasets")
    pl.set_defaults(func=_cmd_datasets)

    py_ = sub.add_parser("sysid", help="System identification on a registered dataset")
    py_.add_argument("--data", required=True, help="Registered dataset name")
    py_.add_argument("--data-arg", action="append", metavar="K=V",
                     help="Loader option (repeatable), e.g. --data-arg years=[2019]")
    py_.add_argument("--states", nargs="+", required=True, help="State channel names")
    py_.add_argument("--inputs", nargs="+", default=[],
                     help="Input channel names, or 'all' for every non-state channel")
    py_.add_argument("--method", choices=["sindyc", "sindy", "dmdc"], default="sindyc")
    py_.add_argument("--center", choices=["causal", "segment"], default="causal")
    py_.add_argument("--op-window", type=float, default=72.0)
    py_.add_argument("--threshold", type=float, default=None)
    py_.add_argument("--alpha", type=float, default=None)
    py_.add_argument("--degree", type=int, default=None)
    py_.add_argument("--out", help="Optional path for a JSON result file")
    py_.set_defaults(func=_cmd_sysid)
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
