"""Evaluate a trained SWE DeepONet: cases C1-C3 + generalization.

    python -m experiments.swe.evaluate --weights outputs/swe/model.weights.h5
"""

from __future__ import annotations

import argparse
import os

from sciml.problems.swe import runners
from sciml.problems.swe.config import SWEConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--weights", default="outputs/swe/model.weights.h5")
    ap.add_argument("--out", default="outputs/swe")
    args = ap.parse_args()
    cfg = SWEConfig.load(args.config) if args.config else SWEConfig()
    prob = runners.prepare_problem(cfg)
    model = prob.build_model("full")
    if os.path.exists(args.weights):
        model.load_weights(args.weights)
        print(f"Loaded weights: {args.weights}")
    else:
        print(f"WARNING: {args.weights} not found; evaluating an untrained model.")
    results = {"cases": runners.evaluate_cases(prob, model, out_dir=args.out),
               "generalization": runners.generalization(prob, model, out_dir=args.out)}
    runners.save_results(results, os.path.join(args.out, "evaluation.json"))
    print(f"Done. Results in {args.out}/evaluation.json")


if __name__ == "__main__":
    main()
