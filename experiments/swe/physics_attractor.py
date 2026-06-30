"""Empirical validation of the F=0 attractor (physics-only DeepONet).

    python -m experiments.swe.physics_attractor --steps 5000
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from sciml.core.metrics import rel_l2
from sciml.methods.deeponet.optim import make_optimizer
from sciml.problems.swe import cases as swe_cases
from sciml.problems.swe import runners
from sciml.problems.swe.config import SWEConfig
from sciml.problems.swe.physics import make_pi_step


def main():
    import tensorflow as tf
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--n-coll", type=int, default=500)
    ap.add_argument("--out", default="outputs/swe")
    args = ap.parse_args()
    cfg = SWEConfig.load(args.config) if args.config else SWEConfig()
    prob = runners.prepare_problem(cfg)

    model = prob.build_model("full")
    opt = make_optimizer(cfg.train.lr, cfg.train.lr_decay_steps, cfg.train.lr_decay_rate)
    step = make_pi_step(model, opt, prob.L)

    print(f"Training PI-DeepONet for {args.steps} steps (no data supervision)...")
    nan_step = None
    for it in range(1, args.steps + 1):
        idx = np.random.choice(prob.n_total, cfg.train.batch, replace=False)
        tbc = np.random.uniform(0, prob.T, cfg.train.n_bc).astype(np.float32)
        xc = tf.constant(np.random.uniform(0, prob.L, args.n_coll).astype(np.float32))
        tc = tf.constant(np.random.uniform(0, prob.T, args.n_coll).astype(np.float32))
        loss, Lpde, Lbc, gn = step(
            tf.constant(prob.H0_s[idx]), tf.constant(prob.B_s[idx]),
            tf.constant(prob.H0_grid[idx]), tf.constant(prob.B_grid[idx]),
            tf.constant(tbc), xc, tc)
        if not np.isfinite(float(loss)):
            nan_step = it
            print(f"  Step {it}: NaN/Inf -- gradient explosion confirmed")
            break
        if it % 1000 == 0:
            print(f"  {it:5d} Lpde={float(Lpde):.3e} Lbc={float(Lbc):.3e}")

    xs, _, h, _ = prob.predict_grid(model, swe_cases.h0_gaussian, swe_cases.b_flat)
    x_ref, snaps = prob.reference(swe_cases.h0_gaussian, swe_cases.b_flat)
    eps = rel_l2(h[-1], np.interp(xs, x_ref, snaps["h"][-1]))
    f0_gap = float(np.mean(np.abs(h[-1] - swe_cases.h0_gaussian(xs))))
    print(f"\nPI-DeepONet eps_h={eps:.3e}  F0-gap={f0_gap:.4f} m")
    print("F=0 attractor confirmed." if f0_gap < 0.02 else "Partial collapse.")
    runners.save_results({"eps_h": eps, "f0_gap": f0_gap, "nan_step": nan_step},
                         os.path.join(args.out, "physics_attractor.json"))


if __name__ == "__main__":
    main()
