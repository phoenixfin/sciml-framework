"""Ablation study: A1 (shared branch), A2 (no IC shortcut), A3 (full).

    python -m experiments.swe.ablation --steps 10000
"""

from __future__ import annotations

import argparse
import os
from typing import TYPE_CHECKING

import numpy as np

from sciml.core.metrics import rel_l2
from sciml.methods.deeponet.optim import make_optimizer
from sciml.methods.deeponet.trainer import Trainer
from sciml.problems.swe import cases as swe_cases
from sciml.problems.swe import runners
from sciml.problems.swe.config import SWEConfig

if TYPE_CHECKING:  # annotations only
    import tensorflow as tf

    from sciml.problems.swe.problem import SWEProblem


def _opt(cfg: "SWEConfig"):
    """Build the optimizer from a config's training block.

    Parameters
    ----------
    cfg : SWEConfig
        Configuration whose ``train`` block sets the schedule.

    Returns
    -------
    tf.keras.optimizers.Optimizer
        Adam with the configured exponential decay.
    """
    return make_optimizer(cfg.train.lr, cfg.train.lr_decay_steps, cfg.train.lr_decay_rate)


def beta_norm(model: "tf.keras.Model", h0s: "tf.Tensor", bs: "tf.Tensor") -> float:
    """Mean branch-coefficient norm, averaged over a batch of sensor profiles.

    A collapsed variant drives its coefficients towards zero, so this is the
    quantity that distinguishes "learnt nothing" from "learnt something wrong".

    Parameters
    ----------
    model : tf.keras.Model
        Either the shared-branch variant (which exposes ``b1`` / ``b2``) or the
        full model (which exposes ``h_op`` / ``hu_op``).
    h0s : tf.Tensor
        Initial depth at the sensors, ``(N, m)``.
    bs : tf.Tensor
        Bathymetry at the sensors, ``(N, m)``.

    Returns
    -------
    float
        Mean over the batch of ``||beta||``, averaged over the two heads when
        the model has separate ones.
    """
    import tensorflow as tf
    if hasattr(model, "b1"):
        return float(tf.reduce_mean(tf.norm(model.b1(h0s) + model.b2(bs), axis=1)))
    bh = model.h_op.coefficients([h0s, bs])
    bhu = model.hu_op.coefficients([h0s, bs])
    return float(tf.reduce_mean((tf.norm(bh, axis=1) + tf.norm(bhu, axis=1)) / 2))


def eval_c1(prob: "SWEProblem", model: "tf.keras.Model") -> tuple:
    """Score a variant on benchmark C1 and test whether it collapsed to h0.

    Parameters
    ----------
    prob : SWEProblem
        Problem holding the reference solver and the prediction helpers.
    model : tf.keras.Model
        Trained variant.

    Returns
    -------
    tuple
        ``(eps_h, eps_hu, collapsed)``: the two relative L2 errors at the final
        time, and whether the depth field stayed within 0.02 m of the initial
        condition everywhere — the signature of the F = 0 attractor.
    """
    xs, _, h, hu = prob.predict_grid(model, swe_cases.h0_gaussian, swe_cases.b_flat)
    x_ref, snaps = prob.reference(swe_cases.h0_gaussian, swe_cases.b_flat)
    eh = rel_l2(h[-1], np.interp(xs, x_ref, snaps["h"][-1]))
    ehu = rel_l2(hu[-1], np.interp(xs, x_ref, snaps["hu"][-1]))
    collapsed = float(np.mean(np.abs(h[-1] - swe_cases.h0_gaussian(xs)))) < 0.02
    return eh, ehu, collapsed


def train_variant(prob: "SWEProblem", cfg: "SWEConfig", variant: str, steps: int):
    """Train one architecture variant from scratch.

    Parameters
    ----------
    prob : SWEProblem
        Problem providing the model builder, step function and batch sampler.
    cfg : SWEConfig
        Configuration supplying the optimizer schedule.
    variant : str
        ``"full"``, ``"shared_branch"`` or ``"no_ic_shortcut"``.
    steps : int
        Optimisation steps; matched across variants so the comparison is at a
        common budget.

    Returns
    -------
    tf.keras.Model
        The trained variant.
    """
    model = prob.build_model(variant)
    opt = _opt(cfg)
    step = prob.make_step(model, opt, variant=variant)
    Trainer(model, opt, step, prob.component_names).fit(
        prob.sample_batch, steps, log_every=max(steps // 5, 1), verbose=True)
    return model


def main():
    """Train A1 and A2, score them against A3, and write ``ablation.json``."""
    import tensorflow as tf
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--full-weights", default="outputs/swe/model.weights.h5")
    ap.add_argument("--out", default="outputs/swe")
    args = ap.parse_args()
    cfg = SWEConfig.load(args.config) if args.config else SWEConfig()
    prob = runners.prepare_problem(cfg)

    results = {}
    for name, variant in [("A1_shared_branch", "shared_branch"),
                          ("A2_no_ic_shortcut", "no_ic_shortcut")]:
        print(f"\n== {name} ({args.steps} steps) ==")
        model = train_variant(prob, cfg, variant, args.steps)
        eh, ehu, collapsed = eval_c1(prob, model)
        rng = np.random.choice(prob.n_total, 32, replace=False)
        bn = beta_norm(model, tf.constant(prob.H0_s[rng]), tf.constant(prob.B_s[rng]))
        results[name] = {"eps_h": eh, "eps_hu": ehu, "collapsed": collapsed, "beta_norm": bn}
        print(f"  eps_h={eh:.3e} eps_hu={ehu:.3e} collapsed={collapsed} ||beta||={bn:.4f}")

    print("\n== A3_full_model ==")
    full = prob.build_model("full")
    if os.path.exists(args.full_weights):
        full.load_weights(args.full_weights)
        print(f"  Loaded {args.full_weights}")
    else:
        full = train_variant(prob, cfg, "full", args.steps)
    eh, ehu, collapsed = eval_c1(prob, full)
    rng = np.random.choice(prob.n_total, 32, replace=False)
    bn = beta_norm(full, tf.constant(prob.H0_s[rng]), tf.constant(prob.B_s[rng]))
    results["A3_full_model"] = {"eps_h": eh, "eps_hu": ehu, "collapsed": collapsed, "beta_norm": bn}
    print(f"  eps_h={eh:.3e} eps_hu={ehu:.3e} ||beta||={bn:.4f}")

    print("\n" + "=" * 68)
    print(f"{'Variant':<22}{'eps_h':>12}{'eps_hu':>12}{'||beta||':>12}{'collapsed':>10}")
    for k, r in results.items():
        print(f"{k:<22}{r['eps_h']:>12.3e}{r['eps_hu']:>12.3e}"
              f"{r['beta_norm']:>12.4f}{str(r['collapsed']):>10}")
    runners.save_results(results, os.path.join(args.out, "ablation.json"))


if __name__ == "__main__":
    main()
