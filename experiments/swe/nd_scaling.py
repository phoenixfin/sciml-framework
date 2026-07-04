"""Error vs. supervised sample count (multi-seed).

    python -m experiments.swe.nd_scaling --nd 10 25 50 100 150 --seeds 5 --steps 15000
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from sciml.core.metrics import rel_l2
from sciml.core.plotting import set_paper_style
from sciml.core.seeding import seed_everything
from sciml.methods.deeponet.optim import make_optimizer
from sciml.methods.deeponet.trainer import Trainer
from sciml.problems.swe import runners
from sciml.problems.swe.config import SWEConfig


def evaluate_pairs(prob, model, H0_test, B_test, n_pairs=25):
    eh, ehu, xq = [], [], np.linspace(0, prob.L, 100)
    for k in range(n_pairs):
        h0k = lambda x, k=k: np.interp(x, prob.x_sensors, H0_test[k])
        bk = lambda x, k=k: np.interp(x, prob.x_sensors, B_test[k])
        _, _, h_pk, hu_pk = prob.predict_grid(model, h0k, bk, nx=100, nt=50)
        xrk, snk = prob.reference(h0k, bk, nx=300, nt=3000)
        eh.append(rel_l2(h_pk[-1], np.interp(xq, xrk, snk["h"][-1])))
        ehu.append(rel_l2(hu_pk[-1], np.interp(xq, xrk, snk["hu"][-1])))
    return float(np.mean(eh)), float(np.mean(ehu))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--nd", type=int, nargs="+", default=[10, 25, 50, 100, 150])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--out", default="outputs/swe")
    args = ap.parse_args()
    cfg = SWEConfig.load(args.config) if args.config else SWEConfig()
    prob = runners.prepare_problem(cfg)

    np.random.seed(0)
    H0_test = prob.h0_sampler.sample(prob.x_sensors, 25)
    B_test = prob.bath_sampler.sample(prob.x_sensors, 25)

    results = {nd: {"eps_h": [], "eps_hu": []} for nd in args.nd}
    for nd in args.nd:
        print(f"\n== Nd={nd} ({args.seeds} seeds) ==")
        prob.generate_dataset(n_data_gp=nd, verbose=False)
        for seed in range(args.seeds):
            seed_everything(seed)
            model = prob.build_model("full")
            opt = make_optimizer(cfg.train.lr, cfg.train.lr_decay_steps, cfg.train.lr_decay_rate)
            step = prob.make_step(model, opt, variant="full")
            Trainer(model, opt, step, prob.component_names).fit(
                prob.sample_batch, args.steps, log_every=args.steps, verbose=False)
            eh, ehu = evaluate_pairs(prob, model, H0_test, B_test)
            results[nd]["eps_h"].append(eh)
            results[nd]["eps_hu"].append(ehu)
            print(f"  seed {seed}: eps_h={eh:.3e} eps_hu={ehu:.3e}")

    set_paper_style()
    import matplotlib.pyplot as plt
    nd_vals = sorted(results)
    eh_m = [np.mean(results[n]["eps_h"]) for n in nd_vals]
    eh_s = [np.std(results[n]["eps_h"]) for n in nd_vals]
    ehu_m = [np.mean(results[n]["eps_hu"]) for n in nd_vals]
    ehu_s = [np.std(results[n]["eps_hu"]) for n in nd_vals]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.errorbar(nd_vals, eh_m, yerr=eh_s, fmt="b-o", lw=1.4, ms=5, capsize=3,
                label=r"$\bar\varepsilon_h$")
    ax.errorbar(nd_vals, ehu_m, yerr=ehu_s, fmt="r-s", lw=1.4, ms=5, capsize=3,
                label=r"$\bar\varepsilon_{hu}$")
    ax.set_yscale("log")
    ax.set(xlabel="Supervised samples $N_d$", ylabel="Mean rel. $L^2$ error",
           title="Error vs. supervised sample count")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3, lw=0.5)
    os.makedirs(args.out, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "fig_nd_scaling.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    runners.save_results({str(k): v for k, v in results.items()},
                         os.path.join(args.out, "nd_scaling.json"))
    print(f"\nDone. Artifacts in {args.out}/")


if __name__ == "__main__":
    main()
