"""Train the SWE DeepONet.

    python -m experiments.swe.train --config configs/swe.yaml --out outputs/swe
"""

from __future__ import annotations

import argparse
import os

from sciml.core.plotting import set_paper_style
from sciml.problems.swe import runners
from sciml.problems.swe.config import SWEConfig


def plot_history(history, out_dir):
    import matplotlib.pyplot as plt
    set_paper_style()
    h = history.to_dict()
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
    axes[0].semilogy(h["iter"], h["Ld"], "b-", lw=1.2, label="Data loss")
    axes[0].semilogy(h["iter"], [max(v, 1e-12) for v in h["Lb"]], "g-", lw=1.2, label="BC loss")
    axes[0].set(xlabel="Iteration", ylabel="Loss", title="Loss components"); axes[0].legend()
    axes[1].plot(h["iter"], h["gnorm"], "k-", lw=0.9)
    axes[1].set(xlabel="Iteration", ylabel="Norm", title="Gradient norm")
    axes[2].semilogy(h["iter"], h["loss"], "k-", lw=1.2)
    axes[2].set(xlabel="Iteration", ylabel="Total", title="Total loss")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_losses.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="outputs/swe")
    ap.add_argument("--variant", default="full",
                    choices=["full", "shared_branch", "no_ic_shortcut"])
    args = ap.parse_args()
    cfg = SWEConfig.load(args.config) if args.config else SWEConfig()
    os.makedirs(args.out, exist_ok=True)
    model, history, prob = runners.train(
        cfg, variant=args.variant, ckpt_dir=os.path.join(args.out, "ckpt"),
        weights_path=os.path.join(args.out, "model.weights.h5"))
    plot_history(history, args.out)
    runners.save_results(history.to_dict(), os.path.join(args.out, "history.json"))
    print(f"Done. Artifacts in {args.out}/")


if __name__ == "__main__":
    main()
