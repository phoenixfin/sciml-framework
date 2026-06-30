"""Train + evaluate the moving-boundary wave PINN.

    python -m experiments.wave_obstacle.run --config configs/wave_obstacle.yaml
    python -m experiments.wave_obstacle.run --no-lbfgs        # Adam phases only
"""

from __future__ import annotations

import argparse

from sciml.problems.wave_obstacle import runners
from sciml.problems.wave_obstacle.config import WaveObstacleConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="outputs/wave_obstacle")
    ap.add_argument("--no-lbfgs", action="store_true")
    args = ap.parse_args()
    cfg = WaveObstacleConfig.load(args.config) if args.config else WaveObstacleConfig()
    prob, trainer = runners.train(cfg, lbfgs=not args.no_lbfgs, out_dir=args.out)
    metrics = runners.evaluate(prob, trainer, out_dir=args.out)
    print(f"Done. e_s={metrics['e_s_pct']:.2f}%  e_u={metrics['e_u_pct']:.2f}%  -> {args.out}/")


if __name__ == "__main__":
    main()
