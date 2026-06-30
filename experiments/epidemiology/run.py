"""Run the dengue beta(t) SINDy pipeline (simulate or real data).

    python -m experiments.epidemiology.run --config configs/dengue.yaml
"""

from __future__ import annotations

import argparse

from sciml.problems.epidemiology import runners
from sciml.problems.epidemiology.config import EpiConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = EpiConfig.load(args.config) if args.config else EpiConfig()
    out = runners.run(cfg, out_dir=args.out)
    print("Done.", out["summary"].get("rmse", {}))


if __name__ == "__main__":
    main()
