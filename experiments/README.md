# Experiment scripts

Runnable studies built on `sciml`. Each package is a thin command-line layer:
the science lives in [`sciml.problems`](../src/sciml/problems) and
[`sciml.methods`](../src/sciml/methods), and the scripts here parse arguments,
call runners, and write artefacts into `outputs/<study>/`.

That is the difference from [`notebooks/`](../notebooks), where the *open*
research happens. A study that settles is packaged into `sciml.problems` and
gets a script here; until then it stays a notebook.

## Shallow water — DeepONet

```bash
python -m experiments.swe.train             --config configs/swe.yaml
python -m experiments.swe.evaluate          --weights outputs/swe/model.weights.h5
python -m experiments.swe.ablation          --steps 10000
python -m experiments.swe.nd_scaling        --nd 10 25 50 100 150 --seeds 5
python -m experiments.swe.physics_attractor --steps 5000
```

`train` fits one architecture variant and saves weights, history and a loss
figure; `evaluate` scores saved weights on benchmarks C1–C3 and on unseen
`(h0, b)` pairs.

> The `ablation`, `nd_scaling` and `physics_attractor` scripts mirror the
> *original* notebook sections. Several of their conclusions do not survive a
> corrected reference solver — read
> [the audit](../notebooks/pi_deeponet_swe/RESULTS.md) before quoting them.

## Moving-boundary wave — PINN, and dengue — SINDy

```bash
python -m experiments.wave_obstacle.run --config configs/wave_obstacle.yaml
python -m experiments.epidemiology.run  --config configs/dengue.yaml
```

Both accept `--config` (YAML or JSON) and fall back to the dataclass defaults;
`wave_obstacle.run --no-lbfgs` stops after the Adam phases.

## Gas transmission network — SINDYc

The one study here that is not a packaged `sciml.problems` example, because its
data is confidential and not in the tree. Its consolidated findings — data
quirks, protocol design, results A1–A4 and B1–B4 — are in
[`wnts/REPORT.md`](wnts/REPORT.md), and its evaluation protocol is the one that
became [`sciml.tasks.sysid`](../src/sciml/tasks/sysid.py).

```bash
python -m experiments.wnts.run              # six-model ladder + baselines + figures
python -m experiments.wnts.multi_year       # A1: per-year + transfer robustness
python -m experiments.wnts.ablation_states  # A3: state-dimension / stability mechanism
python -m experiments.wnts.ablation_library # B1: polynomial vs physics libraries
python -m experiments.wnts.ablation_inputs  # B2: which boundary flows matter
python -m experiments.wnts.sweep_hyper      # B3: threshold/alpha/dt/clip sensitivity
python -m experiments.wnts.benchmark_dmdc   # B4: DMDc null-model comparison
```

Every script starts from `wnts.run.build_parser()` — either parsing the real
command line, or taking `parse_args([])` as a defaults namespace — so the
protocol constants stay identical across the study. Point it at the data with
`--data-dir`, and choose the contract years with `--years` / `--test-years`.

## Conventions

- **Artefacts, not state.** Everything lands under `outputs/` (gitignored):
  figures as `fig_*.png`, metrics as `summary.json` / `<study>.json`.
- **Configs are files.** Anything worth reproducing goes in `configs/` and is
  loaded with `--config`, not edited into the script.
- **Determinism** via `sciml.core.seeding.seed_everything`, and multi-seed
  contrasts wherever a single run could flip a conclusion.
- Docstrings, lint and tests are gated the same way as `src/` — see
  [docs/reference.md](../docs/reference.md).
