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

Moved to its own repository, [`wnts-sindyc`](https://github.com/phoenixfin/wnts-sindyc): the data is
confidential, so the study never belonged with the packaged problems. Its
evaluation protocol is the one that became
[`sciml.tasks.sysid`](../src/sciml/tasks/sysid.py).
