# Configs

One file per packaged problem, holding the settings that reproduce its study.
Anything worth reproducing belongs here rather than edited into a script.

| File | Problem | Reproduces |
|---|---|---|
| [`swe.yaml`](swe.yaml) | `sciml.problems.swe` — DeepONet on the 1D shallow-water equations | the `pi_deeponet_swe_v6` notebook |
| [`wave_obstacle.yaml`](wave_obstacle.yaml) | `sciml.problems.wave_obstacle` — PINN on a moving boundary | `pinn_string_obstacle_original_v4` |
| [`dengue.yaml`](dengue.yaml) | `sciml.problems.epidemiology` — SINDy identification of β(t) | `dengue_beta_estimation` |

## Using one

```bash
sciml swe    --config configs/swe.yaml --timing
sciml wave   --config configs/wave_obstacle.yaml
sciml dengue --config configs/dengue.yaml

python -m experiments.swe.train --config configs/swe.yaml
```

```python
from sciml.problems.swe.config import SWEConfig

cfg = SWEConfig.load("configs/swe.yaml")   # or SWEConfig() for the same defaults
cfg.train.n_iter = 2000                    # ordinary attribute assignment
cfg.save("outputs/swe/config-used.json")   # record what actually ran
```

## How they work

Each file mirrors a tree of `ConfigBase` dataclasses (`domain`, `model`,
`data`, `train`, …) defined in `problems/<name>/config.py`, which is also where
every field is documented — with an inline `#:` comment giving its meaning and
units. The file and the dataclass defaults say the same thing, so a config that
is absent is not a config that is unset.

- **YAML or JSON.** `.yaml` needs `pip install -e ".[yaml]"`; `.json` works with
  no extra dependency and is what `cfg.save()` writes.
- **Partial files are fine.** Anything you leave out keeps its dataclass
  default; only the blocks you name are overridden.
- **Unknown keys fail loudly.** A typo raises `TypeError` at load rather than
  being silently ignored — worth knowing before a 40,000-step run.
- **Seeds live here too** (`train.seed`), and are applied through
  `sciml.core.seeding.seed_everything`.

Adding a new problem means adding a `config.py` and, if it has a study worth
reproducing, a file here. See [docs/extending.md](../docs/extending.md).
