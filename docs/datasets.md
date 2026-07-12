# Datasets & tasks

How to run **your data** through the framework's methods with one call:
the dataset registry (`sciml.data.datasets`) plus task layers
(`sciml.tasks`). This is the recommended path for any new real-world
dataset — bespoke experiment packages (like `experiments/wnts`) are for
full studies, not for routine "data × method" runs.

---

## The registry

```python
from sciml.data.datasets import load, list_datasets, register

list_datasets()
# {'advection_pairs': 'GP initial conditions through the periodic advection-diffusion operator.',
#  'lti_demo':        'Stable 2-state linear system with 2 control inputs (ground truth known).',
#  'wnts':            'WNTS pipeline telemetry as clean, block-averaged segments.'}

data = load("lti_demo", n_segments=4, seg_len=400)      # kwargs go to the loader
```

Loaders resolve **lazily**: a dataset's module (and its extra dependencies —
e.g. pandas for `wnts`) is only imported when that dataset is actually
loaded, so `import sciml` stays cheap and numpy-only datasets work without
extras.

### Built-in datasets

| Name | Container | Notes |
|---|---|---|
| `lti_demo` | `TimeSeriesData` | 2-state stable LTI system + 2 inputs, ground truth in `meta` — the sanity check for `sysid` |
| `advection_pairs` | `FunctionPairData` | GP initial conditions → exact advection-diffusion solution operator |
| `wnts` | `TimeSeriesData` | Confidential gas-pipeline telemetry; needs `data_dir=` or the `SCIML_WNTS_DIR` env var; pandas required |

## The containers

Two shapes, matching the two method families (deliberately **not** one
universal interface):

**`TimeSeriesData`** — system-identification-shaped (SINDy, DMDc, Neural ODE):

- `segments`: list of `(n_i, d)` numpy arrays — contiguous, uniformly
  sampled stretches (gaps between segments are expected: outages, masked
  bad data);
- `channels`: the `d` column names; `select([...])` returns a channel
  subset; `columns([...])` gives indices;
- `dt_hours`: the sample spacing; `index` (optional): per-segment
  timestamps; `meta`: free-form.

**`FunctionPairData`** — operator-learning-shaped (DeepONet, FNO):

- `u` / `s`: `(n, ...)` arrays of paired input/output functions;
- `u_grid` / `s_grid`: sensor and evaluation locations;
- `split(train_frac, seed)` for random splits.

---

## The `sysid` task

`sciml.tasks.sysid.run` is the full identification-and-evaluation protocol
developed in the WNTS study (`experiments/wnts/REPORT.md` documents the
design rationale and every default):

```python
from sciml.tasks import sysid

res = sysid.run(
    data,                              # a TimeSeriesData
    states=["x1", "x2"],               # channel names used as dynamical states
    inputs=["u1", "u2"],               # exogenous inputs (known at forecast time)
    method="sindyc",                   # "sindyc" | "sindy" (autonomous) | "dmdc"
)
print(res.summary())
res.equations        # human-readable identified dynamics
res.coefficients     # {state: {term: value}} (z-scored fluctuation units)
res.r2_test          # one-step-fit R^2 per state, held-out segments
res.metrics          # {"horizons": {24: {"nrmse", "diverged_frac"}, ...}}
res.baselines        # persistence / climatology / daily-repeat NRMSE per horizon
res.model            # the fitted sciml.methods.sindy.SINDy estimator
```

What happens inside (each step is a keyword you can change):

1. **Split** — `split="chrono"` (test = latest segments, honest
   extrapolation in time), `"interleave"`, or `test_data=` another
   `TimeSeriesData` for cross-period transfer.
2. **Operating point** — `center="causal"` references every signal to a
   trailing mean (`op_window_h=72`) computed from past data only; the
   spin-up is trimmed. `center="segment"` is the oracle variant for
   identification-quality (not forecasting) comparisons.
3. **Scale & saturate** — z-scoring with training statistics; library
   inputs clipped at `clip=3` train-stds so polynomial features stay in
   their calibrated range during excursions.
4. **Fit** — `fit="discrete"` (forward-difference targets, consistent
   forward-Euler rollout — the robust default) or `fit="savgol"`
   (classic continuous-time SINDy). Estimator presets per `method`;
   override with `degree=`, `threshold=`, `alpha=`, or a custom
   `library=` (any `sciml.methods.sindy.FeatureLibrary`).
5. **Evaluate** — multi-IC forecast rollouts at `horizons=(6, ..., 72)` h
   in the frame frozen at each forecast start (causality preserved over
   the horizon), NRMSE normalized by test fluctuation std, divergence
   tracking, and the three trivial baselines on the identical protocol.

CLI equivalent:

```bash
sciml datasets
sciml sysid --data wnts --data-arg years=[2019] \
            --states P_up P_orf --inputs all --method sindyc --out results.json
```

Hard-won defaults (from the WNTS study — see REPORT.md §B3): the discrete
fit at a sampling interval matched to the system's physics matters most;
ridge `alpha` barely matters; `threshold` has a broad optimum; clipping is
a safety net. If rollouts diverge, suspect **state collinearity** before
tuning — collapsing near-duplicate channels into one (as WNTS's `P_up`
does) is what restores stability (REPORT.md §A3).

---

## Adding a dataset

One loader function, registered under a name:

```python
# src/sciml/data/datasets/my_plant.py  (or anywhere in your own code)
from sciml.data.datasets import register, TimeSeriesData

@register("my_plant")                       # or list it in _BUILTIN for lazy loading
def load_my_plant(path: str = "data/plant.csv", dt_hours: float = 1.0) -> TimeSeriesData:
    """One-line description shown by list_datasets()."""
    # 1. read raw source
    # 2. mask bad data (see wnts.py for frozen-telemetry detection)
    # 3. split into contiguous clean segments, optionally block-average
    # 4. optionally add derived channels (wnts.py adds P_up)
    return TimeSeriesData(segments=[...], channels=[...], dt_hours=dt_hours)
```

Checklist, learned the hard way on WNTS:

- **Mask stale telemetry** (long runs of identical values), don't just
  drop NaNs.
- **Segments must be contiguous** — a gap means a new segment, never
  interpolate across outages.
- **Match `dt` to the dynamics** — block-average raw high-rate data;
  fitting at measurement-noise timescales ruins rollouts.
- **Precompute physically-motivated derived channels** in the loader
  (pool means, imbalances) so task calls stay declarative.
- Put provenance and confidentiality notes in `meta`.

For operator-learning data, return a `FunctionPairData` instead; the
corresponding task layer is not built yet — pair it with
`sciml.methods.deeponet` / `sciml.methods.fno` directly for now.
