# sciml

[![CI](https://github.com/phoenixfin/sciml-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/phoenixfin/sciml-framework/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9–3.12](https://img.shields.io/badge/python-3.9%E2%80%933.12-blue.svg)](pyproject.toml)
[![Docstrings: NumPy](https://img.shields.io/badge/docstrings-NumPy%20%7C%20100%25-brightgreen.svg)](docs/reference.md)

A small **scientific-machine-learning** research framework. Three method
families share one substrate, each demonstrated on a worked example distilled
from a research notebook:

| Method | Engine | Worked example |
|---|---|---|
| **DeepONet** (operator learning) | `sciml.methods.deeponet` | 1D Shallow Water Equations (`problems.swe`) |
| **PINN** (physics-informed NN) | `sciml.methods.pinn` | moving-boundary wave / obstacle (`problems.wave_obstacle`) |
| **SINDy** (sparse identification) | `sciml.methods.sindy` | dengue β(t) identification (`problems.epidemiology`) |

Additional model engines (generic, no packaged example yet — see `tests/` for usage):

| Method | Engine | Backend |
|---|---|---|
| **FNO** (Fourier Neural Operator, 1D + 2D) | `sciml.methods.fno` | TensorFlow |
| **Neural ODE** (continuous-depth dynamics) | `sciml.methods.neuralode` | TensorFlow |
| **DMD / Koopman** (dynamic mode decomposition) | `sciml.methods.dmd` | pure numpy |

For **real-world data**, a dataset registry and a task layer make
"this dataset x that method" a one-liner (see
[Datasets & tasks](#datasets--tasks-your-data-through-any-method)):

| Layer | Module | What it gives you |
|---|---|---|
| Dataset registry | `sciml.data.datasets` | `load("<name>", **opts)` for any registered dataset |
| System identification | `sciml.tasks.sysid` | SINDy / SINDYc / DMDc + a full forecast-evaluation protocol |

The design goal: the **method engines and the shared substrate are generic**;
each PDE/system is a problem that plugs in, and each dataset is a loader that
registers. Adding a new problem, method or dataset means writing one module,
not forking the repo.

## Documentation

Full docs (with flowcharts) live in [`docs/`](docs/README.md):
[overview](docs/overview.md) · [architecture](docs/architecture.md) ·
[methods](docs/methods.md) · [problems](docs/problems.md) ·
[extending](docs/extending.md) · [reference](docs/reference.md).

---

## Architecture

```
src/sciml/
  core/        config, metrics, plotting, seeding, logging, io, derivatives   (pure numpy)
  data/        gp.py (GP samplers), interp.py                                  (pure numpy)
    datasets/  registry (load/register/list_datasets), containers
               (TimeSeriesData, FunctionPairData), built-ins:
               wnts (confidential, needs pandas), lti_demo, advection_pairs    (numpy; lazy pandas)
  tasks/
    sysid.py   system identification on TimeSeriesData: SINDy/SINDYc/DMDc +
               causal operating point, splits, multi-horizon rollout metrics,
               trivial baselines                                               (pure numpy)
  solvers/     swe_lax_friedrichs, wave_fdm, compartmental                     (pure numpy)
  methods/
    deeponet/  mlp, operator (DeepONetOperator), optim, trainer                (TensorFlow)
    pinn/      layers (Fourier), networks, gradients, training (Adam+L-BFGS),
               sampling (RAR)                                                   (TF + SciPy)
    sindy/     sparse (STRidge), library (Poly/Fourier/Custom), model (SINDy)  (pure numpy)
    fno/       spectral (SpectralConv1D/2D), model (build_fno1d / build_fno2d) (TensorFlow)
    neuralode/ integrators (Euler/RK4 odeint), model (NeuralODE)               (TensorFlow)
    dmd/       dmd (exact DMD / Koopman)                                       (pure numpy)
  problems/
    swe/            DeepONet on the SWE         (config, cases, model, problem, runners)
    wave_obstacle/  PINN on a moving boundary   (config, problem, runners)
    epidemiology/   SINDy on dengue β(t)        (config, reconstruction, estimators, problem, runners)
  cli.py       `sciml {swe,wave,dengue,datasets,sysid}`
configs/       swe.yaml, wave_obstacle.yaml, dengue.yaml (+ JSON also supported)
experiments/   swe/{train,evaluate,ablation,nd_scaling,physics_attractor}, wave_obstacle/run,
               epidemiology/run, wnts/ (gas-network SINDYc study -- see its REPORT.md)
notebooks/
  pi_deeponet_swe/  audit of the SWE study: one combined Kaggle notebook, a
                    well-balanced HLL solver, a port of the v6 pipeline, and
                    RESULTS.md -- see below                     (numpy + TensorFlow)
tests/         numpy tests (always run) + TF-guarded tests (skip without TF)
```

**Backend-light core.** `core`, `data`, `solvers`, all three example *configs*,
and the entire **SINDy** path are pure numpy — they import and run without a
deep-learning backend. TensorFlow (DeepONet, PINN) and SciPy (PINN L-BFGS) are
optional extras, imported lazily so `import sciml` stays cheap.

---

## Install

> **Python version.** TensorFlow ships wheels for CPython **3.9–3.12**. Use one
> of those for the neural methods. The numpy-only utilities and the SINDy
> example run on any supported Python (incl. 3.13/3.14).

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows (POSIX: source .venv/bin/activate)

pip install -e ".[sindy]"      # SINDy example: numpy + scikit-learn + pandas
pip install -e ".[deeponet]"   # DeepONet: + tensorflow
pip install -e ".[pinn]"       # PINN: + tensorflow + scipy
pip install -e ".[all]"        # everything + pyyaml + pytest
pip install -e .               # bare core only (no examples that need a backend)
```

---

## Quickstart

### CLI — one subcommand per example

```bash
sciml dengue                                    # SINDy: simulates data, runs out of the box
sciml swe    --quick                            # DeepONet: tiny smoke run (needs tensorflow)
sciml wave   --quick                            # PINN: short Adam-only run (needs tensorflow)

sciml swe  --config configs/swe.yaml  --timing
sciml wave --config configs/wave_obstacle.yaml
sciml dengue --config configs/dengue.yaml
```

Each writes figures (`fig_*.png`) and JSON result tables into `outputs/<example>/`.

### Python API

```python
# DeepONet / SWE
from sciml.problems.swe.config import SWEConfig
from sciml.problems.swe import runners
model, history, prob = runners.train(SWEConfig(), weights_path="outputs/swe/m.weights.h5")
runners.evaluate_cases(prob, model)           # C1, C2, C3
runners.generalization(prob, model)           # unseen pairs

# PINN / moving-boundary wave
from sciml.problems.wave_obstacle import runners as wave_runners
prob, trainer = wave_runners.train(lbfgs=False)
print(wave_runners.evaluate(prob, trainer))   # e_s, e_u, amplitude/frequency recovery

# SINDy / dengue β(t)
from sciml.problems.epidemiology import runners as epi_runners
epi_runners.run()                             # simulate -> reconstruct S -> identify β(t)
```

### Reusable engine pieces

```python
from sciml.methods.deeponet import DeepONetOperator          # generic operator net
from sciml.methods.pinn import build_mlp, derivatives_2d, PINNTrainer
from sciml.methods.sindy import SINDy, PolynomialLibrary, stridge
from sciml.methods.fno import build_fno1d                     # Fourier Neural Operator
from sciml.methods.neuralode import NeuralODE, build_odefunc  # continuous-depth dynamics
from sciml.methods.dmd import DMD                             # dynamic mode decomposition

# SINDy: identify x' = -0.5 x from data
import numpy as np
t = np.linspace(0, 10, 500); x = np.exp(-0.5*t)[:, None]
model = SINDy(PolynomialLibrary(degree=1), threshold=0.05).fit(x, t=t, input_names=["x"])
print(model.equations(["dx/dt"]))             # -> dx/dt = -0.5000 x

# DMD: extract spatial modes + temporal eigenvalues from snapshots (pure numpy)
X = np.random.rand(64, 100)                   # (n_features, n_time)
dmd = DMD(rank=8).fit(X, dt=0.1)
recon = dmd.reconstruct(X.shape[1])           # dmd.eigenvalues / .omega / .modes
```

---

## Datasets & tasks — your data through any method

Real datasets register once, then pair with any suitable method through a
task layer that fixes the evaluation protocol (so results are comparable
across datasets and methods).

```python
from sciml.data.datasets import load, list_datasets
from sciml.tasks import sysid

print(list_datasets())          # {'advection_pairs': ..., 'lti_demo': ..., 'wnts': ...}

# system identification: states + exogenous inputs -> sparse dynamics + forecast skill
data = load("lti_demo")                                   # or "wnts", or your own
res = sysid.run(data, states=["x1", "x2"], inputs=["u1", "u2"], method="sindyc")
print(res.summary())            # identified equations, R^2, NRMSE vs baselines per horizon
print(res.equations)            # e.g. d/dt x1 = -0.157 x1 +0.068 x2 +0.155 u1
```

Or from the shell:

```bash
sciml datasets                                           # list what's registered
sciml sysid --data lti_demo --states x1 x2 --inputs all  # full protocol, one line
sciml sysid --data wnts --data-arg years=[2019] \
            --states P_up P_orf --inputs all --method sindyc --out results.json
```

The `sysid` protocol is the one developed in the WNTS gas-network study
(`experiments/wnts/REPORT.md` — the study is also the design rationale):
**causal** trailing operating point (no future information), discrete-time
fitting with the consistent Euler rollout, chronological or transfer splits,
multi-horizon forecast NRMSE against persistence / climatology / daily-repeat
baselines, and divergence tracking. Methods: `sindyc` (sparse, with inputs),
`sindy` (sparse, autonomous), `dmdc` (dense linear least squares — the
natural null model).

Containers (`sciml.data.datasets`):

- `TimeSeriesData` — named channels, contiguous segments, uniform `dt`
  (system-identification-shaped; pure numpy).
- `FunctionPairData` — paired input/output functions on grids
  (operator-learning-shaped, for DeepONet/FNO; task layer TBD).

### Adding your own dataset

```python
from sciml.data.datasets import register, TimeSeriesData

@register("my_plant")
def load_my_plant(path: str = "data/plant.csv") -> TimeSeriesData:
    """One-line description shown by list_datasets()."""
    segments, channels = ...   # read, clean, split into contiguous arrays (n_i, d)
    return TimeSeriesData(segments=segments, channels=channels, dt_hours=1.0)
```

That's the whole integration: `load("my_plant")` and every task, metric and
baseline works immediately. Built-in loaders live in
`src/sciml/data/datasets/` (`wnts.py` is the reference for a real, messy
dataset: frozen-telemetry masking, segment extraction, block-averaging,
derived channels).

---

## Experiment scripts

```bash
# DeepONet / SWE (mirror the original notebook sections)
python -m experiments.swe.train             --config configs/swe.yaml
python -m experiments.swe.evaluate          --weights outputs/swe/model.weights.h5
python -m experiments.swe.ablation          --steps 10000
python -m experiments.swe.nd_scaling        --nd 10 25 50 100 150 --seeds 5
python -m experiments.swe.physics_attractor --steps 5000

# PINN / wave-obstacle, SINDy / dengue
python -m experiments.wave_obstacle.run     --config configs/wave_obstacle.yaml
python -m experiments.epidemiology.run      --config configs/dengue.yaml

# SINDYc / WNTS gas network (confidential data; see experiments/wnts/REPORT.md)
python -m experiments.wnts.run              # six-model ladder + baselines + figures
python -m experiments.wnts.multi_year       # A1: per-year + transfer robustness
python -m experiments.wnts.ablation_states  # A3: state-dimension / stability mechanism
python -m experiments.wnts.ablation_library # B1: polynomial vs physics libraries
python -m experiments.wnts.ablation_inputs  # B2: which boundary flows matter
python -m experiments.wnts.sweep_hyper      # B3: threshold/alpha/dt/clip sensitivity
python -m experiments.wnts.benchmark_dmdc   # B4: DMDc null-model comparison
```

The WNTS study's consolidated findings (data quirks, protocol design,
results A1–A4 and B1–B4, and the remaining experiment plan) are in
[`experiments/wnts/REPORT.md`](experiments/wnts/REPORT.md).

> The `experiments/swe/{ablation,nd_scaling,physics_attractor}` scripts mirror the
> *original* notebook sections. Several of their conclusions do not survive a
> corrected reference solver — see the audit below before quoting them.

---

## The SWE study — reference-solver audit

The DeepONet/SWE example is a refactor of `pi_deeponet_swe_v6`. That notebook was
audited end to end — reference solver, theory, metrics, ablations — and the
findings are consolidated in
[`notebooks/pi_deeponet_swe/RESULTS.md`](notebooks/pi_deeponet_swe/RESULTS.md).
Every number there is generated from one unattended run whose raw record
(`results_2026-08-12.json`) sits beside it, and the run itself is reproducible
from a single notebook.

> ### Caveat for anyone using `solvers/swe_lax_friedrichs`
>
> At the settings the example ships with (`nx=400`, `nt=4000`) the **measured CFL
> is 0.038**, not the ~0.45 one might assume — and Lax-Friedrichs viscosity
> *grows* as Δt falls at fixed Δx, so a "safe" small timestep makes it worse.
> Against a converged reference the resulting field is **6.4 × 10⁻²** relative,
> which is **84% of the wave anomaly**, and it flattens peak-to-peak amplitude to
> 0.071 m against a true 0.237 m.
>
> A well-balanced HLL solver (Audusse hydrostatic reconstruction, minmod-MUSCL,
> SSP-RK2) reaching **8.9 × 10⁻³** on the same grid — and exact to machine
> precision on lake-at-rest — is in
> [`notebooks/pi_deeponet_swe/swe_solvers.py`](notebooks/pi_deeponet_swe/swe_solvers.py).
> Prefer it whenever the reference error competes with what you are measuring.

### Headline findings

| | |
|---|---|
| **Reference error dominated the budget** | The training targets were `6.4e-2` relative against a converged solution, versus `8.9e-3` for a well-balanced scheme on the same grid. Operator errors measured against them were flattered accordingly. |
| **Benchmark C1 is not smooth** | It develops a shock at **t ≈ 0.78 s**: `max|∂h/∂x|` doubles at every refinement (2.14 → 30.66 from nx=400 to 6400) instead of saturating. |
| **The PI failure was a property of the residual** | The implemented momentum residual is `∂ₜ(hu)` alone, with no flux divergence or bed source. `F = 0` is its **exact global minimum**, so the reported collapse is guaranteed by construction. With the full residual, training leaves for the lake-at-rest manifold instead. |
| **The stationarity claim is half true** | The trunk gradient and the **mass** residual vanish identically at `F = 0`; the branch gradient does not, unless the state is already lake-at-rest. |
| **The IC shortcut had two defects** | It was off by ε at t=0, and its positivity bound was false (depth reaches **−0.95 m** under stress). Moving ε inside the ELU fixes exactness at **no measured cost** (0.99–1.03× on unseen pairs); multiplicative alternatives cost **2.4×**. |
| **The t=1 s oscillations are Gibbs ringing** | Across the shock, error concentration rises `1.7 → 3.0` and the high-wavenumber share of error power rises `0.11 → 0.47`. Finite trunk resolution would have raised both at *all* times. |
| **Half the architecture ablation survives** | At a matched 40k budget the IC shortcut is worth **1.6×**, but a shared branch is indistinguishable from separate branch pairs, and branch fusion is a **null over five seeds**. |
| **The operator is resolution-free** | ε_h varies by **4.5%** across a 32× range of query grids. Extrapolation past the training horizon is useful for about **10%** of it. |
| **Honest speedup** | **2297×** against a serial reference solver, **356×** against a vectorised one. |

### Method notes worth reusing

- **Single-seed ablations flipped their winners** across runs on identical
  settings. The study reports paired-by-seed contrasts and a
  difference-in-differences against a no-coupling control instead of a ranking.
- **Reference-data error belongs in the error budget.** Quoting an operator error
  without auditing the solver that produced its targets can be off by more than
  the effect under study.
- **Normalisation matters**: `‖h‖`-relative error is flattered by the constant
  background depth by 10–22× depending on the snapshot. The study reports
  anomaly-relative error and dimensional RMSE alongside it.

---

## Adding your own problem

1. Drop a reference solver in `solvers/` (pure numpy).
2. Write `problems/<name>/config.py` (compose `core.config.ConfigBase` dataclasses).
3. Write `problems/<name>/problem.py` wiring the solver + a method engine
   (`methods.deeponet` / `methods.pinn` / `methods.sindy`).
4. Add a thin `runners.py` and an `experiments/<name>/run.py`.

The method engines and `core`/`data`/`solvers` are reused as-is.

---

## Tests

```bash
pip install -e ".[dev]"
pytest        # numpy + SINDy tests always run; TF tests skip when tensorflow is absent
```

## Provenance

The three examples are refactors of research notebooks:
`pi_deeponet_swe_v6` (DeepONet/SWE), `pinn_string_obstacle_original_v4`
(PINN/wave), and `dengue_beta_estimation` (SINDy/epidemiology).

`pi_deeponet_swe_v6.ipynb` is no longer in the tree — its pipeline is ported to
[`notebooks/pi_deeponet_swe/pi_deeponet_v6.py`](notebooks/pi_deeponet_swe/pi_deeponet_v6.py)
and audited in [`RESULTS.md`](notebooks/pi_deeponet_swe/RESULTS.md). To check the
port against the original:

```bash
git show ff45b6b^:notebooks/pi_deeponet_swe/pi_deeponet_swe_v6.ipynb > v6.ipynb
```

## License

MIT.
