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

The design goal: the **method engines and the shared substrate are generic**;
each PDE/system is a problem that plugs in. Adding a new problem (or a new
method) means writing one module, not forking the repo.

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
  cli.py       `sciml {swe,wave,dengue}`
configs/       swe.yaml, wave_obstacle.yaml, dengue.yaml (+ JSON also supported)
experiments/   swe/{train,evaluate,ablation,nd_scaling,physics_attractor}, wave_obstacle/run, epidemiology/run
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
```

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

## License

MIT.
