# Architecture

## Layered structure

sciml is organized as a **shared substrate**, a set of **method engines** on top
of it, **problems** that wire an engine to a specific system, and thin
**interfaces** (CLI, experiment scripts, examples) on top of those.

```mermaid
graph TD
  subgraph Substrate["Shared substrate — pure numpy / pandas"]
    core["core/<br/>config · metrics · plotting<br/>seeding · logging · io · derivatives"]
    data["data/<br/>GP samplers · interpolation"]
    solvers["solvers/<br/>reference numerical solvers"]
  end
  subgraph Engines["methods/ — method engines"]
    deeponet["deeponet"]
    fno["fno"]
    pinn["pinn"]
    neuralode["neuralode"]
    sindy["sindy"]
    dmd["dmd"]
  end
  subgraph Problems["problems/ — worked problems"]
    swe["swe → DeepONet"]
    wave["wave_obstacle → PINN"]
    epi["epidemiology → SINDy"]
  end
  Interfaces["cli.py&nbsp;·&nbsp;experiments/&nbsp;·&nbsp;examples/"]

  Engines --> core
  solvers --> core
  data --> core
  Problems --> Engines
  Problems --> solvers
  Problems --> data
  Interfaces --> Problems
  Interfaces --> Engines
  Interfaces --> solvers
```

**Dependency rule:** arrows point *downward only*. Problems depend on engines and
the substrate; engines depend on the substrate; the substrate depends on nothing
internal. This keeps the core reusable and the coupling one-directional.

## Directory map

```
src/sciml/
├── __init__.py            # version only; imports nothing heavy
├── cli.py                 # `sciml {swe,wave,dengue}` entry point
├── tf_utils.py            # TF-only helpers (grid_interp, grad norm)
│
├── core/                  # ── pure numpy substrate ──
│   ├── config.py          # ConfigBase mixin (dict/JSON/YAML) + DomainConfig
│   ├── metrics.py         # rel_l2, rmse, abs_error, …
│   ├── derivatives.py     # Savitzky–Golay smoothing / differentiation
│   ├── plotting.py        # matplotlib paper style
│   ├── seeding.py         # seed_everything (numpy + optional TF)
│   ├── logging.py         # get_logger
│   └── io.py              # save_json / load_json
│
├── data/                  # gp.py (GP samplers), interp.py
│
├── solvers/               # ── reference solvers (numpy) ──
│   ├── swe_lax_friedrichs.py   wave_fdm.py        compartmental.py
│   ├── dynamical.py            heat.py            burgers.py
│   ├── transport.py            wave1d.py
│   ├── kuramoto_sivashinsky.py darcy.py
│
├── methods/               # ── engines ──
│   ├── deeponet/   mlp · operator (DeepONetOperator) · optim · trainer   (TF)
│   ├── fno/        spectral (Conv1D/2D) · model (build_fno1d/2d)         (TF)
│   ├── pinn/       layers · networks · gradients · training · sampling   (TF/SciPy)
│   ├── neuralode/  integrators (odeint) · model (NeuralODE)              (TF)
│   ├── sindy/      sparse (STRidge) · library · model (SINDy)            (numpy)
│   └── dmd/        dmd (exact DMD / Koopman)                             (numpy)
│
└── problems/              # ── worked problems ──
    ├── base.py            # Problem ABC
    ├── swe/               # config · cases · model · problem · physics · runners
    ├── wave_obstacle/     # config · problem · runners
    └── epidemiology/      # config · reconstruction · estimators · problem · runners

configs/       swe.yaml · wave_obstacle.yaml · dengue.yaml (+ JSON)
experiments/   swe/{train,evaluate,ablation,nd_scaling,physics_attractor} · wave_obstacle · epidemiology
examples/      01–13 graded gallery
tests/         numpy tests (always run) + TF-guarded tests
docs/          this documentation
```

## Backend strategy

Only some layers need a deep-learning backend. The split is deliberate so the
core is testable and importable anywhere.

```mermaid
graph LR
  subgraph numpy["Pure numpy / pandas — always available"]
    core2["core · data · solvers"]
    sindy2["methods/sindy"]
    dmd2["methods/dmd"]
    cfg["all problem configs"]
  end
  subgraph tf["Needs TensorFlow"]
    dn["methods/deeponet"]
    fn["methods/fno"]
    nd["methods/neuralode"]
  end
  subgraph tfsp["Needs TensorFlow + SciPy"]
    pn["methods/pinn (L-BFGS)"]
  end
  subgraph skl["Needs scikit-learn / pandas"]
    epi2["epidemiology (LASSO / xlsx)"]
  end
```

Install only what you use: `pip install -e ".[sindy]"`, `".[deeponet]"`,
`".[pinn]"`, or `".[all]"`. `import sciml` never imports TensorFlow.

## Data flow — operator learning (DeepONet / FNO)

Learn a mapping between functions: sample input functions, solve the PDE to get
targets, train the operator, evaluate.

```mermaid
flowchart LR
  A["GP sampler<br/>data/gp.py"] -->|"input functions u₀"| B["reference solver<br/>solvers/*.py (numpy)"]
  B -->|"targets u(·,T)"| C["dataset<br/>(input, output) pairs"]
  A --> C
  C --> D["engine<br/>DeepONet / FNO"]
  D --> E["train<br/>Adam + MSE"]
  E --> F["evaluate<br/>rel L2 · figures"]
```

## Data flow — equation discovery (SINDy)

No neural network: estimate derivatives, build a candidate-term library, solve a
sparse regression, read off the equations.

```mermaid
flowchart LR
  A["trajectory X(t)<br/>(data or solver)"] --> B["derivatives Ẋ<br/>core/derivatives.py"]
  A --> C["feature library Θ(X)<br/>methods/sindy/library.py"]
  B --> D["sparse regression<br/>STRidge"]
  C --> D
  D --> E["coefficients Ξ"]
  E --> F["symbolic equations<br/>model.equations()"]
```

## Data flow — modal analysis (DMD)

```mermaid
flowchart LR
  A["snapshots X<br/>(features × time)"] --> B["SVD + rank truncation"]
  B --> C["reduced operator Ã"]
  C --> D["eigen-decomposition"]
  D --> E["modes Φ · eigenvalues λ<br/>ω = log(λ)/dt"]
  E --> F["reconstruct / predict"]
```

## Training — the generic loop (DeepONet)

The `Trainer` is problem-agnostic: a problem supplies a compiled `step_fn` and a
`sample_batch(iteration)` producer; the Trainer runs the loop, records history,
and checkpoints.

```mermaid
flowchart TD
  P["Problem.make_step(model, opt)"] --> S["step_fn(*batch)<br/>@tf.function"]
  P2["Problem.sample_batch(it)"] --> B["batch tensors"]
  S --> T{"Trainer.fit"}
  B --> T
  T -->|"every step"| S
  T -->|"log_every"| L["History + log"]
  T -->|"ckpt_every"| K["checkpoint"]
  T --> H["return History"]
```

## Training — PINN multi-phase schedule

The PINN trainer drives Adam phases (with a causal-weight anneal and optional
adaptive resampling) followed by SciPy L-BFGS refinement.

```mermaid
flowchart LR
  L["loss_fn()<br/>PDE + BC + IC (+ mask)"] --> A1["Adam phase 1a…2c<br/>anneal ε_causal, RAR"]
  A1 --> B1["L-BFGS<br/>(SciPy)"]
  B1 --> B2["L-BFGS restart<br/>from best weights"]
  B2 --> E["evaluate<br/>e_s, e_u"]
```

## The `Problem` contract

A problem couples a reference solver + data to a method. `problems/base.py`
fixes only the minimum (hold a config, produce a `reference()`); each concrete
problem adds method-specific responsibilities — e.g. `SWEProblem` (DeepONet)
exposes `build_model` / `make_step` / `sample_batch`, while `EpiProblem`
(SINDy) exposes `reconstruct` / `estimate`.

```mermaid
classDiagram
  class Problem {
    <<abstract>>
    +config
    +reference()*
  }
  class SWEProblem {
    +prepare()
    +generate_dataset()
    +build_model(variant)
    +make_step(model, opt)
    +sample_batch(it)
    +predict_grid(...)
  }
  class WaveObstacleProblem {
    +make_loss()
    +pde_residuals()
    +evaluate()
  }
  class EpiProblem {
    +load_or_simulate()
    +reconstruct()
    +estimate()
  }
  Problem <|-- SWEProblem
  Problem <|-- WaveObstacleProblem
  Problem <|-- EpiProblem
```

## Configuration

Configs are nested `dataclasses` deriving from `core.config.ConfigBase`, which
adds `to_dict` / `from_dict` / `load` / `save`. `from_dict` recurses into nested
configs **and** lists of configs (e.g. the PINN training `phases`). Unknown keys
raise `TypeError`, so a typo in a YAML file fails loudly.

```python
from sciml.problems.swe.config import SWEConfig
cfg = SWEConfig.load("configs/swe.yaml")   # or SWEConfig() for defaults
cfg.train.n_iter = 20000
cfg.save("outputs/used_config.json")
```

Next: [methods.md](methods.md) for each engine, or [problems.md](problems.md)
for the worked studies.
