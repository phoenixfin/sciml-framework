# sciml documentation

Full documentation for the **sciml** scientific-machine-learning framework.

| Document | What's inside |
|---|---|
| [overview.md](overview.md) | What the project is, the design philosophy, the big picture |
| [architecture.md](architecture.md) | Layers, package structure, dependency & data-flow **flowcharts** |
| [methods.md](methods.md) | The six method engines — math, when to use, API, examples |
| [problems.md](problems.md) | The three packaged problems + the examples gallery |
| [datasets.md](datasets.md) | The dataset registry + task layers: run your own data through any method |
| [extending.md](extending.md) | How to add a new problem or a new method |
| [reference.md](reference.md) | API-reference generation, testing, docstring coverage |

Study write-ups (findings, not API) — the index is
[notebooks/README.md](../notebooks/README.md):

| Document | What's inside |
|---|---|
| [pi_deeponet_swe/RESULTS.md](../notebooks/pi_deeponet_swe/RESULTS.md) | End-to-end audit of the SWE study: reference-solver error budget, the shock in benchmark C1, the attractor result, metrics, ablations |
| [wnts/REPORT.md](../experiments/wnts/REPORT.md) | Gas-network SINDYc study: data quirks, protocol design, results A1–A4 / B1–B4 |
| [pinn_boussinesq/README.md](../notebooks/pinn_boussinesq/README.md) | PINN on the dispersive Boussinesq (VBM) system: the five run-up benchmarks, the exact Carrier–Greenspan reference, and the seed sweep that sets the noise floor |
| [financialdist/RESULTS.md](../notebooks/financialdist/RESULTS.md) | Financial distress as a first-passage problem: a pre-registered design and the negative result it returned |
| [sindy/README.md](../notebooks/sindy/README.md) | Dengue structure discovery: six ways to get a state vector out of case counts alone |

Where to run things:

| Document | What's inside |
|---|---|
| [examples/README.md](../examples/README.md) | The graded gallery, 01–13 |
| [experiments/README.md](../experiments/README.md) | Scripted studies over the framework, and their conventions |
| [notebooks/README.md](../notebooks/README.md) | The four research studies, and how a Kaggle run is organised |

New here? Read **overview** → **architecture** → the **method** you care about.
Using `problems/swe` or `solvers/swe_lax_friedrichs`? Read the SWE audit first —
it revises several claims made elsewhere in these docs.

Quick links: [top-level README](../README.md) · [examples gallery](../examples/README.md)
· [GitHub repo](https://github.com/phoenixfin/sciml-framework)
