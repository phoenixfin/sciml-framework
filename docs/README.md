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

Study write-ups (findings, not API):

| Document | What's inside |
|---|---|
| [pi_deeponet_swe/RESULTS.md](../notebooks/pi_deeponet_swe/RESULTS.md) | End-to-end audit of the SWE study: reference-solver error budget, the shock in benchmark C1, the attractor result, metrics, ablations |
| [wnts/REPORT.md](../experiments/wnts/REPORT.md) | Gas-network SINDYc study: data quirks, protocol design, results A1–A4 / B1–B4 |

New here? Read **overview** → **architecture** → the **method** you care about.
Using `problems/swe` or `solvers/swe_lax_friedrichs`? Read the SWE audit first —
it revises several claims made elsewhere in these docs.

Quick links: [top-level README](../README.md) · [examples gallery](../examples/README.md)
· [GitHub repo](https://github.com/phoenixfin/sciml-framework)
