# Research studies

Four studies live here. Each is a self-contained piece of research — a question,
the notebooks that answer it, and a write-up of what came back — as opposed to
[`src/sciml/`](../src/sciml), which holds the reusable framework, and
[`experiments/`](../experiments), which holds scripted runs of that framework.

The relationship runs in both directions: a study that settles becomes a
packaged problem under `sciml.problems` (the SWE, wave and dengue examples all
arrived that way), and a study that needs a solver, a metric or a method engine
takes it from `sciml` rather than re-implementing it.

| Study | What it asks | Status | Start with |
|---|---|---|---|
| [`pi_deeponet_swe/`](pi_deeponet_swe) | Do the PI-DeepONet/SWE manuscript's claims survive a corrected reference solver? | Closed; every claim re-measured | [RESULTS.md](pi_deeponet_swe/RESULTS.md) |
| [`pinn_boussinesq/`](pinn_boussinesq) | Can a PINN solve the dispersive Boussinesq (VBM) system on the standard run-up benchmarks? | Active | [README.md](pinn_boussinesq/README.md) |
| [`financialdist/`](financialdist) | Is corporate financial distress a first-passage problem on a discoverable vector field? | Closed; pre-registered negative result | [RESULTS.md](financialdist/RESULTS.md) |
| [`sindy/`](sindy) | Can SINDy recover dengue transmission dynamics from case counts alone? | Active | [research plan](sindy/research_plan_sindy_dengue.md) |

## How these studies are run

Three of the four run on **Kaggle**, because they need a GPU for hours at a
time. The shape is the same in each:

- one *driver* notebook (`kaggle_*.ipynb`) that runs the whole programme
  unattended and writes a single `results_<date>.json`;
- a `NOTEBOOK_VERSION` string printed first and stored in that JSON — Kaggle
  re-runs the notebook saved in *its* editor, not the file on disk, so a log
  that disagrees with the repo means the run predates the current code;
- a `QUICK` switch that shrinks every expensive knob to a few minutes, for
  validating the environment before committing to the real run;
- `RUN_PART*` switches, so a section can be re-run without repeating the ones
  that already settled;
- a write-up (`RESULTS.md`) generated from — or checked against — that JSON,
  never transcribed by hand.

## Conventions

- **Numbers come from a named run.** Every quantitative claim in a `RESULTS.md`
  points at the `results_*.json` beside it. `pi_deeponet_swe/build_report.py`
  goes further and *generates* the write-up from the JSON, so the two cannot
  drift.
- **Extracted code is library code.** The `.py` files here are held to the same
  standard as `src/`: NumPy-style docstrings on every public object, linted and
  gated in CI (see [docs/reference.md](../docs/reference.md)).
- **Notebook cells are not.** `.ipynb` files are excluded from the linter on
  purpose; their cells are terse and are re-ordered by hand between runs.
- **Outputs are gitignored.** Figures, checkpoints and `.npz` data are
  regenerated, not committed. The `results_*.json` records are the exception —
  they are the evidence.

## Data

`financialdist/` ships its own data (the cleaned panel, as CSV and XLSX) because
the study is unreproducible without it. The other studies generate their data
from solvers in `sciml.solvers`, except the WNTS gas-network study under
[`experiments/wnts/`](../experiments/wnts), whose source data is confidential and
not in the tree.
