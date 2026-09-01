# PI-DeepONet / SWE — the reference-solver audit

Do the claims of the `pi_deeponet_swe_v6` manuscript survive a **corrected
reference solver**? The study audits the whole chain — solver, theory, metrics,
ablations — and re-measures every number that turned out to depend on the
training targets rather than on the operator.

The findings are in [**RESULTS.md**](RESULTS.md); the short version is in the
[repo README](../../README.md#the-swe-study--reference-solver-audit). Two of
them set the tone: the reference data carried a **6.4e-02** relative error, 84%
of the wave anomaly, against **8.9e-03** for a well-balanced scheme on the same
grid; and the manuscript's physics-informed failure was a property of its
residual, whose exact global minimum is the state it "collapsed" to.

## The files

| File | Role |
|---|---|
| [`RESULTS.md`](RESULTS.md) | The write-up. **Generated**, not written — see below. |
| [`results_2026-08-12.json`](results_2026-08-12.json) | The raw record of the run every number comes from. |
| [`kaggle_swe_revision_all.ipynb`](kaggle_swe_revision_all.ipynb) | The run: four independent parts, each behind a `RUN_PART*` switch. Part 1 (solver, CFL, convergence, shock, data regeneration) writes `swe_data_wb.npz`, which Parts 3 and 4 consume; Part 2 (the corrected Proposition 1) is independent of both. |
| [`swe_solvers.py`](swe_solvers.py) | The manuscript's Lax-Friedrichs scheme, kept verbatim for the audit, beside a well-balanced Audusse/HLL solver (MUSCL + SSP-RK2) that is exact on lake-at-rest. **Use this one** whenever the reference error competes with what you are measuring. |
| [`deeponet_tf.py`](deeponet_tf.py) | A minimal four-branch DeepONet built for diagnostics: switchable branch fusion and five initial-condition shortcuts, the full autodiff SWE residual, and the trunk/branch gradient split at `F = 0`. Used by Parts 2 and 3. |
| [`pi_deeponet_v6.py`](pi_deeponet_v6.py) | A faithful port of the paper's *own* architecture, loss and training loop, so Part 4's numbers drop straight into the manuscript. Adds only what the audit needs: a data `Bundle` instead of module globals, an `ic_mode` switch, and three PDE residuals side by side. |
| [`build_report.py`](build_report.py) | Generates `RESULTS.md` and the README's findings block from the results JSON. |

## Regenerating the write-up

```bash
python notebooks/pi_deeponet_swe/build_report.py
```

That rewrites `RESULTS.md` **and** the block between the `swe-findings` markers
in the repo README, both from `results_*.json` (newest by name, or `--results
<file>`; `--no-readme` skips the README). No figure in either document is
transcribed by hand, which is what keeps the two from drifting.

## Method notes worth reusing

- **Single-seed ablations flipped their winners** across runs on identical
  settings. The study reports paired-by-seed contrasts and a
  difference-in-differences against a no-coupling control instead of a ranking.
- **Reference-data error belongs in the error budget.** Quoting an operator
  error without auditing the solver that produced its targets can be off by more
  than the effect under study.
- **Normalisation matters.** `‖h‖`-relative error is flattered by the constant
  background depth by 10–22× depending on the snapshot; anomaly-relative error
  and dimensional RMSE are reported alongside it.

## Provenance

`pi_deeponet_swe_v6.ipynb` is no longer in the tree — its pipeline is ported to
`pi_deeponet_v6.py`. To check the port against the original:

```bash
git show ff45b6b^:notebooks/pi_deeponet_swe/pi_deeponet_swe_v6.ipynb > v6.ipynb
```

## Status

Closed. The packaged refactor lives in
[`sciml.problems.swe`](../../src/sciml/problems/swe); the audit's caveats about
`sciml.solvers.swe_lax_friedrichs` apply there too, and the
`experiments/swe/{ablation,nd_scaling,physics_attractor}` scripts mirror the
*original* notebook sections — several of their conclusions do not survive this
audit.
