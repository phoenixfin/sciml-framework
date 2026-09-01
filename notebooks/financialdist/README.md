# Corporate financial distress as a first-passage problem

Financial distress is conventionally a **classification** problem: predict a
binary label from a cross-section of ratios. This study treats it instead as a
**first-passage problem on a discovered vector field**. The label is not a label
— it is the sign of a smooth state, `IEQ = Equity / Total Liability`, and
"distress" is the event that a firm's trajectory crosses `IEQ = 0` and stays
there.

That reframing turns three vague questions into sharp ones: what sparse map
governs a firm's motion through ratio space, do firms that eventually cross live
on a *structurally different* map, and does the discovered map — run forward as a
stochastic system — reproduce the observed distribution of crossing times?

**Data:** 324 Indonesian listed firms × 11 years (2013–2023), balanced; 3,240
one-step pairs, 116 distress firm-years, 28 ever-distressed firms, 22 entries
into distress, 12 recoveries.

## The documents

| File | What's inside |
|---|---|
| [`RESEARCH_DESIGN.md`](RESEARCH_DESIGN.md) | The full design, **pre-registered**: the reformulation, the five constraints the data imposes, the panel-aware estimator, the experiment suite E0–E7, and — in §7 — the four outcomes the study committed in advance to accepting. |
| [`DATA_REPORT.md`](DATA_REPORT.md) | What was wrong with the source workbook and what was done about it. The serious one: the corruption-index column was desynchronised from `YEAR` by a spreadsheet re-sort, diagnosable because the value multiset matched the true national series exactly. |
| [`RESULTS.md`](RESULTS.md) | The run of 2026-08-09: the verdict against the pre-registration, panel diagnostics, the estimator gate, E1/E2/E3/E5, what it implies for the plan, and the caveats. |
| [`kaggle_panel_sindy.ipynb`](kaggle_panel_sindy.ipynb) | The run itself, top to bottom, unattended: state coordinates, the `panel_stlsq` estimator, the synthetic-recovery gate, then E1, E2, E3 and E5. |
| [`prepare_sindy_data.py`](prepare_sindy_data.py) | The cleaning pipeline that produced the data files below, one repair at a time, each logged. |

## The verdict, in one line

The run lands in **row 4 of the pre-registered outcome table** — degree-2 does
not beat AR(1), and the cohorts are not distinguishable — which the design had
already committed to treating as "still publishable, as a negative result done
properly". Read `RESULTS.md` for what that rests on.

The notebook's Part 3 is a gate in the strict sense: if the estimator cannot
recover a *known* sparse map at this N, T and noise, nothing in the later parts
means anything. It runs first and prints a verdict.

## The data files

Committed, because the study is unreproducible without them:

| File | Contents |
|---|---|
| `panel_clean.csv` | The cleaned panel in raw units, with quality flags. |
| `panel_sindy_ready.csv` | The same panel with asinh-transformed, standardised states. |
| `quality_log.csv` | One row per repair — code, firm, year, column, old value, new value, and why. |
| `datadrtpm_clean.xlsx` | All of the above in one workbook, plus the transform metadata and the CPI reference. |
| `results_2026-08-09.json` | The raw record behind every number in `RESULTS.md`. |

Regenerate the first four from the full-precision source with:

```bash
python notebooks/financialdist/prepare_sindy_data.py path/to/datadrtpm0Rev1.xlsx
```

Every repair is recorded in `quality_log.csv` and reversible from the source
file; nothing is silently overwritten. Figures land in `outputs/figures/` and
are gitignored — rerun the notebook to regenerate them.

## Status

Closed, as a pre-registered negative result. The open follow-ups (E4
unsupervised regime discovery, E6 SINDyc with year effects, E7 the Δt/τ
admissibility study) are listed in the design and have not been run.
