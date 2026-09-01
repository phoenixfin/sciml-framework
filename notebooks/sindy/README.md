# SINDy for dengue transmission dynamics

Can the governing equations of dengue transmission be recovered from **weekly
case counts alone** — no susceptible counts, no mosquito density, no assumed
compartmental structure?

Surveillance gives `I(t)` and nothing else, while SIR/SEIR models need the
unobserved compartments as inputs. The existing answers each pay for that
somewhere: SPADE4 works from `I(t)` alone but returns uninterpretable random
features; Horrocks & Bauch return interpretable equations but need `S(t)`
reconstructed by a grid search. This study asks whether delay embedding plus an
epidemiological candidate library can have both.

The full argument, the two-paper plan and the reference list are in
[`research_plan_sindy_dengue.md`](research_plan_sindy_dengue.md).

## The notebooks

| Notebook | Role |
|---|---|
| [`sindy_dengue_all_variants.ipynb`](sindy_dengue_all_variants.ipynb) | The variant sweep. Six ways of getting a state vector out of `I(t)` (§5), each fitted by the same STLS core over the same grid search (§4, §6), then compared head to head (§14). |
| [`sindy_dengue_paper_analysis.ipynb`](sindy_dengue_paper_analysis.ipynb) | The three components a paper needs for V1 and V2: forward simulation of the recovered equations against held-out weeks, structural comparison with canonical SIR/SEIR, and bootstrap confidence intervals on the coefficients. |

## The six variants

| | Approach |
|---|---|
| **V1** | Delayed SIR |
| **V2** | Delayed SEIR |
| **V3** | Semi-analytic SEIR, following Puspita et al. (2023) |
| **V4** | Semi-analytic SEIR plus a cumulative auxiliary state |
| **V5** | Cumulative-only — no semi-analytic reconstruction |
| **V6** | Takens delay embedding |

V6 is the one the plan builds Paper 1 on: it reconstructs the hidden dimensions
from lagged values of `I(t)` alone, so no `S(t)` reconstruction is needed, and it
asks whether the selected lags line up with the known dengue biology (intrinsic
incubation ≈ 1 week, extrinsic incubation in *Aedes* ≈ 1–2 weeks).

## Running them

Both notebooks run top to bottom on a CPU. Settings live in §0/§1: population
`Nh`, the rate constants, the smoothing width, the STLS `THRESHOLD`, the library
order, and the delay grids.

**Data.** `DATA_PATH = None` generates synthetic data with known coefficients —
which is also the honest way to read the results, since a variant that cannot
recover a known map has nothing to say about a real one. Point `DATA_PATH` at a
CSV of weekly cases to run on real surveillance data.

## Relationship to `sciml`

The packaged dengue example in
[`sciml.problems.epidemiology`](../../src/sciml/problems/epidemiology) is a
*different* question on the same disease: it reconstructs `S(t)` and identifies a
time-varying transmission rate `β(t)`, and it is settled. This study is about
recovering the equation structure itself, and is still open.

## Status

Active, and pre-publication: the variant sweep runs and the paper-analysis
notebook produces the three required components, but no run has been written up
against the plan's research questions the way the other studies here are.
