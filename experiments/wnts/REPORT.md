# WNTS Gas-Pipeline Network System Identification — Progress Report

*Status: 2026-07-09 · Code: `experiments/wnts/` · Results: `outputs/wnts/`*
*Data: `D:\repository\wnts_hourly` (confidential — do not redistribute)*

## 1. Objective

Discover an interpretable dynamical system connecting the operational states
(pressure P, flow q) at the endpoint nodes of the West Natuna Transportation
System (WNTS) gas pipeline network, using sparse regression (SINDy family).
Target publication narrative: **standard autonomous SINDy fails on
boundary-observed network data; a physically-informed SINDy-with-control
formulation succeeds.**

Only purely operational data is used (pressure, energy rate). Gas-composition
columns (C1–C9, N2, CO2, H2O, HCDP) and temperature are excluded.

## 2. Dataset

Hourly measurements at the network's endpoint metering stations only:

| Asset ID | Node | Role |
|---|---|---|
| 133001 | Anoa | source |
| 133002 | Kakap | source |
| 133003 | Hang Tuah | source |
| 133004 | Gajah Baru | source |
| 133060 | ORF | sink (Singapore delivery) |
| 133070, 133071 | — | duplicate delivery meters, dropped |

Files `2013.csv`–`2021.csv` are **contract years (Aug–Jul)**; e.g. `2019.csv`
covers Aug 2018 – Jul 2019. ~8.7k rows per asset per file, no NaNs in P/q.

### Data characteristics that shaped the method

- **Operational regime**: source flows are near-flat nominations; ORF offtake
  has a strong daily cycle (Singapore power demand). The 471 km, 28-inch line
  absorbs the difference as line pack — the pipeline "breathes" daily.
- **Unobserved interior**: only boundary nodes are measured. The distributed
  line-pack state is latent, so endpoint states alone are not Markovian.
- **Collinearity**: the four source pressures are ~99.9 % correlated (one
  hydraulically-coupled upstream pool).
- **Persistent imbalance**: sources deliver ~9 % more energy than the sink
  meters (fuel gas / shrinkage / meter basis). Naive mass balance fails;
  corr(dP/dt, flow imbalance) ≈ 0 at lag 0 (integrator phase shift +
  pressure control).
- **Telemetry freezes**: sensors repeat exact float values for hours–days
  (e.g. Kakap, Oct 2018). Must be masked.
- **Regime drift**: operating points drift over weeks–years (field depletion,
  renominations). A test window can sit many σ from a train window.
- **Excursions**: isolated events reach > 7σ (e.g. Hang Tuah flow spike in
  the Mar–Apr 2019 test segment).

## 3. Pipeline (`data.py`)

1. Load P, q per node (operational columns only), reindex to a gap-free
   hourly grid.
2. Mask rows with NaNs or any frozen-sensor run (≥ 6 identical values).
3. Extract contiguous clean segments (default ≥ 10 days). One contract year
   yields ~7 segments / ~128 clean days.
4. Block-average to a 3 h grid.
5. **Per-segment centering**: model fluctuations about each segment's
   operating point (removes multi-week drift).
6. z-score with pooled training statistics; saturate library inputs at ±3σ
   (`--clip`).
7. Optional latent line-pack proxy `L(t) = Σ (q_in − q_out − bias)·dt`,
   bias estimated from training data.

## 4. Models and protocol (`run.py`)

Six-model ladder, all fitted with STRidge on a polynomial library
(degree 2, threshold 0.02, ridge α = 1e-2):

| Key | States | Inputs |
|---|---|---|
| `sindy_P` | 5 node pressures | — (autonomous) |
| `sindy_Pq` | 5 pressures + 5 flows | — (autonomous) |
| `sindyc` | 5 pressures | 5 flows |
| `sindyc_L` | 5 pressures + L | 5 flows |
| `sindyc_red` | **p_up** (mean source pressure), **p_orf** | 5 flows |
| `sindyc_redL` | p_up, p_orf, L | 5 flows |

**Fitting**: discrete-time SINDy — targets are forward differences
`(x_{k+1} − x_k)/dt`, rollouts use the consistent forward-Euler map.
(Continuous Savitzky-Golay + RK4 is available via `--fit savgol` but was
found unstable on this data.)

**Evaluation** (all out-of-sample, later-in-time segments):
- one-step-fit R² on pressure states, train vs test;
- multi-IC forecast rollouts (72 h horizon, ICs every 48 h): divergence
  fraction and median NRMSE at 24 h / 72 h, normalized by test std;
- trivial baselines on the same protocol: persistence, climatology
  (operating point), daily-repeat.

## 5. Results (contract year 2019; train Sep 2018–Mar 2019, test Mar–Apr 2019)

*Note: this table used the original oracle (per-segment) centering; the
honest causal-protocol numbers are in section A2 below. The ranking of
models is unchanged.*

| Model | R² test (dP/dt) | NRMSE 24 h | NRMSE 72 h | diverged |
|---|---|---|---|---|
| persistence | — | 0.80 | 1.23 | — |
| climatology | — | 0.96 | 0.87 | — |
| daily-repeat | — | 0.69 | 1.11 | — |
| `sindy_P` | −0.04 | ∞ | ∞ | 100 % |
| `sindy_Pq` | −4.6 | ∞ | ∞ | 100 % |
| `sindyc` | −4.6 | ∞ | ∞ | 100 % |
| `sindyc_L` | −3.7 | ∞ | ∞ | 100 % |
| **`sindyc_red`** | **+0.24** | **0.46** | **0.54** | **0 %** |
| `sindyc_redL` | +0.24 | 0.46 | 0.54 | 0 % |

Identified reduced model (z-scored, per-hour units):

```
d/dt p_up  = −0.04 p_up + 0.04 q_orf + (small terms)
d/dt p_orf = +0.06 p_up − 0.07 p_orf − 0.08 q_orf + (small terms)
```

Physically readable: delivery pressure relaxes toward the upstream pool
(transport) and is drawn down by offtake — textbook line-pack behaviour.

Figures: `outputs/wnts/fig_rollout.png` (all naive variants diverge < 24 h;
SINDYc-2 tracks the ORF daily cycle over 72 h), `outputs/wnts/fig_r2.png`,
`outputs/wnts/equations.txt`, `outputs/wnts/summary.json`.

## 6. Lessons learned (methods findings)

1. **Continuous-time SINDy is unstable here.** Savitzky-Golay derivatives +
   RK4 rollouts diverge in every configuration tried. Discrete-time fitting
   with the consistent Euler map is the working recipe.
2. **Collinearity, not just missing inputs, kills the full-state models.**
   With four ~identical source pressures the regression distributes
   indeterminate cross-feedback among them; iterated, it explodes — even
   with control inputs and ridge regularization. Collapsing the sources into
   one pool pressure (`p_up`) stabilizes everything.
3. **The pool pressure *is* the line-pack state.** The explicit latent
   integral L is pruned by STRidge at the operating threshold and adds no
   forecast skill: p_up already observes stored mass. (An earlier scratch
   result suggesting L helped was a metric artifact — the easily-predicted L
   state had been averaged into the NRMSE.)
4. **Model fluctuations, not absolute levels.** Chronological extrapolation
   of absolute pressures fails (−54σ excursions relative to a narrow train
   window). Per-segment centering plus library-input clipping makes
   out-of-sample evaluation meaningful.
5. **Data hygiene is half the work**: frozen-telemetry masking and segment
   extraction determine everything downstream.

## 7. Current caveats

- ~~Per-segment centering uses the test segment's own mean~~ — resolved by
  A2: causal trailing operating point is now the default.
- ~~Results shown for one contract year~~ — resolved by A1: six usable
  years, within-year + transfer + pooled evaluations.
- `q_orf` used as a known input means the model answers "given demand and
  nominations, what do pressures do?" — the right question for operations,
  but it should be stated explicitly in the paper.
- Temperature and volume rate unused; energy rate only.
- Contract year 2021 does not fit the identified 2016–2020 dynamics (see
  A1) — open diagnostic question.

---

# Results of experiments A1–A4 (2026-07-09)

## A2 — Causal operating point (leakage removed) ✅

Per-segment (oracle) centering was replaced by a **trailing operating
point**: every signal is referenced to a causal trailing mean
(`--op-window 72` h; the spin-up period of each segment is trimmed so no
partially-formed window enters training). Forecasts are evaluated in the
frame **frozen at the forecast start**, so no future information enters
anywhere. This is now the default (`--center causal`; `--center segment`
retains the oracle variant).

2019 contract year, SINDYc-2 (causal vs oracle in parentheses):

| metric | causal | (oracle) |
|---|---|---|
| test R² (dP/dt) | **+0.22** | (+0.24) |
| NRMSE 24 h | **0.67** | (0.46) |
| NRMSE 72 h | **0.69** | (0.56) |
| best trivial baseline 24 h / 72 h | 0.75 / 1.16 | 0.54 / 0.92 |
| diverged | 0 % | (0 %) |

The honest numbers are worse in absolute terms (the frozen frame is a
harder target) but the **skill vs baselines survives**: the model beats
every baseline at 24 h and roughly halves the 72 h error. Naive/full-state
models still diverge 60–100 %.

## A4 — Horizon-skill curves ✅

Median NRMSE vs forecast horizon (2019, causal protocol; figure
`outputs/wnts/fig_horizon.png` is produced by every run):

| horizon [h] | 6 | 12 | 24 | 48 | 72 | 120 | 168 |
|---|---|---|---|---|---|---|---|
| SINDYc-2 | 0.25 | 0.47 | **0.67** | **0.72** | **0.68** | 0.83 | 1.08 |
| persistence | 0.21 | 0.43 | 0.84 | 1.03 | 1.24 | 1.31 | 1.23 |
| daily-repeat | 0.36 | 0.60 | 0.75 | 0.90 | 1.16 | 1.18 | 1.04 |
| climatology | 0.55 | 0.72 | 1.09 | 1.38 | 1.23 | 1.28 | 1.33 |

Reading: at ≤ 12 h persistence is unbeatable (as expected); the model
pulls ahead from 24 h, has its largest advantage at 48–72 h (the line-pack
time scale), and loses skill at ~5 days where daily-repeat catches up.
**Skill horizon ≈ 5 days given boundary flows.**

## A3 — State-space ablation: the instability mechanism ✅

All variants are SINDYc (flows as inputs); only the pressure-state
definition changes. ρ = spectral radius of the identified linear one-step
map `I + Δt·A` (`outputs/wnts_A3/fig_states.png`):

| states | dim | ρ(I+ΔtA) | test R² | diverged 72 h | NRMSE 24 h / 72 h |
|---|---|---|---|---|---|
| 5 raw pressures | 5 | **1.102** | −5.4 | 100 % | 1.14 / ∞ |
| 2 source PCs + p_orf | 3 | **1.053** | −4.3 | 80 % | 1.21 / 0.80 |
| p_up + p_orf | 2 | **0.974** | +0.22 | 0 % | 0.67 / 0.69 |
| p_orf only | 1 | **0.731** | +0.36 | 0 % | 0.68 / 0.54 |

The spectral radius crosses the stability boundary exactly where rollouts
start diverging — the quantitative mechanism behind "collinear source
pressures make the identified feedback explosive". Source-pressure PCA:
PC1 = 83 %, PC2 = 11 % of variance, i.e. the upstream pool is nearly (not
exactly) one mode; keeping the second PC already destabilizes the model.

Notable: the **1-state model (p_orf + flow inputs) is the best pure
forecaster** (72 h NRMSE 0.54, R² +0.36). The 2-state form remains the
interpretability sweet spot (it captures the source–sink coupling), and
that trade-off is worth a paragraph in the paper.

## A1 — Multi-year robustness ✅ (with data-quality findings)

**2014 and 2015 are excluded**: source telemetry is frozen/stale much of
the time (Hang Tuah 43–63 % frozen; in 2015 all source channels degraded,
92 % of rows unusable). The usable study period is **2016–2021**
(the ORF meter is clean in all years). Full results:
`outputs/wnts_A1/table.md`; per-year coefficient stability:
`outputs/wnts_A1/fig_coeff_stability.png`. skill = 1 − NRMSE/best-baseline
(positive = beats every trivial baseline).

Within-year splits are noisy (often only 1–2 test segments); the
**transfer runs** (train year Y → test all clean segments of year Y+1,
14–54 rollouts) and the **pooled runs** (train on all prior years) are the
reliable estimates:

| run | n | skill 24 h | skill 72 h | diverged |
|---|---|---|---|---|
| 2016→2017 | 14 | +0.17 | −0.60 | 14 % |
| 2017→2018 | 25 | +0.04 | +0.05 | 20 % |
| 2018→2019 | 54 | +0.15 | +0.09 | 9 % |
| 2019→2020 | 24 | **+0.47** | **+0.20** | 0 % |
| 2020→2021 | 37 | −0.63 | +0.05 | 35 % |
| **pooled 2016–19 → 2020** | 24 | **+0.41** | **+0.25** | **0 %** |
| pooled 2016–20 → 2021 | 37 | −0.10 | −0.03 | 32 % |

Reading (honest):

- **Pooling history helps.** Trained on 2016–2019, the model forecasts
  2020 with the best numbers of the whole study (24 h NRMSE 0.19 vs 0.32
  for the best baseline, zero divergence).
- **Skill is positive but modest in most year-pairs** (+0.04…+0.47 at
  24 h in 4 of 6 reliable runs).
- **Contract year 2021 (Aug 2020–Jul 2021) breaks the model** — even with
  all history pooled, skill is slightly negative and a third of rollouts
  diverge. Something changed in the operating regime (COVID-recovery
  demand pattern and/or field/nomination changes). This is a genuine
  finding, not a tuning issue: the identified 2016–2020 dynamics does not
  transfer to 2021. Diagnosing it (regime detection, event windows — see
  C1) belongs in the paper's discussion.
- The naive/full-state variants diverge 60–100 % in **every** year and
  configuration — the core claim is robust across all six years.

## B1 — Library ablation: the fluctuation dynamics is linear ✅

Four candidate libraries on the SINDYc-2 form (linear; quadratic;
linear + physics terms `p_up²−p_orf²`, `q_orf|q_orf|`, `p·q_orf`;
quadratic + friction), on 2019 and pooled→2020
(`outputs/wnts_B1/`):

**All four libraries collapse to the identical 8-term linear model.**
STRidge (threshold 0.02) prunes every nonlinear term — generic polynomial
and physics-informed alike — in both configurations, with identical
metrics. Interpretation for the paper: after referencing to the operating
point, the dynamics of the *fluctuations* is effectively linear; the
Weymouth nonlinearity lives *across* operating points and is absorbed by
the centering. Two consequences: (1) the headline model is a sparse
linear state-space model with inputs, which connects SINDYc directly to
DMDc (B4 becomes the natural comparison); (2) claims about "discovering
nonlinear physics" should not be made — the discovery here is the
*structure* (which states, which inputs, stability), not nonlinearity.

## B2 — Input ablation: demand dominates ✅

Same form, quadratic library, varying only the input set
(`outputs/wnts_B2/`). 2019 configuration:

| inputs | NRMSE 24 h | NRMSE 72 h |
|---|---|---|
| all 5 flows | **0.60** | **0.69** |
| q_orf only (demand) | 0.60 | 0.79 |
| 4 source flows | 0.82 | 1.59 |
| net imbalance only | 0.80 | 1.10 |
| none (autonomous) | 0.80 | 1.14 |

**The ORF offtake (demand signal) carries almost all of the input skill**;
at 24 h it is indistinguishable from the full input set, and source flows
alone are barely better than no inputs. Of the four source flows only
Gajah Baru survives thresholding in the full model. The final identified
model (2019; pooled→2020 finds nearly identical coefficients — a free
coefficient-stability check):

```
d/dt p_up  = −0.057 p_up + 0.052 p_orf + 0.041 q_gajahbaru + 0.098 q_orf
d/dt p_orf = +0.087 p_up − 0.104 p_orf − 0.060 q_gajahbaru − 0.132 q_orf
```

(z-scored fluctuation units per hour). In the pooled→2020 test period the
differences between input sets shrink (quiet operating year: even the
autonomous 2-state model beats the trivial baselines there; inputs add
~25 % at 24 h).

## B3 — Hyperparameter sensitivity ✅

Grids on 2019, causal protocol (`outputs/wnts_B3/fig_heatmaps.png`):

**Threshold × ridge-α (dt = 3 h, clip = 3):**

- The ridge penalty α is **irrelevant** across three orders of magnitude
  (identical results for α = 1e-3…1) — with n ≈ 900 samples and ≤ 36
  features the regression is well-conditioned once the state is reduced.
- The STRidge threshold has a clear optimum at **0.02** (72 h NRMSE 0.69,
  0 % divergence). 0.01 under-prunes (0.96, 11 % divergence), 0.05–0.1
  over-prune (0.89 / 0.76) — but no setting is catastrophic for the
  reduced model.
- The **naive 5-state model diverges 33–100 % over the entire grid** — no
  hyperparameter tuning rescues it. This pre-empts the "you didn't tune
  the baseline" review.

**dt × clip (threshold/α at defaults), SINDYc-2, 72 h NRMSE:**

| | clip off | 2σ | 3σ | 5σ |
|---|---|---|---|---|
| dt = 1 h | 3.31 | 3.35 | 3.54 | 3.75 |
| **dt = 3 h** | 0.76 | 0.92 | **0.69** | 0.71 |
| dt = 6 h | 1.30 | 1.21 | 0.86 | 0.85 |

- **The 3 h sampling interval is doing real work**: hourly fitting is
  ~5× worse (the one-step map is dominated by measurement noise; 72
  noisy Euler steps compound), and 6 h is too coarse for the daily
  cycle. The identification timescale must match the line-pack physics.
- Clipping matters little at dt = 3 h (0.69–0.92); 3σ is a good default,
  and even clip-off works — the clip is a safety net for excursion
  events, not a driver of the result.

## B4 — DMDc benchmark: the state reduction is the discovery ✅

Since B1 showed the identified model is linear, DMDc is the natural null.
Implemented as unthresholded, unregularized linear least squares inside
the same discrete one-step machinery (mathematically DMDc in companion
form), so data handling and the rollout protocol are identical — only the
estimator changes (`outputs/wnts_B4/`):

| model | active terms | 2019: 24 h / 72 h | pool→2020: 24 h / 72 h |
|---|---|---|---|
| DMDc, naive 5 pressures | 55 | 0.53 / **0.98** | 0.25 / 0.41 (R² −1.0) |
| DMDc, reduced (p_up, p_orf) | 16 | 0.54 / 0.70 | 0.19 / 0.40 |
| ridge linear, reduced | 16 | 0.54 / 0.70 | 0.19 / 0.40 |
| SINDYc-2 (ridge + STRidge) | **8** | 0.60 / 0.69 | 0.19 / 0.41 |

Honest reading, and the paper's sharpest formulation:

- **Given the reduced state, DMDc matches SINDYc-2 in forecast skill.**
  Sparse regression buys parsimony (8 vs 16 terms) and the interpretable
  discovery pipeline — not accuracy.
- **The state-space reduction is the load-bearing discovery.** DMDc on
  the naive 5-pressure state reaches only baseline-level skill at 72 h
  (0.98 ≈ best trivial baseline) and its regression transfers terribly
  across years (test R² −1.0 pooled), even though linearity + clipping
  keep it from outright diverging. So the failure of naive identification
  is estimator-independent; the fix (collapse the collinear pool) is what
  matters. (A3's 100 %-divergence naive runs used the quadratic library;
  linear + clipping fails more gracefully — worth one sentence in the
  paper.)

---

# Framework: dataset registry + generic task layer (2026-07-11)

To run further datasets through any method without hand-building a new
package each time, the protocol developed here was generalized into the
`sciml` package (validated by reproducing the A2 headline exactly):

- **`sciml.data.datasets`** — a lazy registry with two containers:
  `TimeSeriesData` (numpy segments + named channels; system-id-shaped) and
  `FunctionPairData` (input/output function pairs; DeepONet/FNO-shaped).
  Built-ins: `wnts` (this study's loader, incl. the derived `P_up`
  channel; pandas imported only on demand), `lti_demo` (ground-truth
  linear system for testing), `advection_pairs` (GP → advection-diffusion
  operator pairs). A new dataset = one loader function + `@register`.
- **`sciml.tasks.sysid`** — the full WNTS protocol as one call: causal
  operating-point referencing, spin-up trimming, chronological /
  interleaved / explicit-transfer splits, discrete-SINDy / SINDYc / DMDc
  fitting, multi-horizon rollout metrics and trivial baselines.

```python
from sciml.data.datasets import load
from sciml.tasks import sysid

data = load("wnts", years=[2019])
res = sysid.run(data, states=["P_up", "P_orf"],
                inputs=[c for c in data.channels if c.startswith("q_")])
print(res.summary())   # reproduces the A2 numbers exactly
```

CLI: `sciml datasets` and
`sciml sysid --data wnts --states P_up P_orf --inputs all`.
Covered by `tests/test_datasets.py` / `tests/test_sysid.py` (including
coefficient recovery against the known `lti_demo` ground truth).
The `experiments/wnts` package remains as the frozen record of the study;
new datasets should go through the registry.

---

# Experiment design plan

Priority A = needed for the paper's core claims; B = strengthens it;
C = extensions / follow-up work. Status: **A1–A4 and B1–B4 done** (above);
B5 (bootstrap coefficient uncertainty) and the C tier remain. The 2021
regime-break diagnostic (C1) is the highest-value remaining item for the
discussion section.

## A1. Multi-year robustness matrix

**Question**: does the SINDYc-2 result hold across operating eras?
**Design**: for each contract year 2014–2021, run the full ladder with the
within-year chronological split; additionally train on year Y, test on year
Y+1 (cross-year transfer, using per-segment centering to absorb the level
shift). Report the results table per year plus mean ± spread.
**Output**: robustness table + a coefficient-stability plot (identified
coefficients per year with error bars — are the equations the *same
physics* every year?).
**Effort**: small — `--years` already supports it; add a driver script that
loops years and aggregates `summary.json` files.

## A2. Causal operating point (remove the leakage caveat)

**Question**: does skill survive when the operating point is estimated
causally?
**Design**: replace per-segment mean with a trailing mean (e.g. previous
3–7 days, computed only from data before each rollout IC). Compare NRMSE
against the current (oracle-centered) protocol.
**Output**: one table row per centering scheme; the honest headline numbers.
**Effort**: moderate — centering moves from preprocessing into the rollout
loop.

## A3. State-space ablation (the paper's core argument, made quantitative)

**Question**: *why* does the reduced form work — where exactly between 5
states and 2 states does stability appear?
**Design**: sweep state definitions: (a) 5 raw pressures; (b) 3 states
(PCA of source pressures' first PC + p_orf + second PC); (c) 2 states
(p_up, p_orf); (d) 1 state (p_orf only). For each: eigenvalues of the
identified linear part (stability margin), divergence fraction, NRMSE.
**Output**: figure of spectral radius / divergence vs state dimension —
directly visualizes the collinearity → instability mechanism.
**Effort**: moderate — mostly new `seg_data` variants; eigenvalue extraction
from the linear coefficients is a few lines.

## A4. Horizon-skill curves

**Question**: how far ahead is the model useful?
**Design**: NRMSE vs horizon (6, 12, 24, 48, 72, 120, 168 h) for SINDYc-2
and all baselines. Identify the skill horizon (crossing with climatology).
**Output**: the second money figure; operationally meaningful claim
("pressure forecast beats operating-point assumption out to ~N days given
nominations").
**Effort**: small — loop over horizons in `rollout_metrics`.

## B1. Library ablation / physics library

**Question**: does a physics-informed library beat a generic polynomial, and
what do the discovered terms mean?
**Design**: compare (a) linear; (b) quadratic (current); (c) physics
library: ΔP² terms (Weymouth: q² ∝ P_up² − P_orf²), q·|q| friction terms,
P·q crosses; (d) quadratic + physics. Same protocol; also report term count
and which physics terms survive thresholding.
**Output**: sparsity-vs-skill table; interpretation section for the paper.
**Effort**: small — `CustomLibrary` already exists in `sciml`.

## B2. Input ablation

**Question**: which boundary flows actually drive the pressures?
**Design**: inputs = {q_orf only} vs {4 source flows} vs {all 5} vs
{net imbalance only}. Expectation: q_orf dominates (demand-driven system).
**Output**: table; simplifies the final model and sharpens the story.
**Effort**: small.

## B3. Hyperparameter sensitivity and stability map

**Question**: how sensitive are the conclusions to threshold, α, dt, clip?
**Design**: grid: threshold {0.01, 0.02, 0.05, 0.1} × α {1e-3, 1e-2, 1e-1,
1} × dt {1, 3, 6} h × clip {2, 3, 5, off}. Heatmaps of test NRMSE and
divergence fraction for SINDYc-2; verify the naive models fail across the
whole grid (pre-empts the "you didn't tune the baseline" review).
**Output**: appendix heatmaps + a sentence in methods.
**Effort**: small–moderate — driver script; ~200 cheap runs.

## B4. Benchmark against other methods in the `sciml` package

**Question**: how does sparse regression compare with other data-driven
models on the same protocol?
**Design**: same states/inputs/protocol with (a) DMDc (linear operator with
control — the natural null model for SINDYc); (b) a small Neural ODE with
inputs; (c) optionally DeepONet mapping input histories → pressure
trajectories (different problem shape; frame as operator-learning
comparison).
**Output**: comparison table; positions the paper within the SciML
framework and justifies "interpretable + competitive".
**Effort**: moderate — DMDc is nearly free (module exists); Neural
ODE/DeepONet need TF environment (not available in the current Python 3.14
setup — run in the project's TF environment or defer).

## B5. Ensemble/bootstrap SINDy (coefficient uncertainty)

**Question**: are the identified coefficients stable under resampling?
**Design**: E-SINDy style — refit on bootstrap resamples of training
segments (segment-level block bootstrap); report coefficient inclusion
probabilities and confidence intervals; optionally ensemble-median model for
forecasting.
**Output**: coefficient CI table/figure — reviewers like discovered-equation
uncertainty.
**Effort**: moderate — loop around `ModelSpec.fit`.

## C1. Event robustness

Evaluate separately on windows containing excursions (shut-ins, spikes)
vs quiet windows; study clip's role during events. Motivates future
regime-switching or hybrid models.

## C2. Delay embedding as an alternative latent state

Add lagged p_orf (and/or lagged q_orf) as extra states instead of L;
compare with A3 variants. Connects to Takens/HAVOK literature and tests
whether any unobserved dynamics remains.

## C3. Volume-rate and temperature variants

Repeat headline experiments with volume rate instead of energy rate
(removes composition-driven energy-content variation from the flow signal);
optionally add T as an exogenous input.

## C4. Physical-units model and operational deliverable

Un-normalize the final equations into psig / BBtu·d⁻¹ units, validate
against known line-pack capacity figures if available; package as a simple
pressure-forecast tool (given nomination schedule → 24–72 h pressure bands).

## Suggested execution order

1. **A2** (causal centering) — it changes the headline numbers; do it before
   collecting the full result matrix.
2. **A1 + A4** (multi-year matrix + horizon curves) — the paper's main
   tables/figures.
3. **A3** (state-space ablation) — the mechanism figure.
4. **B1, B2, B3** — ablations and appendix.
5. **B4, B5** — comparisons and uncertainty, as time allows.
