# Structure Discovery in Corporate Financial Distress
## A first-passage formulation with panel-aware sparse regression

**Data:** 324 Indonesian listed firms × 11 years (2013–2023), balanced. 3,240 one-step pairs.
116 distress firm-years, 28 ever-distressed firms, 22 entries into distress, 12 recoveries.

---

# 0. Thesis in one paragraph

Financial distress is conventionally treated as a **classification** problem: predict a binary label from a cross-section of ratios. This project treats it instead as a **first-passage problem on a discovered vector field**. The label is not a label — it is the sign of a smooth state variable, `IEQ = Equity/Total Liability`, and "distress" is the event that a firm's trajectory crosses the hyperplane `IEQ = 0` and stays there. The scientific questions then become: (i) what sparse map governs the firm's motion through ratio space, (ii) do firms that eventually cross live on a *structurally different* map, and (iii) does the discovered map, run forward as a stochastic dynamical system, reproduce the observed distribution of crossing times? SINDy is the estimator for (i); permutation inference on coefficient supports answers (ii); a Monte-Carlo hitting-time study answers (iii).

The contribution is threefold — a **reformulation** (distress as boundary crossing), a **methodological adaptation** (SINDy for short-*T*, large-*N* panels with unit heterogeneity, which the SINDy literature has essentially not addressed), and an **honest empirical result** about how much dynamical structure this class of data can actually support.

---

# 1. Concept

## 1.1 The reframing

```
CONVENTIONAL                          PROPOSED

  x_i(t) ──► classifier ──► ŷ ∈{0,1}    x_i(t+1) = F(x_i(t)) + firm effect + noise
  rare-event classification                        │
  3.2% positive rate                               ▼
  contemporaneous, no time structure      boundary S = { x : IEQ(x) = 0 }
  IEQ sign ⇒ Y at 99.8%  → leakage        distress = first passage to S
                                          τ_i = inf{ t : IEQ_i(t) < 0, IEQ_i(t+1) < 0 }
```

Two things this buys:

1. **It dissolves the leakage problem.** "IEQ < 0 twice ⇒ Y" is 99.8% accurate, which makes any classifier using leverage near-tautological and any reported AUC meaningless. Under the first-passage view that identity is not a bug to be hidden — it is the *definition of the boundary*, stated openly and used constructively.

2. **It dissolves the class-imbalance problem.** There is no longer a 3.2% positive class to oversample. Every one of the 3,240 transitions carries information about the vector field; the 22 crossings are then a *validation target* for the fitted dynamics, not a training signal to be fought over.

## 1.2 Why SINDy specifically

Honest answer, to be stated in the paper: SINDy here is not chosen because financial ratios obey a hidden ODE. It is chosen because it is the natural estimator when you want (a) a **nonlinear** function class, (b) **sparsity as the model-selection principle** rather than post-hoc feature importance, and (c) an object that is a **dynamical system** — something you can integrate forward, compute hitting times from, and linearise around a boundary. A random forest gives you none of (b) or (c). A panel VAR gives you (c) but not (a).

The framing must be *structure discovery*, not prediction. The paper's claim is about **which interaction terms are supported by the data**, with confidence intervals, not about beating a benchmark AUC.

---

# 2. Formal setup

**State.** `x_i(t) ∈ ℝ¹²`, the asinh-transformed standardised ratios in `panel_sindy_ready.csv`
(`X1–X8, IEQ, X10, X11, X12`; `X9` removed as its reciprocal, `X13–X15` held out as parameters/input).

**Model class.** Discrete-time map with firm heterogeneity:

```
x_i(t+1) = α_i + Θ(x_i(t)) Ξ + ε_i(t)          i = 1..N,  t = 1..T-1
           ↑     ↑            ↑
           firm  library      sparse coefficients
           effect (91 terms   (91 × 12)
                  at deg 2)
```

`α_i ∈ ℝ¹²` are firm fixed effects. **These are not a nuisance — omitting them is the single largest error available in this project** (§3.1).

**Boundary.** `S = { x : IEQ(x) = 0 }`, a hyperplane in the transformed coordinates because
`asinh` is odd and maps 0 → 0 exactly.

> ⚠ **This holds for the `*_t` columns, not the `*_z` columns the data file ships.** Standardising
> after the transform subtracts the mean, which moves the boundary to `IEQ_z = −z_mean/z_std =
> −0.6727`. Reading `IEQ_z < 0` as "distressed" labels **2,580 of 3,564** firm-years against a true
> count of **146**. Every model here is therefore fitted on a state in which `IEQ` is *scaled but
> not centred* — see `kaggle_panel_sindy.ipynb` Part 1, which rebuilds the state and asserts the
> two counts agree. In the **within** transform the boundary is firm-specific, at `−mean_i(IEQ)`;
> E5 and E9 must carry that offset rather than assume zero.

Distress onset for firm *i*:

```
τ_i = min{ t : IEQ_i(t) < 0 ∧ IEQ_i(t+1) < 0 }
```

**First-passage functional.** Given fitted `(Ξ̂, α̂_i, Σ̂_ε)`, define

```
P_i(h) = Pr[ τ_i ≤ t₀ + h | x_i(t₀) ]
```

estimated by Monte-Carlo rollout. The empirical 22 crossings are the test set for `P`.

**Continuous-time link (the perturbation-theory thread).** The fitted map `F̂` is, by backward
error analysis, the exact time-Δt flow of a *modified* vector field:

```
F̂ = exp(Δt · L_f̃),      f̃ = f₀ + Δt f₁ + Δt² f₂ + ⋯
```

At Δt = 1 yr and within-firm relaxation times τ ∈ [1.1, 2.3] yr, `Δt/τ ∈ [0.43, 0.91]` — the
expansion parameter is **O(1), not small**. This is a rigorous statement of why continuous-time
SINDy is inadmissible here, and it is quantifiable rather than hand-waved (E7).

---

# 3. Five constraints the data imposes on the design

Each of these is measured, not assumed. They are the reason the design looks the way it does.

## 3.1 Between-firm variance dominates — fixed effects are mandatory

| | IEQ |
|---|---|
| between-firm share of variance | **0.762** |
| pooled AR(1) coefficient | 0.930 |
| within-firm AR(1) coefficient | **0.634** (τ = 2.2 yr) |

A pooled model achieving R² = 0.82 is reporting that firms stay near their own long-run level.
That is a fact about *cross-sectional heterogeneity*, not about *dynamics*, and it would be an
embarrassing thing for a referee to point out. Every model in this project is fitted in the
**within transform** (firm-demeaned), and the pooled result is reported only as a cautionary
baseline.

**Cost:** the within estimator has Nickell bias of order `−(1+ρ)/(T−1) ≈ −0.16` at T = 11.
The gap between 0.930 and 0.634 brackets the truth; the design must correct it (§4.3).

## 3.2 Within-firm predictability is low — the signal budget is small

Out-of-sample within-firm one-step R², firm-level 70/30 split:

| variable | scalar AR(1) | linear (12-term) | degree-2 STLSQ (thr 0.05) |
|---|---|---|---|
| X8 leverage asset | 0.439 | 0.437 | 0.443 (6 terms) |
| X6 receivable turnover | 0.389 | 0.401 | 0.367 (21 terms) |
| **IEQ** | 0.350 | 0.357 | 0.253 (20 terms) |
| X11 asset coverage | 0.321 | 0.329 | 0.308 (32 terms) |
| X7 inventory turnover | 0.306 | 0.266 | 0.093 (43 terms) |
| X1 gross profit | 0.085 | 0.044 | **−0.236 (43 terms)** |

Degree-2 STLSQ at a fixed threshold **overfits catastrophically** — worse than a one-parameter
AR(1) on eleven of twelve variables. This is not a verdict on SINDy; it is a verdict on an
untuned threshold with a 91-term library and an honest R² ceiling near 0.4. The design responds
with nested threshold selection, library restriction, and group sparsity (§4.2, E1).

## 3.3 Sampling is at the relaxation timescale

Within-firm ρ₁ and implied τ = −Δt/ln ρ₁:

```
X1 0.44 (1.2y)  X2 0.40 (1.1y)  X3 0.53 (1.6y)  X4 0.50 (1.5y)
X5 0.43 (1.2y)  X6 0.57 (1.8y)  X7 0.54 (1.6y)  X8 0.65 (2.3y)
IEQ 0.65 (2.3y) X10 0.47 (1.3y) X11 0.60 (2.0y) X12 0.49 (1.4y)
```

Δt/τ ≈ 0.4–0.9. Finite-difference derivatives are not derivatives. **Discrete-time SINDy only.**

(The IEQ entry above and the 0.634 in §3.1 are the same quantity under two estimators — the mean of
per-firm autocorrelations against the pooled within-OLS slope. The pooled within slope is 0.634,
τ = 2.20 yr; the notebook recomputes all twelve and prints them, so quote from that run.)

## 3.4 N is large, T is tiny — this is the ensemble regime

324 trajectories × 10 transitions. Good news: bootstrap over firms is cheap and statistically
clean, so E-SINDy inclusion probabilities are the natural inferential object. Bad news: no single
trajectory can identify anything, so all claims are population-level, and firm-specific vector
fields are out of reach without pooling.

## 3.5 The boundary is sparsely visited

22 entries and 12 recoveries. Any hitting-time validation has an effective sample size of 22.
Design accordingly: report intervals, use exact/permutation inference, never asymptotics.

---

# 4. Implementation architecture

**What exists today:** `kaggle_panel_sindy.ipynb` implements the estimator (§4.1), the evaluation
protocol (§4.2, §6), the §10 synthetic recovery gate, and E1, E2, E3 and E5 end to end. E4, E6,
E7, E8 and E9 are not written yet; E7 in particular is a standalone simulation study with no
dependence on this data and belongs in its own notebook. The layout below is the target for when
this outgrows one file.

```
sindy_distress/
├── data/
│   ├── prepare.py            # the pipeline already built
│   └── transforms.py         # asinh, within/between decomposition, invertibility
├── model/
│   ├── library.py            # polynomial, restricted, rational (SINDy-PI), custom
│   ├── panel_sindy.py        # ★ core contribution: FE + group-sparse STLSQ
│   ├── bias.py               # half-panel jackknife, Arellano–Bond GMM check
│   ├── ensemble.py           # E-SINDy: bootstrap over FIRMS, inclusion probabilities
│   └── baselines.py          # persistence, AR(1), VAR(1), panel-VAR, RF, LSTM
├── analysis/
│   ├── cohort.py             # distressed vs healthy vector fields + permutation test
│   ├── passage.py            # Monte-Carlo first-passage, KS/calibration
│   ├── modified_eq.py        # backward-error expansion, coarse-Δt bias study
│   └── stability.py          # fixed points, Jacobian spectra, distance to boundary
├── experiments/              # E0 … E9, one script each, seeded, logged
└── report/                   # tables + figures, auto-regenerated
```

## 4.1 Core estimator — `panel_sindy.py`

```python
def panel_stlsq(X, Y, firm_id, library, threshold, alpha,
                fixed_effects=True, group_sparse=True, max_iter=20):
    """
    Discrete-time sparse regression with unit heterogeneity.

    fixed_effects : demean X and Y within firm before regression.
                    Equivalent to profiling out alpha_i.
    group_sparse  : a library term is kept or dropped for ALL 12 equations
                    jointly (group-lasso style), not per-equation. This is
                    the right prior: "does receivable turnover x leverage
                    matter to this system", not "...to equation 7".
                    It also cuts the effective parameter count by ~12x.
    """
```

Two design choices worth defending in the paper:

- **Group sparsity across equations.** Standard SINDy thresholds each equation independently, which
  at 91 terms × 12 equations means 1,092 free parameters against 3,240 observations with R² ≈ 0.3.
  Group sparsity reduces the *selection* problem to 91 binary decisions. This is the single
  highest-leverage modification and is, as far as I can tell, not standard in the SINDy literature.

- **Fixed effects inside the sparse loop, not before it.** Demeaning and thresholding interact:
  a term that looks significant pooled can vanish within. Profiling `α_i` at each iteration is
  cheap and keeps the two consistent.

  *Settled during implementation:* one demeaning at the start **is** the profiled estimator, exactly.
  Because `Θ(x)` does not depend on `Ξ`, the profiled optimum for any support *S* is
  `α_i* = mean_i(Y) − mean_i(Θ_S) Ξ_S`, and restricting demeaned columns to *S* is the same object as
  demeaning the restricted columns. So there is no iteration here to get wrong, and no cost to pay.

## 4.2 Hyperparameter selection

Nested, firm-blocked:

```
outer loop: 5 folds, split BY FIRM (never by row — rows within a firm are dependent)
  inner loop: 4 folds on training firms
    grid: threshold ∈ logspace(-3, 0, 25)
          library ∈ {linear, deg2-full, deg2-restricted, deg2+rational}
          ridge alpha ∈ {1e-4, 1e-3, 1e-2, 1e-1}
    select by: mean multi-step rollout error at horizon 3
               (not one-step R² — one-step rewards persistence)
  refit on all training firms, evaluate on held-out firms
```

**Selecting on horizon-3 rollout rather than one-step R² is deliberate.** One-step error is
dominated by the trivially-predictable persistent component; a model can win it by doing nothing.
Rollout error is where structure has to actually earn its place.

## 4.3 Bias correction

Report the within estimate, the half-panel jackknife correction (Dhaene–Jochmans), and an
Arellano–Bond GMM estimate of the AR block as a cross-check. If all three agree on the *support*
of `Ξ̂` (which terms are nonzero) even where they disagree on magnitudes, the structural claim
survives — and that is the claim being made.

---

# 5. Experiment suite

Each experiment states a hypothesis, a protocol, a primary metric, and **what result would
falsify it**. The last column is the part that keeps this honest.

## E0 — Cleaning ablation *(reproducibility, half a day)*

**H:** The X15 desynchronisation materially changed prior findings.
**Protocol:** Run the full pipeline on (a) raw source, (b) cleaned data. Compare `Ξ̂` supports and
any published coefficient on corruption.
**Metric:** Jaccard overlap of selected supports; sign flips.
**Falsified if:** supports are identical — in which case say so plainly and move on.

*Value: this is a short, publishable methodological note in its own right if the effect is large,
and it inoculates the main paper against "did you check your data" reviews.*

## E1 — Library and identifiability study *(the foundation)*

**H:** There exists a threshold/library combination where degree-2 terms beat AR(1) on
horizon-3 rollout in the within transform.
**Protocol:** Full nested sweep of §4.2. Produce a **Pareto frontier**: number of active terms vs
held-out rollout error, one curve per library class. Overlay AR(1) and panel-VAR(1).
**Metric:** Pareto dominance; term count at the knee.
**Falsified if:** no degree-2 configuration dominates AR(1) anywhere on the frontier.

> **This is the gate experiment.** Given §3.2, there is a real chance it fails. If it does, the
> paper is still viable — see §7 — but the narrative changes, so run this first.

## E2 — Baseline ladder *(credibility)*

Persistence → per-variable AR(1) → VAR(1) pooled → panel-VAR with FE → linear SINDy →
degree-2 group-sparse SINDy → rational library (SINDy-PI) → random forest → small LSTM.

**Metric:** held-out horizon-1/3/5 rollout RMSE, firm-blocked, 20 seeds, with paired bootstrap CIs.
**Purpose:** the paper must show SINDy's position honestly, including where it loses. A sparse
model that ties a random forest at 1/50th the parameter count is a *good* result and should be
framed as one.

## E3 — Cohort contrast *(the primary scientific claim)*

**H:** Ever-distressed firms (n=28) occupy a structurally different vector field from the rest (n=296).

**Protocol:**
1. Fit `Ξ̂_D` on distressed cohort, `Ξ̂_H` on healthy cohort, same library and threshold.
2. Test statistic `T = ‖Ξ̂_D − Ξ̂_H‖_F` and per-term inclusion-probability differences from E-SINDy
   (500 bootstraps over firms within each cohort).
3. **Permutation null:** re-assign the cohort label across firms 5,000 times, preserving cohort
   sizes, refit, recompute `T`. This controls for the fact that a 28-firm fit is noisier than a
   296-firm fit — which a naive comparison would mistake for structure.
4. Report per-term two-sided p-values with Benjamini–Hochberg control.

**Metric:** permutation p-value on `T`; the list of terms with FDR < 0.10.
**Falsified if:** p > 0.10 — meaning distressed firms follow the same dynamics and merely start
closer to the boundary. **That is itself a publishable and somewhat striking finding**: it would say
distress is a matter of *initial condition and noise*, not of *distinct mechanism*.

## E4 — Unsupervised regime discovery *(the strong version of E3)*

**H:** The cohort split in E3 emerges from the data without using the distress label.

**Protocol:** Mixture-of-vector-fields / cluster-SINDy — EM over K ∈ {1..5} regimes, each with its
own `Ξ_k`, firms soft-assigned. Fit **without** the label. Then measure alignment between the
recovered partition and `ever_distressed`.
**Metric:** adjusted Rand index; BIC across K.
**Falsified if:** BIC selects K=1, or ARI ≈ 0.

*If this works it is the strongest result in the project — a label-free discovery of the distress
regime. If it fails, E3 still stands.*

## E5 — First-passage validation *(the reformulation's payoff)*

**H:** The fitted stochastic map reproduces the observed distribution of crossing times.

**Protocol:**
1. Fit on firms observed 2013–2018.
2. For each firm, Monte-Carlo rollout 2019→2023 (10,000 paths, residual bootstrap for `ε`,
   preserving the cross-equation covariance `Σ̂_ε`).
3. Predicted `P_i(h)` vs realised crossings.
**Metrics:** KS test on predicted vs empirical hitting times; reliability diagram and Brier score
for `P_i(5)`; **and** a leakage-free comparison against the trivial rule "rank by current IEQ".
**Falsified if:** the dynamical model fails to beat ranking by current IEQ. Report this either way —
the trivial baseline is the honest bar and it is a strong one.

## E6 — Exogenous input and the identification caveat *(SINDyc, with teeth)*

**H:** X15 carries information beyond a year fixed effect. *Expected: it does not.*
**Protocol:** Fit (a) SINDyc with `u(t) = CPI`, (b) the same model with 10 year dummies. Compare
fit and coefficient stability.
**Metric:** likelihood-ratio / rollout difference between (a) and (b).
**Expected result and how to use it:** the two will be statistically indistinguishable, because CPI
is a deterministic function of year across all firms. **Report this as a methodological warning**:
any panel study attributing effects to a national annual index without within-year variation is
identifying a time effect and calling it corruption. Combined with E0, this is a genuinely useful
cautionary contribution to the empirical-finance literature.

## E7 — Coarse sampling and the modified equation *(the distinctive contribution)*

**H:** At Δt/τ ≈ 0.5, the discrete map recovered by SINDy is a biased representative of any
underlying continuous vector field, and the bias is characterisable.

**Protocol (synthetic, ground truth known):**
1. Construct a 12-dimensional ODE with known sparse quadratic `f`, tuned so its relaxation times
   match the empirical τ ∈ [1.1, 2.3].
2. Sample N=324 trajectories at T=11 points, Δt ∈ {0.1, 0.25, 0.5, 1, 2} × τ, with noise matched
   to the empirical residual scale and firm-level random intercepts matched to the 0.762
   between-variance share.
3. Recover the discrete map; attempt the inverse `f̃ = log(F̂)/Δt`; expand in Δt.
4. Measure support recovery (precision/recall of the true nonzero terms) and coefficient bias
   as a function of Δt/τ.

**Metric:** support F1 and relative coefficient bias vs Δt/τ; the value of Δt/τ at which support
recovery degrades below 0.8.
**Deliverable:** a **practitioner's admissibility curve** — "for SINDy on panel data, you need
Δt/τ below *this*." That is a reusable, citable result independent of the finance application,
and it is exactly the intersection of perturbation analysis and ML that this project is best
positioned to produce.

## E8 — Governance as bifurcation parameter

**H:** Board independence (X13) indexes distinct vector fields.
**Protocol:** Stratify firms by time-averaged X13 into tertiles; fit `Ξ̂` per stratum; permutation
test as in E3. Also treat X13 as a continuous modulation: augment the library with `X13 × Θ(x)`
interaction terms and let group sparsity decide.
**Metric:** permutation p-value; which interaction terms survive FDR.
**Falsified if:** no stratum difference — likely, given X13's near-constancy, and worth reporting
as a bound.

## E9 — Stability and boundary geometry *(interpretation layer)*

Not a hypothesis test; the interpretive payoff.
- Fixed points of `F̂` in the within transform; Jacobian eigenvalues; is the healthy state a stable
  node and is there a saddle near the boundary?
- Distance from each firm's fitted attractor to `S`, as an interpretable risk coordinate.
- Whether the distressed cohort's field has an eigenvalue crossing the unit circle — i.e. a genuine
  **bifurcation** interpretation of distress onset.
**Deliverable:** the figure the paper is remembered for — phase portrait projected onto
(IEQ, leverage), boundary `S` drawn, the 22 observed crossings overlaid on the fitted flow.

---

# 6. Evaluation protocol (fixed in advance, applied everywhere)

| Choice | Value | Why |
|---|---|---|
| Split unit | **firm**, never row | rows within a firm are dependent; row splits leak |
| Folds | 5 outer × 4 inner, nested | threshold must not be chosen on test firms |
| Repetitions | 20 seeds | 28-firm cohorts are small; single splits are noise |
| Primary metric | horizon-3 rollout RMSE | one-step rewards doing nothing (§4.2) |
| Secondary | support stability (E-SINDy inclusion prob.) | the claim is about structure, not accuracy |
| Inference | permutation + BH-FDR | n=22 crossings; asymptotics are not available |
| Reporting | paired bootstrap CIs on all deltas | point estimates alone will not survive review |
| Uncertainty | E-SINDy over 500 firm-bootstraps | coefficients reported as distributions |

**Two leakage traps, both live in this dataset:**
1. `IEQ < 0` twice ⇒ `Y` at 99.8%. Never report a distress-classification metric that uses
   contemporaneous IEQ without stating this identity in the same sentence.
2. Firm identity. Any pooled fit without fixed effects is partly memorising which firm it is
   looking at (§3.1).

---

# 7. Pre-registered interpretation — including the null

The value of stating this now is that the project cannot fail silently.

| E1 gate | E3 cohort | Resulting paper |
|---|---|---|
| deg-2 beats AR(1) | significant | **Strong.** "Nonlinear structure exists and differs by cohort." Full three-layer story, first-passage validation as the capstone. |
| deg-2 beats AR(1) | not significant | **Good.** "Shared nonlinear dynamics; distress is initial condition + noise." Reframes distress as a stochastic-boundary phenomenon rather than a mechanistic one. |
| deg-2 does *not* beat AR(1) | significant | **Good.** "Linear dynamics, cohort-dependent coefficients." Then the contribution is panel-SINDy methodology + the cohort finding. |
| deg-2 does *not* beat AR(1) | not significant | **Still publishable, as a negative result done properly.** "Annual financial-ratio panels do not support nonlinear dynamical structure at Δt = 1 yr, and here is the identifiability analysis (E7) showing why." Paired with E0 and E6 this is a *methodological caution* paper — and given how much of the ML-in-finance literature reports inflated numbers from row-level splits and pooled fits, it has a real audience. |

The fourth row is the likely one on current evidence. **Design the project so that outcome is a
contribution rather than a disappointment** — which means E0, E6, and E7 must be executed to
publication standard regardless of what E1 and E3 return, because they carry the paper in that case.

---

# 8. Narrative and venue

**Option A — methods-forward** (recommended given §7).
*"Sparse structure discovery in short panels: identifiability limits and a first-passage
formulation of corporate distress."* Contributions: panel-SINDy with FE and group sparsity; the
Δt/τ admissibility curve; the first-passage reformulation; the empirical application as
demonstration. Venues: *Chaos*, *Physica A*, *Journal of Computational Science*, *Nonlinear
Dynamics*.

**Option B — application-forward** (only if E3 or E4 comes back significant).
*"Financial distress as a boundary-crossing phenomenon: evidence from Indonesian listed firms."*
Venues: *Journal of Economic Dynamics and Control*, *Emerging Markets Review*, *Quantitative
Finance*, *International Review of Financial Analysis*.

**Option C — the short cautionary note** (E0 + E6, standalone, fast).
A 4–6 page note on the desynchronisation error and the year-effect identification trap. Fast to
write, cites cleanly, and can be spun out immediately without waiting for the main study.

---

# 9. Order of operations

```
Week 1     E0  cleaning ablation                          → Option C draft in parallel
Weeks 2-3  panel_sindy.py + baselines.py + eval harness
Week 4     E1  GATE — library/identifiability sweep       → decides the narrative
Week 5     E2  baseline ladder
Weeks 6-7  E3  cohort contrast + permutation inference
Week 8     E4  unsupervised regime discovery
Weeks 9-10 E7  synthetic coarse-sampling study            → runs independent of E1/E3
Week 11    E5  first-passage validation
Week 12    E6, E8, E9 + figures
```

E7 is deliberately parallel: it does not depend on the outcome of the gate, and it is the
component most likely to be cited on its own.

---

# 10. First thing to build

Before any of E1–E9: a **synthetic-data recovery test** for `panel_sindy.py`. Generate data from a
known sparse map with N=324, T=11, firm intercepts calibrated to the 0.762 between-variance share,
and noise at the empirical residual scale. If the estimator cannot recover a known map under
exactly this design, no result on the real data means anything. This is half a day and it is the
difference between a study and a set of numbers.
