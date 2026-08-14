# Panel-SINDy results — run of 2026-08-09

Produced by `kaggle_panel_sindy.ipynb` with `QUICK = False`, `SEED = 0`, 169 s on CPU — no GPU, no
Kaggle. Raw numbers in `results_2026-08-09.json`; figures in `outputs/figures/`
(`synthetic_recovery`, `E1_pareto`, `E2_ladder`, `E5_first_passage`; gitignored — rerun to
regenerate). Every number below is from that run, not from memory.

---

## Verdict, against the pre-registration

`RESEARCH_DESIGN.md` §7 pre-registers four outcomes. **This run lands in row 4** — degree-2 does not
beat AR(1), and the cohorts are not distinguishable — which the design already committed to
treating as "still publishable, as a negative result done properly". The methods-forward narrative
(Option A) is the one the evidence supports.

| Experiment | Result |
|---|---|
| Estimator gate (§10) | **PASS** — support F1 = 1.000 at threshold 0.0346, recovering all 20 true terms, coefficient MAE 0.019 |
| E1 gate | **NOT passed** — best nonlinear library is 0.0003 better than AR(1)+FE, interval spans zero |
| E2 ladder | Only VAR(1) pooled beats AR(1)+FE. Degree-2 is clearly *worse* |
| E3 cohort | p = 0.7245 on 5,000 permutations; 0 terms at FDR < 0.10 |
| E5 first passage | Ranks better than the trivial rule (AUC 0.672 vs 0.654), calibrates worse than the base rate (Brier 0.0691 vs 0.0452) |

## The one finding that changes what to do next

**Most of what looks like model error is firm-effect estimation noise.** Held-out firms get their
`α_i` from a six-year burn-in — five pairs. Give the same panel-VAR(1)+FE model an oracle `α_i`
(read off the whole sample, so it is not a reportable result) and horizon-3 RMSE drops from
**0.8161 to 0.5549**. That gap of 0.262 is **four times** the largest gap between any competitor and
AR(1)+FE in the whole ladder (0.066, degree-2 group-sparse) and **twice** the spread from the best
competitor to the worst (0.124).

So the ladder is not measuring what it looks like it is measuring. With 76% of variance between
firms, the binding constraint on this data is not the functional form of `F` — it is how precisely
`α_i` can be pinned down from a short window. That reframes the project: the payoff is in
shrinkage or hierarchical pooling of the firm effects, not in a richer library. It also explains
why VAR(1) pooled wins the ladder and why plain persistence beats AR(1)+FE — both sidestep the
noisy intercept.

This is a stronger version of §3.1's warning. §3.1 says a pooled fit is partly memorising firm
identity; this run says that on a *held-out* firm, not knowing the identity precisely enough costs
more than every modelling choice combined.

Appendix A2 puts a number on "combined". At high thresholds every library collapses to `Ξ̂ = 0`,
which predicts the firm's burn-in mean and nothing else; that scores **0.8658**, against **0.8380**
for the best configuration in the entire sweep. All the dynamics in this project are worth
**0.028**. The firm effect is worth **0.261** — **9.4× more**. Whatever the next experiment is, it
should be aimed at the 0.261.

## Panel diagnostics

324 firms × 11 years = 3,564 firm-years, 3,240 one-step pairs, 12 states. 116 distress firm-years,
28 ever-distressed firms.

**Boundary confirmed.** The shipped `IEQ_z` column places `IEQ = 0` at `z = −0.6727`, so its sign is
not the distress condition: **146** rows have `IEQ < 0`, but **2,580** have `IEQ_z < 0`. The state
used here rebuilds `IEQ` uncentred and the notebook asserts the counts agree. See `DATA_REPORT.md`
§3.

| state | between share | pooled ρ | within ρ | τ (yr) |
|---|---|---|---|---|
| X1 | 0.646 | 0.869 | 0.556 | 1.71 |
| X2 | 0.410 | 0.622 | 0.344 | 0.94 |
| X3 | 0.688 | 0.872 | 0.516 | 1.51 |
| X4 | 0.660 | 0.852 | 0.511 | 1.49 |
| X5 | 0.689 | 0.864 | 0.521 | 1.54 |
| X6 | 0.764 | 0.917 | 0.613 | 2.05 |
| X7 | 0.813 | 0.899 | 0.434 | 1.20 |
| X8 | 0.841 | 0.959 | 0.672 | 2.52 |
| **IEQ** | **0.760** | **0.930** | **0.634** | **2.20** |
| X10 | 0.625 | 0.795 | 0.411 | 1.13 |
| X11 | 0.692 | 0.890 | 0.582 | 1.85 |
| X12 | 0.658 | 0.766 | 0.295 | 0.82 |

IEQ reproduces `RESEARCH_DESIGN.md` §3.1 (0.762 / 0.930 / 0.634) to three decimals. **This table
supersedes §3.3**, whose twelve within-ρ values were computed under a different estimator and
disagree with §3.1 on IEQ itself (0.65 vs 0.634); quote these.

Δt/τ ranges over **0.40 to 1.22** across the twelve states — O(1) everywhere, so the coarse-sampling
argument of §3.3 holds for every variable, not just the slow ones. Continuous-time SINDy stays
inadmissible.

## Estimator gate — synthetic recovery

Known sparse quadratic map, 20 true terms of 90, N = 324, T = 11, between-share 0.83 realised
against a 0.76 target, 0.046% of simulated states clipped (low enough that the recovered map is the
map that generated the data).

| threshold | terms | precision | recall | F1 | coef MAE |
|---|---|---|---|---|---|
| 0.0010 – 0.0092 | 90 | 0.222 | 1.000 | 0.364 | 0.0200 |
| 0.0143 | 67 | 0.299 | 1.000 | 0.460 | 0.0199 |
| 0.0222 | 21 | 0.952 | 1.000 | 0.976 | 0.0191 |
| **0.0346** | **20** | **1.000** | **1.000** | **1.000** | **0.0191** |
| 0.0538 | 13 | 1.000 | 0.650 | 0.788 | 0.0186 |
| 0.0838 | 8 | 1.000 | 0.400 | 0.571 | 0.0202 |
| 0.1304+ | 0 | — | — | 0.000 | 0.0285 |

Exact recovery, but over a **narrow window**: perfect at 0.0346, and by 0.0838 recall has fallen to
0.40. Recall collapses faster than precision, so on real data an over-aggressive threshold silently
deletes true terms rather than admitting false ones. That is the argument for the nested selection
of §4.2 rather than a fixed threshold.

Worth reading alongside E1: the threshold nested CV settles on for the real panel is **0.0024**,
which on this synthetic study is deep in the no-thresholding regime (all 90 terms retained). On the
real data that threshold is applied to a 12-term linear library and so drops nothing either. In
other words the selected model is not sparse *because sparsity helped* — it is sparse because the
library it selected was already small.

## E1 — library sweep

Nested 5×4 firm-blocked CV, 144 configurations. Mean held-out horizon-3 RMSE **0.8398 ± 0.0430**
(sd over folds). All five folds selected `linear`.

Best configuration per library, paired against AR(1)+FE fold by fold (bar: 0.8385):

| library | threshold | ridge | terms | RMSE | Δ vs AR(1) | ±2 s.e. | beats |
|---|---|---|---|---|---|---|---|
| linear | 0.0010 | 1e-2 | 12.0 | 0.8380 | −0.0005 | [−0.0048, +0.0039] | no |
| lin+sq | 0.0433 | 1e-2 | 12.0 | 0.8382 | −0.0003 | [−0.0080, +0.0075] | no |
| deg2r | 0.0433 | 1e-2 | 15.6 | 0.8426 | +0.0041 | [+0.0018, +0.0064] | no |
| deg2 | 0.0811 | 1e-2 | 5.4 | 0.8439 | +0.0055 | [−0.0058, +0.0167] | no |

Note `lin+sq` keeps **12** terms at its best threshold: every square has been thresholded away, so
the "best nonlinear library" is the linear model wearing a different name. `deg2r` is the only
library that reliably retains quadratic terms, and it is reliably *worse* — its interval excludes
zero on the wrong side.

## E2 — baseline ladder

20 firm-blocked splits; Δ and CI are horizon-3, paired bootstrap over 324 firms, 2,000 draws.

| model | terms | h1 | h3 | h5 | Δ vs AR(1) | 95% CI |
|---|---|---|---|---|---|---|
| VAR(1) pooled | 12 | 0.5545 | **0.7562** | 0.8317 | −0.0559 | [−0.0856, −0.0263] |
| persistence | 12 | 0.5859 | 0.8001 | 0.8745 | −0.0130 | [−0.0445, +0.0191] |
| linear SINDy | 12 | 0.5979 | 0.8160 | 0.8632 | +0.0010 | [−0.0035, +0.0057] |
| AR(1)+FE | 12 | 0.5938 | 0.8161 | 0.8645 | — | — |
| panel-VAR(1)+FE | 12 | 0.5978 | 0.8161 | 0.8635 | +0.0011 | [−0.0034, +0.0059] |
| random forest | — | 0.6677 | 0.8286 | 0.8692 | +0.0081 | [−0.0132, +0.0298] |
| deg2 per-equation | 90 | 0.6651 | 0.8805 | 1.0104 | +0.0654 | [+0.0375, +0.1034] |
| deg2 group-sparse | 90 | 0.6651 | 0.8805 | 1.0106 | +0.0656 | [+0.0376, +0.1036] |
| *panel-VAR(1)+FE, oracle α* | *12* | *0.4889* | *0.5549* | *0.5397* | *−0.2622* | *[−0.2949, −0.2307]* |

The oracle row is italicised because it is not a competitor — it sees the evaluation window.

Two things worth reporting as-is. A **random forest does not beat a 12-parameter linear map** here,
which is the honest ceiling reference §5 E2 asks for. And group-sparse and per-equation STLSQ are
indistinguishable at degree 2 (0.8805 both) — group sparsity costs nothing in accuracy while
cutting selection from 1,080 decisions to 90, which is the methodological claim of §4.1 surviving
its own test even though the library it selects is not useful on this data.

## E3 — cohort contrast

`linear` library, threshold 0.0024, cohorts 28 / 296. Observed `T = ‖Ξ̂_D − Ξ̂_H‖_F = 1.6895`
against a permutation null with mean 1.8916 and 95th percentile 2.4699 — the observed statistic is
*below* the null mean. **p = 0.7245** over 5,000 permutations; no term survives BH at FDR 0.10
(smallest q = 0.802). E-SINDy inclusion probability is 1.000 for all twelve terms in both cohorts,
so there is no support difference to find either.

The permutation null is doing exactly the work §5 E3 anticipated: a naive reading of `T = 1.69`
would look like a large difference, and it is smaller than what label-shuffling produces by chance.

**Interpretation:** ever-distressed firms follow the same dynamics as everyone else. They start
closer to the boundary. Per §7 that is a publishable finding, and it is the one that makes the
first-passage reformulation load-bearing rather than decorative — if the mechanism were different,
you would model the mechanism; because it is not, the interesting object is the boundary geometry.

## E5 — first passage

Fitted on 2013–2018, 10,000 paths per firm, whole-row residual resampling. 316 firms at risk at
2018, 15 realised entries in 2019–2023.

* discrimination: **AUC 0.672** vs **0.654** for ranking by current IEQ — a real but slim margin on
  15 events;
* calibration: **Brier 0.0691** vs **0.0452** for predicting the base rate for everyone. Predicted
  mean P(entry within 5 yr) is 0.1537 against a realised 0.0475 — **over-predicting by 3.2×**.

The model knows roughly *who* is at risk and is badly wrong about *how much*. The over-prediction
is the expected consequence of rolling a linear map forward with pooled residuals: nothing in the
fitted dynamics reproduces the mean reversion that keeps real firms off the boundary, so paths
diffuse across it too easily. Reporting AUC alone here would be misleading, which is why §5 E5 asks
for the reliability diagram.

## What this implies for the plan

1. **Firm-effect shrinkage is now the highest-value next experiment**, ahead of E4. It is not in the
   design document; the oracle-α result argues it should be. The target is explicit: 0.261 of
   held-out RMSE sits in `α_i` estimation error, against 0.028 in the dynamics. Empirical-Bayes
   shrinkage of `α_i` towards the cohort mean, or a hierarchical prior with the between/within
   ratio estimated jointly, are the obvious first attempts — both are cheap and neither needs a
   richer library.
2. **E7 (the Δt/τ admissibility curve) is unaffected** and remains the most citable standalone
   component — §9 already schedules it in parallel for exactly this reason.
3. **E4 is unlikely to pay.** If a *supervised* cohort split shows no structural difference at
   p = 0.72, an unsupervised search for the same split has little to find. Worth one run at low
   priority to state the bound, not worth two weeks.
4. **E6 and E0 carry the paper** in the row-4 outcome, as §7 predicted. Neither is written yet.

## Caveats

* `X4` and `X5` are in the state vector with denominators that `DATA_REPORT.md` §2 flags as wrong —
  coefficients on them are not interpretable until the columns are recomputed from raw financials.
* Nickell bias is not yet corrected (§4.3): no half-panel jackknife, no Arellano–Bond cross-check.
  The support claims are unaffected, the magnitudes are not.
* Rational / SINDy-PI libraries are not implemented, so "no nonlinear structure" is a statement
  about polynomial libraries up to degree 2.
* 5 outer folds make the E1 gate interval wide. A tighter test needs more folds or more seeds; it
  would not change a −0.0003 margin into a finding.

---

# Appendix

## A1 — E3 per-term statistics, all twelve terms

Library `linear`, threshold 0.0024. `t_stat` is the per-term contribution to
`T = ||Xi_D - Xi_H||_F`, i.e. the row norm of the coefficient difference; `p` is its two-sided
permutation p-value over the same 5,000 label shuffles; `q_bh` is Benjamini-Hochberg adjusted.
Inclusion probabilities come from 500 bootstraps over firms *within* each cohort.

| term | t_stat | p | q_bh | incl. distressed | incl. healthy | Δ incl. |
|---|---|---|---|---|---|---|
| X11 | 0.4890 | 0.2026 | 0.8020 | 1.000 | 1.000 | +0.000 |
| X8 | 0.6423 | 0.2076 | 0.8020 | 1.000 | 1.000 | +0.000 |
| X5 | 0.5878 | 0.2657 | 0.8020 | 1.000 | 1.000 | +0.000 |
| X12 | 0.3791 | 0.3327 | 0.8020 | 1.000 | 1.000 | +0.000 |
| X3 | 0.8177 | 0.4283 | 0.8020 | 1.000 | 1.000 | +0.000 |
| X10 | 0.3648 | 0.5283 | 0.8020 | 1.000 | 1.000 | +0.000 |
| X6 | 0.3072 | 0.6089 | 0.8020 | 1.000 | 1.000 | +0.000 |
| X2 | 0.2627 | 0.6321 | 0.8020 | 1.000 | 1.000 | +0.000 |
| IEQ | 0.4505 | 0.6645 | 0.8020 | 1.000 | 1.000 | +0.000 |
| X1 | 0.3930 | 0.6993 | 0.8020 | 1.000 | 1.000 | +0.000 |
| X7 | 0.2963 | 0.7908 | 0.8020 | 1.000 | 1.000 | +0.000 |
| X4 | 0.5506 | 0.8020 | 0.8020 | 1.000 | 1.000 | +0.000 |

Nothing is close. The smallest raw p is 0.203 on X11, which does not survive twelve comparisons;
the smallest q is 0.802. Every term is selected in 100% of bootstraps in both cohorts, so the
cohorts do not differ in *which* terms are supported either — only, marginally, in their values.

## A2 — E1 descriptive frontier, all 144 configurations

Held-out horizon-3 rollout RMSE, averaged over the 5 outer folds, for every configuration scored
directly on the outer test firms. **Optimistic by construction** — the nested-CV estimate of
0.8398 is the honest number; this table is for reading where the knee is. `terms` is the mean
active-term count over folds.

Baselines on the same folds:

| baseline | terms | RMSE |
|---|---|---|
| persistence | 12.0 | 0.8255 |
| panelVAR+FE | 12.0 | 0.8381 |
| AR(1)+FE | 12.0 | 0.8385 |

Per-fold AR(1)+FE: 0.8109, 0.9030, 0.7872, 0.8357, 0.8557.

### linear (36 configurations)

| threshold | ridge 1e-04 | ridge 1e-03 | ridge 1e-02 |
|---|---|---|---|
| 0.0010 | 0.8381 (12) | 0.8381 (12) | 0.8380 (12) |
| 0.0019 | 0.8381 (12) | 0.8381 (12) | 0.8380 (12) |
| 0.0035 | 0.8381 (12) | 0.8381 (12) | 0.8380 (12) |
| 0.0066 | 0.8381 (12) | 0.8381 (12) | 0.8380 (12) |
| 0.0123 | 0.8381 (12) | 0.8381 (12) | 0.8380 (12) |
| 0.0231 | 0.8381 (12) | 0.8381 (12) | 0.8380 (12) |
| 0.0433 | 0.8381 (12) | 0.8381 (12) | 0.8380 (12) |
| 0.0811 | 0.8462 (5) | 0.8462 (5) | 0.8468 (5) |
| 0.1520 | 0.8658 (0) | 0.8658 (0) | 0.8658 (0) |
| 0.2848 | 0.8658 (0) | 0.8658 (0) | 0.8658 (0) |
| 0.5337 | 0.8658 (0) | 0.8658 (0) | 0.8658 (0) |
| 1.0000 | 0.8658 (0) | 0.8658 (0) | 0.8658 (0) |

### lin+sq (36 configurations)

| threshold | ridge 1e-04 | ridge 1e-03 | ridge 1e-02 |
|---|---|---|---|
| 0.0010 | 0.8425 (24) | 0.8422 (24) | 0.8403 (24) |
| 0.0019 | 0.8425 (24) | 0.8422 (24) | 0.8403 (24) |
| 0.0035 | 0.8425 (24) | 0.8422 (24) | 0.8403 (24) |
| 0.0066 | 0.8425 (24) | 0.8422 (24) | 0.8403 (24) |
| 0.0123 | 0.8424 (24) | 0.8421 (24) | 0.8402 (24) |
| 0.0231 | 0.8411 (20) | 0.8408 (20) | 0.8390 (20) |
| 0.0433 | 0.8415 (12) | 0.8391 (12) | 0.8382 (12) |
| 0.0811 | 0.8487 (5) | 0.8493 (5) | 0.8495 (4) |
| 0.1520 | 0.8658 (0) | 0.8658 (0) | 0.8658 (0) |
| 0.2848 | 0.8658 (0) | 0.8658 (0) | 0.8658 (0) |
| 0.5337 | 0.8658 (0) | 0.8658 (0) | 0.8658 (0) |
| 1.0000 | 0.8658 (0) | 0.8658 (0) | 0.8658 (0) |

### deg2r (36 configurations)

| threshold | ridge 1e-04 | ridge 1e-03 | ridge 1e-02 |
|---|---|---|---|
| 0.0010 | 0.8568 (45) | 0.8558 (45) | 0.8504 (45) |
| 0.0019 | 0.8568 (45) | 0.8558 (45) | 0.8504 (45) |
| 0.0035 | 0.8568 (45) | 0.8558 (45) | 0.8504 (45) |
| 0.0066 | 0.8568 (45) | 0.8558 (45) | 0.8504 (45) |
| 0.0123 | 0.8568 (45) | 0.8558 (45) | 0.8503 (44) |
| 0.0231 | 0.8541 (37) | 0.8534 (36) | 0.8486 (35) |
| 0.0433 | 0.8475 (20) | 0.8443 (20) | 0.8426 (16) |
| 0.0811 | 0.8489 (7) | 0.8484 (6) | 0.8520 (5) |
| 0.1520 | 0.8658 (0) | 0.8658 (0) | 0.8658 (0) |
| 0.2848 | 0.8658 (0) | 0.8658 (0) | 0.8658 (0) |
| 0.5337 | 0.8658 (0) | 0.8658 (0) | 0.8658 (0) |
| 1.0000 | 0.8658 (0) | 0.8658 (0) | 0.8658 (0) |

### deg2 (36 configurations)

| threshold | ridge 1e-04 | ridge 1e-03 | ridge 1e-02 |
|---|---|---|---|
| 0.0010 | 0.9107 (90) | 0.9070 (90) | 0.8875 (90) |
| 0.0019 | 0.9107 (90) | 0.9070 (90) | 0.8875 (90) |
| 0.0035 | 0.9107 (90) | 0.9070 (90) | 0.8875 (90) |
| 0.0066 | 0.9107 (90) | 0.9070 (90) | 0.8873 (90) |
| 0.0123 | 0.9097 (89) | 0.9061 (89) | 0.8862 (88) |
| 0.0231 | 0.9103 (71) | 0.9061 (71) | 0.8816 (67) |
| 0.0433 | 0.8645 (38) | 0.8616 (38) | 0.8459 (29) |
| 0.0811 | 0.8519 (13) | 0.8525 (13) | 0.8439 (5) |
| 0.1520 | 0.8630 (0) | 0.8658 (0) | 0.8658 (0) |
| 0.2848 | 0.8658 (0) | 0.8658 (0) | 0.8658 (0) |
| 0.5337 | 0.8658 (0) | 0.8658 (0) | 0.8658 (0) |
| 1.0000 | 0.8658 (0) | 0.8658 (0) | 0.8658 (0) |

Cells are `RMSE (mean active terms)`. Read down a column to see thresholding bite: on `deg2`
the term count falls from 90 to 0 across the grid while RMSE never drops below the AR(1) bar of
0.8385, which is the E1 result in its rawest form.

**The rows worth the most attention are the ones with zero terms.** At high thresholds every
library collapses to `Ξ̂ = 0`, which makes the prediction `x̂(t+h) = α_i` — the firm's burn-in mean,
held constant forever. That scores **0.8658**, identically across all 47 such configurations. The
best configuration anywhere in the sweep scores **0.8380**.

So the entire dynamical content of the best model — every coefficient, every library choice, the
whole sweep — is worth **0.0278** in horizon-3 RMSE over predicting a constant. Knowing `α_i`
precisely is worth **0.2612** (§ *The one finding*). **Locating the firm is 9.4× more valuable than
knowing how it moves.**

That is the sharpest form of this run's result, and it does not depend on the E1 gate verdict, the
choice of library, or the permutation test. It comes straight off the raw grid.

Two protocol notes for anyone cross-reading tables: the baselines here use the 5 outer folds of E1,
while §E2 uses 20 random 70/30 splits, so absolute values differ between the two (persistence is
0.8255 here, 0.8001 there). Compare within a table, not across. And the frontier is scored on the
outer test firms directly, so it is optimistic; 0.8398 from nested CV remains the honest headline.

