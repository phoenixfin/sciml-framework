# Financial Distress Panel — Data Preparation Report

**Source:** `datadrtpm0Rev1.xlsx`, sheet `2013-2023` (the `.csv` is the same data at 3-decimal precision — not used)
**Output:** 324 firms × 11 years = 3,564 firm-years, balanced, no gaps, no duplicate firm-years
**Purpose:** structure discovery with SINDy

---

## 1. The one serious error: X15 was desynchronised from YEAR

The Corruption Perceptions Index is a **single national figure per year**, but the source file contained **six distinct values in every single year**, with near-identical distributions across all eleven years.

The diagnosis is unambiguous. The multiset of X15 values across the panel is exactly

| value | 32 | 34 | 36 | 37 | 38 | 40 |
|---|---|---|---|---|---|---|
| occurrences ÷ 325 firms | 1 | 3 | 1 | 3 | 2 | 1 |

which is precisely the multiplicity structure of Indonesia's CPI series for 2013–2023. The correct values were all present — they were attached to the wrong rows. This is the signature of a spreadsheet re-sort that moved the firm rows while leaving the CPI column in place.

**80.1% of rows (2,856 of 3,564) carried the wrong CPI value.** The column has been rebuilt deterministically from `YEAR` using the published series:

```
2013:32  2014:34  2015:36  2016:37  2017:37  2018:38
2019:40  2020:37  2021:38  2022:34  2023:34
```

Source: Transparency International. Cross-checks: 2019 = 40 is Indonesia's all-time high; 2022 = 34 was the steepest single-year fall since 1995; 2023 = 34, unchanged.

If any prior analysis used X15, it needs rerunning.

---

## 2. Other repairs (all logged in `quality_log.csv`, 24 rows)

| Code | Count | What |
|---|---|---|
| `DUP_FIRM_BLOCK` | 11 | PYFA entered twice as two complete 11-year blocks (`NO` 5737–5747 and 5761–5771). Financial ratios agree exactly; governance values disagree in 5 firm-years. Kept the lower-`NO` block. **Verify against PYFA's annual reports.** |
| `X15_DESYNC` | 1 | See above. |
| `PROP_GT_1` | 5 | X13 > 1 at GEMS 2018 (3), ASRI 2021 (2), DNET 2019 (2.5); X14 > 1 at MBSS 2023 (1.14), LTLS 2019 (5.0). These look like undivided head counts. Set to NaN. |
| `DIV_BY_ZERO` | 2 | X13 was `#DIV/0!` at UNIC 2023 and LTLS 2021 — total commissioners recorded as 0. Recoverable from the annual reports. |
| `NEGATIVE_CASH` | 4 | X5 < 0 for ARTI 2020–2023. Cash cannot be negative. Set to NaN. |
| `ZERO_LIABILITY` | 1 | X9 = 0 (zero total liabilities), so Equity/Liability is undefined. |

**Not repaired, but you should know:**

- **X4 and X5 don't match `rumus_variabel.docx`.** As written there, (TA − inventory)/TA and Cash/TA must lie in [0,1]. The data reaches ~247 (DNET 2014 in both). They were computed against *current liabilities*, not total assets. Either fix the formula document or recompute the columns — I can't recompute without the raw financials.
- **Unit inconsistency.** X1, X2, X8, X9 are in percent (X8 median 25.71, i.e. 25.7% leverage); X3–X7 and X10–X12 are plain ratios. Fraction-scale copies (`*_frac`) are included for interpretability. SINDy standardises anyway, so this doesn't affect fitting — only interpretation of coefficients.
- **Extreme values are flagged, not removed.** X1 reaches −130,563% and X2 −149,189%. These are real firms in real trouble; deleting them would delete the phenomenon. 270 rows carry ≥1 flag at the 0.5/99.5 percentile. The asinh transform handles the tails without discarding anything.

---

## 3. Modelling decisions baked into `panel_sindy_ready.csv`

**X9 replaced by `IEQ = Equity / Total Liability`.** X9 = TL/E has a pole at E = 0, and Y is *defined* as E < 0 for two consecutive years — the pole sits exactly where the event happens, and no polynomial library can represent a pole. The reciprocal is bounded near the event, passes smoothly through zero, and its sign *is* the distress condition. Verified: "IEQ < 0 for two consecutive years" reproduces Y in **99.8%** of comparable rows. Distress is now a **zero crossing of a smooth state**, not a rare binary label.

**Excluded from the state vector:**
- **X13, X14** — within-firm lag-1 autocorrelation ≈ −0.1, near-constant with rare jumps. They are parameters that index *which* vector field a firm lives on, not states. Retained as columns.
- **X15** — identical across all firms in a given year, hence perfectly collinear with a year fixed effect. Usable as an exogenous input u(t) in SINDyc, but any coefficient on it absorbs COVID-2020, commodity cycles, everything macro. You cannot attribute it to corruption. Retained as a column.

**Transform.** `asinh(x / robust_sd)` then standardised. asinh is smooth, odd, sign-preserving, linear near zero and logarithmic in the tails — unlike log it needs no shift and it maps the distress boundary IEQ = 0 to exactly 0. Scales are in the `transform_meta` sheet so the transform is invertible.

> ⚠ **The `*_z` columns break that last property, for IEQ only.** Standardising subtracts the mean, so in `IEQ_equity_to_liability_z` the boundary sits at `−z_mean/z_std = −0.6727`, not at 0. Concretely: **146** rows have `IEQ < 0`, but **2,580** rows have `IEQ_z < 0`. Do not read the sign of `IEQ_z` as the distress condition. Model on the `*_t` columns with IEQ scaled but not centred — `kaggle_panel_sindy.ipynb` Part 1 does exactly this and asserts the two counts agree. The other eleven states are unaffected; only IEQ has a meaningful zero.

**Five cells imputed** by within-firm linear interpolation so trajectories stay contiguous (PySINDy rejects NaN). All five are in the log.

---

## 4. Feasibility check on the cleaned data

Discrete-time STLSQ, degree-2 library, 3,240 one-step pairs, 70/30 split **by firm**, out-of-sample R²:

| variable | SINDy | persistence | linear |
|---|---|---|---|
| X1 gross profit | 0.724 | **0.795** | 0.800 |
| X3 current ratio | 0.759 | 0.752 | 0.766 |
| X8 leverage asset | 0.888 | 0.886 | 0.888 |
| **IEQ equity/liability** | 0.822 | 0.816 | 0.822 |
| X10 DSCR | 0.667 | **0.692** | 0.705 |

The verdict from before is unchanged and now cleaner: **the quadratic terms do not earn their place.** SINDy never beats plain linear, and on three variables it loses to doing nothing. The recovered equation for the boundary variable is a single term:

```
IEQ(t+1) = 0.936 · IEQ(t)
```

That is not a failure — it's a result. What it also says is that a *pooled* model over all 324 firms has no nonlinear structure to find.

> ⚠ **Read the whole of this section as pooled, and therefore as an upper bound on nothing.** 76.0% of IEQ's variance is *between* firms, so an R² of 0.822 is mostly the model reporting that firms stay near their own long-run level. The same coefficient estimated within firm is **0.634**, i.e. τ = 2.2 yr, not the 15 years the pooled 0.936 implies — a materially different statement about how fast distress can develop. `RESEARCH_DESIGN.md` §3.1 is the corrected version and every model in the project is fitted in the within transform. The "near-random-walk" reading of these numbers does not survive the correction; the first-passage framing does, and for a better reason — τ = 2.2 yr with Δt = 1 yr is the coarse-sampling regime of §3.3.

---

## 5. Where I'd point the search next

1. **Cohort-contrast, not prediction.** Fit separate vector fields for the 28 ever-distressed firms (`ever_distressed == 1`) and the 296 others, then compare the sparse coefficient sets. "Distressed firms occupy a structurally different vector field" is testable and doesn't fight the 3.2% class imbalance.
2. **First-passage.** With IEQ(t+1) = a·IEQ(t) + noise as the null, estimate the hitting-time distribution for IEQ = 0 and ask which firms deviate from it. Deviation from the null *is* the signal.
3. **E-SINDy for confidence intervals.** 324 trajectories is the one real advantage of this dataset — bootstrap over firms to get coefficient distributions rather than point estimates.
4. **The sampling-rate ceiling.** Within-firm lag-1 autocorrelation is 0.33–0.65, i.e. a relaxation time of 1–2 years at Δt = 1 year. Continuous-time SINDy is not defensible here; discrete-time is. **If quarterly filings are obtainable, that single change would do more for this project than any modelling choice.**

---

## Files

| File | Contents |
|---|---|
| `panel_clean.csv` | Cleaned panel, original units, `*_frac` copies, `flag_extreme_*` columns |
| `panel_sindy_ready.csv` | State variables raw (`X`), asinh (`X_t`), standardised (`X_z`), plus parameters and `ever_distressed` |
| `quality_log.csv` | One row per repair — firm, year, column, old value, new value, reason |
| `datadrtpm_clean.xlsx` | All of the above plus `README`, `transform_meta`, `cpi_reference` |
| `prepare_sindy_data.py` | The pipeline. Rerun it to regenerate everything from source: `python prepare_sindy_data.py path/to/datadrtpm0Rev1.xlsx [outdir]`. |
| `kaggle_panel_sindy.ipynb` | Panel-SINDy: the estimator, the synthetic recovery gate, and E1/E2/E3/E5. Runs unattended on Kaggle CPU (~3 min). |
| `RESULTS.md` | What the run of 2026-08-09 found, read against the pre-registration. |
| `results_2026-08-09.json` | That run's raw numbers, as the notebook emitted them. |

Load the state — **not** the `*_z` columns, for the reason in §3:

```python
import numpy as np, pandas as pd

STATE = ["X1_gross_profit", "X2_net_profit_margin", "X3_current_ratio", "X4_quick_ratio",
         "X5_cash_ratio", "X6_receivable_turnover", "X7_inventory_turnover",
         "X8_leverage_asset", "IEQ_equity_to_liability", "X10_dscr",
         "X11_asset_coverage", "X12_mbv"]

s = pd.read_csv("panel_sindy_ready.csv").sort_values(["SHARE_CODE", "YEAR"])
Tt = s[[c + "_t" for c in STATE]].to_numpy(float)       # asinh, uncentred
center = Tt.mean(axis=0)
center[STATE.index("IEQ_equity_to_liability")] = 0.0    # keep the boundary at zero
Z = (Tt - center) / Tt.std(axis=0, ddof=1)
X = Z.reshape(s.SHARE_CODE.nunique(), -1, len(STATE))   # (firms, years, states)
```

PySINDy's `STLSQ` will fit this, but it thresholds each equation independently and has no notion of
firm effects — with 91 terms × 12 equations against 3,240 dependent observations, that is the
overfitting reported in `RESEARCH_DESIGN.md` §3.2. Use `panel_stlsq` in the notebook instead, which
profiles out `α_i` and thresholds terms jointly across equations. Either way: `discrete_time=True`,
never continuous (§4 and design §3.3).
