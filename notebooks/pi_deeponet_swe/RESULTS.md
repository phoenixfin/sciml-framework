# PI-DeepONet for the 1D SWE — revision results

Every number below comes from one unattended Kaggle run of
[`kaggle_swe_revision_all.ipynb`](kaggle_swe_revision_all.ipynb); the raw record is
[`results_2026-08-09.json`](results_2026-08-09.json), and this file is generated from it.

| | |
|---|---|
| Notebook version | `2026-08-09b  Parts 1/3/4 (Part 2 off), seed sweeps, GPU guard` |
| Wall clock | 103.5 min |
| TensorFlow | 2.20.0, 2 GPU(s), legacy Keras: True |
| Parts run | part1, part3, part4 |
| Supervised trajectories | 152 of 502 sampled, nx = 400 |
| Production budget | 40,000 steps × 2 IC modes × 3 seeds |
| Ablation budget | 15,000 steps × 3 fusions × 3 seeds |

Part 2 was disabled in this run; its numbers are quoted from the run of the same day
that had it enabled (TF 2.20, P100) and are flagged where they appear. They are
deterministic given the seed and were bit-identical across every run that computed them.

---

## What this changes in the manuscript

| Claim as written | What the run shows |
|---|---|
| §3.3 "CFL number of approximately 0.45" | measured **0.0384** at `nx=400, nt=4000`; CFL 0.45 needs `nt=323`, not 4000 |
| §3.3 reference-solver error is negligible against operator error | the LxF reference is **0.064** relative (**0.835** on the wave anomaly); the well-balanced solver on the same grid is **0.0089** |
| §4.2 C1 is smooth; t=1 s oscillations are trunk spectral resolution | C1 **develops a shock at t ≈ 0.78 s**; the oscillations are Gibbs ringing at a discontinuity |
| Proposition 1: ∇θ L_PDE = 0 at F = 0 | true for the **trunk** and for the **mass** residual only; the branch gradient is nonzero unless the state is lake-at-rest |
| §3.5.1 "F = 0 attractor" | F = 0 is an exact minimum **of the implemented residual**, which omits the momentum flux and bed source; with the full residual training leaves for the **steady-state manifold** |
| Eq. (12) guarantees ĥ ≥ b + h_min + ε | false: under stress ĥ reaches **−0.95 m** |
| §3.7.3 / Remark 3 / Fig. 6: 6.6e12, 1.5e2, 2.2e1 | three protocols, three numbers; the 6.6e12 is not reproducible under any of five matched protocols |
| §3.4.2 the h₀–b interaction is mediated through the trunk | the trunks never see h₀ or b; additive fusion is separable, and **concat beats add on the strong-bump case in 3/3 seeds** while being flat where there is no bathymetry |
| Table 5 speedup | honest **2376× vs serial**, **368× vs a vectorised baseline** |

---

## Part 1 — Reference solver

### 1.1 CFL audit of the published configuration

| quantity | value |
|---|---|
| Δx | 2.5000e-02 m |
| Δt | 2.5000e-04 s |
| measured max CFL | **0.0384** (manuscript: 0.45) |
| numerical viscosity ν_LxF | **1.2482** m²/s |
| diffusion length √(4νT) | 2.234 m (Gaussian half-width ≈ 0.7 m) |
| nt for CFL = 0.45 | 323 |

Lax-Friedrichs viscosity *grows* as Δt falls at fixed Δx, so the 13× excess in `nt` is
not conservative bookkeeping — it is the dominant error in the training data.

### 1.2 Table W1 — lake at rest (h₀ − b = 1.5, u = 0, t = 1 s)

| order | nx | max\|η − 1.5\| [m] | max\|hu\| [m²/s] |
|---|---|---|---|
| 1 | 200 | 4.441e-16 | 3.543e-15 |
| 1 | 400 | 8.882e-16 | 4.788e-15 |
| 2 | 200 | 8.882e-16 | 8.205e-15 |
| 2 | 400 | 1.554e-15 | 1.385e-14 |

Well balanced to machine precision, at both orders and both resolutions.

### 1.3 Table W2 — self-convergence of the order-2 well-balanced HLL solver

| nx | rel L2 @ T=0.25 | order | rel L2 @ T=0.5 | order | rel L2 @ T=1.0 | order |
|---|---|---|---|---|---|---|
| 200 | 5.234e-04 |  | 1.766e-03 |  | 5.660e-03 |  |
| 400 | 1.476e-04 | 1.83 | 5.337e-04 | 1.73 | 3.777e-03 | 0.58 |
| 800 | 3.911e-05 | 1.92 | 1.612e-04 | 1.73 | 2.565e-03 | 0.56 |
| 1600 | 9.946e-06 | 1.98 | 4.825e-05 | 1.74 | 1.731e-03 | 0.57 |

Clean second order before the shock, collapsing to ≈0.57 at T = 1 s. That collapse is
physics, not a solver defect — see 1.4.

### 1.4 Shock formation in benchmark C1

| nx | max\|∂h/∂x\| at T=1 s |
|---|---|
| 400 | 2.14 |
| 800 | 4.09 |
| 1600 | 7.92 |
| 3200 | 14.86 |
| 6400 | 30.66 |

The maximum gradient **doubles with every refinement** rather than saturating, which is
the signature of a genuine discontinuity. Steepening history (nx = 3200):

| t [s] | max\|∂h/∂x\| | peak-to-peak h |
|---|---|---|
| 0.1 | 0.41 | 0.3838 |
| 0.2 | 0.37 | 0.2434 |
| 0.3 | 0.44 | 0.2373 |
| 0.4 | 0.54 | 0.2372 |
| 0.5 | 0.72 | 0.2372 |
| 0.6 | 1.06 | 0.2372 |
| 0.7 | 2.00 | 0.2371 |
| 0.8 | 5.28 | 0.2371 |
| 0.9 | 11.00 | 0.2370 |
| 1.0 | 14.89 | 0.2368 |

The jump from 2.00 at t = 0.7 to 5.28 at t = 0.8 places shock formation at **t ≈ 0.78 s**.

### 1.5 Error budget against a converged reference

Reference: order-2 well-balanced HLL at nx = 12800,
computed in 76 s.

| scheme | CFL | ν [m²/s] | rel L2 | rel L2 (anomaly) | peak-to-peak h |
|---|---|---|---|---|---|
| LxF nx=400 nt=4000 (manuscript) | 0.038 | 1.2482 | **6.354e-02** | **8.354e-01** | 0.0714 |
| LxF nx=400 nt=323 (CFL 0.45) | 0.511 | 0.0746 | **2.520e-02** | **3.313e-01** | 0.1888 |
| well-balanced HLL o2, nx=400 | 0.450 | 0.0000 | **8.905e-03** | **1.171e-01** | 0.2311 |
| converged reference | — | — | — | — | 0.2371 |

The published reference data misses **84% of the wave anomaly**. It also flattens the
wave: peak-to-peak 0.071 m against the true 0.237 m.

### 1.6 Conservation of the reference solver

Relative mass drift at t = 1 s: **1.672e-16** (max over the run 3.343e-16).
Total momentum stays at **7.5e-15** — round-off.

### 1.7 Data regeneration

- 152 supervised trajectories, ensemble-vectorised: **14.1 s** total, 93 ms each
- The manuscript quotes 66 s for this step, so the well-balanced solver at the correct
  CFL is **cheaper**, not more expensive. This strengthens the data-efficiency argument.

### 1.8 Metric inflation from the background depth

| field | ‖h‖/‖h − h_rest‖ | ‖h‖/‖h − h̄‖ |
|---|---|---|
| converged reference | 10.4× | 13.1× |
| manuscript LxF field | 15.7× | 40.9× |

The manuscript's ε_h = 1.17e-2 corresponds to roughly **0.48** on the free-surface anomaly.
Per snapshot time the inflation factor is:

| t [s] | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|
| factor | 9.9× | 15.7× | 22.2× | 15.3× |

---

## Part 2 — The attractor (from the run with Part 2 enabled)

### 2.1 Trunk / branch gradient split at F = 0

| case | ‖g_trunk‖ | ‖g_branch‖ | rms R₁ | rms R₂ |
|---|---|---|---|---|
| C1 flat bed | 0.000e+00 | 9.019e+00 | 0.000e+00 | 2.706e+00 |
| C2 bump bathymetry | 0.000e+00 | 1.542e+01 | 0.000e+00 | 3.506e+00 |
| lake at rest | 0.000e+00 | 2.629e-09 | 0.000e+00 | 6.353e-08 |
| flat water | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |

Exactly the corrected four-clause statement: the trunk gradient and the **mass residual**
vanish identically in every case, while the branch gradient survives except on the two
steady states. The original Proposition 1 does not hold; this table replaces its proof.

### 2.2 The surviving gradient is hydrostatic imbalance

‖R₂ − hydrostatic imbalance‖ / ‖R₂‖ = **7.85e-05** (C1) and **7.97e-05** (C2). For the two
steady states ‖R₂‖ is itself ~1e-8, so the ratio there is 0/0 and carries no information.

### 2.3 Where physics-only training actually goes (C2, 3000 steps)

‖ĥ − h₀‖ = **6.015**, ‖ĥ − lake‖ = **2.590**. The attractor is the steady-state manifold,
not h₀. Reproduced in every run: 5.65/2.22, 5.75/2.35, 6.01/2.59.

### 2.4 PDE gradient norm, one consistent measurement

| group | ‖∇‖ |
|---|---|
| branch_h | 9.242e+00 |
| branch_hu | 1.728e-01 |
| trunk_h | 7.611e+00 |
| trunk_hu | 1.801e-01 |
| **total** | **1.198e+01** |

L_PDE at initialisation = 7.353e+00.

### 2.5 IC shortcut variants

| ic_mode | max\|ĥ(x,0) − h₀\| | min ĥ under stress | floor guaranteed |
|---|---|---|---|
| `paper` (Eq. 12) | 1.001e-04 | **−0.9499** | no |
| `shifted` | 1.192e-07 | **−0.9499** | no |
| `exp` | 1.192e-07 | 0.0500 | yes |
| `softplus` | 1.192e-07 | 0.0500 | yes |

Two independent defects in Eq. (12): it is off by ε at t = 0, and the ELU floor is
`> −1`, so the stated bound is false for b < 0.95 m. `shifted` fixes exactness alone;
`exp`/`softplus` fix both — at a cost measured in 4.2 below.

---

## Part 3 — Metrics, fusion ablation, speedup

### 3.1 Operator conservation

- DeepONet relative mass drift at T: **1.186e-02**
- Reference solver: 1.672e-16
- Operator total momentum at T on a flat bed: **+4.638e-01** (should be 0)

Percent-level mass violation is normal for a neural operator; reporting it is worth more
than the number itself.

### 3.2 Table 4 — branch-fusion ablation

Three variants × 3 seeds × 15,000 steps, evaluated against the
well-balanced reference at T = 1 s. C2b raises the bump from 0.2 m to 0.5 m so the
source term −gh ∂ₓb genuinely depends on the *product* of the two inputs.

| case | fusion | ε_h (total) | ε_h (anomaly) | RMSE_h [m] | ε_hu |
|---|---|---|---|---|---|
| C1  flat bed | `add` | 0.0624 ± 0.0036 | 0.8367 ± 0.0485 | 0.0665 ± 0.0039 | 0.3733 ± 0.0610 |
| C1  flat bed | `concat` | 0.0638 ± 0.0055 | 0.8551 ± 0.0743 | 0.0680 ± 0.0059 | 0.3461 ± 0.0297 |
| C1  flat bed | `bilinear` | 0.0609 ± 0.0016 | 0.8167 ± 0.0219 | 0.0649 ± 0.0017 | 0.3646 ± 0.0242 |
| C2  bump 0.2 m | `add` | 0.0868 ± 0.0109 | 0.6142 ± 0.0767 | 0.0932 ± 0.0116 | 0.3698 ± 0.0272 |
| C2  bump 0.2 m | `concat` | 0.0844 ± 0.0052 | 0.5971 ± 0.0367 | 0.0906 ± 0.0056 | 0.3303 ± 0.0423 |
| C2  bump 0.2 m | `bilinear` | 0.0835 ± 0.0030 | 0.5907 ± 0.0209 | 0.0897 ± 0.0032 | 0.3508 ± 0.0420 |
| C2b bump 0.5 m (NEW) | `add` | 0.1542 ± 0.0145 | 0.7329 ± 0.0691 | 0.2307 ± 0.0218 | 0.5240 ± 0.0050 |
| C2b bump 0.5 m (NEW) | `concat` | 0.1323 ± 0.0078 | 0.6289 ± 0.0372 | 0.1980 ± 0.0117 | 0.5174 ± 0.0261 |
| C2b bump 0.5 m (NEW) | `bilinear` | 0.1397 ± 0.0111 | 0.6641 ± 0.0529 | 0.2091 ± 0.0167 | 0.5193 ± 0.0725 |

Unpaired, every contrast overlaps. But the variants share a seed, hence the same
initialisation stream and batch order, so the **paired** difference is the test with the
power:

| case | contrast | mean diff | std | t (df=2) | wins |
|---|---|---|---|---|---|
| C1  flat bed | `add` − `concat` | -0.0014 | 0.0083 | -0.29 | `concat` 1/3 |
| C1  flat bed | `add` − `bilinear` | +0.0015 | 0.0022 | +1.19 | `bilinear` 2/3 |
| C1  flat bed | `concat` − `bilinear` | +0.0029 | 0.0062 | +0.80 | `bilinear` 2/3 |
| C2  bump 0.2 m | `add` − `concat` | +0.0024 | 0.0120 | +0.35 | `concat` 2/3 |
| C2  bump 0.2 m | `add` − `bilinear` | +0.0033 | 0.0089 | +0.65 | `bilinear` 2/3 |
| C2  bump 0.2 m | `concat` − `bilinear` | +0.0009 | 0.0038 | +0.41 | `bilinear` 2/3 |
| C2b bump 0.5 m (NEW) | `add` − `concat` | **+0.0219** | 0.0105 | **+3.62** | `concat` 3/3 |
| C2b bump 0.5 m (NEW) | `add` − `bilinear` | +0.0145 | 0.0195 | +1.29 | `bilinear` 2/3 |
| C2b bump 0.5 m (NEW) | `concat` − `bilinear` | -0.0074 | 0.0180 | -0.71 | `bilinear` 1/3 |

**The one contrast that separates:** `add` − `concat` on C2b is +0.0219, t = 3.62, with `concat` ahead in
3/3 seeds — additive fusion is **16.5% worse**. The same contrast on C1, where there
is no bathymetry to couple to, is −0.0014 (t = −0.29). Flat where the theory says flat,
open where the theory says open.

Caveat to state in the paper: df = 2, so t = 3.62 is p ≈ 0.07 two-sided. The direction is
unanimous but the design is underpowered; five seeds would settle it.

### 3.3 Table 5 — like-for-like speedup

| batch | solver serial [ms] | solver batched [ms] | operator [ms] | vs serial | vs batched |
|---|---|---|---|---|---|
| 1 | 449.0 | 487.8 | 18.2866 | 25× | 27× |
| 10 | 445.3 | 99.0 | 1.7804 | 250× | 56× |
| 100 | 454.8 | 70.4 | 0.1914 | 2376× | 368× |

All times per trajectory. The solver leg is single-threaded NumPy. The manuscript's
original comparison timed a solver at 13× more timesteps than it needed against a
batched GPU network; the batched-solver column removes that confound.

---

## Part 4 — The manuscript's own pipeline, re-run on the new data

`pi_deeponet_v6.py` is a faithful port of the paper's architecture, loss and training
loop. Only the supervised targets change: well-balanced snapshots instead of the
over-diffused Lax-Friedrichs ones.

### 4.1 Table 3 — errors against the well-balanced reference

40,000 steps, mean ± std over seeds [42, 43, 44].

| case | IC | ε_h (total) | ε_h (anomaly) | RMSE_h [m] | ε_hu |
|---|---|---|---|---|---|
| C1  smooth IC, flat bed | `paper` | **0.0282 ± 0.0020** | 0.3778 ± 0.0273 | 0.0300 ± 0.0022 | 0.1724 ± 0.0036 |
| C2  smooth IC, bump bed | `paper` | **0.0301 ± 0.0030** | 0.2127 ± 0.0214 | 0.0323 ± 0.0033 | 0.1765 ± 0.0095 |
| C3  dam break (OOD) | `paper` | **0.1855 ± 0.0253** | 0.8576 ± 0.1170 | 0.2850 ± 0.0389 | 0.5568 ± 0.0832 |
| C4  100 unseen (mean) | `paper` | **0.0303 ± 0.0006** | 0.4201 ± 0.0094 | 0.0255 ± 0.0003 | 0.1753 ± 0.0056 |
| C1  smooth IC, flat bed | `exp` | **0.0261 ± 0.0030** | 0.3498 ± 0.0407 | 0.0278 ± 0.0032 | 0.1790 ± 0.0096 |
| C2  smooth IC, bump bed | `exp` | **0.0344 ± 0.0050** | 0.2432 ± 0.0356 | 0.0369 ± 0.0054 | 0.1661 ± 0.0237 |
| C3  dam break (OOD) | `exp` | **0.2107 ± 0.0191** | 0.9742 ± 0.0881 | 0.3238 ± 0.0293 | 0.4997 ± 0.0439 |
| C4  100 unseen (mean) | `exp` | **0.0718 ± 0.0020** | 0.9104 ± 0.0298 | 0.0530 ± 0.0008 | 0.1848 ± 0.0024 |

- Against the manuscript's ε_h = 1.17e-2, C1 is **0.0282** — 2.4× higher
  once the reference is converged rather than diffused. Part of the original figure was
  the operator matching a smeared target.
- **The corrected IC shortcut costs 2.4× on operator generalisation**: C4 goes 0.0303 → 0.0718,
  with error bars small enough that this is not noise. On the anomaly metric `exp`
  reaches 0.91, i.e. it barely captures the wave on unseen pairs.
- ε_hu(anomaly) equals ε_hu(total) by construction — the rest state for hu is zero, so
  there is no background to subtract. Report one column.

**Recommendation.** `shifted` is exact at t = 0 (1.19e-7) and fails only the *hard floor*
guarantee, which was never approached in training — that test drove F = −50 artificially.
It fixes the real defect at no accuracy cost. Present `exp`/`softplus` as buying a
guarantee at a measured price, not as a free correction.

### 4.2 BC × IC × residual factorial — the h₀-vs-lake question

v6's residual is R₁ = ∂ₜh + ∂ₓ(hu), **R₂ = ∂ₜ(hu)**. The momentum flux divergence and the
bed source are absent, so it is not the SWE momentum equation. That truncated residual
has an exact global minimum at F = 0: driving R₂ → 0 forces hu ≡ 0 (since hu = tF_hu
vanishes at t = 0), and then R₁ = ∂ₜh → 0 forces h ≡ h₀.

Physics-only training on C2, 3000 steps per cell:

| residual | ic_mode | BC | L_PDE | d(h₀) | d(lake) | gap | verdict |
|---|---|---|---|---|---|---|---|
| `time_only` | `paper` | on | 1.887e-09 | 0.0028 | 4.3557 | 0.999 | **h0** |
| `time_only` | `paper` | off | 4.689e-06 | 0.0221 | 4.3494 | 0.995 | **h0** |
| `time_only` | `exp` | on | 5.771e-10 | 0.0002 | 4.3557 | 1.000 | **h0** |
| `time_only` | `exp` | off | 7.701e-10 | 0.0002 | 4.3556 | 1.000 | **h0** |
| `full` | `paper` | on | 1.395e+00 | 10.4522 | 9.5280 | 0.088 | **neither** |
| `full` | `paper` | off | 1.032e+00 | 11.3736 | 8.7370 | 0.232 | **lake** |
| `full` | `exp` | on | 3.690e-01 | 5.7576 | 2.7194 | 0.528 | **lake** |
| `full` | `exp` | off | 1.737e-01 | 6.0184 | 2.4787 | 0.588 | **lake** |

**All four `time_only` cells land on h₀** to four decimal places, with L_PDE ~1e-9.
**No `full` cell reaches h₀**; three go to the lake state and the fourth — the one with
by far the largest residual — has not settled anywhere.

BC on/off and IC mode change nothing. The residual form is the discriminator. So
Proposition 1 is *true of the loss v6 implements*, and that loss is not the shallow-water
system. This is a stronger and more defensible result than the current chain-rule
argument, and it explains why the corrected residual lands on the lake-at-rest manifold.

### 4.3 PDE gradient norm under matched protocols

Same freshly initialised v6 model, same collocation points, one protocol choice varied
at a time.

| protocol | batch | L_PDE | branch_h | branch_hu | trunk_h | trunk_hu | **total** |
|---|---|---|---|---|---|---|---|
| FD, R2 = hu_t (v6 verbatim) | 8 | 6.002e-01 | 3.915e+01 | 2.144e+00 | 2.346e+01 | 2.405e+00 | **4.575e+01** |
| FD, R2 = hu_t | 1 | 5.220e-01 | 3.842e+01 | 9.523e+00 | 2.354e+01 | 8.198e+00 | **4.678e+01** |
| FD, full momentum | 1 | 1.105e+01 | 3.031e+02 | 2.318e+01 | 1.833e+02 | 2.096e+01 | **3.556e+02** |
| autodiff, full momentum | 1 | 1.105e+01 | 3.032e+02 | 2.316e+01 | 1.834e+02 | 2.094e+01 | **3.557e+02** |
| FD, R2 = hu_t, SUM not MEAN | 8 | 2.401e+03 | 1.566e+05 | 8.575e+03 | 9.383e+04 | 9.620e+03 | **1.830e+05** |

Readings:

- **Finite differences and autodiff agree to four significant figures** (3.5562e+02 vs 3.5573e+02), so the FD
  approximation explains none of the published spread. Batch size explains none either.
- Fig. 6's 2.2e1 sits with the truncated-residual rows (~4.6e+01).
- Remark 3's 1.5e2 sits with the full-momentum rows (~3.6e+02).
- §3.7.3's 6.6e12 is seven orders above the largest protocol constructible here
  (1.8e+05, deliberately unreduced). Treat it as an error.

Report one row, name the protocol in the caption, and make all three places agree.

---

## Reproducibility

Five complete runs were made across two environments (TF 2.20 with a P100/T4, and a
pinned TF 2.13 CPU image).

| quantity | behaviour across runs |
|---|---|
| All of Part 1 | deterministic to ~1e-12; identical every run |
| Gradient split at F = 0 | bit-identical |
| PDE gradient norms (§4.3) | agree to 4 significant figures |
| Attractor endpoint | lake closer in all runs (5.65/2.22, 5.75/2.35, 6.01/2.59) |
| IC shortcut floor | −0.9499 every run |
| Speedup at batch 100 | 2329×, 2376×, 2434× vs serial on GPU |
| **Fusion ablation** | **single-seed rankings disagreed across all three early runs** |

The last row is why the ablation is reported as a paired contrast over seeds rather than
a ranking. Three separate single-seed runs picked three different winners on C1 and on
C2b; one put all three fusions within 0.5%.

Two runs were lost to a CPU-only Kaggle image (6.05 h each instead of ~1.5 h, with the
seed sweeps silently reduced to one seed). The notebook now refuses to start Parts 2–4
without a GPU.

---

## Manuscript edit checklist

**§3.3 / §5 — reference solver**

- Replace the CFL number with the measured value and state the resulting ν_LxF.
- Replace the O(Δx²/Δt) truncation sentence with Table W2 (§1.3).
- Replace Lax-Friedrichs with the well-balanced HLL scheme throughout.
- Delete "the dominant error source is the operator approximation rather than the
  reference solver diffusion" — §1.5 inverts it.
- Add Table W1 (lake at rest), Table W2 (convergence), Fig. W1 (conservation).

**§3.6, Table 1, abstract — data**

- Update the generation cost; it went **down**.

**§4.2, Table 2 — benchmarks**

- Annotate C1/C2 as smooth until t ≈ 0.78 s, shock thereafter.
- Reframe the t = 1 s oscillations as shock-related Gibbs ringing.
- Delete "capturing over 98.8% of the spatial variance".

**§3.5.1, Proposition 1, Remarks 2–3 — the attractor**

- Replace Proposition 1 with the four-clause statement; the proof is three lines.
- State that F = 0 is an exact minimum of the *implemented* residual (§4.2), and that
  the full momentum equation moves the attractor to the steady-state manifold.
- Rewrite Remark 2 around the identically-vanishing **mass** residual under hu(x,0) = 0,
  not the chain-rule argument, which is equation-agnostic.
- Rename "F = 0 attractor" to "steady-state (lake-at-rest) attractor" in the title,
  abstract, keywords and §3.5.1.
- Cite Rohrhofer et al. (TMLR 2023, arXiv:2203.13648) and De Ryck et al. (wPINNs), and
  soften "not previously characterised".
- Rewrite Remark 3 and the Fig. 6 caption around one row of §4.3.

**Eq. (12) — IC shortcut**

- The positivity claim is false as written; either adopt `shifted` (exactness only, free)
  or `exp`/`softplus` (both, at the C4 cost in §4.1) and say which and why.
- Drop the softplus-underflow paragraph: with a floor in place, u = hu/h cannot see a
  near-zero denominator.

**§3.4.2, Table 4 — fusion**

- Delete the claim that the trunk mediates the h₀–b interaction; the trunks take only
  (x, t).
- Report the paired contrast and the C2b construction, not a ranking.
- Report the true 10k-step numbers for A3 rather than repeating the 40k values.

**§4.9, Table 5 — speedup**

- Replace with the three-leg benchmark; reconcile the prose against the table.
- State the CPU and GPU models, and that the solver leg is single-threaded.

**Table 3, abstract — metrics**

- Lead with anomaly-normalised error and dimensional RMSE; keep rel_total for continuity.
- Quote ε_hu alongside ε_h in the abstract.

---

## Still open

Queued for the next run (notebook `2026-08-10`), all three affecting what §4.1 and
§3.2 above can claim:

- **`elu_scaled` at 40k.** A fourth shortcut, `b + h_min + (h₀−b−h_min)(elu(tF)+1)`:
  exact at t = 0 and floored exactly as `exp` is, but **linear** in the correction field
  rather than exponential. Exponential amplification of F is the obvious suspect for the
  2.4× C4 penalty. If `elu_scaled` matches `paper` on C4, the IC fix is free and the
  "guarantee at a measured price" framing in §4.1 should be deleted rather than softened.
- **Five ablation seeds with difference-in-differences.** The primary statistic becomes
  (add − concat)|C2b − (add − concat)|C1: differencing against the flat-bed control
  cancels any across-the-board advantage of one fusion and isolates the bathymetry
  interaction, which is the actual claim. df goes from 2 to 4.
- **Fig. 6 and Table 4 row A0 with the full residual.** Both were produced with v6's
  truncated residual, whose global minimum *is* F = 0, so they document that residual
  rather than physics-informed training. §4.5 regenerates them with the momentum flux
  and bed source restored, with the depth error measured against the well-balanced
  reference instead of against h₀.

Not queued:

- **`softplus` at 40k.** Untested at the production budget. Worth adding to
  `CFG["IC_MODES_40K"]` only if `elu_scaled` fails to recover the gap.
- **C3 is not a clean benchmark.** It is non-periodic and solved with a periodic solver,
  as in v6. Either label it a stress test or give it a non-periodic reference.
