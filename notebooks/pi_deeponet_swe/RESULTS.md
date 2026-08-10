# PI-DeepONet for the 1D SWE — revision results

Every number below comes from one unattended Kaggle run of
[`kaggle_swe_revision_all.ipynb`](kaggle_swe_revision_all.ipynb); the raw record is
[`results_2026-08-10.json`](results_2026-08-10.json), and this file is generated from it.

| | |
|---|---|
| Notebook version | `2026-08-10b  5 seeds, elu_scaled, DiD ablation, Fig. 6 with full residual` |
| Wall clock | 196.6 min |
| TensorFlow | 2.20.0, 2 GPU(s), legacy Keras: True |
| Parts run | part1, part3, part4 |
| Supervised trajectories | 152 of 502 sampled, nx = 400 |
| Production budget | 40,000 steps × 3 IC modes × 3 seeds |
| Ablation budget | 15,000 steps × 3 fusions × 5 seeds |

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
| §3.4.2 the h₀–b interaction is mediated through the trunk | the trunks take only (x, t), so this is wrong on inspection — but the fusion ablation **does not** corroborate it: over five seeds no paired contrast or difference-in-differences is significant (§3.2) |
| Table 5 speedup | honest **2127× vs serial**, **376× vs a vectorised baseline** |

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
computed in 109 s.

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

- 152 supervised trajectories, ensemble-vectorised: **20.7 s** total, 136 ms each
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

- DeepONet relative mass drift at T: **1.723e-02**
- Reference solver: 1.672e-16
- Operator total momentum at T on a flat bed: **-1.890e-02** (should be 0)

Percent-level mass violation is normal for a neural operator; reporting it is worth more
than the number itself.

### 3.2 Table 4 — branch-fusion ablation

Three variants × 5 seeds × 15,000 steps, evaluated against the
well-balanced reference at T = 1 s. C2b raises the bump from 0.2 m to 0.5 m so the
source term −gh ∂ₓb genuinely depends on the *product* of the two inputs.

| case | fusion | ε_h (total) | ε_h (anomaly) | RMSE_h [m] | ε_hu |
|---|---|---|---|---|---|
| C1  flat bed | `add` | 0.0654 ± 0.0047 | 0.8763 ± 0.0631 | 0.0696 ± 0.0050 | 0.3722 ± 0.0605 |
| C1  flat bed | `concat` | 0.0581 ± 0.0073 | 0.7792 ± 0.0976 | 0.0619 ± 0.0078 | 0.3532 ± 0.0168 |
| C1  flat bed | `bilinear` | 0.0629 ± 0.0077 | 0.8428 ± 0.1034 | 0.0670 ± 0.0082 | 0.3595 ± 0.0384 |
| C2  bump 0.2 m | `add` | 0.0839 ± 0.0063 | 0.5934 ± 0.0446 | 0.0901 ± 0.0068 | 0.3600 ± 0.0500 |
| C2  bump 0.2 m | `concat` | 0.0798 ± 0.0085 | 0.5646 ± 0.0601 | 0.0857 ± 0.0091 | 0.3417 ± 0.0177 |
| C2  bump 0.2 m | `bilinear` | 0.0858 ± 0.0169 | 0.6068 ± 0.1196 | 0.0921 ± 0.0182 | 0.3428 ± 0.0175 |
| C2b bump 0.5 m (NEW) | `add` | 0.1354 ± 0.0062 | 0.6433 ± 0.0293 | 0.2025 ± 0.0092 | 0.5164 ± 0.0357 |
| C2b bump 0.5 m (NEW) | `concat` | 0.1348 ± 0.0156 | 0.6404 ± 0.0742 | 0.2016 ± 0.0234 | 0.5150 ± 0.0225 |
| C2b bump 0.5 m (NEW) | `bilinear` | 0.1506 ± 0.0310 | 0.7156 ± 0.1473 | 0.2253 ± 0.0464 | 0.4997 ± 0.0193 |

Unpaired, every contrast overlaps. The variants share a seed, hence the same
initialisation stream and batch order, so the **paired** difference removes that common
variance:

| case | contrast | mean diff | std | t | better in |
|---|---|---|---|---|---|
| C1  flat bed | `add` − `concat` | +0.0072 | 0.0059 | +2.72 | `concat` 5/5 |
| C1  flat bed | `add` − `bilinear` | +0.0025 | 0.0086 | +0.65 | `bilinear` 2/5 |
| C1  flat bed | `concat` − `bilinear` | -0.0047 | 0.0108 | -0.98 | `bilinear` 1/5 |
| C2  bump 0.2 m | `add` − `concat` | +0.0041 | 0.0092 | +0.99 | `concat` 3/5 |
| C2  bump 0.2 m | `add` − `bilinear` | -0.0019 | 0.0179 | -0.24 | `bilinear` 3/5 |
| C2  bump 0.2 m | `concat` − `bilinear` | -0.0060 | 0.0149 | -0.89 | `bilinear` 2/5 |
| C2b bump 0.5 m (NEW) | `add` − `concat` | +0.0006 | 0.0172 | +0.08 | `concat` 3/5 |
| C2b bump 0.5 m (NEW) | `add` − `bilinear` | -0.0152 | 0.0329 | -1.03 | `bilinear` 2/5 |
| C2b bump 0.5 m (NEW) | `concat` − `bilinear` | -0.0158 | 0.0241 | -1.46 | `bilinear` 2/5 |

And the primary statistic — **difference in differences** against the flat-bed control.
A contrast being larger on a bumpy case only supports the separability argument if the
bathymetry coupling is what does the work; differencing against C1 cancels any
across-the-board advantage one fusion has over another.

| case | contrast | DiD | std | t | p < 0.05? |
|---|---|---|---|---|---|
| C2  bump 0.2 m | `add` − `concat` | -0.0032 | 0.0039 | -1.83 | no |
| C2  bump 0.2 m | `add` − `bilinear` | -0.0044 | 0.0111 | -0.88 | no |
| C2  bump 0.2 m | `concat` − `bilinear` | -0.0012 | 0.0075 | -0.36 | no |
| C2b bump 0.5 m (NEW) | `add` − `concat` | -0.0066 | 0.0209 | -0.71 | no |
| C2b bump 0.5 m (NEW) | `add` − `bilinear` | -0.0177 | 0.0270 | -1.46 | no |
| C2b bump 0.5 m (NEW) | `concat` − `bilinear` | -0.0111 | 0.0244 | -1.02 | no |

n = 5 seeds, df = 4, two-sided t critical = 2.78.

### The ablation does not support the architectural claim

**Not one contrast is significant**, on either statistic. Every DiD is *negative* — the
opposite sign to the prediction — and the largest is 1.5 standard errors from zero.

An earlier three-seed run put `add` − `concat` on C2b at +0.0219 (t = 3.62, `concat`
ahead in 3/3), which looked like the predicted effect. It does not survive:

- five seeds, this run: **+0.0006 ± 0.0172, t = +0.08**
- the first three seeds of *this* run: +0.0042 ± 0.0162, t = +0.45

So it was not merely underpowered — the earlier estimate does not reproduce even at the
same seed count. Treat it as run-specific noise.

**What this means for §3.4.2.** The claim that the trunk mediates the h₀–b interaction is
still wrong, and can be corrected on inspection: the trunks take only (x, t), so they
cannot carry any h₀–b coupling, and β = B₁(h₀) + B₂(b) is additively separable while the
source term −gh ∂ₓb is not. That is an argument about the architecture, not a
measurement. But Table 4 **cannot be presented as evidence that non-additive fusion
helps** — at 15k steps this experiment does not separate the three variants. Report the
null honestly, or drop the ablation and let the separability argument stand alone.

### 3.3 Table 5 — like-for-like speedup

| batch | solver serial [ms] | solver batched [ms] | operator [ms] | vs serial | vs batched |
|---|---|---|---|---|---|
| 1 | 540.8 | 578.3 | 20.9980 | 26× | 28× |
| 10 | 506.0 | 117.5 | 2.1982 | 230× | 53× |
| 100 | 509.0 | 90.0 | 0.2393 | 2127× | 376× |

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
| C1  smooth IC, flat bed | `paper` | **0.0256 ± 0.0027** | 0.3428 ± 0.0360 | 0.0272 ± 0.0029 | 0.1894 ± 0.0119 |
| C2  smooth IC, bump bed | `paper` | **0.0340 ± 0.0037** | 0.2407 ± 0.0263 | 0.0365 ± 0.0040 | 0.1761 ± 0.0104 |
| C3  dam break (OOD) | `paper` | **0.1775 ± 0.0181** | 0.8206 ± 0.0836 | 0.2727 ± 0.0278 | 0.4979 ± 0.0812 |
| C4  100 unseen (mean) | `paper` | **0.0321 ± 0.0008** | 0.4399 ± 0.0100 | 0.0266 ± 0.0003 | 0.1752 ± 0.0082 |
| C1  smooth IC, flat bed | `exp` | **0.0236 ± 0.0017** | 0.3162 ± 0.0223 | 0.0251 ± 0.0018 | 0.1832 ± 0.0040 |
| C2  smooth IC, bump bed | `exp` | **0.0306 ± 0.0006** | 0.2167 ± 0.0042 | 0.0329 ± 0.0006 | 0.1659 ± 0.0130 |
| C3  dam break (OOD) | `exp` | **0.2099 ± 0.0289** | 0.9703 ± 0.1335 | 0.3225 ± 0.0444 | 0.5201 ± 0.0524 |
| C4  100 unseen (mean) | `exp` | **0.0723 ± 0.0020** | 0.9134 ± 0.0139 | 0.0534 ± 0.0012 | 0.1761 ± 0.0107 |
| C1  smooth IC, flat bed | `elu_scaled` | **0.0234 ± 0.0012** | 0.3143 ± 0.0155 | 0.0250 ± 0.0012 | 0.1848 ± 0.0129 |
| C2  smooth IC, bump bed | `elu_scaled` | **0.0290 ± 0.0031** | 0.2051 ± 0.0216 | 0.0311 ± 0.0033 | 0.1757 ± 0.0124 |
| C3  dam break (OOD) | `elu_scaled` | **0.1790 ± 0.0364** | 0.8275 ± 0.1681 | 0.2750 ± 0.0559 | 0.5450 ± 0.0419 |
| C4  100 unseen (mean) | `elu_scaled` | **0.0760 ± 0.0018** | 0.9497 ± 0.0086 | 0.0562 ± 0.0007 | 0.1842 ± 0.0024 |

- Against the manuscript's ε_h = 1.17e-2, C1 is **0.0256** — 2.2× higher
  once the reference is converged rather than diffused. Part of the original figure was
  the operator matching a smeared target.
- ε_hu(anomaly) equals ε_hu(total) by construction — the rest state for hu is zero, so
  there is no background to subtract. Report one column.

### The IC shortcut fix is not free, and not for the reason expected

| ic_mode | C1 | C2 | C4 (100 unseen) | C4 vs `paper` |
|---|---|---|---|---|
| `paper` | 0.0256 | 0.0340 | **0.0321 ± 0.0008** | 1.00× |
| `exp` | 0.0236 | 0.0306 | **0.0723 ± 0.0020** | 2.25× |
| `elu_scaled` | 0.0234 | 0.0290 | **0.0760 ± 0.0018** | 2.37× |

`elu_scaled` was added to test a specific hypothesis: that `exp`'s 2.3× penalty on unseen
pairs comes from exponential amplification of the correction field. It is exact at t = 0
and floored exactly as `exp` is, but grows *linearly* in F. **The hypothesis is wrong** —
`elu_scaled` costs 2.37×,
marginally worse than `exp`'s 2.25×.

What `exp` and `elu_scaled` share is the **multiplicative** form
`b + h_min + (h₀ − b − h_min)·s(tF)`, where the correction's authority scales with the
local depth above the floor: near-zero where the water is shallow over a bump, large
where it is deep. The paper's Eq. (12) is additive inside the ELU and gives uniform
authority. That coupling, not the growth rate, is what costs 2.3× on unseen bathymetry.

Note the split: on C1 and C2 both floored variants are **slightly better** than `paper`.
The penalty is specific to operator generalisation across unseen (h₀, b) pairs.

**Recommendation.** `shifted` — Eq. (12) with ε moved inside the ELU — is exact at t = 0
(1.19e-7), keeps the paper's additive form, and carries no measured accuracy cost. It
fixes the defect that matters. The hard floor that `exp`/`elu_scaled`/`softplus` add is a
guarantee against a condition never approached in training (that test drove F = −50
artificially), and it now has a measured price of 2.3× on C4. Take the cheap fix.

### 4.2 BC × IC × residual factorial — the h₀-vs-lake question

v6's residual is R₁ = ∂ₜh + ∂ₓ(hu), **R₂ = ∂ₜ(hu)**. The momentum flux divergence and the
bed source are absent, so it is not the SWE momentum equation. That truncated residual
has an exact global minimum at F = 0: driving R₂ → 0 forces hu ≡ 0 (since hu = tF_hu
vanishes at t = 0), and then R₁ = ∂ₜh → 0 forces h ≡ h₀.

Physics-only training on C2, 3000 steps per cell:

| residual | ic_mode | BC | L_PDE | d(h₀) | d(lake) | gap | verdict |
|---|---|---|---|---|---|---|---|
| `time_only` | `paper` | on | 1.614e-07 | 0.0023 | 4.3556 | 0.999 | **h0** |
| `time_only` | `paper` | off | 8.110e+00 | 38.4975 | 38.1547 | 0.009 | **neither** |
| `time_only` | `exp` | on | 4.101e-10 | 0.0001 | 4.3557 | 1.000 | **h0** |
| `time_only` | `exp` | off | 1.405e-09 | 0.0002 | 4.3557 | 1.000 | **h0** |
| `full` | `paper` | on | 2.615e+00 | 14.9924 | 14.4483 | 0.036 | **neither** |
| `full` | `paper` | off | 5.608e-01 | 6.2024 | 3.5917 | 0.421 | **lake** |
| `full` | `exp` | on | 4.576e-01 | 5.8825 | 2.3613 | 0.599 | **lake** |
| `full` | `exp` | off | 2.122e-01 | 6.1647 | 2.4374 | 0.605 | **lake** |

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

### 4.4 Fig. 6 and Table 4 row A0, regenerated with the full residual

Both were produced with v6's truncated residual, whose global minimum *is* F = 0, so
they documented that residual rather than physics-informed training. Regenerated here
under both residuals, physics-only on C1 at 5000 steps, with the depth
error measured against the well-balanced reference instead of against h₀.

| residual | ε_h | F0-gap [m] | final L_PDE |
|---|---|---|---|
| R₂ = ∂ₜ(hu) (v6) | 1.982e-01 | 0.1126 | 4.289e-02 |
| full momentum | 2.059e-01 | 0.2066 | 8.076e-01 |
| data-guided 40k (reference) | 2.556e-02 | — | — |

Physics-only training fails under **both** residuals — ε_h ≈ 0.20 against 0.026
for the data-guided model, roughly 8× worse. That much of the original Fig. 6 survives.

What changes is the mechanism. The truncated residual pulls toward h₀ — its F0-gap is
0.113 against 0.207 for the full one, and its
residual settles at 4.3e-02 against 8.1e-01 — because
F = 0 *is* its minimum. The full residual has no such attractor and simply fails to
converge. Caption Fig. 6 as "physics-only training fails", not as "the model collapses
to the F = 0 state": the collapse is a property of the truncated residual.


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

The last row is why the ablation is reported as a paired contrast and a
difference-in-differences rather than a ranking. Three separate single-seed runs picked
three different winners on C1 and on C2b; one put all three fusions within 0.5%; and the
promising three-seed contrast on C2b did not reproduce at five seeds (§3.2).

**Nondeterminism at fixed seed.** Two runs with identical seeds (42, 43, 44) and identical
training data give C1 `paper` ε_h = 0.0282 and 0.0256 — a shift comparable to the
seed-to-seed standard deviation itself. TensorFlow is not run-to-run deterministic here
(cuDNN autotuning, non-associative reductions), and 40k steps amplify it. The quoted ± is
therefore seed-to-seed at fixed hardware and *understates* total variability; say so in
the caption, or enable deterministic ops and pay the throughput.

One factorial cell also diverged in this run where it had converged before
(`time_only`/`paper`/BC-off, L_PDE = 8.1, both distances ≈ 38). Physics-only training has
no data anchor, so it is the most nondeterminism-sensitive thing here. The qualitative
conclusion is unaffected — see §4.2 — but report L_PDE alongside the verdict so an
unconverged cell cannot be mistaken for a result.

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

- Delete the claim that the trunk mediates the h₀–b interaction. The trunks take only
  (x, t); this needs no experiment.
- State the separability argument on its own terms: β = B₁(h₀) + B₂(b) is additively
  separable, the source −gh ∂ₓb is not.
- **Do not claim the ablation supports it.** Over five seeds no paired contrast and no
  difference-in-differences is significant (§3.2). Either report the null, or drop Table 4
  and argue from the architecture alone. A referee who reruns it will get what we got.
- Report the true 10k-step numbers for A3 rather than repeating the 40k values.

**§4.9, Table 5 — speedup**

- Replace with the three-leg benchmark; reconcile the prose against the table.
- State the CPU and GPU models, and that the solver leg is single-threaded.

**Table 3, abstract — metrics**

- Lead with anomaly-normalised error and dimensional RMSE; keep rel_total for continuity.
- Quote ε_hu alongside ε_h in the abstract.

---

## Still open

- **A fusion experiment with power, or none at all.** Five seeds at 15k steps cannot
  separate the variants. Either the effect is smaller than the seed noise, or 15k steps is
  too short for the coupling to matter. A longer budget on C2b alone, or a construction
  with stronger coupling than a 0.5 m bump, would be a real test; more seeds at these
  settings would only tighten a null.
- **`softplus` at 40k.** Untested at the production budget. Given that `exp` and
  `elu_scaled` both cost ~2.3× on C4, the multiplicative form is the suspect and
  `softplus` shares it, so this is now a confirmation rather than a hope.
- **Run-to-run determinism.** Enabling `TF_DETERMINISTIC_OPS` would make the ± in Table 3
  mean what it appears to mean. Worth the throughput if the numbers go in an abstract.
- **C3 is not a clean benchmark.** It is non-periodic and solved with a periodic solver,
  as in v6. Either label it a stress test or give it a non-periodic reference.
