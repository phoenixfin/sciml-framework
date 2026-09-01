# PINN for the dispersive Boussinesq (VBM) system

Can a physics-informed neural network solve the **variational Boussinesq model**
on the standard shallow-water run-up benchmarks — and can the result be checked
against something other than a picture?

The system is the weakly nonlinear VBM of Adytia et al. (2019),
*Computational Geosciences*, eqs. (13)–(15):

```
∂ₜη = -∂ₓ(hu) - ∂ₓ(β ∂ₓΨ)
∂ₜu = -g ∂ₓη - u ∂ₓu - R_B
-∂ₓ(α ∂ₓΨ) + γΨ = ∂ₓ(β u)
```

with `h = d(x) + η`, and `α, β, γ` the vertical moments of the VBM profile
evaluated at the local depth. The third equation is elliptic — there is no time
derivative of `Ψ` — which is what makes the system dispersive and what a PINN has
to satisfy pointwise rather than by time-stepping.

## The notebooks

| Notebook | Role |
|---|---|
| [`pinn_boussinesq_benchmarks.ipynb`](pinn_boussinesq_benchmarks.ipynb) | The study. One benchmark case per run, selected by `ACTIVE_CASE` in §2. Sections 3–5d build the case (config, VBM coefficients, bathymetry, exact solution, sponge layer, breaking closure); 6–11 sample, build and train (Adam, then L-BFGS-B); 12–18 plot, score and save. Publishes `CASE_RESULT`. |
| [`kaggle_pinn_boussinesq_all.ipynb`](kaggle_pinn_boussinesq_all.ipynb) | The driver. Executes the benchmarks notebook's code cells once per case in a fresh namespace holding only `ACTIVE_CASE` and `CFG_OVERRIDES`, clearing the Keras session between runs, and prints a summary table. Also runs the **seed sweep** (below). |
| [`pinn_complete.ipynb`](pinn_complete.ipynb) | A separate, self-contained study: the **inverse** problem — recover the bathymetry `d(x)` from surface observations. Forward PINN generates non-circular observations (§6), then prototype inverse run, 5-seed ensemble UQ, loss-weight ablation, a K × σ ablation, κ₁ sensitivity, residual fields and a cost table. |

## The five benchmark cases

| Key | Case | Reference | How it is verified |
|---|---|---|---|
| `carrier_greenspan` | Regular wave run-up on a plane beach | Carrier & Greenspan (1958) | **Exact nonlinear solution**, solved via the hodograph transform (§5b) and used for the offshore boundary, the initial condition, and the RMSE/correlation scoring in §17 |
| `solitary_nonbreaking` | Solitary run-up, H/d = 0.0185 | Synolakis (1987) | Snapshots; needs digitised lab data for a quantitative score |
| `solitary_breaking` | Solitary run-up, H/d = 0.3 | Synolakis (1987) | Snapshots; eddy-viscosity breaking closure (§5d) |
| `beji_battjes` | Harmonic wave over a submerged bar | Beji & Battjes (1993) | Wave-gauge signals; sponge layer + ramped wave-maker (§5c) |
| `flat_cosine` | Cosine wave over a flat bottom | — | Sanity check, ~2 min |

**Carrier–Greenspan is the only case with a true reference.** Its exact solution
is verified inside the notebook against the nonlinear shallow-water equations by
finite differences before it is used, and the run-up, the single-valuedness of
the hodograph map, and the breaking parameter are asserted rather than assumed.

> **Scope note.** Carrier–Greenspan solves the *non-dispersive* nonlinear
> shallow-water equations while the VBM is dispersive. At this configuration
> `kd ≈ 0.14` offshore, so the two differ by `O((kd)²/3) ≈ 0.7%`. That is the
> noise floor of the comparison — not machine precision, and not a PINN error.

## The seed sweep, and why it exists

Every comparison in this study was a single realisation until the sweep was
added. An accidental control — `flat_cosine`, whose sampling was unchanged
between two runs — still moved its final loss by **48%** purely from a shifted
RNG stream. The driver's `SEED_SWEEP = ("carrier_greenspan", [0, 1, 2])` repeats
one case across seeds and prints the mean, spread and CV of the final loss, the
`η`/`u` correlations, the relative RMSE and the run-up.

Read the resulting line literally: **a change smaller than about two standard
deviations is not evidence of anything.**

## Running it

The driver is a Kaggle kernel (`kernel-metadata.json`, GPU + internet on). Its
knobs sit in §0:

- `QUICK = True` — ~3 min over all five cases. Trains nothing useful, but
  exercises every physics assert and every plot path. Use it to validate the
  environment first.
- `REQUIRE_GPU = True` — stops before training if no GPU is visible; on CPU the
  full queue is 8–15× slower. `QUICK` is exempt.
- `CASE_QUEUE` — ordered cheapest and most diagnostic first, so a systematic
  problem surfaces in minutes rather than after the first long train
  (rough GPU estimate: ~80 min for all five).
- `SEED_SWEEP` — set to `(case, seeds)` to run the sweep instead of the queue;
  `None` for the normal queue.

Each case writes to `pinn_results/<case>/`. A case that raises is caught, its
traceback recorded, and the queue continues.

To run one case interactively instead, open the benchmarks notebook and set
`ACTIVE_CASE` in §2.

## Status

Active. The physics is verified where a reference exists (Carrier–Greenspan,
lake-at-rest behaviour of the sponge, the breaking closure's activation), the
other three cases run and produce plausible fields but are scored visually. What
is still missing is digitised laboratory data for the Synolakis and
Beji–Battjes cases, without which those three cases cannot be given a number.
