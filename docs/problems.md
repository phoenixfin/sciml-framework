# Problems & examples

Two tiers of worked material: **packaged problems** (full config + runners +
figures, driven by the `sciml` CLI) and the **examples gallery** (13 small,
self-contained scripts).

---

## Packaged problems

### 1D Shallow Water Equations — DeepONet (`problems/swe`)

Learn the solution operator of the 1D SWE: map an initial depth `h₀(x)` and
bathymetry `b(x)` (sampled at M=100 sensors) to the depth/momentum fields
`h(x,t), hu(x,t)`. Two design choices encode the physics:

- **Analytical IC shortcut** — outputs are `t`-scaled corrections so
  `h(x,0)=h₀`, `hu(x,0)=0` exactly, with positivity via ELU.
- **Separate branch pairs** for `h` and `hu` — prevents a coupled
  boundary-condition collapse.

Training combines a **data** loss (vs Lax–Friedrichs reference snapshots) with a
periodic **boundary** loss. Ships evaluation cases C1–C3, an operator
generalization test (C4), ablation variants, an N_d-scaling study, and an
`F=0`-attractor physics experiment.

```bash
sciml swe --config configs/swe.yaml --timing
python -m experiments.swe.ablation --steps 10000
```

### Moving-boundary wave (obstacle) — PINN (`problems/wave_obstacle`)

A string vibrating on the moving domain `s(τ) < x̄ < 1`. Two networks train
jointly: `Nu(x̄,τ)` (displacement, Fourier-embedded) and `Ns(τ)` (the free
boundary). The masked physical-PDE loss has **nine weighted terms** (PDE,
exclusion, Dirichlet/Neumann at the obstacle, fixed end, IC displacement/velocity,
anchor, boundary-velocity) with causal weighting and RAR. Trained with a
multi-phase Adam schedule + L-BFGS; evaluated against an FDM reference (e_s, e_u).

```bash
sciml wave --config configs/wave_obstacle.yaml
```

### Dengue β(t) identification — SINDy (`problems/epidemiology`)

Identify a time-varying transmission rate `β(t)` of an SI/SIR/SIRS model from a
weekly case series. Pipeline: **load or simulate** → **reconstruct S(t)**
(cumulative / ODE / EKF) → **estimate β(t)** (direct ratio, windowed SINDy,
global Poly+Fourier LASSO). Defaults simulate data, so it runs with no
downloads and no backend.

```bash
sciml dengue --config configs/dengue.yaml
```

```mermaid
flowchart LR
  R["weekly cases I(t)<br/>(real or simulated)"] --> S["reconstruct S(t)<br/>cumulative / ODE / EKF"]
  S --> L["local β(t)<br/>direct · windowed-SINDy"]
  L --> G["global β(t)<br/>Poly+Fourier sparse fit"]
```

---

## Examples gallery

Thirteen graded scripts under `examples/`, balanced across **ODE** and **PDE**
and covering all six engines. Numpy examples run anywhere; TF examples need
`pip install -e ".[all]"`.

**ODE track:** 01 linear decay (SINDy) · 02 harmonic oscillator (DMD) ·
03 Lotka–Volterra (SINDy) · 12 Van der Pol (SINDy) · 13 FitzHugh–Nagumo (SINDy) ·
04 Lorenz, chaotic (SINDy) · 07 Lotka–Volterra (Neural ODE).

**PDE track:** 09 1D wave (PINN) · 05 heat operator (FNO) ·
08 advection–diffusion (DeepONet) · 06 Burgers (FNO) ·
10 Kuramoto–Sivashinsky (DMD) · 11 2D Darcy (FNO-2D).

See [examples/README.md](../examples/README.md) for the full table. Each script
prints results and writes a figure to `outputs/examples/`.

---

## Reference solvers

The `solvers/` package provides the pure-numpy ground truth used by problems and
examples:

| Solver | Equation / system |
|---|---|
| `swe_lax_friedrichs` | 1D shallow water equations |
| `wave_fdm` | moving-boundary wave (reference field) |
| `compartmental` | SI/SIR/SIRS ODEs (RK4) + `rk4_integrate` |
| `dynamical` | linear decay, harmonic, Lotka–Volterra, Lorenz, Van der Pol, FitzHugh–Nagumo |
| `heat` | periodic heat equation (spectral) |
| `burgers` | viscous Burgers (pseudo-spectral) |
| `transport` | linear advection–diffusion (spectral) |
| `wave1d` | 1D wave d'Alembert reference |
| `kuramoto_sivashinsky` | KS equation (ETDRK4) |
| `darcy` | 2D Darcy flow (finite differences) |
