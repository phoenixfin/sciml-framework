# Examples gallery

A graded tour of the framework. Two balanced tracks — **ODE systems** and
**PDE / operators** — each ordered simplest → most complex. Every script is
self-contained, prints results, and saves a figure to `outputs/examples/`.

```bash
pip install -e ".[all]"
python examples/01_linear_ode_sindy.py
```

### ODE systems

| # | Example | Method | What it shows | Backend |
|---|---|---|---|---|
| 01 | `01_linear_ode_sindy.py` | SINDy | identify `x' = -k x` | **numpy** |
| 02 | `02_harmonic_oscillator_dmd.py` | DMD | recover oscillator frequency/modes | **numpy** |
| 03 | `03_lotka_volterra_sindy.py` | SINDy | nonlinear predator–prey ODE | **numpy** |
| 12 | `12_van_der_pol_sindy.py` | SINDy | limit cycle with a cubic term | **numpy** |
| 13 | `13_fitzhugh_nagumo_sindy.py` | SINDy | excitable neuron model | **numpy** |
| 04 | `04_lorenz_sindy.py` | SINDy | identify the chaotic Lorenz system | **numpy** |
| 07 | `07_lotka_volterra_neural_ode.py` | Neural ODE | learn black-box dynamics from a trajectory | TensorFlow |

### PDE / operators

| # | Example | Method | What it shows | Backend |
|---|---|---|---|---|
| 09 | `09_wave1d_pinn.py` | PINN | solve the 1D wave equation | TensorFlow |
| 05 | `05_heat_fno.py` | FNO (1D) | learn the heat solution operator | TensorFlow |
| 08 | `08_advection_diffusion_deeponet.py` | DeepONet | learn the advection–diffusion operator | TensorFlow |
| 06 | `06_burgers_fno.py` | FNO (1D) | Burgers operator (near-shocks) | TensorFlow |
| 10 | `10_kuramoto_sivashinsky_dmd.py` | DMD | modal analysis of a chaotic PDE | **numpy** |
| 11 | `11_darcy_fno2d.py` | FNO (2D) | learn the 2D Darcy-flow operator | TensorFlow |

The **numpy** examples run anywhere (and the SINDy ones genuinely recover the
governing equations from data — Lorenz, Van der Pol, FitzHugh–Nagumo, …). The
TensorFlow examples need `pip install -e ".[all]"` on a Python 3.10–3.12
environment. Every example takes CLI flags (`--epochs`, `--steps`, `--n`, …) to
scale the run.

## Coverage

All six method engines appear in the gallery: **SINDy, DMD, FNO (1D + 2D),
DeepONet, PINN, Neural ODE**. For the heavier, fully-packaged problems (config +
runners + publication figures) see the `problems/` packages and `experiments/`
scripts:

| Problem | Method | Entry point |
|---|---|---|
| 1D Shallow Water Equations | DeepONet | `sciml swe` / `experiments/swe/` |
| Moving-boundary wave (obstacle) | PINN | `sciml wave` / `experiments/wave_obstacle/` |
| Dengue β(t) identification | SINDy | `sciml dengue` / `experiments/epidemiology/` |
