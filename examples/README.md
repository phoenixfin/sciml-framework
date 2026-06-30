# Examples gallery

A graded tour of the framework, simplest → most complex. Each script is
self-contained, prints results, and saves a figure to `outputs/examples/`.

Run any of them after installing the package (`pip install -e ".[all]"`):

```bash
python examples/01_linear_ode_sindy.py
```

| # | Example | Method | Problem | Backend |
|---|---|---|---|---|
| 01 | `01_linear_ode_sindy.py` | SINDy | identify `x' = -k x` | **numpy** |
| 02 | `02_harmonic_oscillator_dmd.py` | DMD | recover oscillator frequency/modes | **numpy** |
| 03 | `03_lotka_volterra_sindy.py` | SINDy | identify predator–prey nonlinear ODE | **numpy** |
| 04 | `04_lorenz_sindy.py` | SINDy | identify the chaotic Lorenz system | **numpy** |
| 05 | `05_heat_fno.py` | FNO | learn the heat-equation solution operator | TensorFlow |
| 06 | `06_burgers_fno.py` | FNO | learn the Burgers operator (near-shocks) | TensorFlow |
| 07 | `07_lotka_volterra_neural_ode.py` | Neural ODE | learn predator–prey dynamics from a trajectory | TensorFlow |

The **numpy** examples (01–04) run anywhere. Examples 05–07 need TensorFlow
(`pip install -e ".[all]"`; use a Python 3.10–3.12 environment).

For the heavier, fully-packaged problems (config + runners + figures), see the
`problems/` packages and `experiments/` scripts:

| Problem | Method | Entry point |
|---|---|---|
| 1D Shallow Water Equations | DeepONet | `sciml swe` / `experiments/swe/` |
| Moving-boundary wave (obstacle) | PINN | `sciml wave` / `experiments/wave_obstacle/` |
| Dengue β(t) identification | SINDy | `sciml dengue` / `experiments/epidemiology/` |
