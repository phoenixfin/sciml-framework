import numpy as np

from sciml.problems.swe.cases import b_flat, h0_gaussian
from sciml.solvers.swe_lax_friedrichs import lax_friedrichs_swe
from sciml.solvers.compartmental import simulate_compartmental, rk4_integrate
from sciml.solvers.wave_fdm import wave_moving_boundary_fdm


def test_swe_snapshots_and_mass():
    t_out = [0.25, 0.5, 0.75, 1.0]
    x, snaps = lax_friedrichs_swe(h0_gaussian, b_flat, nx=300, nt=3000, t_out=t_out)
    assert snaps["t"] == t_out and x.shape == (300,)
    m0, mf = h0_gaussian(x).sum(), snaps["h"][-1].sum()
    assert abs(mf - m0) / m0 < 0.02 and snaps["h"][-1].min() > 0


def test_rk4_exponential():
    # y' = -y, y(0)=1 -> y(t)=exp(-t).
    t = np.linspace(0, 2, 401)
    y = rk4_integrate(lambda tt, yy: -yy, np.array([1.0]), t)[:, 0]
    assert np.allclose(y, np.exp(-t), atol=1e-4)


def test_sir_simulation_conserves_population():
    sim = simulate_compartmental("SIR", N=1000, I0=10, n_weeks=100,
                                 beta_fn=lambda t: 0.3, gamma=0.1)
    total = sim["S"] + sim["I"] + sim["R"]
    assert np.allclose(total, 1000, atol=1.0)
    assert sim["I"].max() > 10  # an outbreak occurs


def test_wave_fdm_shapes():
    s_y = 0.16
    out = wave_moving_boundary_fdm(
        lambda tau: s_y + 0.05 * np.cos(np.asarray(tau)),
        lambda xbar: 0.3 * (np.asarray(xbar) - 1.0), s_y,
        delta=0.04, t_final=2.0, nx=120, n_snaps=40)
    assert 1 <= len(out["t"]) <= 40
    assert out["xbar"][0].shape == (121,) and len(out["u"]) == len(out["t"])
