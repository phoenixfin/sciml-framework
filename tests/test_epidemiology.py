"""Tests for the dengue pipeline: S reconstruction and beta(t) estimation."""

import numpy as np

from sciml.problems.epidemiology.config import EpiConfig
from sciml.problems.epidemiology.problem import EpiProblem
from sciml.problems.epidemiology.reconstruction import reconstruct_S_cumulative


def _short_cfg(model: str = "SIR") -> EpiConfig:
    """Build a short, noise-free config so the tests run quickly.

    Parameters
    ----------
    model : str
        Compartmental model name, e.g. ``"SIR"`` or ``"SIRS"``.

    Returns
    -------
    EpiConfig
        Config over 156 simulated weeks with no observation noise and only the
        two local estimators enabled.
    """
    cfg = EpiConfig()
    cfg.model.model = model
    cfg.data.sim_weeks = 156
    cfg.data.sim_noise = 0.0
    cfg.estim.local_methods = ["direct", "windowed"]
    return cfg


def test_cumulative_reconstruction_monotone_R():
    """Reconstructed recoveries are non-decreasing and S never exceeds the population."""
    cfg = _short_cfg()
    prob = EpiProblem(cfg)
    raw = prob.load_or_simulate(rng=np.random.default_rng(0))
    S, R = reconstruct_S_cumulative(raw["t"], raw["I"], cfg.model.N, cfg.model.gamma)
    assert np.all(np.diff(R) >= -1e-6)          # cumulative recovered is non-decreasing
    assert S.max() <= cfg.model.N + 1e-6


def test_pipeline_recovers_plausible_beta():
    """The full pipeline returns both local estimators, with beta in a plausible range."""
    cfg = _short_cfg("SIR")
    prob = EpiProblem(cfg)
    prob.load_or_simulate(rng=np.random.default_rng(0))
    prob.reconstruct()
    out = prob.estimate()
    assert "direct" in out["local"] and "windowed" in out["local"]
    beta = out["local"]["direct"]["beta"]
    valid = beta[~np.isnan(beta)]
    assert valid.size > 0
    # seasonal beta oscillates around 0.25; the mean estimate should be in range.
    assert 0.05 < np.mean(valid) < 0.6


def test_global_time_basis_runs():
    """The global time-basis estimator runs and returns beta on the data's time grid."""
    cfg = _short_cfg("SIR")
    prob = EpiProblem(cfg)
    prob.load_or_simulate(rng=np.random.default_rng(1))
    prob.reconstruct()
    out = prob.estimate()
    assert out["global"] is not None
    assert out["global"]["beta"].shape == prob.data["t"].shape
