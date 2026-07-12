"""Tests for the system-identification task layer on ground-truth data."""

from __future__ import annotations

import numpy as np
import pytest

from sciml.data.datasets import load
from sciml.data.datasets.synthetic import A_TRUE, B_TRUE
from sciml.tasks import sysid


@pytest.fixture(scope="module")
def lti():
    return load("lti_demo", n_segments=4, seg_len=400, noise=0.005, seed=3)


def test_sindyc_recovers_lti(lti):
    res = sysid.run(lti, states=["x1", "x2"], inputs=["u1", "u2"], method="sindyc",
                    center="segment", horizons=(24, 72))
    # One-step fit should be excellent on nearly noise-free linear data.
    assert min(res.r2_test.values()) > 0.8
    hz = res.metrics["horizons"]
    assert hz[72]["diverged_frac"] == 0.0
    assert hz[72]["nrmse"] < min(v[72] for v in res.baselines.values())


def test_sindyc_coefficient_recovery(lti):
    res = sysid.run(lti, states=["x1", "x2"], inputs=["u1", "u2"], method="sindyc",
                    center="segment", horizons=(24,))
    # Compare identified linear coefficients (z-units) to the truth mapped
    # through the same scaling: dz_i/dt = sum_j A_ij (s_j / s_i) z_j + ...
    fl = np.vstack([s[:, :2] - s[:, :2].mean(0) for s in lti.segments])
    fu = np.vstack([s[:, 2:] - s[:, 2:].mean(0) for s in lti.segments])
    s_x, s_u = fl.std(0), fu.std(0)
    names = ["x1", "x2", "u1", "u2"]
    for i, st in enumerate(["x1", "x2"]):
        got = res.coefficients[st]
        for j, nm in enumerate(names):
            true_c = (A_TRUE[i, j] * s_x[j] if j < 2 else B_TRUE[i, j - 2] * s_u[j - 2])
            true_c /= s_x[i]
            if abs(true_c) > 0.05:
                assert nm in got, f"{st}: expected term {nm}"
                assert got[nm] == pytest.approx(true_c, rel=0.35)


def test_dmdc_matches_protocol(lti):
    res = sysid.run(lti, states=["x1", "x2"], inputs=["u1", "u2"], method="dmdc",
                    center="segment", horizons=(24,))
    assert res.details["threshold"] == 0.0 and res.details["degree"] == 1
    assert min(res.r2_test.values()) > 0.8


def test_autonomous_sindy_runs(lti):
    res = sysid.run(lti, states=["x1", "x2"], method="sindy", center="segment",
                    horizons=(24,))
    assert res.inputs == []
    assert res.metrics["n_rollouts"] > 0


def test_causal_centering_runs(lti):
    res = sysid.run(lti, states=["x1", "x2"], inputs=["u1", "u2"], method="sindyc",
                    center="causal", op_window_h=48, horizons=(24,))
    assert np.isfinite(res.metrics["horizons"][24]["nrmse"])


def test_explicit_test_data(lti):
    other = load("lti_demo", n_segments=1, seg_len=300, noise=0.005, seed=99)
    res = sysid.run(lti, states=["x1", "x2"], inputs=["u1", "u2"], method="sindyc",
                    center="segment", test_data=other, horizons=(24,))
    assert res.details["n_test_segments"] == 1


def test_unknown_method_raises(lti):
    with pytest.raises(ValueError):
        sysid.run(lti, states=["x1"], method="nope")


def test_summary_is_text(lti):
    res = sysid.run(lti, states=["x1", "x2"], inputs=["u1", "u2"], center="segment",
                    horizons=(24,))
    s = res.summary()
    assert "equations:" in s and "NRMSE" in s
