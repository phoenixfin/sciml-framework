"""Tests for the dataset registry and containers."""

from __future__ import annotations

import numpy as np
import pytest

from sciml.data.datasets import FunctionPairData, TimeSeriesData, list_datasets, load, register


def test_registry_lists_builtins():
    names = list_datasets()
    assert "lti_demo" in names and "advection_pairs" in names and "wnts" in names


def test_register_and_load_custom():
    @register("unit_test_ds")
    def _loader(k: int = 3) -> TimeSeriesData:
        """One tiny constant segment."""
        return TimeSeriesData([np.ones((10, k))], [f"c{i}" for i in range(k)], 1.0)

    d = load("unit_test_ds", k=2)
    assert d.channels == ["c0", "c1"] and d.n_samples == 10


def test_unknown_dataset_raises():
    with pytest.raises(KeyError):
        load("no_such_dataset")


def test_lti_demo_shapes():
    d = load("lti_demo", n_segments=2, seg_len=50, seed=1)
    assert d.n_segments == 2 and d.channels == ["x1", "x2", "u1", "u2"]
    assert all(s.shape == (50, 4) for s in d.segments)
    sub = d.select(["x1", "u2"])
    assert sub.channels == ["x1", "u2"] and sub.segments[0].shape == (50, 2)
    np.testing.assert_allclose(sub.segments[0][:, 0], d.segments[0][:, 0])


def test_lti_demo_is_deterministic():
    a = load("lti_demo", n_segments=1, seg_len=30, seed=7)
    b = load("lti_demo", n_segments=1, seg_len=30, seed=7)
    np.testing.assert_allclose(a.segments[0], b.segments[0])


def test_select_unknown_channel_raises():
    d = load("lti_demo", n_segments=1, seg_len=30)
    with pytest.raises(KeyError):
        d.select(["nope"])


def test_advection_pairs():
    d = load("advection_pairs", n_samples=8, grid=64, seed=2)
    assert d.n_samples == 8 and d.u.shape == d.s.shape == (8, 64)
    # The operator preserves the spatial mean (advection shifts, diffusion decays k!=0).
    np.testing.assert_allclose(d.u.mean(axis=1), d.s.mean(axis=1), atol=1e-8)
    tr, te = d.split(train_frac=0.75, seed=0)
    assert tr.n_samples == 6 and te.n_samples == 2


def test_function_pair_mismatch_raises():
    with pytest.raises(ValueError):
        FunctionPairData(u=np.zeros((3, 4)), s=np.zeros((2, 4)))
