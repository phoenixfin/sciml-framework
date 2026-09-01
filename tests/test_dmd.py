"""Tests for exact DMD against a known linear operator."""

import numpy as np
import pytest

from sciml.methods.dmd import DMD


def _linear_snapshots(n_time: int = 40, dt: float = 1.0, seed: int = 0) -> tuple:
    """Generate snapshots from a known linear operator.

    The operator is an orthogonal conjugation of two rotation-scaling blocks,
    so its eigenvalues are ``r_j * exp(+/- i theta_j)`` — known exactly, which
    is what makes the recovery tests sharp.

    Parameters
    ----------
    n_time : int
        Number of snapshots.
    dt : float
        Sampling interval, passed through to the caller.
    seed : int
        Seed for the random conjugation and the initial state.

    Returns
    -------
    tuple
        ``(X, A, dt, (theta1, theta2))``: snapshots ``(4, n_time)``, the
        operator, the interval, and the two rotation angles.
    """
    rng = np.random.default_rng(seed)
    r1, th1 = 0.98, 0.3
    r2, th2 = 0.95, 0.8

    def block(r, th):
        return r * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

    D = np.zeros((4, 4))
    D[:2, :2] = block(r1, th1)
    D[2:, 2:] = block(r2, th2)
    Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    A = Q @ D @ Q.T
    x = rng.standard_normal(4)
    X = np.zeros((4, n_time))
    for k in range(n_time):
        X[:, k] = x
        x = A @ x
    return X, A, dt, (th1, th2)


def test_dmd_recovers_eigenvalues():
    """DMD recovers the eigenvalues of the operator that generated the data."""
    X, A, dt, _ = _linear_snapshots()
    dmd = DMD(rank=4).fit(X, dt=dt)
    assert np.allclose(np.sort_complex(dmd.eigenvalues),
                       np.sort_complex(np.linalg.eigvals(A)), atol=1e-6)


def test_dmd_reconstructs():
    """The rank-4 fit reconstructs the snapshots it was fitted on."""
    X, _, dt, _ = _linear_snapshots()
    dmd = DMD(rank=4).fit(X, dt=dt)
    Xr = dmd.reconstruct(X.shape[1])
    assert np.linalg.norm(Xr - X) / np.linalg.norm(X) < 1e-6


def test_dmd_recovers_frequencies():
    """The continuous-time frequencies match the rotation angles used."""
    X, _, dt, (th1, th2) = _linear_snapshots()
    dmd = DMD(rank=4).fit(X, dt=dt)
    abs_imag = np.abs(dmd.omega.imag)
    for th in (th1, th2):
        assert np.min(np.abs(abs_imag - th)) < 1e-3


def test_dmd_decay_eigenvalue():
    """A single decaying mode gives back its decay rate."""
    v = np.random.default_rng(0).standard_normal(10)
    dt = 0.1
    t = np.arange(60) * dt
    X = np.outer(v, np.exp(-0.3 * t))
    dmd = DMD(rank=1).fit(X, dt=dt)
    assert abs(dmd.omega[0].real - (-0.3)) < 1e-3


def test_dmd_predict_shape_and_validation():
    """``predict`` is correctly shaped, and a single snapshot is rejected."""
    X, _, dt, _ = _linear_snapshots()
    dmd = DMD(rank=4).fit(X, dt=dt)
    assert dmd.predict(np.linspace(0, 1, 20)).shape == (4, 20)
    with pytest.raises(ValueError):
        DMD().fit(np.zeros((5, 1)))
