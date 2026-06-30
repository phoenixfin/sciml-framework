import numpy as np
import pytest

from sciml.methods.dmd import DMD


def _linear_snapshots(n_time=40, dt=1.0, seed=0):
    """Snapshots x_{k+1} = A x_k from a known operator with two rotation-scaling
    blocks -> eigenvalues r_j * exp(+/- i theta_j)."""
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
    X, A, dt, _ = _linear_snapshots()
    dmd = DMD(rank=4).fit(X, dt=dt)
    assert np.allclose(np.sort_complex(dmd.eigenvalues),
                       np.sort_complex(np.linalg.eigvals(A)), atol=1e-6)


def test_dmd_reconstructs():
    X, _, dt, _ = _linear_snapshots()
    dmd = DMD(rank=4).fit(X, dt=dt)
    Xr = dmd.reconstruct(X.shape[1])
    assert np.linalg.norm(Xr - X) / np.linalg.norm(X) < 1e-6


def test_dmd_recovers_frequencies():
    X, _, dt, (th1, th2) = _linear_snapshots()
    dmd = DMD(rank=4).fit(X, dt=dt)
    abs_imag = np.abs(dmd.omega.imag)
    for th in (th1, th2):
        assert np.min(np.abs(abs_imag - th)) < 1e-3


def test_dmd_decay_eigenvalue():
    v = np.random.default_rng(0).standard_normal(10)
    dt = 0.1
    t = np.arange(60) * dt
    X = np.outer(v, np.exp(-0.3 * t))
    dmd = DMD(rank=1).fit(X, dt=dt)
    assert abs(dmd.omega[0].real - (-0.3)) < 1e-3


def test_dmd_predict_shape_and_validation():
    X, _, dt, _ = _linear_snapshots()
    dmd = DMD(rank=4).fit(X, dt=dt)
    assert dmd.predict(np.linspace(0, 1, 20)).shape == (4, 20)
    with pytest.raises(ValueError):
        DMD().fit(np.zeros((5, 1)))
