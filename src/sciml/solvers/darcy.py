"""2D Darcy flow on the unit square (pure numpy, finite differences).

Solve ``-div(a(x,y) grad u) = f`` with homogeneous Dirichlet BCs on a uniform
``m x m`` grid. The operator ``a -> u`` (permeability -> pressure) is the classic
2D Fourier-Neural-Operator benchmark. Grids are kept small (dense solve) since
this is for examples/tests.
"""

from __future__ import annotations

import numpy as np


def solve_darcy_2d(a: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Solve for the pressure ``u`` given permeability ``a`` and forcing ``f``.

    ``a``, ``f`` are ``(m, m)`` node arrays on ``[0,1]^2``. Returns ``u`` ``(m, m)``
    with zeros on the boundary.

    Parameters
    ----------
    a : np.ndarray
        Permeability field of shape ``(m, m)`` on ``[0,1]^2``.
    f : np.ndarray
        Forcing field of shape ``(m, m)`` on ``[0,1]^2``.

    Returns
    -------
    np.ndarray
        Pressure field ``u`` of shape ``(m, m)`` with zeros on the boundary.
    """
    a = np.asarray(a, dtype=float)
    f = np.asarray(f, dtype=float)
    m = a.shape[0]
    h = 1.0 / (m - 1)
    inner = [(i, j) for i in range(1, m - 1) for j in range(1, m - 1)]
    index = {ij: n for n, ij in enumerate(inner)}
    ni = len(inner)
    A = np.zeros((ni, ni))
    b = np.zeros(ni)

    def aface(p, q):
        return 0.5 * (a[p] + a[q])

    for (i, j), n in index.items():
        aE = aface((i, j), (i + 1, j)); aW = aface((i, j), (i - 1, j))
        aN = aface((i, j), (i, j + 1)); aS = aface((i, j), (i, j - 1))
        A[n, n] = aE + aW + aN + aS
        for (ii, jj), coef in (((i + 1, j), aE), ((i - 1, j), aW),
                               ((i, j + 1), aN), ((i, j - 1), aS)):
            if (ii, jj) in index:                  # interior neighbour
                A[n, index[(ii, jj)]] = -coef
            # else: Dirichlet 0, drops out
        b[n] = h * h * f[i, j]

    u_inner = np.linalg.solve(A, b)
    u = np.zeros((m, m))
    for (i, j), n in index.items():
        u[i, j] = u_inner[n]
    return u


def sample_permeability(m: int, n: int, rng: np.random.Generator,
                        length_scale: float = 0.15, log_amp: float = 1.0) -> np.ndarray:
    """Sample ``n`` smooth positive permeability fields ``(n, m, m)`` via a
    Fourier-filtered Gaussian random field exponentiated to stay positive.

    Parameters
    ----------
    m : int
        Grid resolution (fields are ``m x m``).
    n : int
        Number of fields to sample.
    rng : np.random.Generator
        Random generator used to draw the underlying Gaussian noise.
    length_scale : float
        Correlation length scale of the Gaussian random field.
    log_amp : float
        Amplitude applied in log-space before exponentiating.

    Returns
    -------
    np.ndarray
        Batch of positive permeability fields with shape ``(n, m, m)`` as
        ``float32``.
    """
    kx = np.fft.fftfreq(m)[:, None] * m
    ky = np.fft.fftfreq(m)[None, :] * m
    k2 = kx**2 + ky**2
    filt = np.exp(-0.5 * (length_scale**2) * (2 * np.pi)**2 * k2)
    fields = np.empty((n, m, m), dtype=np.float32)
    for s in range(n):
        noise = rng.standard_normal((m, m))
        g = np.real(np.fft.ifft2(np.fft.fft2(noise) * filt))
        g = (g - g.mean()) / (g.std() + 1e-8)
        fields[s] = np.exp(log_amp * g).astype(np.float32)   # positive
    return fields


def darcy_dataset(a_batch: np.ndarray, f_const: float = 1.0) -> np.ndarray:
    """Solve Darcy for a batch of permeability fields ``(n, m, m) -> (n, m, m)``.

    Parameters
    ----------
    a_batch : np.ndarray
        Batch of permeability fields with shape ``(n, m, m)``.
    f_const : float
        Constant forcing value applied uniformly over the domain.

    Returns
    -------
    np.ndarray
        Batch of pressure fields with shape ``(n, m, m)`` as ``float32``.
    """
    m = a_batch.shape[1]
    f = np.full((m, m), f_const)
    return np.stack([solve_darcy_2d(a_batch[i], f)
                     for i in range(len(a_batch))]).astype(np.float32)
