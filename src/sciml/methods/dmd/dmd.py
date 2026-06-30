"""Exact Dynamic Mode Decomposition (Tu et al., 2014)."""

from __future__ import annotations

from typing import Optional

import numpy as np


class DMD:
    """Exact DMD on a snapshot matrix.

    Parameters
    ----------
    rank : SVD truncation rank ``r`` (None keeps all non-negligible singular
        values). Truncation is the main regularizer / denoiser.
    tol : relative singular-value tolerance used when ``rank`` is None.

    After :meth:`fit`, exposes ``eigenvalues`` (discrete), ``omega``
    (continuous growth/frequency = ``log(eig)/dt``), ``modes`` (spatial DMD
    modes Phi), and ``amplitudes`` (b).
    """

    def __init__(self, rank: Optional[int] = None, tol: float = 1e-10):
        self.rank = rank
        self.tol = tol
        self.eigenvalues = None
        self.omega = None
        self.modes = None
        self.amplitudes = None
        self._dt = 1.0
        self._x0 = None

    def fit(self, X: np.ndarray, dt: float = 1.0) -> "DMD":
        """Fit on snapshots ``X`` of shape ``(n_features, n_time)``."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] < 2:
            raise ValueError("X must be (n_features, n_time) with n_time >= 2")
        self._dt = float(dt)
        self._x0 = X[:, 0].astype(complex)
        X1, X2 = X[:, :-1], X[:, 1:]

        U, s, Vh = np.linalg.svd(X1, full_matrices=False)
        r = self.rank
        if r is None:
            r = int(np.sum(s > self.tol * s[0])) if s[0] > 0 else len(s)
        r = max(1, min(r, len(s)))
        Ur, sr, Vr = U[:, :r], s[:r], Vh[:r].conj().T

        # Reduced operator A_tilde = Ur^H X2 Vr Sr^{-1}.
        A_tilde = (Ur.conj().T @ X2 @ Vr) / sr
        eigs, W = np.linalg.eig(A_tilde)
        # Exact DMD modes.
        Phi = (X2 @ Vr / sr) @ W

        self.eigenvalues = eigs
        self.modes = Phi
        with np.errstate(divide="ignore"):
            self.omega = np.log(eigs.astype(complex)) / self._dt
        self.amplitudes = np.linalg.lstsq(Phi, self._x0, rcond=None)[0]
        return self

    @property
    def frequencies(self) -> np.ndarray:
        """Oscillation frequencies in Hz-like units (imag(omega) / 2 pi)."""
        return self.omega.imag / (2 * np.pi)

    def predict(self, t: np.ndarray) -> np.ndarray:
        """Reconstruct the (real) state at times ``t`` -> ``(n_features, len(t))``."""
        if self.modes is None:
            raise RuntimeError("Call fit() before predict().")
        t = np.asarray(t, dtype=float)
        dynamics = np.exp(np.outer(self.omega, t)) * self.amplitudes[:, None]
        return (self.modes @ dynamics).real

    def reconstruct(self, n_time: int) -> np.ndarray:
        """Reconstruct the training window of ``n_time`` snapshots."""
        return self.predict(np.arange(n_time) * self._dt)
