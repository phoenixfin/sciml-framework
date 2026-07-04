"""Sparse regression solvers for SINDy (pure numpy)."""

from __future__ import annotations

import numpy as np


def ridge_regression(Theta: np.ndarray, y: np.ndarray, alpha: float = 0.0) -> np.ndarray:
    """Closed-form ridge: ``(Theta^T Theta + alpha I)^-1 Theta^T y``.

    ``alpha = 0`` reduces to ordinary least squares (via ``lstsq`` for
    stability). ``y`` may be 1D ``(m,)`` or 2D ``(m, k)``.

    Parameters
    ----------
    Theta : np.ndarray
        Feature matrix of shape ``(m, p)``.
    y : np.ndarray
        Target values of shape ``(m,)`` or ``(m, k)``.
    alpha : float
        Ridge penalty; ``alpha <= 0`` falls back to least squares.

    Returns
    -------
    np.ndarray
        The fitted coefficient vector/matrix.
    """
    Theta = np.asarray(Theta, dtype=float)
    y = np.asarray(y, dtype=float)
    if alpha <= 0:
        return np.linalg.lstsq(Theta, y, rcond=None)[0]
    p = Theta.shape[1]
    A = Theta.T @ Theta + alpha * np.eye(p)
    return np.linalg.solve(A, Theta.T @ y)


def stridge(Theta: np.ndarray, y: np.ndarray, threshold: float = 0.01,
            alpha: float = 0.0, max_iter: int = 20) -> np.ndarray:
    """Sequential Thresholded Ridge regression (STRidge / STLSQ).

    Repeatedly fits ridge regression and zeros out coefficients with magnitude
    below ``threshold``, iterating until the active set stabilizes. Returns the
    coefficient vector ``xi`` of shape ``(n_features,)`` (1D target) or
    ``(n_features, k)`` (multi-target).

    Parameters
    ----------
    Theta : np.ndarray
        Feature matrix of shape ``(m, n_features)``.
    y : np.ndarray
        Target values of shape ``(m,)`` or ``(m, k)``.
    threshold : float
        Magnitude below which coefficients are zeroed each iteration.
    alpha : float
        Ridge penalty used in each inner fit.
    max_iter : int
        Maximum number of thresholding iterations.

    Returns
    -------
    np.ndarray
        The sparse coefficient vector ``(n_features,)`` or matrix
        ``(n_features, k)``.
    """
    Theta = np.asarray(Theta, dtype=float)
    y = np.asarray(y, dtype=float)
    xi = np.linalg.lstsq(Theta, y, rcond=None)[0]

    if xi.ndim == 1:
        for _ in range(max_iter):
            big = np.abs(xi) >= threshold
            if big.sum() == 0:
                break
            new = np.zeros_like(xi)
            if big.any():
                new[big] = ridge_regression(Theta[:, big], y, alpha)
            if np.array_equal(big, np.abs(new) >= threshold):
                xi = new
                break
            xi = new
        return xi

    # Multi-target: solve each column independently (per-column active sets).
    out = np.zeros_like(xi)
    for j in range(xi.shape[1]):
        out[:, j] = stridge(Theta, y[:, j], threshold, alpha, max_iter)
    return out
