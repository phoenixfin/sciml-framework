"""High-level SINDy estimator and a generic windowed-regression helper."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from ...core.derivatives import savgol_derivative
from .library import FeatureLibrary
from .sparse import stridge


class SINDy:
    """Fit ``Xdot ~ Theta(X) @ Xi`` with a thresholded-ridge sparse solver."""

    def __init__(self, library: FeatureLibrary, threshold: float = 0.01,
                 alpha: float = 0.0):
        """Configure the feature library and STRidge hyper-parameters.

        Parameters
        ----------
        library : FeatureLibrary
            The candidate-term feature library.
        threshold : float
            STRidge sparsity threshold.
        alpha : float
            Ridge penalty used inside STRidge.
        """
        self.library = library
        self.threshold = threshold
        self.alpha = alpha
        self.coef_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None

    def fit(self, X: np.ndarray, Xdot: Optional[np.ndarray] = None,
            t: Optional[np.ndarray] = None,
            input_names: Optional[Sequence[str]] = None) -> "SINDy":
        """Fit sparse coefficients ``Xi``; derivatives are estimated if ``Xdot`` is None.

        Parameters
        ----------
        X : np.ndarray
            State data matrix of shape ``(m, d)``.
        Xdot : Optional[np.ndarray]
            Time derivatives; estimated from ``X`` and ``t`` if None.
        t : Optional[np.ndarray]
            Time points used to estimate derivatives when ``Xdot`` is None.
        input_names : Optional[Sequence[str]]
            Names of the ``d`` state variables.

        Returns
        -------
        SINDy
            The fitted estimator (``self``).
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if Xdot is None:
            # Estimate derivatives column-wise via Savitzky-Golay.
            Xdot = np.column_stack([savgol_derivative(X[:, j], t)
                                    for j in range(X.shape[1])])
        Theta = self.library.transform(X)
        self.coef_ = stridge(Theta, np.asarray(Xdot, dtype=float),
                             self.threshold, self.alpha)
        try:
            self.feature_names_ = self.library.names(input_names)
        except Exception:
            self.feature_names_ = [f"f{i}" for i in range(Theta.shape[1])]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the derivatives ``Theta(X) @ Xi`` at states ``X``.

        Parameters
        ----------
        X : np.ndarray
            State data matrix of shape ``(m, d)``.

        Returns
        -------
        np.ndarray
            The predicted derivatives ``Theta(X) @ Xi``.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if self.coef_ is None:
            raise RuntimeError("Call fit() before predict().")
        return self.library.transform(np.atleast_2d(X)) @ self.coef_

    def equations(self, target_names: Optional[Sequence[str]] = None,
                  precision: int = 4) -> List[str]:
        """Human-readable identified equations (one per target).

        Parameters
        ----------
        target_names : Optional[Sequence[str]]
            Left-hand-side names for each target equation.
        precision : int
            Number of decimal places used when formatting coefficients.

        Returns
        -------
        List[str]
            One formatted equation string per target.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if self.coef_ is None:
            raise RuntimeError("Call fit() before equations().")
        coef = self.coef_
        if coef.ndim == 1:
            coef = coef[:, None]
        names = self.feature_names_ or [f"f{i}" for i in range(coef.shape[0])]
        eqs = []
        for j in range(coef.shape[1]):
            terms = [f"{coef[i, j]:+.{precision}f} {names[i]}"
                     for i in range(coef.shape[0]) if abs(coef[i, j]) > 0]
            lhs = (target_names[j] if target_names else f"d/dt x{j}")
            eqs.append(f"{lhs} = " + (" ".join(terms) if terms else "0"))
        return eqs


def windowed_coefficients(Theta: np.ndarray, y: np.ndarray, t: np.ndarray, *,
                          window: int = 8, step: int = 1, threshold: float = 0.01,
                          alpha: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Sliding-window STRidge: fit ``y ~ Theta @ xi`` on each window.

    Returns ``(centers, coeffs)`` where ``centers`` are the window-midpoint
    times and ``coeffs`` has shape ``(n_windows, n_features)``.

    Parameters
    ----------
    Theta : np.ndarray
        Feature matrix of shape ``(n, n_features)``.
    y : np.ndarray
        Target values aligned with the rows of ``Theta``.
    t : np.ndarray
        Time points aligned with the rows of ``Theta``.
    window : int
        Number of samples per sliding window.
    step : int
        Stride between successive windows.
    threshold : float
        STRidge sparsity threshold.
    alpha : float
        Ridge penalty used inside STRidge.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The window-midpoint times and the per-window coefficients of shape
        ``(n_windows, n_features)``.
    """
    Theta = np.atleast_2d(np.asarray(Theta, dtype=float))
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    n = len(t)
    centers, coeffs = [], []
    for start in range(0, n - window + 1, step):
        sl = slice(start, start + window)
        xi = stridge(Theta[sl], y[sl], threshold, alpha)
        centers.append(t[start + window // 2])
        coeffs.append(np.atleast_1d(xi))
    return np.array(centers), np.array(coeffs)
