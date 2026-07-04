"""beta(t) estimators built on the SINDy core (numpy; sklearn optional).

Local estimators (per time point): ``direct`` ratio and ``windowed`` SINDy.
Global identification: a Poly+Fourier time basis fit with a sparse solver.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ...core.derivatives import savgol, savgol_derivative
from ...methods.sindy.model import windowed_coefficients
from ...methods.sindy.sparse import stridge


def _sparse_fit(Theta: np.ndarray, y: np.ndarray, *, method: str = "stridge",
                alpha: float = 0.01, threshold: float = 0.01,
                lasso_alpha: float = 1e-5) -> np.ndarray:
    """Sparse least squares with a numpy STRidge default or optional sklearn LASSO.

    Parameters
    ----------
    Theta : np.ndarray
        Library/feature matrix.
    y : np.ndarray
        Target vector.
    method : str
        Solver to use, ``"stridge"`` (default) or ``"lasso"``.
    alpha : float
        Ridge regularization strength for STRidge.
    threshold : float
        Sparsity threshold for STRidge.
    lasso_alpha : float
        L1 regularization strength for the sklearn LASSO solver.

    Returns
    -------
    np.ndarray
        The fitted sparse coefficient vector.

    Raises
    ------
    ImportError
        If ``method="lasso"`` but scikit-learn is not installed.
    """
    if method == "lasso":
        try:
            from sklearn.linear_model import Lasso
        except ImportError as exc:  # pragma: no cover
            raise ImportError("method='lasso' requires scikit-learn "
                              "(`pip install scikit-learn`).") from exc
        m = Lasso(alpha=lasso_alpha, fit_intercept=False, max_iter=20000)
        m.fit(Theta, y)
        return m.coef_
    return stridge(Theta, y, threshold=threshold, alpha=alpha)


def _mask(arr: np.ndarray, cond: np.ndarray) -> np.ndarray:
    out = arr.astype(float).copy()
    out[~cond] = np.nan
    return out


def estimate_beta_direct(data: Dict, *, mu: float = 0.0, omega: float = 0.0,
                         mask_eps: float = 1e-4, sg_window: int = 11, sg_poly: int = 3
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Ratio estimator ``beta = -(dS - births + deaths - waning) / (S I / N)``.

    Parameters
    ----------
    data : Dict
        Reconstructed series with keys ``t``, ``S``, ``I``, ``N``, ``R``.
    mu : float
        Birth/death rate.
    omega : float
        Waning-immunity rate.
    mask_eps : float
        Relative threshold below which the denominator is treated as invalid.
    sg_window : int
        Savitzky-Golay smoothing window length.
    sg_poly : int
        Savitzky-Golay polynomial order.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The time array and the estimated ``beta(t)`` (NaN where invalid).
    """
    t, S, I, N, R = data["t"], data["S"], data["I"], data["N"], data["R"]
    dS = savgol_derivative(S, t, window=sg_window, poly=sg_poly)
    denom = S * I / N
    valid = denom > mask_eps * N
    dS_corr = dS - mu * N + mu * S - omega * R
    beta = _mask(-dS_corr / np.where(denom == 0, np.nan, denom), valid)
    beta = _mask(beta, beta > 0)
    beta = _mask(beta, beta < 10)
    beta = savgol(np.where(np.isnan(beta), 0.0, beta), window=sg_window, poly=sg_poly)
    return t, _mask(beta, valid)


def estimate_beta_windowed(data: Dict, *, mu: float = 0.0, omega: float = 0.0,
                           window: int = 8, step: int = 1, threshold: float = 0.01,
                           alpha: float = 0.01, mask_eps: float = 1e-4,
                           sg_window: int = 11, sg_poly: int = 3
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Sliding-window SINDy. Library ``[S I / N]``; ``beta = -xi_0``.

    Parameters
    ----------
    data : Dict
        Reconstructed series with keys ``t``, ``S``, ``I``, ``N``, ``R``.
    mu : float
        Birth/death rate.
    omega : float
        Waning-immunity rate.
    window : int
        Sliding-window length (in samples).
    step : int
        Stride between successive windows.
    threshold : float
        STRidge sparsity threshold.
    alpha : float
        STRidge ridge regularization strength.
    mask_eps : float
        Relative threshold below which a value is treated as invalid.
    sg_window : int
        Savitzky-Golay smoothing window length.
    sg_poly : int
        Savitzky-Golay polynomial order.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The window-center times and the estimated ``beta(t)`` at those centers.
    """
    t, S, I, N, R = data["t"], data["S"], data["I"], data["N"], data["R"]
    dS = savgol_derivative(S, t, window=sg_window, poly=sg_poly)
    Theta = (S * I / N).reshape(-1, 1)
    dS_corr = dS - mu * N + mu * S - omega * R
    centers, coeffs = windowed_coefficients(Theta, dS_corr, t, window=window,
                                            step=step, threshold=threshold, alpha=alpha)
    betas = -coeffs[:, 0]
    keep = (betas > 0) & (betas < 10)
    centers, betas = centers[keep], betas[keep]
    if len(betas) > sg_window:
        betas = savgol(betas, window=sg_window, poly=sg_poly)
    return centers, betas


def _time_basis(t: np.ndarray, t_ref_min: float, t_ref_max: float,
                poly_degree: int, n_fourier: int, period: float
                ) -> Tuple[np.ndarray, List[str]]:
    t_norm = (t - t_ref_min) / max(t_ref_max - t_ref_min, 1.0)
    cols, names = [np.ones(len(t))], ["1"]
    for d in range(1, poly_degree + 1):
        cols.append(t_norm ** d); names.append(f"t^{d}")
    for k in range(1, n_fourier + 1):
        w = 2 * np.pi * k / period
        cols.append(np.sin(w * t)); names.append(f"sin({k}w t)")
        cols.append(np.cos(w * t)); names.append(f"cos({k}w t)")
    return np.column_stack(cols), names


def fit_global_time_basis(t_local: np.ndarray, beta_local: np.ndarray, t_full: np.ndarray, *,
                          poly_degree: int = 3, n_fourier: int = 5, fourier_period: float = 52.0,
                          method: str = "stridge", alpha: float = 0.01, threshold: float = 1e-4,
                          lasso_alpha: float = 1e-5
                          ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Fit ``beta(t) = Phi(t) @ xi`` on a Poly+Fourier basis with a sparse solver.

    Parameters
    ----------
    t_local : np.ndarray
        Times of the local (per-point) beta estimates.
    beta_local : np.ndarray
        Local beta estimates (may contain NaN for invalid points).
    t_full : np.ndarray
        Times at which to evaluate the fitted global ``beta(t)``.
    poly_degree : int
        Degree of the polynomial part of the time basis.
    n_fourier : int
        Number of Fourier harmonics in the time basis.
    fourier_period : float
        Fundamental period of the Fourier terms.
    method : str
        Sparse solver to use, ``"stridge"`` or ``"lasso"``.
    alpha : float
        Ridge regularization strength for STRidge.
    threshold : float
        Sparsity threshold for STRidge.
    lasso_alpha : float
        L1 regularization strength for the LASSO solver.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, List[str]]
        The global ``beta(t)`` on ``t_full``, the sparse coefficient vector, and
        the basis-term names.
    """
    valid = ~np.isnan(beta_local) & (beta_local > 0)
    tv, bv = t_local[valid], beta_local[valid]
    Phi_fit, names = _time_basis(tv, tv.min(), tv.max(), poly_degree, n_fourier, fourier_period)
    xi = _sparse_fit(Phi_fit, bv, method=method, alpha=alpha, threshold=threshold,
                     lasso_alpha=lasso_alpha)
    Phi_full, _ = _time_basis(t_full, tv.min(), tv.max(), poly_degree, n_fourier, fourier_period)
    beta_global = np.clip(Phi_full @ xi, 0, 10)
    return beta_global, xi, names
