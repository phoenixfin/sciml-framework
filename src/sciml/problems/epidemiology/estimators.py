"""beta(t) estimators built on the SINDy core (numpy; sklearn optional).

Local estimators (per time point): ``direct`` ratio and ``windowed`` SINDy.
Global identification: a Poly+Fourier time basis fit with a sparse solver.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ...core.derivatives import savgol, savgol_derivative
from ...methods.sindy.model import windowed_coefficients
from ...methods.sindy.sparse import stridge


def _sparse_fit(Theta: np.ndarray, y: np.ndarray, *, method: str = "stridge",
                alpha: float = 0.01, threshold: float = 0.01,
                lasso_alpha: float = 1e-5) -> np.ndarray:
    """Sparse least squares with a numpy STRidge default or optional sklearn LASSO."""
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


def estimate_beta_direct(data: Dict, *, mu=0.0, omega=0.0, mask_eps=1e-4,
                         sg_window=11, sg_poly=3) -> Tuple[np.ndarray, np.ndarray]:
    """Ratio estimator ``beta = -(dS - births + deaths - waning) / (S I / N)``."""
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


def estimate_beta_windowed(data: Dict, *, mu=0.0, omega=0.0, window=8, step=1,
                           threshold=0.01, alpha=0.01, mask_eps=1e-4,
                           sg_window=11, sg_poly=3) -> Tuple[np.ndarray, np.ndarray]:
    """Sliding-window SINDy. Library ``[S I / N]``; ``beta = -xi_0``."""
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
                          poly_degree=3, n_fourier=5, fourier_period=52.0,
                          method="stridge", alpha=0.01, threshold=1e-4, lasso_alpha=1e-5
                          ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Fit ``beta(t) = Phi(t) @ xi`` on a Poly+Fourier basis with a sparse solver."""
    valid = ~np.isnan(beta_local) & (beta_local > 0)
    tv, bv = t_local[valid], beta_local[valid]
    Phi_fit, names = _time_basis(tv, tv.min(), tv.max(), poly_degree, n_fourier, fourier_period)
    xi = _sparse_fit(Phi_fit, bv, method=method, alpha=alpha, threshold=threshold,
                     lasso_alpha=lasso_alpha)
    Phi_full, _ = _time_basis(t_full, tv.min(), tv.max(), poly_degree, n_fourier, fourier_period)
    beta_global = np.clip(Phi_full @ xi, 0, 10)
    return beta_global, xi, names
