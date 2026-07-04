"""S(t) reconstruction from an observed infected series I(t) (pure numpy).

Three options matching the notebook:
* ``cumulative`` -- R = integral of gamma*I, S = N - I - R (fast, recommended).
* ``ode``        -- forward-integrate the compartments (RK4) with beta back-computed.
* ``ekf``        -- Extended Kalman Filter jointly estimating S, R and beta.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ...core.derivatives import savgol_derivative
from ...solvers.compartmental import rk4_integrate


def _cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y, dtype=float)
    out[1:] = np.cumsum((y[1:] + y[:-1]) / 2.0 * np.diff(x))
    return out


def reconstruct_S_cumulative(t: np.ndarray, I: np.ndarray, N: float,
                             gamma: float) -> Tuple[np.ndarray, np.ndarray]:
    """``R(t) = int gamma*I``, ``S = N - I - R`` (constant gamma, single strain).

    Parameters
    ----------
    t : np.ndarray
        Time array.
    I : np.ndarray
        Infected series.
    N : float
        Total population size.
    gamma : float
        Recovery rate.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The reconstructed susceptible ``S(t)`` and recovered ``R(t)`` series.
    """
    R = _cumtrapz(gamma * I, t)
    S = np.clip(N - I - R, 0, N)
    return S, R


def reconstruct_S_ode(t: np.ndarray, I: np.ndarray, N: float, gamma: float,
                      mu: float = 0.0, omega: float = 0.0, model: str = "SIR",
                      sg_window: int = 11, sg_poly: int = 3, mask_eps: float = 1e-4
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Forward-integrate compartments using I(t) as forcing; beta back-computed.

    Parameters
    ----------
    t : np.ndarray
        Time array.
    I : np.ndarray
        Infected series (used as forcing).
    N : float
        Total population size.
    gamma : float
        Recovery rate.
    mu : float
        Birth/death rate.
    omega : float
        Waning-immunity rate.
    model : str
        Compartmental model, one of ``"SI"``, ``"SIR"`` or ``"SIRS"``.
    sg_window : int
        Savitzky-Golay smoothing window length for the derivative of ``I``.
    sg_poly : int
        Savitzky-Golay polynomial order.
    mask_eps : float
        Relative floor on the ``S I / N`` denominator when back-computing beta.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The reconstructed susceptible ``S(t)`` and recovered ``R(t)`` series.
    """
    dI = savgol_derivative(I, t, window=sg_window, poly=sg_poly)

    def I_at(time):
        return float(np.interp(time, t, I))

    def dI_at(time):
        return float(np.interp(time, t, dI))

    def rhs(time, y):
        S = y[0]
        R = y[1] if model != "SI" else 0.0
        Iv, dIv = I_at(time), dI_at(time)
        denom = max(S * Iv / N, mask_eps * N)
        beta_t = np.clip((dIv + gamma * Iv) / denom, 0, 5)
        if model == "SI":
            return np.array([-beta_t * S * Iv / N])
        if model == "SIR":
            return np.array([-beta_t * S * Iv / N, gamma * Iv])
        return np.array([mu * N - beta_t * S * Iv / N - mu * S + omega * R,
                         gamma * Iv - mu * R - omega * R])

    y0 = np.array([N - I[0]]) if model == "SI" else np.array([N - I[0], 0.0])
    ys = rk4_integrate(rhs, y0, t)
    S = np.clip(ys[:, 0], 0, N)
    R = np.clip(ys[:, 1], 0, N) if model != "SI" else np.zeros_like(S)
    return S, R


def reconstruct_S_ekf(t: np.ndarray, I_obs: np.ndarray, N: float, gamma: float,
                      mu: float = 0.0, omega: float = 0.0, model: str = "SIR",
                      Q_diag: Tuple = (1e4, 1e4, 1e4), R_obs: Optional[float] = None,
                      beta_init: float = 0.3
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extended Kalman Filter over state ``[S, I, R, beta]`` (beta random walk).

    Returns ``(S_hist, R_hist, beta_hist)`` -- the EKF's own beta estimate too.

    Parameters
    ----------
    t : np.ndarray
        Time array.
    I_obs : np.ndarray
        Observed infected series.
    N : float
        Total population size.
    gamma : float
        Recovery rate.
    mu : float
        Birth/death rate.
    omega : float
        Waning-immunity rate.
    model : str
        Compartmental model, one of ``"SI"``, ``"SIR"`` or ``"SIRS"``.
    Q_diag : Tuple
        Process-noise variances for the ``S``, ``I`` and ``R`` states.
    R_obs : Optional[float]
        Observation-noise variance; defaults to ``var(I_obs) / 10`` when ``None``.
    beta_init : float
        Initial guess for the transmission rate beta.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        The filtered susceptible ``S(t)``, recovered ``R(t)`` and beta histories.
    """
    dt = float(np.mean(np.diff(t)))
    if R_obs is None:
        R_obs = np.var(I_obs) / 10.0
    n = 4
    x = np.array([N - I_obs[0], I_obs[0], 0.0, beta_init])
    P = np.diag([N**2 * 0.01, N**2 * 0.01, N**2 * 0.01, 0.01])
    Q = np.diag(list(Q_diag) + [1e-4])
    H = np.array([[0, 1, 0, 0]], dtype=float)
    R_mat = np.array([[R_obs]], dtype=float)
    S_hist, R_hist, beta_hist = (np.zeros(len(t)) for _ in range(3))

    for k in range(len(t)):
        S_k, I_k, R_k, b_k = x
        b_k = np.clip(b_k, 0, 5)
        if model == "SI":
            dS, dI, dR = -b_k * S_k * I_k / N, b_k * S_k * I_k / N, 0.0
        elif model == "SIR":
            dS, dI, dR = (-b_k * S_k * I_k / N, b_k * S_k * I_k / N - gamma * I_k, gamma * I_k)
        else:
            dS = mu * N - b_k * S_k * I_k / N - mu * S_k + omega * R_k
            dI = b_k * S_k * I_k / N - gamma * I_k - mu * I_k
            dR = gamma * I_k - mu * R_k - omega * R_k
        x_pred = np.array([np.clip(S_k + dS * dt, 0, N), np.clip(I_k + dI * dt, 0, N),
                           np.clip(R_k + dR * dt, 0, N), b_k])
        F = np.eye(n)
        F[0, 0] += -b_k * I_k / N * dt; F[0, 1] += -b_k * S_k / N * dt
        F[0, 3] += -S_k * I_k / N * dt; F[1, 0] += b_k * I_k / N * dt
        F[1, 1] += ((b_k * S_k / N - gamma) * dt if model != "SI" else b_k * S_k / N * dt)
        F[1, 3] += S_k * I_k / N * dt
        P_pred = F @ P @ F.T + Q
        y_innov = I_obs[k] - x_pred[1]
        S_innov = H @ P_pred @ H.T + R_mat
        K = P_pred @ H.T @ np.linalg.inv(S_innov)
        x = x_pred + (K @ np.array([[y_innov]])).flatten()
        x[:3] = np.clip(x[:3], 0, N); x[3] = np.clip(x[3], 0, 5)
        P = (np.eye(n) - K @ H) @ P_pred
        S_hist[k], _, R_hist[k], beta_hist[k] = x[0], x[1], x[2], x[3]
    return S_hist, R_hist, beta_hist
