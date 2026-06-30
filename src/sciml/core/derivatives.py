"""Savitzky-Golay smoothing and differentiation -- pure numpy.

A compact reimplementation of ``scipy.signal.savgol_filter`` (mode='interp'
edge handling) so the SINDy core has no SciPy dependency. For each point a
local polynomial of degree ``poly`` is least-squares fit to a window of
``window`` samples and its ``deriv``-th derivative is evaluated.
"""

from __future__ import annotations

from math import factorial

import numpy as np


def _odd_window(window: int, n: int, poly: int) -> int:
    window = int(window)
    if window % 2 == 0:
        window += 1
    if window > n:
        window = n if n % 2 == 1 else n - 1
    if window <= poly:
        window = poly + 1 if (poly + 1) % 2 == 1 else poly + 2
    return max(window, 1)


def _coeffs(window: int, poly: int, deriv: int, delta: float, pos: int) -> np.ndarray:
    """Convolution coefficients giving the ``deriv``-th derivative of the local
    polynomial fit, evaluated at index ``pos`` within the window."""
    x = np.arange(window) - pos
    A = x[:, None] ** np.arange(poly + 1)[None, :]      # Vandermonde (window, poly+1)
    pinv = np.linalg.pinv(A)                            # (poly+1, window)
    return (factorial(deriv) / delta ** deriv) * pinv[deriv]


def _savgol(y: np.ndarray, window: int, poly: int, deriv: int, delta: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return y.copy()
    window = _odd_window(window, n, poly)
    if window < 2:
        return y.copy()
    half = window // 2
    out = np.empty(n, dtype=float)

    # Interior points: shared central coefficients via correlation.
    c_center = _coeffs(window, poly, deriv, delta, half)
    if n - 2 * half > 0:
        out[half:n - half] = np.correlate(y, c_center, mode="valid")

    # Left/right edges: one-sided polynomial fits.
    for pos in range(half):
        out[pos] = _coeffs(window, poly, deriv, delta, pos) @ y[:window]
    for k in range(half):
        out[n - half + k] = _coeffs(window, poly, deriv, delta, half + 1 + k) @ y[n - window:]
    return out


def savgol(y: np.ndarray, window: int = 11, poly: int = 3) -> np.ndarray:
    """Savitzky-Golay smoothing (0th derivative)."""
    return _savgol(y, window, poly, deriv=0, delta=1.0)


def savgol_derivative(y: np.ndarray, t: np.ndarray | None = None,
                      window: int = 11, poly: int = 3) -> np.ndarray:
    """Savitzky-Golay first derivative.

    ``t`` may be the (uniformly spaced) sample locations; its spacing sets the
    finite-difference ``delta``. If ``None``, unit spacing is assumed.
    """
    delta = 1.0 if t is None else float(np.mean(np.diff(np.asarray(t, dtype=float))))
    return _savgol(y, window, poly, deriv=1, delta=delta)
