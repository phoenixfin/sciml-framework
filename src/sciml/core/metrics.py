"""Error metrics for comparing predictions against references (pure numpy)."""

from __future__ import annotations

import numpy as np


def rel_l2(pred: np.ndarray, ref: np.ndarray, eps: float = 1e-10) -> float:
    """Relative L2 error ``||pred - ref|| / (||ref|| + eps)``.

    Parameters
    ----------
    pred : np.ndarray
        Predicted values.
    ref : np.ndarray
        Reference (ground-truth) values, same shape as ``pred``.
    eps : float
        Small constant added to the denominator to avoid division by zero.

    Returns
    -------
    float
        The scalar relative L2 error.
    """
    pred = np.asarray(pred)
    ref = np.asarray(ref)
    return float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + eps))


def rel_l2_pct(pred: np.ndarray, ref: np.ndarray, eps: float = 1e-10) -> float:
    """Relative L2 error expressed as a percentage.

    Parameters
    ----------
    pred : np.ndarray
        Predicted values.
    ref : np.ndarray
        Reference values, same shape as ``pred``.
    eps : float
        Small constant added to the denominator to avoid division by zero.

    Returns
    -------
    float
        ``100 * rel_l2(pred, ref, eps)``.
    """
    return 100.0 * rel_l2(pred, ref, eps)


def abs_error(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Pointwise absolute error ``|pred - ref|``.

    Parameters
    ----------
    pred : np.ndarray
        Predicted values.
    ref : np.ndarray
        Reference values, same shape as ``pred``.

    Returns
    -------
    np.ndarray
        Elementwise absolute differences, same shape as the inputs.
    """
    return np.abs(np.asarray(pred) - np.asarray(ref))


def rel_l2_batch(pred: np.ndarray, ref: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Per-sample relative L2 error for batched arrays.

    Parameters
    ----------
    pred : np.ndarray
        Predicted values of shape ``(n, ...)``; the leading axis is the batch.
    ref : np.ndarray
        Reference values, same shape as ``pred``.
    eps : float
        Small constant added to each denominator to avoid division by zero.

    Returns
    -------
    np.ndarray
        A length-``n`` array with the relative L2 error of each sample.
    """
    pred = np.asarray(pred)
    ref = np.asarray(ref)
    num = np.linalg.norm((pred - ref).reshape(len(pred), -1), axis=1)
    den = np.linalg.norm(ref.reshape(len(ref), -1), axis=1) + eps
    return num / den


def rmse(pred: np.ndarray, ref: np.ndarray) -> float:
    """Root-mean-square error.

    Parameters
    ----------
    pred : np.ndarray
        Predicted values.
    ref : np.ndarray
        Reference values, same shape as ``pred``.

    Returns
    -------
    float
        ``sqrt(mean((pred - ref) ** 2))``.
    """
    pred = np.asarray(pred)
    ref = np.asarray(ref)
    return float(np.sqrt(np.mean((pred - ref) ** 2)))
