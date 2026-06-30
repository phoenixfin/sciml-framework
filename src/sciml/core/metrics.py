"""Error metrics (pure numpy)."""

from __future__ import annotations

import numpy as np


def rel_l2(pred: np.ndarray, ref: np.ndarray, eps: float = 1e-10) -> float:
    """Relative L2 error ``||pred - ref|| / (||ref|| + eps)``."""
    pred = np.asarray(pred); ref = np.asarray(ref)
    return float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + eps))


def rel_l2_pct(pred: np.ndarray, ref: np.ndarray, eps: float = 1e-10) -> float:
    """Relative L2 error as a percentage."""
    return 100.0 * rel_l2(pred, ref, eps)


def abs_error(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Pointwise absolute error ``|pred - ref|``."""
    return np.abs(np.asarray(pred) - np.asarray(ref))


def rel_l2_batch(pred: np.ndarray, ref: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Per-sample relative L2 error for batched ``(n, ...)`` arrays."""
    pred = np.asarray(pred); ref = np.asarray(ref)
    num = np.linalg.norm((pred - ref).reshape(len(pred), -1), axis=1)
    den = np.linalg.norm(ref.reshape(len(ref), -1), axis=1) + eps
    return num / den


def rmse(pred: np.ndarray, ref: np.ndarray) -> float:
    """Root-mean-square error."""
    pred = np.asarray(pred); ref = np.asarray(ref)
    return float(np.sqrt(np.mean((pred - ref) ** 2)))
