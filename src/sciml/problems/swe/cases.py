"""Named initial conditions / bathymetry for the SWE example (pure numpy)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def h0_gaussian(x: np.ndarray, center: float = 5.0, amp: float = 0.5,
                width: float = 2.0, base: float = 1.0) -> np.ndarray:
    """Smooth Gaussian-bump depth (C1/C2).

    Parameters
    ----------
    x : np.ndarray
        Spatial coordinates.
    center : float
        Center of the Gaussian bump.
    amp : float
        Bump amplitude above the base level.
    width : float
        Inverse-variance width factor of the bump.
    base : float
        Baseline depth away from the bump.

    Returns
    -------
    np.ndarray
        The initial depth profile at ``x``.
    """
    return (base + amp * np.exp(-width * (x - center) ** 2)).astype(np.float32)


def h0_dambreak(x: np.ndarray, center: float = 5.0, height: float = 1.5,
                drop: float = 0.5, sharpness: float = 5.0) -> np.ndarray:
    """Partial dam-break (tanh step) depth (C3, OOD).

    Parameters
    ----------
    x : np.ndarray
        Spatial coordinates.
    center : float
        Location of the tanh step.
    height : float
        Mean depth level of the step.
    drop : float
        Half the depth difference across the step.
    sharpness : float
        Sharpness of the tanh transition.

    Returns
    -------
    np.ndarray
        The initial depth profile at ``x``.
    """
    return (height - drop * np.tanh(sharpness * (x - center))).astype(np.float32)


def b_flat(x: np.ndarray) -> np.ndarray:
    """Flat bed bathymetry (zeros).

    Parameters
    ----------
    x : np.ndarray
        Spatial coordinates.

    Returns
    -------
    np.ndarray
        Zero bathymetry at ``x``.
    """
    return np.zeros_like(x, dtype=np.float32)


def b_bump(x: np.ndarray, center: float = 5.0, amp: float = 0.2) -> np.ndarray:
    """Gaussian-bump bathymetry.

    Parameters
    ----------
    x : np.ndarray
        Spatial coordinates.
    center : float
        Center of the bathymetry bump.
    amp : float
        Bump amplitude.

    Returns
    -------
    np.ndarray
        The bathymetry profile at ``x``.
    """
    return (amp * np.exp(-(x - center) ** 2)).astype(np.float32)


@dataclass(frozen=True)
class Case:
    """A named evaluation case: initial depth, bathymetry, description and plot color."""

    name: str
    h0: Callable[[np.ndarray], np.ndarray]
    bath: Callable[[np.ndarray], np.ndarray]
    description: str = ""
    color: str = "b"


C1 = Case("C1", h0_gaussian, b_flat, "Smooth IC, flat bed (in-distribution)", "b")
C2 = Case("C2", h0_gaussian, b_bump, "Smooth IC, bump bathymetry (in-distribution)", "b")
C3 = Case("C3", h0_dambreak, b_flat, "Partial dam-break (OOD)", "r")

CASES = {c.name: c for c in (C1, C2, C3)}
