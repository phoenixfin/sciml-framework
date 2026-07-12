"""Data containers shared by all registered datasets.

Two container families cover the method families in this package:

- :class:`TimeSeriesData` -- multivariate signals over time, possibly in
  several disjoint segments (system identification: SINDy/SINDYc, DMD/DMDc,
  Neural ODE). Numpy-first: named channels + one array per segment.
- :class:`FunctionPairData` -- paired input/output function samples on fixed
  grids (operator learning: DeepONet, FNO).

Deliberately *not* one universal interface: the two shapes are genuinely
different, and forcing them together helps nobody (see
:mod:`sciml.problems.base` for the same argument about problems).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class TimeSeriesData:
    """Multivariate time series in contiguous, uniformly-sampled segments.

    Each segment is a ``(n_i, d)`` array over the same ``d`` named channels;
    segments are disjoint in time (gaps between them are allowed and
    expected -- e.g. masked sensor outages).
    """

    segments: List[np.ndarray]                    #: per-segment arrays ``(n_i, d)``
    channels: List[str]                           #: the ``d`` channel names
    dt_hours: float                               #: sample spacing in hours
    index: Optional[List[Any]] = None             #: optional per-segment time stamps
    meta: Dict[str, Any] = field(default_factory=dict)  #: free-form dataset metadata

    def __post_init__(self):
        """Coerce segments to 2D float arrays and validate their shape.

        Raises
        ------
        ValueError
            If any segment's column count differs from ``len(channels)``.
        """
        self.segments = [np.atleast_2d(np.asarray(s, dtype=float)) for s in self.segments]
        d = len(self.channels)
        for i, s in enumerate(self.segments):
            if s.shape[1] != d:
                raise ValueError(
                    f"segment {i} has {s.shape[1]} columns, expected {d} "
                    f"(channels: {self.channels})")

    @property
    def n_segments(self) -> int:
        """Number of contiguous segments.

        Returns
        -------
        int
            The number of segments.
        """
        return len(self.segments)

    @property
    def n_samples(self) -> int:
        """Total number of samples across all segments.

        Returns
        -------
        int
            The summed segment lengths.
        """
        return int(sum(len(s) for s in self.segments))

    def columns(self, names: Sequence[str]) -> List[int]:
        """Column indices of the given channel names.

        Parameters
        ----------
        names : Sequence[str]
            Channel names to look up.

        Returns
        -------
        List[int]
            The column index of each name.

        Raises
        ------
        KeyError
            If a name is not a channel of this dataset.
        """
        idx = []
        for n in names:
            if n not in self.channels:
                raise KeyError(f"unknown channel {n!r}; available: {self.channels}")
            idx.append(self.channels.index(n))
        return idx

    def select(self, names: Sequence[str]) -> "TimeSeriesData":
        """A new dataset restricted to the given channels (same segments).

        Parameters
        ----------
        names : Sequence[str]
            Channel names to keep, in the requested order.

        Returns
        -------
        TimeSeriesData
            The channel-subset view (arrays are copies).
        """
        cols = self.columns(names)
        return TimeSeriesData(
            segments=[s[:, cols] for s in self.segments], channels=list(names),
            dt_hours=self.dt_hours, index=self.index, meta=dict(self.meta))


@dataclass
class FunctionPairData:
    """Paired input/output function samples for operator learning.

    ``u[i]`` is the i-th input function sampled on ``u_grid`` and ``s[i]``
    the corresponding output function on ``s_grid`` (DeepONet/FNO-shaped).
    """

    u: np.ndarray                                  #: input functions ``(n, *u_shape)``
    s: np.ndarray                                  #: output functions ``(n, *s_shape)``
    u_grid: Optional[np.ndarray] = None            #: sensor locations of ``u``
    s_grid: Optional[np.ndarray] = None            #: evaluation locations of ``s``
    meta: Dict[str, Any] = field(default_factory=dict)  #: free-form dataset metadata

    def __post_init__(self):
        """Coerce arrays to float and validate the pairing.

        Raises
        ------
        ValueError
            If ``u`` and ``s`` disagree on the number of samples.
        """
        self.u = np.asarray(self.u, dtype=float)
        self.s = np.asarray(self.s, dtype=float)
        if len(self.u) != len(self.s):
            raise ValueError(f"u has {len(self.u)} samples but s has {len(self.s)}")

    @property
    def n_samples(self) -> int:
        """Number of function pairs.

        Returns
        -------
        int
            The number of (u, s) pairs.
        """
        return len(self.u)

    def split(self, train_frac: float = 0.8,
              seed: Optional[int] = None) -> Tuple["FunctionPairData", "FunctionPairData"]:
        """Random train/test split of the function pairs.

        Parameters
        ----------
        train_frac : float
            Fraction of samples assigned to the training split.
        seed : Optional[int]
            Seed for the permutation; None keeps the original order.

        Returns
        -------
        Tuple[FunctionPairData, FunctionPairData]
            The (train, test) datasets sharing this dataset's grids.
        """
        n = self.n_samples
        order = np.arange(n) if seed is None else np.random.default_rng(seed).permutation(n)
        cut = int(round(train_frac * n))
        mk = lambda ix: FunctionPairData(self.u[ix], self.s[ix], self.u_grid,
                                         self.s_grid, dict(self.meta))
        return mk(order[:cut]), mk(order[cut:])
