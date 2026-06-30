"""Composable feature libraries for SINDy / sparse regression (pure numpy).

A library maps a data matrix ``X`` of shape ``(m, d)`` to a feature matrix
``Theta`` of shape ``(m, p)`` and exposes human-readable feature names. Build
the candidate-term dictionary by composing libraries with
:class:`ConcatLibrary` (or the ``+`` operator).
"""

from __future__ import annotations

from itertools import combinations_with_replacement
from typing import Callable, List, Optional, Sequence

import numpy as np


class FeatureLibrary:
    """Base class. Subclasses implement :meth:`transform` and :meth:`names`."""

    def transform(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def names(self, input_names: Optional[Sequence[str]] = None) -> List[str]:  # pragma: no cover
        raise NotImplementedError

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.transform(np.atleast_2d(np.asarray(X, dtype=float)))

    def __add__(self, other: "FeatureLibrary") -> "ConcatLibrary":
        return ConcatLibrary([self, other])

    @staticmethod
    def _input_names(d: int, input_names: Optional[Sequence[str]]) -> List[str]:
        if input_names is not None:
            return list(input_names)
        return [f"x{i}" for i in range(d)]


class PolynomialLibrary(FeatureLibrary):
    """All monomials up to ``degree`` over the ``d`` input variables."""

    def __init__(self, degree: int = 2, include_bias: bool = True,
                 interaction_only: bool = False):
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only

    def _combos(self, d: int):
        combos = []
        start = 0 if self.include_bias else 1
        for deg in range(start, self.degree + 1):
            if deg == 0:
                combos.append(())
                continue
            for c in combinations_with_replacement(range(d), deg):
                if self.interaction_only and len(set(c)) != len(c):
                    continue
                combos.append(c)
        return combos

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        d = X.shape[1]
        cols = []
        for combo in self._combos(d):
            if not combo:
                cols.append(np.ones(len(X)))
            else:
                col = np.ones(len(X))
                for i in combo:
                    col = col * X[:, i]
                cols.append(col)
        return np.column_stack(cols)

    def names(self, input_names: Optional[Sequence[str]] = None) -> List[str]:
        # d inferred lazily: names need d; default to single var if unknown.
        raise_if_unknown = input_names is None
        d = 1 if raise_if_unknown else len(input_names)
        names = []
        for combo in self._combos(d):
            if not combo:
                names.append("1")
            else:
                counts = {i: combo.count(i) for i in set(combo)}
                parts = []
                for i, c in sorted(counts.items()):
                    nm = self._input_names(d, input_names)[i]
                    parts.append(nm if c == 1 else f"{nm}^{c}")
                names.append("*".join(parts))
        return names


class FourierLibrary(FeatureLibrary):
    """Sine/cosine harmonics of each input column (a time/seasonal basis).

    For each column ``x`` and harmonic ``k=1..n_frequencies`` produces
    ``sin(2 pi k x / period)`` and ``cos(2 pi k x / period)``.
    """

    def __init__(self, n_frequencies: int = 3, period: float = 1.0):
        self.n_frequencies = n_frequencies
        self.period = period

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        cols = []
        for j in range(X.shape[1]):
            for k in range(1, self.n_frequencies + 1):
                w = 2 * np.pi * k / self.period
                cols.append(np.sin(w * X[:, j]))
                cols.append(np.cos(w * X[:, j]))
        return np.column_stack(cols)

    def names(self, input_names: Optional[Sequence[str]] = None) -> List[str]:
        d = 1 if input_names is None else len(input_names)
        nm = self._input_names(d, input_names)
        names = []
        for j in range(d):
            for k in range(1, self.n_frequencies + 1):
                names.append(f"sin({k}w*{nm[j]})")
                names.append(f"cos({k}w*{nm[j]})")
        return names


class CustomLibrary(FeatureLibrary):
    """User-supplied term functions ``f(X) -> (m,)`` with matching names."""

    def __init__(self, functions: Sequence[Callable[[np.ndarray], np.ndarray]],
                 names: Sequence[str]):
        if len(functions) != len(names):
            raise ValueError("functions and names must have equal length")
        self.functions = list(functions)
        self._names = list(names)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return np.column_stack([f(X) for f in self.functions])

    def names(self, input_names: Optional[Sequence[str]] = None) -> List[str]:
        return list(self._names)


class ConcatLibrary(FeatureLibrary):
    """Horizontally concatenate several libraries."""

    def __init__(self, libraries: Sequence[FeatureLibrary]):
        self.libraries: List[FeatureLibrary] = list(libraries)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([lib.transform(np.atleast_2d(X)) for lib in self.libraries])

    def names(self, input_names: Optional[Sequence[str]] = None) -> List[str]:
        out: List[str] = []
        for lib in self.libraries:
            out.extend(lib.names(input_names))
        return out
