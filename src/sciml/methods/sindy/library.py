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
        """Map data ``X`` ``(m, d)`` to the feature matrix ``Theta`` ``(m, p)``.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape ``(m, d)``.

        Returns
        -------
        np.ndarray
            The feature matrix ``Theta`` of shape ``(m, p)``.

        Raises
        ------
        NotImplementedError
            Always; subclasses must override this method.
        """
        raise NotImplementedError

    def names(self, input_names: Optional[Sequence[str]] = None) -> List[str]:  # pragma: no cover
        """Return the ``p`` human-readable feature names.

        Parameters
        ----------
        input_names : Optional[Sequence[str]]
            Names of the ``d`` input variables.

        Returns
        -------
        List[str]
            The ``p`` feature names.

        Raises
        ------
        NotImplementedError
            Always; subclasses must override this method.
        """
        raise NotImplementedError

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Transform ``X`` after coercing it to a 2D float array.

        Parameters
        ----------
        X : np.ndarray
            Data matrix (coerced to at least 2D float).

        Returns
        -------
        np.ndarray
            The feature matrix ``Theta``.
        """
        return self.transform(np.atleast_2d(np.asarray(X, dtype=float)))

    def __add__(self, other: "FeatureLibrary") -> "ConcatLibrary":
        """Concatenate this library with ``other``.

        Parameters
        ----------
        other : "FeatureLibrary"
            The library to concatenate with.

        Returns
        -------
        "ConcatLibrary"
            A concatenated library combining both.
        """
        return ConcatLibrary([self, other])

    @staticmethod
    def _input_names(d: int, input_names: Optional[Sequence[str]]) -> List[str]:
        """Resolve input-variable names, defaulting to ``x0, x1, ...``.

        Parameters
        ----------
        d : int
            Number of input variables.
        input_names : Optional[Sequence[str]]
            Explicit names, or None to auto-generate.

        Returns
        -------
        List[str]
            The resolved input-variable names.
        """
        if input_names is not None:
            return list(input_names)
        return [f"x{i}" for i in range(d)]


class PolynomialLibrary(FeatureLibrary):
    """All monomials up to ``degree`` over the ``d`` input variables."""

    def __init__(self, degree: int = 2, include_bias: bool = True,
                 interaction_only: bool = False):
        """Configure the polynomial degree and term options.

        Parameters
        ----------
        degree : int
            Maximum total monomial degree.
        include_bias : bool
            Whether to include the constant (degree-0) term.
        interaction_only : bool
            If True, exclude terms with repeated variables (pure powers).
        """
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only

    def _combos(self, d: int) -> list:
        """Enumerate the variable-index combinations for each monomial.

        Parameters
        ----------
        d : int
            Number of input variables.

        Returns
        -------
        list
            One tuple of variable indices per monomial term.
        """
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
        """Evaluate every monomial term column-wise -> ``(m, p)``.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape ``(m, d)``.

        Returns
        -------
        np.ndarray
            The monomial feature matrix of shape ``(m, p)``.
        """
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
        """Monomial names such as ``1``, ``x``, ``x^2``, ``x*y`` (needs ``input_names``).

        Parameters
        ----------
        input_names : Optional[Sequence[str]]
            Names of the ``d`` input variables.

        Returns
        -------
        List[str]
            The monomial term names.
        """
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
        """Configure the number of harmonics and the base period.

        Parameters
        ----------
        n_frequencies : int
            Number of harmonics ``k = 1..n_frequencies`` per column.
        period : float
            Base period used to set the fundamental frequency.
        """
        self.n_frequencies = n_frequencies
        self.period = period

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Stack sine/cosine harmonics of every column -> ``(m, 2 * d * n_freq)``.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape ``(m, d)``.

        Returns
        -------
        np.ndarray
            The harmonic feature matrix of shape ``(m, 2 * d * n_freq)``.
        """
        X = np.atleast_2d(X)
        cols = []
        for j in range(X.shape[1]):
            for k in range(1, self.n_frequencies + 1):
                w = 2 * np.pi * k / self.period
                cols.append(np.sin(w * X[:, j]))
                cols.append(np.cos(w * X[:, j]))
        return np.column_stack(cols)

    def names(self, input_names: Optional[Sequence[str]] = None) -> List[str]:
        """Harmonic names such as ``sin(1w*x)``, ``cos(1w*x)``.

        Parameters
        ----------
        input_names : Optional[Sequence[str]]
            Names of the ``d`` input variables.

        Returns
        -------
        List[str]
            The harmonic term names.
        """
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
        """Store the user term functions and their matching names.

        Parameters
        ----------
        functions : Sequence[Callable[[np.ndarray], np.ndarray]]
            Term functions mapping ``X`` to a column ``(m,)``.
        names : Sequence[str]
            One name per term function.

        Raises
        ------
        ValueError
            If ``functions`` and ``names`` differ in length.
        """
        if len(functions) != len(names):
            raise ValueError("functions and names must have equal length")
        self.functions = list(functions)
        self._names = list(names)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply each user function to ``X`` and stack the columns.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape ``(m, d)``.

        Returns
        -------
        np.ndarray
            The stacked feature matrix of shape ``(m, len(functions))``.
        """
        X = np.atleast_2d(X)
        return np.column_stack([f(X) for f in self.functions])

    def names(self, input_names: Optional[Sequence[str]] = None) -> List[str]:
        """Return the user-provided term names.

        Parameters
        ----------
        input_names : Optional[Sequence[str]]
            Ignored; present for interface compatibility.

        Returns
        -------
        List[str]
            The user-provided term names.
        """
        return list(self._names)


class ConcatLibrary(FeatureLibrary):
    """Horizontally concatenate several libraries."""

    def __init__(self, libraries: Sequence[FeatureLibrary]):
        """Store the sub-libraries to concatenate.

        Parameters
        ----------
        libraries : Sequence[FeatureLibrary]
            The feature libraries to concatenate horizontally.
        """
        self.libraries: List[FeatureLibrary] = list(libraries)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Concatenate the feature matrices of all sub-libraries.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape ``(m, d)``.

        Returns
        -------
        np.ndarray
            The horizontally concatenated feature matrix.
        """
        return np.column_stack([lib.transform(np.atleast_2d(X)) for lib in self.libraries])

    def names(self, input_names: Optional[Sequence[str]] = None) -> List[str]:
        """Concatenate the feature names of all sub-libraries.

        Parameters
        ----------
        input_names : Optional[Sequence[str]]
            Names of the ``d`` input variables.

        Returns
        -------
        List[str]
            The concatenated feature names.
        """
        out: List[str] = []
        for lib in self.libraries:
            out.extend(lib.names(input_names))
        return out
