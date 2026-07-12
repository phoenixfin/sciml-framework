"""Named-dataset registry: ``load("<name>", **options)`` for any dataset.

Built-in datasets resolve lazily -- a dataset's module (and its extra
dependencies, e.g. pandas for WNTS) is only imported when that dataset is
actually loaded. Register project-specific datasets with the
:func:`register` decorator::

    from sciml.data.datasets import register, TimeSeriesData

    @register("my_plant")
    def load_my_plant(path: str = "...") -> TimeSeriesData:
        ...

Then anywhere else: ``load("my_plant", path=...)``.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict

from .base import FunctionPairData, TimeSeriesData

#: Built-in dataset name -> "module:function", resolved on first use.
_BUILTIN: Dict[str, str] = {
    "wnts": "sciml.data.datasets.wnts:load_wnts",
    "lti_demo": "sciml.data.datasets.synthetic:load_lti_demo",
    "advection_pairs": "sciml.data.datasets.synthetic:load_advection_pairs",
}

#: Datasets registered at runtime via :func:`register`.
_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator registering a loader function under a dataset name.

    Parameters
    ----------
    name : str
        The dataset name used with :func:`load`.

    Returns
    -------
    Callable[[Callable[..., Any]], Callable[..., Any]]
        A decorator that stores the loader and returns it unchanged.
    """
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        """Store ``fn`` in the runtime registry.

        Parameters
        ----------
        fn : Callable[..., Any]
            The loader function to register.

        Returns
        -------
        Callable[..., Any]
            The same loader, unchanged.
        """
        _REGISTRY[name] = fn
        return fn
    return deco


def _resolve(name: str) -> Callable[..., Any]:
    """Look up a loader by name (runtime registry first, then built-ins).

    Parameters
    ----------
    name : str
        The dataset name.

    Returns
    -------
    Callable[..., Any]
        The loader function.

    Raises
    ------
    KeyError
        If the name is neither registered nor built-in.
    """
    if name in _REGISTRY:
        return _REGISTRY[name]
    if name in _BUILTIN:
        mod_name, fn_name = _BUILTIN[name].split(":")
        return getattr(importlib.import_module(mod_name), fn_name)
    known = sorted(set(_REGISTRY) | set(_BUILTIN))
    raise KeyError(f"unknown dataset {name!r}; available: {known}")


def load(name: str, **kwargs: Any) -> Any:
    """Load a dataset by name.

    Parameters
    ----------
    name : str
        A registered or built-in dataset name (see :func:`list_datasets`).
    **kwargs : Any
        Options forwarded to the dataset's loader function.

    Returns
    -------
    Any
        The loaded dataset container (:class:`TimeSeriesData` or
        :class:`FunctionPairData`, depending on the dataset).
    """
    return _resolve(name)(**kwargs)


def list_datasets() -> Dict[str, str]:
    """All known dataset names with a one-line description.

    Returns
    -------
    Dict[str, str]
        Mapping of dataset name to the first line of its loader's
        docstring (or a note when the loader's dependencies are missing).
    """
    out: Dict[str, str] = {}
    for name in sorted(set(_REGISTRY) | set(_BUILTIN)):
        try:
            doc = _resolve(name).__doc__ or ""
            out[name] = doc.strip().splitlines()[0] if doc.strip() else ""
        except ImportError as exc:
            out[name] = f"(unavailable: {exc.name} not installed)"
    return out


__all__ = ["TimeSeriesData", "FunctionPairData", "register", "load", "list_datasets"]
