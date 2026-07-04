"""Small JSON IO helpers for experiment results."""

from __future__ import annotations

import json
import os
from typing import Any


def save_json(obj: Any, path: str) -> None:
    """Write ``obj`` to ``path`` as indented JSON, creating parent directories.

    Parameters
    ----------
    obj : Any
        A JSON-serializable object (numpy scalars/arrays are handled).
    path : str
        Destination file path.

    Returns
    -------
    None
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_default)


def load_json(path: str) -> Any:
    """Load and return the JSON object stored at ``path``.

    Parameters
    ----------
    path : str
        Path to a JSON file.

    Returns
    -------
    Any
        The decoded JSON object.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _default(o: Any):
    """Fallback encoder making numpy scalars/arrays JSON-serializable.

    Parameters
    ----------
    o : Any
        The object ``json`` could not serialize natively.

    Returns
    -------
    Any
        A JSON-native representation of ``o``.

    Raises
    ------
    TypeError
        If ``o`` is not a supported numpy type.
    """
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")
