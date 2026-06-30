"""Small JSON IO helpers for experiment results."""

from __future__ import annotations

import json
import os
from typing import Any


def save_json(obj: Any, path: str) -> None:
    """Write ``obj`` to ``path`` as indented JSON, creating parent dirs."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_default)


def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _default(o):
    # Make numpy scalars/arrays JSON-serializable.
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")
