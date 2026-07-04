"""Minimal logging helper with a consistent project-wide format."""

from __future__ import annotations

import logging

_CONFIGURED = False


def get_logger(name: str = "sciml", level: int = logging.INFO) -> logging.Logger:
    """Return a logger under the ``sciml`` root, configuring a handler once.

    Parameters
    ----------
    name : str
        Logger name; a ``sciml.`` prefix is added if missing.
    level : int
        Logging level applied when the root handler is first configured.

    Returns
    -------
    logging.Logger
        The configured logger instance.
    """
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%H:%M:%S"))
        root = logging.getLogger("sciml")
        root.addHandler(handler)
        root.setLevel(level)
        root.propagate = False
        _CONFIGURED = True
    if not name.startswith("sciml"):
        name = f"sciml.{name}"
    return logging.getLogger(name)
