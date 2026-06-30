"""Reproducible seeding across numpy and (optionally) TensorFlow."""

from __future__ import annotations

import os
import random


def seed_everything(seed: int = 42, *, tensorflow: bool = True) -> None:
    """Seed Python ``random``, numpy and -- if installed -- TensorFlow.

    TensorFlow is imported lazily so this is usable in pure-numpy code paths.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    import numpy as np
    np.random.seed(seed)

    if tensorflow:
        try:
            import tensorflow as tf  # type: ignore
        except ImportError:
            return
        tf.random.set_seed(seed)
