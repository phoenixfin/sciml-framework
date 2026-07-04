"""A light, optional base for problem definitions.

Problems differ enough across method families (operator learning vs. PINN vs.
system identification) that a single heavy interface would be a poor fit. This
base only fixes the common shape: a problem holds a config and can produce a
reference (ground-truth) solution. Method-specific responsibilities (build a
model, a training step, estimators) are added by each concrete problem.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..core.config import ConfigBase


class Problem(ABC):
    """Base for problem definitions: holds a config and yields a reference solution."""

    name: str = "problem"

    def __init__(self, config: ConfigBase):
        self.config = config

    @abstractmethod
    def reference(self, *args, **kwargs) -> Any:
        """Compute a ground-truth reference solution for evaluation."""
