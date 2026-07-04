"""A generic, problem-agnostic training loop for injected-step models.

The :class:`Trainer` drives an injected ``step_fn(*batch) -> (loss, *components)``
(typically a ``@tf.function`` that itself applies gradients) and a
``sample_batch(iteration) -> tuple`` producer. It owns the loop, timing,
history and checkpointing -- not any PDE specifics.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ...core.logging import get_logger

_log = get_logger(__name__)


@dataclass
class History:
    """Recorded scalar training histories keyed by component name."""

    iters: List[int] = field(default_factory=list)
    values: Dict[str, List[float]] = field(default_factory=dict)

    def record(self, it: int, components: Dict[str, float]) -> None:
        """Append the scalar ``components`` recorded at iteration ``it``."""
        self.iters.append(it)
        for k, v in components.items():
            self.values.setdefault(k, []).append(float(v))

    def to_dict(self) -> Dict[str, List[float]]:
        """Return the history as a plain dict (``iter`` plus each component)."""
        return {"iter": list(self.iters), **{k: list(v) for k, v in self.values.items()}}


class Trainer:
    """Drive a training loop from an injected ``step_fn`` and ``sample_batch``."""

    def __init__(self, model, optimizer, step_fn: Callable[..., Tuple],
                 component_names: Optional[Sequence[str]] = None):
        self.model = model
        self.optimizer = optimizer
        self.step_fn = step_fn
        self.component_names = list(component_names) if component_names else None

    def _label(self, outputs: Tuple) -> Dict[str, float]:
        comps = {"loss": float(outputs[0])}
        extras = outputs[1:]
        names = self.component_names or [f"c{i}" for i in range(len(extras))]
        for name, val in zip(names, extras):
            comps[name] = float(val)
        return comps

    def fit(self, sample_batch: Callable[[int], Tuple], n_iter: int, *,
            log_every: int = 1000, history_every: int = 1000,
            ckpt_dir: Optional[str] = None, ckpt_every: int = 2000,
            warmup: int = 1, verbose: bool = True) -> History:
        """Run ``n_iter`` training steps, logging/checkpointing, returning the History."""
        import tensorflow as tf

        for w in range(warmup):
            self.step_fn(*sample_batch(-1 - w))
        t0 = time.time()
        for _ in range(min(5, n_iter)):
            self.step_fn(*sample_batch(-100))
        ms = (time.time() - t0) / max(1, min(5, n_iter)) * 1000
        if verbose:
            _log.info("Step: %.0f ms | ETA: %.1f min", ms, ms * n_iter / 60000)

        mgr = None
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
            mgr = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=2)

        history = History()
        t_start = time.time()
        for it in range(1, n_iter + 1):
            outputs = self.step_fn(*sample_batch(it))
            if it % history_every == 0:
                comps = self._label(outputs)
                history.record(it, comps)
                if verbose and it % log_every == 0:
                    elapsed = time.time() - t_start
                    eta = elapsed / it * (n_iter - it)
                    parts = " ".join(f"{k}={v:.3e}" for k, v in comps.items())
                    _log.info("%6d/%d %s | %.1f/%.0f min", it, n_iter, parts,
                              elapsed / 60, (elapsed + eta) / 60)
            if mgr and it % ckpt_every == 0:
                mgr.save(checkpoint_number=it // ckpt_every)
        if verbose:
            _log.info("Training done in %.1f min", (time.time() - t_start) / 60)
        return history
