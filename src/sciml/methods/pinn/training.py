"""Multi-phase PINN training: Adam phases + (SciPy) L-BFGS, with best-weight tracking.

The :class:`PINNTrainer` is problem-agnostic. It drives a user ``loss_fn()``
that returns a scalar loss tensor (closing over the model variables and any
collocation sampling). Per-step scheduling (e.g. annealing a causal weight) and
adaptive resampling are injected as ``on_step`` callbacks.
"""

from __future__ import annotations

import time
from typing import Callable, List, Optional

import numpy as np

from ...core.logging import get_logger

_log = get_logger(__name__)


def causal_weight(tau, eps_causal, t_final):
    """Causal training weight ``exp(-eps_causal * tau / T)`` (down-weights late
    times until earlier ones are learned). Works on numpy or TF tensors."""
    import tensorflow as tf
    return tf.exp(-eps_causal * tau / t_final)


class PINNTrainer:
    """Drive Adam phases and an optional L-BFGS phase over a variable list."""

    def __init__(self, variables, loss_fn: Callable[[], "object"]):
        self.variables = list(variables)
        self.loss_fn = loss_fn
        self.history: List[float] = []
        self.best_loss = np.inf
        self.best_weights: Optional[np.ndarray] = None

    # -- flat weight (de)serialization -----------------------------------
    def get_weights(self) -> np.ndarray:
        """Return all trainable variables flattened into one float64 vector."""
        return np.concatenate([v.numpy().ravel() for v in self.variables]).astype(np.float64)

    def set_weights(self, w: np.ndarray) -> None:
        """Assign a flat float64 vector ``w`` back into the trainable variables."""
        idx = 0
        for v in self.variables:
            n = v.numpy().size
            v.assign(w[idx:idx + n].reshape(v.shape).astype(np.float32))
            idx += n

    def _update_best(self, loss_val: float, weights: Optional[np.ndarray] = None) -> None:
        if loss_val < self.best_loss:
            self.best_loss = loss_val
            self.best_weights = weights if weights is not None else self.get_weights()

    # -- Adam ------------------------------------------------------------
    def run_adam(self, n_steps: int, lr: float, *,
                 on_step: Optional[Callable[[int], None]] = None,
                 print_every: int = 500, verbose: bool = True) -> None:
        """Run ``n_steps`` Adam steps; ``on_step(step)`` runs before each (scheduling/RAR)."""
        import tensorflow as tf
        opt = tf.keras.optimizers.Adam(lr)

        @tf.function
        def train_step():
            with tf.GradientTape() as tape:
                total = self.loss_fn()
            opt.apply_gradients(zip(tape.gradient(total, self.variables), self.variables))
            return total

        t0 = time.time()
        for ep in range(1, n_steps + 1):
            if on_step is not None:
                on_step(ep)
            loss_val = float(train_step())
            self.history.append(loss_val)
            self._update_best(loss_val)
            if verbose and (ep % print_every == 0 or ep == 1):
                _log.info("  [%5d/%d] loss=%.3e (%.0fs)", ep, n_steps, loss_val,
                          time.time() - t0)

    # -- L-BFGS (SciPy) --------------------------------------------------
    def run_lbfgs(self, maxiter: int = 6000, *, restart_from_best: bool = False,
                  ftol: float = 1e-16, gtol: float = 1e-11, verbose: bool = True) -> None:
        """Refine with SciPy L-BFGS-B; optionally restart from the best-so-far weights."""
        try:
            from scipy.optimize import minimize
        except ImportError as exc:  # pragma: no cover
            raise ImportError("The L-BFGS phase requires SciPy "
                              "(`pip install scipy`).") from exc
        import tensorflow as tf

        if restart_from_best and self.best_weights is not None:
            self.set_weights(self.best_weights)

        def loss_and_grad(w):
            self.set_weights(w)
            with tf.GradientTape() as tape:
                total = self.loss_fn()
            grads = tape.gradient(total, self.variables)
            g = np.concatenate([gi.numpy().ravel() for gi in grads]).astype(np.float64)
            lv = float(total)
            self.history.append(lv)
            self._update_best(lv, w.copy())
            return lv, g

        t0 = time.time()
        res = minimize(loss_and_grad, self.get_weights(), method="L-BFGS-B", jac=True,
                       options={"maxiter": maxiter, "ftol": ftol, "gtol": gtol})
        if self.best_weights is not None:
            self.set_weights(self.best_weights)
        if verbose:
            _log.info("  L-BFGS: best=%.3e iters=%d (%.0fs)", self.best_loss,
                      res.nit, time.time() - t0)
