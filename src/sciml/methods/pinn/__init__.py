"""Physics-informed neural network (PINN) engine (TensorFlow).

Reusable building blocks distilled from the moving-boundary wave example:

* :class:`FourierEmbedding`, :class:`ScaledSigmoid` -- spectral-bias layers.
* :func:`build_mlp` -- MLP with optional Fourier feature embedding.
* :func:`derivatives_2d` -- first/second derivatives for 2nd-order PDEs in (x,t).
* :class:`PINNTrainer` -- multi-phase Adam + (SciPy) L-BFGS with best-weight tracking.
* :func:`beta_max_sampling`, :func:`replace_high_residual` -- RAR helpers.

The SciPy L-BFGS phase is optional; the Adam phases need only TensorFlow.
"""

from .layers import FourierEmbedding, ScaledSigmoid
from .networks import build_mlp
from .gradients import derivatives_2d
from .training import PINNTrainer, causal_weight
from .sampling import beta_max_sampling, replace_high_residual

__all__ = [
    "FourierEmbedding", "ScaledSigmoid", "build_mlp", "derivatives_2d",
    "PINNTrainer", "causal_weight", "beta_max_sampling", "replace_high_residual",
]
