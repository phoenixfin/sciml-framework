"""sciml -- a scientific-machine-learning research framework.

Three method families share one substrate:

* :mod:`sciml.methods.deeponet` -- operator learning (DeepONet).
* :mod:`sciml.methods.pinn`     -- physics-informed neural networks.
* :mod:`sciml.methods.sindy`    -- sparse identification of dynamics.

The shared substrate is backend-light on purpose:

* :mod:`sciml.core`     -- config, metrics, plotting, seeding, logging, derivatives
* :mod:`sciml.data`     -- function samplers + interpolation
* :mod:`sciml.solvers`  -- numerical reference solvers (numpy)
* :mod:`sciml.problems` -- worked examples wiring a problem to a method

``import sciml`` pulls in only pure-python/numpy code; TensorFlow, SciPy and
scikit-learn are optional extras used by individual methods/examples.

Worked examples (one per method)
--------------------------------
* DeepONet -> 1D Shallow Water Equations         (:mod:`sciml.problems.swe`)
* PINN     -> moving-boundary wave (obstacle)    (:mod:`sciml.problems.wave_obstacle`)
* SINDy    -> dengue beta(t) identification      (:mod:`sciml.problems.epidemiology`)
"""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = ["__version__"]
