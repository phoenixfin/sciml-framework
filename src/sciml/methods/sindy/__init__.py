"""Sparse identification of nonlinear dynamics (SINDy).

The core is pure numpy: a sequential-thresholded-ridge sparse solver
(:func:`stridge`), composable feature libraries, and a :class:`SINDy` estimator
that fits ``Xdot ~ Theta(X) @ Xi``. scikit-learn is optional and only used by
the LASSO-based variants in the epidemiology example.
"""

from .library import (
                      ConcatLibrary,
                      CustomLibrary,
                      FeatureLibrary,
                      FourierLibrary,
                      PolynomialLibrary,
)
from .model import SINDy, windowed_coefficients
from .sparse import ridge_regression, stridge

__all__ = [
    "stridge", "ridge_regression",
    "FeatureLibrary", "PolynomialLibrary", "FourierLibrary",
    "ConcatLibrary", "CustomLibrary",
    "SINDy", "windowed_coefficients",
]
