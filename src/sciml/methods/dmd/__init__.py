"""Dynamic Mode Decomposition / Koopman engine (pure numpy).

DMD finds a best-fit linear operator ``A`` such that ``x_{k+1} ~= A x_k`` from
snapshot data, then diagonalizes it into spatial modes with exponential time
dynamics -- a data-driven, equation-free counterpart to SINDy for (approximately)
linear/Koopman dynamics. Pure numpy, so it is fully testable without a backend.
"""

from .dmd import DMD

__all__ = ["DMD"]
