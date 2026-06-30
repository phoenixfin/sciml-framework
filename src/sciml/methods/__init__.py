"""Method families. Each subpackage is a self-contained engine.

* :mod:`sciml.methods.deeponet` -- operator learning (TensorFlow).
* :mod:`sciml.methods.pinn`     -- physics-informed NNs (TensorFlow + SciPy L-BFGS).
* :mod:`sciml.methods.sindy`    -- sparse identification (pure numpy; sklearn optional).

Importing a method subpackage pulls its backend; this package itself does not.
"""
