"""Runnable experiment scripts, one package per study.

Each package is a thin command-line layer over ``sciml``: the science lives in
``sciml.problems`` and ``sciml.methods``, and the scripts here only parse
arguments, call runners, and write artefacts into ``outputs/``.

======================  ====================================================
``swe``                 DeepONet on the 1D shallow-water equations: training,
                        evaluation, and the three follow-up studies.
``wave_obstacle``       PINN on the moving-boundary wave problem.
``epidemiology``        SINDy identification of a dengue transmission rate.
``wnts``                SINDYc on a gas transmission network (see REPORT.md).
======================  ====================================================
"""
