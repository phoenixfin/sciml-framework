"""Task layers: reusable evaluation pipelines over registered datasets.

Each task family pairs a dataset container with a standard protocol so
results are comparable across datasets:

- :mod:`sciml.tasks.sysid` -- system identification on
  :class:`~sciml.data.datasets.TimeSeriesData` (SINDy / SINDYc / DMDc):
  causal operating-point referencing, chronological splits, multi-horizon
  forecast rollouts and trivial-baseline comparisons.
"""

from . import sysid

__all__ = ["sysid"]
