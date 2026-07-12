"""WNTS gas-pipeline dataset loader (confidential; requires pandas).

Hourly pressure/energy-rate telemetry at the endpoint metering stations of
the West Natuna Transportation System: four sources (Anoa, Kakap, Hang
Tuah, Gajah Baru) and one sink (ORF). Only purely operational channels are
loaded; gas-composition columns are excluded. See
``experiments/wnts/REPORT.md`` for the dataset study.

Data files are contract years (``2019.csv`` covers Aug 2018 -- Jul 2019).
The data directory is taken from the ``SCIML_WNTS_DIR`` environment
variable when not passed explicitly. The data must not be redistributed
outside the research.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from .base import TimeSeriesData

NODES = {133001: "anoa", 133002: "kakap", 133003: "hangtuah",
         133004: "gajahbaru", 133060: "orf"}
SOURCES = ["anoa", "kakap", "hangtuah", "gajahbaru"]
SINK = "orf"
NODE_ORDER = SOURCES + [SINK]
P_COLS = [f"P_{n}" for n in NODE_ORDER]
Q_COLS = [f"q_{n}" for n in NODE_ORDER]


def _load_wide(data_dir: str, years: Sequence[int]) -> pd.DataFrame:
    """Contract-year CSVs as one gap-free hourly wide frame of P/q per node.

    Parameters
    ----------
    data_dir : str
        Directory holding the ``<year>.csv`` files.
    years : Sequence[int]
        Contract-year file names to load.

    Returns
    -------
    pd.DataFrame
        Hourly frame with ``P_<node>``/``q_<node>`` columns (NaN rows at
        missing hours).
    """
    usecols = ["DATE_STAMP", "ASSET_ID", "PRESSURE", "ENERGY_RATE"]
    frames = []
    for y in years:
        df = pd.read_csv(f"{data_dir}/{y}.csv", usecols=usecols, parse_dates=["DATE_STAMP"])
        frames.append(df[df["ASSET_ID"].isin(NODES)])
    df = pd.concat(frames, ignore_index=True)
    df["node"] = df["ASSET_ID"].map(NODES)
    piv = df.pivot_table(index="DATE_STAMP", columns="node",
                         values=["PRESSURE", "ENERGY_RATE"], aggfunc="mean")
    out = pd.DataFrame(index=piv.index)
    for name in NODE_ORDER:
        out[f"P_{name}"] = piv[("PRESSURE", name)]
        out[f"q_{name}"] = piv[("ENERGY_RATE", name)]
    full = pd.date_range(out.index.min(), out.index.max(), freq="h")
    return out.reindex(full)


def _frozen_runs(x: np.ndarray, min_run: int) -> np.ndarray:
    """Mask samples inside runs of ``min_run``+ identical values (stale sensor).

    Parameters
    ----------
    x : np.ndarray
        Signal (1D).
    min_run : int
        Minimum run length considered frozen.

    Returns
    -------
    np.ndarray
        Boolean mask, True inside frozen runs.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return np.zeros(0, dtype=bool)
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = x[1:] != x[:-1]
    run_id = np.cumsum(change)
    _, counts = np.unique(run_id, return_counts=True)
    return counts[run_id - 1] >= min_run


def _clean_segments(wide: pd.DataFrame, min_run: int, min_len: int) -> List[pd.DataFrame]:
    """Contiguous stretches free of NaNs and frozen sensors, chronological.

    Parameters
    ----------
    wide : pd.DataFrame
        Output of :func:`_load_wide`.
    min_run : int
        Minimum frozen-run length (hours).
    min_len : int
        Minimum segment length (hours) to keep.

    Returns
    -------
    List[pd.DataFrame]
        Clean contiguous sub-frames in chronological order.
    """
    bad = wide.isna().any(axis=1).to_numpy().copy()
    for c in wide.columns:
        bad |= _frozen_runs(wide[c].to_numpy(), min_run)
    segs, n, i = [], len(bad), 0
    while i < n:
        if bad[i]:
            i += 1
            continue
        j = i
        while j < n and not bad[j]:
            j += 1
        if j - i >= min_len:
            segs.append(wide.iloc[i:j])
        i = j
    return segs


def _block_mean(seg: pd.DataFrame, hours: int) -> pd.DataFrame:
    """Downsample a contiguous hourly segment by block-averaging.

    Parameters
    ----------
    seg : pd.DataFrame
        A contiguous hourly segment.
    hours : int
        Block size in hours (1 returns the segment unchanged).

    Returns
    -------
    pd.DataFrame
        Block means indexed by each block's first timestamp.
    """
    if hours <= 1:
        return seg
    n = (len(seg) // hours) * hours
    blocks = seg.iloc[:n].groupby(np.arange(n) // hours).mean()
    blocks.index = seg.index[:n:hours]
    return blocks


def load_wnts(data_dir: Optional[str] = None, years: Sequence[int] = (2019,),
              dt_hours: int = 3, min_seg_days: int = 10,
              min_run: int = 6) -> TimeSeriesData:
    """WNTS pipeline telemetry as clean, block-averaged segments.

    Channels: ``P_<node>`` and ``q_<node>`` for the five endpoint nodes,
    plus the derived upstream pool pressure ``P_up`` (mean of the four
    collinear source pressures -- see the A3 state-space ablation).

    Parameters
    ----------
    data_dir : Optional[str]
        Data directory; defaults to the ``SCIML_WNTS_DIR`` environment
        variable.
    years : Sequence[int]
        Contract-year files to load (2014--2015 have largely frozen source
        telemetry and should be avoided).
    dt_hours : int
        Block-averaging interval in hours.
    min_seg_days : int
        Minimum clean-segment length in days.
    min_run : int
        Frozen-sensor detection: minimum run of identical hourly values.

    Returns
    -------
    TimeSeriesData
        The segmented dataset (chronological segments, ``dt_hours``
        sampling).

    Raises
    ------
    ValueError
        If no data directory is given and ``SCIML_WNTS_DIR`` is unset.
    """
    data_dir = data_dir or os.environ.get("SCIML_WNTS_DIR")
    if not data_dir:
        raise ValueError("pass data_dir=... or set the SCIML_WNTS_DIR environment variable")
    wide = _load_wide(data_dir, years)
    segs = [_block_mean(s, dt_hours) for s in
            _clean_segments(wide, min_run=min_run, min_len=min_seg_days * 24)]
    channels = P_COLS + Q_COLS + ["P_up"]
    arrays, index = [], []
    for s in segs:
        arr = s[P_COLS + Q_COLS].to_numpy()
        p_up = arr[:, :4].mean(axis=1, keepdims=True)
        arrays.append(np.hstack([arr, p_up]))
        index.append(s.index)
    return TimeSeriesData(
        segments=arrays, channels=channels, dt_hours=float(dt_hours), index=index,
        meta={"sources": SOURCES, "sink": SINK, "years": list(years),
              "confidential": True,
              "note": "P_up = mean of the four collinear source pressures"})
