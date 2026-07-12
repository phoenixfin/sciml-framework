"""Loading and preprocessing for the WNTS gas-pipeline dataset.

The West Natuna Transportation System (WNTS) dataset holds hourly
measurements (pressure, temperature, energy/volume rate, composition) at
the network's endpoint metering stations only: four sources (Anoa, Kakap,
Hang Tuah, Gajah Baru) and one sink (ORF). Asset IDs 133070/133071
duplicate the ORF and Gajah Baru flow meters and are dropped.

The data is confidential -- do not redistribute outside the research.

Notes on quirks handled here
----------------------------
- Files are contract years (Aug--Jul): ``2019.csv`` covers Aug 2018--Jul 2019.
- Telemetry freezes: sensors repeat the exact same float value for hours or
  days. Rows inside such runs are masked out.
- Sources persistently deliver ~9% more energy than the sink meters
  (fuel gas / shrinkage / meter basis); the line-pack proxy removes this
  bias before integrating the flow imbalance.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

NODES = {
    133001: "anoa",
    133002: "kakap",
    133003: "hangtuah",
    133004: "gajahbaru",
    133060: "orf",
}
SOURCES = ["anoa", "kakap", "hangtuah", "gajahbaru"]
SINK = "orf"
NODE_ORDER = SOURCES + [SINK]

P_COLS = [f"P_{n}" for n in NODE_ORDER]
Q_COLS = [f"q_{n}" for n in NODE_ORDER]


def load_wide(data_dir: str, years: Sequence[int]) -> pd.DataFrame:
    """Load contract-year CSVs into a wide hourly frame of P/q per node.

    Parameters
    ----------
    data_dir : str
        Directory holding the ``<year>.csv`` files.
    years : Sequence[int]
        Contract-year file names to load (e.g. ``[2019]`` loads
        Aug 2018--Jul 2019).

    Returns
    -------
    pd.DataFrame
        Frame indexed by a gap-free hourly ``DatetimeIndex`` with columns
        ``P_<node>`` and ``q_<node>`` (missing hours are NaN rows).

    Notes
    -----
    Only purely operational columns are read (pressure and energy rate).
    The gas-composition columns (C1..C9, N2, CO2, H2O, HCDP) are excluded
    from the study.
    """
    usecols = ["DATE_STAMP", "ASSET_ID", "PRESSURE", "ENERGY_RATE"]
    frames = []
    for y in years:
        df = pd.read_csv(f"{data_dir}/{y}.csv", usecols=usecols, parse_dates=["DATE_STAMP"])
        frames.append(df[df["ASSET_ID"].isin(NODES)])
    df = pd.concat(frames, ignore_index=True)
    df["node"] = df["ASSET_ID"].map(NODES)
    piv = df.pivot_table(
        index="DATE_STAMP", columns="node", values=["PRESSURE", "ENERGY_RATE"], aggfunc="mean"
    )
    out = pd.DataFrame(index=piv.index)
    for name in NODE_ORDER:
        out[f"P_{name}"] = piv[("PRESSURE", name)]
        out[f"q_{name}"] = piv[("ENERGY_RATE", name)]
    full = pd.date_range(out.index.min(), out.index.max(), freq="h")
    return out.reindex(full)


def frozen_runs(x: np.ndarray, min_run: int = 6) -> np.ndarray:
    """Mask samples belonging to a run of ``min_run``+ identical values.

    Real telemetry virtually never repeats the exact same float for many
    consecutive hours; such runs indicate a frozen/stale sensor.

    Parameters
    ----------
    x : np.ndarray
        Signal (1D).
    min_run : int
        Minimum run length (in samples) considered frozen.

    Returns
    -------
    np.ndarray
        Boolean mask, True where the sample lies inside a frozen run.
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


def bad_rows(wide: pd.DataFrame, min_run: int = 6) -> np.ndarray:
    """Rows unusable for identification: any NaN or any frozen sensor.

    Parameters
    ----------
    wide : pd.DataFrame
        Output of :func:`load_wide`.
    min_run : int
        Minimum frozen-run length passed to :func:`frozen_runs`.

    Returns
    -------
    np.ndarray
        Boolean mask over rows, True where the row is unusable.
    """
    bad = wide.isna().any(axis=1).to_numpy().copy()
    for c in wide.columns:
        bad |= frozen_runs(wide[c].to_numpy(), min_run)
    return bad


def clean_segments(
    wide: pd.DataFrame, min_run: int = 6, min_len: int = 21 * 24
) -> List[pd.DataFrame]:
    """Split the frame into contiguous clean segments, longest first.

    Parameters
    ----------
    wide : pd.DataFrame
        Output of :func:`load_wide`.
    min_run : int
        Minimum frozen-run length passed to :func:`bad_rows`.
    min_len : int
        Minimum segment length in hours to keep.

    Returns
    -------
    List[pd.DataFrame]
        Clean contiguous sub-frames sorted by length (descending).
    """
    bad = bad_rows(wide, min_run)
    segs: List[pd.DataFrame] = []
    n, i = len(bad), 0
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
    segs.sort(key=len, reverse=True)
    return segs


def block_mean(seg: pd.DataFrame, hours: int) -> pd.DataFrame:
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


def linepack_proxy(
    seg: pd.DataFrame, bias: Optional[float] = None, dt: float = 1.0
) -> Tuple[np.ndarray, float]:
    """Latent line-pack proxy: integrated, bias-corrected flow imbalance.

    ``L(t) = sum_t (q_in - q_out - bias) * dt``, then mean-centred over the
    segment (line pack is only observable up to an unknown offset, so each
    segment is referenced to its own operating level).

    Parameters
    ----------
    seg : pd.DataFrame
        A clean segment with ``q_<node>`` columns.
    bias : Optional[float]
        Systematic source-sink imbalance (fuel gas / shrinkage) to remove
        before integrating. Estimated from ``seg`` itself when None --
        pass the training-segment value when transforming test data.
    dt : float
        Sample spacing in hours.

    Returns
    -------
    Tuple[np.ndarray, float]
        The mean-centred line-pack proxy ``(m,)`` and the bias used.
    """
    imb = seg[[f"q_{s}" for s in SOURCES]].sum(axis=1) - seg[f"q_{SINK}"]
    if bias is None:
        bias = float(imb.mean())
    L = ((imb - bias) * dt).cumsum().to_numpy()
    return L - L.mean(), bias


class Scaler:
    """Column-wise z-score scaler (so STRidge thresholds are comparable)."""

    def fit(self, X: np.ndarray) -> "Scaler":
        """Store per-column mean and standard deviation.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape ``(m, d)``.

        Returns
        -------
        Scaler
            The fitted scaler (``self``).
        """
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standardize columns with the fitted statistics.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape ``(m, d)``.

        Returns
        -------
        np.ndarray
            The standardized data.
        """
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_

    def inverse(self, Z: np.ndarray) -> np.ndarray:
        """Undo the standardization.

        Parameters
        ----------
        Z : np.ndarray
            Standardized data matrix of shape ``(m, d)``.

        Returns
        -------
        np.ndarray
            Data in original units.
        """
        return np.asarray(Z, dtype=float) * self.std_ + self.mean_
