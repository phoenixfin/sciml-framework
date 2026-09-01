#!/usr/bin/env python3
"""Reprocess the financial-distress panel into a clean, SINDy-ready dataset.

The panel is Indonesian listed firms, 2013-2023, one row per firm-year. The
script loads the full-precision source workbook, repairs the four defects
documented in ``DATA_REPORT.md`` (a duplicated firm block, a desynchronised
corruption-index column, impossible proportions, negative cash ratios), derives
the boundary variable the study is built around, and writes both the raw-unit
panel and a transformed, standardised copy for SINDy.

Input
-----
``datadrtpm0Rev1.xlsx`` (sheet ``"2013-2023"``) -- the full-precision source.
Override with ``argv[1]`` or the ``FINDIST_SRC`` environment variable.

Output (into ``argv[2]`` / ``FINDIST_OUT``, default: beside this script)
------
``panel_clean.csv``
    Cleaned panel in raw units, with quality flags.
``panel_sindy_ready.csv``
    asinh-transformed and standardised states.
``quality_log.csv``
    One row per repair, fully auditable.
``datadrtpm_clean.xlsx``
    All of the above in one workbook, plus the transform metadata.

Every repair is (a) recorded in ``quality_log.csv`` and (b) reversible from the
source file. Nothing is silently overwritten.

Author: prepared for A. F. Ihsan
"""

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Usage: python prepare_sindy_data.py [source.xlsx] [outdir]
# Falls back to the FINDIST_SRC / FINDIST_OUT environment variables, then to the
# source sitting next to this file and writing the outputs alongside it.
_HERE = Path(__file__).resolve().parent
SRC = Path(sys.argv[1] if len(sys.argv) > 1
           else os.environ.get("FINDIST_SRC", _HERE / "datadrtpm0Rev1.xlsx"))
OUTDIR = Path(sys.argv[2] if len(sys.argv) > 2
              else os.environ.get("FINDIST_OUT", _HERE))
if not SRC.is_file():
    sys.exit(f"source workbook not found: {SRC}\n"
             f"pass it as the first argument or set FINDIST_SRC.")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- constants

# Transparency International CPI score for Indonesia, 0-100 scale.
# Source: https://www.transparency.org/en/countries/indonesia
# Cross-check: 2019 = 40 is Indonesia's all-time high; 2022 = 34 was the
# steepest single-year fall since 1995; 2023 = 34 (unchanged).
CPI_INDONESIA = {
    2013: 32, 2014: 34, 2015: 36, 2016: 37, 2017: 37, 2018: 38,
    2019: 40, 2020: 37, 2021: 38, 2022: 34, 2023: 34,
}

# Short machine-friendly names, in source order.
RENAME = {
    "Y Financial Distress ": "Y_distress",
    "X1 Gross Profit": "X1_gross_profit",
    "X2 Net Profit Margin ": "X2_net_profit_margin",
    "X3 Current Ratio ": "X3_current_ratio",
    "X4 Quick ratio": "X4_quick_ratio",
    "X5 Cash Ratio": "X5_cash_ratio",
    "X6 Perputaran piutang ": "X6_receivable_turnover",
    "X7 Perputaran Persedian ": "X7_inventory_turnover",
    "X8 Leverage Aset": "X8_leverage_asset",
    "X9 Leverage Equiy": "X9_leverage_equity",
    "X10 Debt Service Coverage Ratio": "X10_dscr",
    "X11 Aset Coverge Ratio": "X11_asset_coverage",
    "X12 MBV": "X12_mbv",
    "X13 Independent Commissioner": "X13_indep_commissioner",
    "X14 Board Diversity ": "X14_board_diversity",
    "X15 Corruption Index": "X15_corruption_index",
}

# Variables that enter the SINDy state vector.
# X13/X14 are excluded: within-firm lag-1 autocorrelation is approx -0.1 and they
#   are near-constant with rare jumps -> parameters, not states.
# X15 is excluded: identical across all firms in a year -> perfectly collinear
#   with a year fixed effect; use it as an exogenous input u(t) in SINDyc only,
#   with that caveat stated.
# X9 is excluded and replaced by IEQ = 1/X9 (see below).
STATE_VARS = [
    "X1_gross_profit", "X2_net_profit_margin", "X3_current_ratio",
    "X4_quick_ratio", "X5_cash_ratio", "X6_receivable_turnover",
    "X7_inventory_turnover", "X8_leverage_asset", "IEQ_equity_to_liability",
    "X10_dscr", "X11_asset_coverage", "X12_mbv",
]

log_rows = []


def log(code: str, share: str, year: Any, column: str,
        old: Any, new: Any, reason: str) -> None:
    """Record one repair in the quality log.

    Parameters
    ----------
    code : str
        Repair code, e.g. ``"X15_DESYNC"``; groups repairs of the same kind.
    share : str
        Share code of the firm affected, or a marker such as ``"(all firms)"``.
    year : Any
        Firm-year affected (int), or a range string for panel-wide repairs.
    column : str
        Column repaired, or ``"(entire row)"`` for a dropped row.
    old : Any
        Value (or description of the state) before the repair.
    new : Any
        Value written in its place.
    reason : str
        Why the original value cannot be right, and what would be needed to
        recover it from the source documents.
    """
    log_rows.append(dict(repair_code=code, share_code=share, year=year,
                         column=column, old_value=old, new_value=new,
                         reason=reason))


# ---------------------------------------------------------------- 1. load

df = pd.read_excel(SRC, sheet_name="2013-2023")
df = df.rename(columns=RENAME)
df["SHARE_CODE"] = df["SHARE CODE"]
df = df.drop(columns=["SHARE CODE"])
print(f"[1] loaded {df.shape[0]} rows x {df.shape[1]} cols")


# ------------------------------------------- 2. drop duplicated PYFA block

# PYFA appears as TWO complete 11-year blocks (NO 5737-5747 and NO 5761-5771).
# All financial ratios agree exactly; the blocks disagree on governance
# variables in 5 firm-years. There is no internal evidence favouring either,
# so we keep the lower-NO block and flag it for source verification.
dup_block = df[(df.SHARE_CODE == "PYFA") & (df.NO >= 5761)]
for _, r in dup_block.iterrows():
    log("DUP_FIRM_BLOCK", "PYFA", r.YEAR, "(entire row)", f"NO={r.NO}", "dropped",
        "PYFA entered twice as two full 11-year blocks; kept NO 5737-5747. "
        "Governance values disagree in 5 firm-years - verify against annual reports.")
df = df[~df.index.isin(dup_block.index)].copy()
print(f"[2] dropped duplicated PYFA block -> {df.shape[0]} rows, "
      f"{df.SHARE_CODE.nunique()} firms")


# ------------------------------------------------ 3. rebuild X15 from YEAR

# The source X15 column contains SIX distinct values in EVERY year, yet the
# CPI is a single national figure per year. The multiset of values across the
# panel is exactly {32:1, 34:3, 36:1, 37:3, 38:2, 40:1} per firm -- precisely
# the 2013-2023 Indonesian CPI series. The column is therefore the correct
# value set scrambled across rows: a classic spreadsheet re-sort that moved
# the firm rows but not the CPI column. Rebuild it deterministically.
n_wrong = int((df.X15_corruption_index != df.YEAR.map(CPI_INDONESIA)).sum())
old15 = df.X15_corruption_index.copy()
df["X15_corruption_index"] = df.YEAR.map(CPI_INDONESIA).astype(int)
log("X15_DESYNC", "(all firms)", "2013-2023", "X15_corruption_index",
    f"{n_wrong} of {len(df)} rows mismatched", "rebuilt from YEAR",
    "CPI is a single national value per year but source had 6 distinct values "
    "per year. Value multiset matched the true series exactly -> row/column "
    "desynchronisation. Rebuilt from Transparency International.")
print(f"[3] rebuilt X15: {n_wrong}/{len(df)} rows ({100*n_wrong/len(df):.1f}%) "
      f"were misassigned")


# ------------------------------------------- 4. null impossible proportions

# X13 = independent commissioners / total commissioners  -> must lie in [0, 1]
# X14 = female directors / total directors               -> must lie in [0, 1]
for col in ["X13_indep_commissioner", "X14_board_diversity"]:
    bad = df[df[col] > 1]
    for _, r in bad.iterrows():
        log("PROP_GT_1", r.SHARE_CODE, r.YEAR, col, r[col], "NaN",
            "Proportion exceeds 1 - value looks like an undivided head count. "
            "Not recoverable from this file; recompute from the annual report.")
    df.loc[df[col] > 1, col] = np.nan

# X13 also carried two Excel #DIV/0! cells (total commissioners recorded as 0),
# which pandas already read as NaN. Record them so they are not mistaken for
# genuinely missing data.
for _, r in df[df.X13_indep_commissioner.isna()].iterrows():
    if not any(l["share_code"] == r.SHARE_CODE and l["year"] == r.YEAR
               and l["column"] == "X13_indep_commissioner" for l in log_rows):
        log("DIV_BY_ZERO", r.SHARE_CODE, r.YEAR, "X13_indep_commissioner",
            "#DIV/0!", "NaN",
            "Zero denominator: total commissioners recorded as 0. "
            "Data-entry gap, recoverable from the annual report.")

# X5 = cash ratio -> cash cannot be negative
bad = df[df.X5_cash_ratio < 0]
for _, r in bad.iterrows():
    log("NEGATIVE_CASH", r.SHARE_CODE, r.YEAR, "X5_cash_ratio",
        r.X5_cash_ratio, "NaN",
        "Cash ratio is negative, which is impossible unless 'cash' is net of "
        "bank overdrafts. Nulled; interpolated in the SINDy file.")
df.loc[df.X5_cash_ratio < 0, "X5_cash_ratio"] = np.nan
print(f"[4] nulled {len(log_rows) - 1 - len(dup_block)} impossible values")


# ------------------------------------------ 5. derive the boundary variable

# X9 = Total Liability / Equity has a POLE at Equity = 0, and Y is defined as
# Equity < 0 for two consecutive years. The pole sits exactly where the event
# of interest happens, and no polynomial library can represent a pole.
#
# The reciprocal IEQ = Equity / Total Liability is bounded near the event,
# passes SMOOTHLY through zero, and its sign IS the distress condition.
# This turns "predict a rare binary label" into "find when a smooth state
# crosses zero" -- a first-passage / free-boundary problem.
#
# X9 is recorded in percent (median 43.65 -> TL/E = 0.4365), so:
lev = df.X9_leverage_equity / 100.0                    # TL / E, fraction
with np.errstate(divide="ignore", invalid="ignore"):
    df["IEQ_equity_to_liability"] = np.where(lev != 0, 1.0 / lev, np.nan)

zero_lev = df[df.X9_leverage_equity == 0]
for _, r in zero_lev.iterrows():
    log("ZERO_LIABILITY", r.SHARE_CODE, r.YEAR, "IEQ_equity_to_liability",
        "X9 = 0", "NaN",
        "X9 = TL/E = 0 means zero total liabilities, so E/TL is undefined "
        "(debt-free firm). Interpolated in the SINDy file.")

# Explicit boundary-crossing bookkeeping.
df = df.sort_values(["SHARE_CODE", "YEAR"]).reset_index(drop=True)
df["equity_negative"] = (df.IEQ_equity_to_liability < 0).astype("Int64")
df["equity_negative_2yr"] = (
    df.groupby("SHARE_CODE")["equity_negative"]
      .transform(lambda s: s.rolling(2).min()).astype("Float64")
)
agree = (df.equity_negative_2yr == df.Y_distress).mean()
print(f"[5] derived IEQ; 'equity negative 2 yrs' reproduces Y in "
      f"{100*agree:.1f}% of rows")

# Percent -> fraction copies for the four columns recorded in percent,
# for interpretability only (SINDy standardises anyway).
for col, frac in [("X1_gross_profit", "X1_gross_profit_frac"),
                  ("X2_net_profit_margin", "X2_net_profit_margin_frac"),
                  ("X8_leverage_asset", "X8_leverage_asset_frac"),
                  ("X9_leverage_equity", "X9_leverage_equity_frac")]:
    df[frac] = df[col] / 100.0


# ---------------------------------------------------- 6. outlier flagging

# Flag but DO NOT remove: the extreme values are real firms in real trouble,
# and deleting them would delete the phenomenon. The asinh transform in the
# SINDy file handles the tails without discarding anything.
flag_cols = [c for c in STATE_VARS if c in df.columns]
for c in flag_cols:
    lo, hi = df[c].quantile([0.005, 0.995])
    df[f"flag_extreme_{c}"] = ((df[c] < lo) | (df[c] > hi)).astype(int)
df["n_extreme_flags"] = df[[f"flag_extreme_{c}" for c in flag_cols]].sum(axis=1)
print(f"[6] flagged extremes (0.5%/99.5%); "
      f"{(df.n_extreme_flags > 0).sum()} rows carry >=1 flag")


# ------------------------------------------------- 7. panel sanity checks

assert df.duplicated(["SHARE_CODE", "YEAR"]).sum() == 0, "duplicate firm-years"
counts = df.groupby("SHARE_CODE").size()
assert (counts == 11).all(), "panel is not balanced"
assert df.groupby("SHARE_CODE").YEAR.apply(
    lambda s: (s.sort_values().diff().dropna() == 1).all()).all(), "year gaps"
print(f"[7] panel OK: {counts.shape[0]} firms x 11 years, balanced, no gaps")


# --------------------------------------------- 8. build the SINDy-ready set

sindy = df[["SHARE_CODE", "YEAR", "Y_distress", "equity_negative",
            "equity_negative_2yr"] + STATE_VARS].copy()

# Impute the handful of nulled cells within firm so trajectories stay
# contiguous (PySINDy cannot accept NaN). Count is tiny and fully logged.
n_before = int(sindy[STATE_VARS].isna().sum().sum())
sindy[STATE_VARS] = (
    sindy.groupby("SHARE_CODE")[STATE_VARS]
         .transform(lambda g: g.interpolate(limit_direction="both"))
)
sindy[STATE_VARS] = sindy[STATE_VARS].fillna(sindy[STATE_VARS].median())
print(f"[8] imputed {n_before} cells by within-firm interpolation")

# asinh transform: smooth, odd, sign-preserving, logarithmic in the tails and
# linear near zero. Unlike log it needs no shift and keeps the sign of the
# boundary variable, so IEQ = 0 (the distress boundary) maps to exactly 0.
# Scale by a robust spread so the linear region covers the bulk of the data.
scales = {}
for c in STATE_VARS:
    s = (sindy[c].quantile(0.75) - sindy[c].quantile(0.25)) / 1.349  # robust sd
    s = s if s > 0 else sindy[c].std()
    scales[c] = float(s)
    sindy[c + "_t"] = np.arcsinh(sindy[c] / s)

# Standardise the transformed states (SINDy thresholds are scale-dependent).
norm = {}
for c in STATE_VARS:
    m, sd = sindy[c + "_t"].mean(), sindy[c + "_t"].std()
    norm[c] = (float(m), float(sd))
    sindy[c + "_z"] = (sindy[c + "_t"] - m) / sd

sindy = sindy.sort_values(["SHARE_CODE", "YEAR"]).reset_index(drop=True)
sindy["X15_corruption_index"] = sindy.YEAR.map(CPI_INDONESIA)   # exogenous u(t)
sindy["X13_indep_commissioner"] = df["X13_indep_commissioner"].values  # parameter
sindy["X14_board_diversity"] = df["X14_board_diversity"].values        # parameter
sindy["ever_distressed"] = sindy.groupby("SHARE_CODE")["Y_distress"].transform("max")

transform_meta = pd.DataFrame([
    dict(variable=c, asinh_scale=scales[c], z_mean=norm[c][0], z_std=norm[c][1])
    for c in STATE_VARS
])


# ------------------------------------------------------------- 9. write out

qlog = pd.DataFrame(log_rows)

df.to_csv(OUTDIR / "panel_clean.csv", index=False)
sindy.to_csv(OUTDIR / "panel_sindy_ready.csv", index=False)
qlog.to_csv(OUTDIR / "quality_log.csv", index=False)

readme = pd.DataFrame({
    "item": [
        "source file", "rows in", "rows out", "firms", "years",
        "distress observations", "ever-distressed firms",
        "PYFA duplicate block", "X15 rows rebuilt",
        "impossible proportions nulled", "negative cash ratios nulled",
        "cells imputed for SINDy", "state variables",
        "excluded from state", "boundary variable",
        "recommended SINDy mode",
    ],
    "value": [
        "datadrtpm0Rev1.xlsx (sheet 2013-2023)", 3575, len(df),
        df.SHARE_CODE.nunique(), "2013-2023",
        int(df.Y_distress.sum()), int((df.groupby('SHARE_CODE').Y_distress.max() == 1).sum()),
        "dropped NO 5761-5771 (verify governance vs annual reports)",
        f"{n_wrong} of {len(df)}",
        int(qlog.repair_code.eq('PROP_GT_1').sum()),
        int(qlog.repair_code.eq('NEGATIVE_CASH').sum()),
        n_before, ", ".join(STATE_VARS),
        "X9 (pole), X13, X14 (quasi-static -> parameters), X15 (collinear with year FE)",
        "IEQ = Equity/Total Liability; distress = zero crossing",
        "discrete-time map x_{t+1} = Theta(x_t) Xi, multiple_trajectories=True",
    ],
})

with pd.ExcelWriter(OUTDIR / "datadrtpm_clean.xlsx", engine="openpyxl") as xw:
    readme.to_excel(xw, sheet_name="README", index=False)
    df.to_excel(xw, sheet_name="panel_clean", index=False)
    sindy.to_excel(xw, sheet_name="sindy_ready", index=False)
    qlog.to_excel(xw, sheet_name="quality_log", index=False)
    transform_meta.to_excel(xw, sheet_name="transform_meta", index=False)
    pd.DataFrame({"YEAR": list(CPI_INDONESIA), "CPI": list(CPI_INDONESIA.values())}
                 ).to_excel(xw, sheet_name="cpi_reference", index=False)

print(f"[9] wrote panel_clean.csv, panel_sindy_ready.csv, quality_log.csv, "
      f"datadrtpm_clean.xlsx  ({len(qlog)} repairs logged)")
