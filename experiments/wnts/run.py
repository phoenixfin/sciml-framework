"""WNTS system identification: why standard SINDy fails and SINDYc works.

Fits six models of increasing physical honesty to endpoint data of the
WNTS gas pipeline network and evaluates them with a common protocol
(one-step-fit R^2 on held-out segments + multi-IC, multi-horizon forecast
rollouts, compared against persistence / climatology / daily-repeat
baselines):

- ``sindy_P``     : vanilla autonomous SINDy, states = node pressures.
- ``sindy_Pq``    : vanilla autonomous SINDy, states = pressures and flows.
- ``sindyc``      : SINDYc, states = pressures, control inputs = flows.
- ``sindyc_L``    : SINDYc plus a latent line-pack state (integrated,
  bias-corrected flow imbalance).
- ``sindyc_red``  : SINDYc on a reduced 2-state form -- the four collinear
  source pressures collapse into one upstream pool pressure. This is the
  model that is stable in rollout: the pool pressure acts as the observed
  line-pack state.
- ``sindyc_redL`` : the reduced form plus the latent line-pack state.

Operating point / centering (``--center``):

- ``causal`` (default): every signal is referenced to a *trailing* mean
  (window ``--op-window`` hours, computed from past data only). Forecasts
  are evaluated in a frame frozen at the forecast start, so no future
  information enters the operating point -- honest forecasting numbers.
- ``segment``: reference each clean segment to its own mean (oracle
  operating point; slightly optimistic, useful for identification-quality
  comparisons).

Fitting modes (``--fit``): ``discrete`` (default) fits forward-difference
targets and rolls out with the consistent forward-Euler map; ``savgol``
is classic continuous-time SINDy (Savitzky-Golay derivatives + RK4).

Data splits: chronological by default (test = latest clean segments), or
``--split interleave``; ``--test-years`` trains on ``--years`` and tests
on entirely different contract years.

Usage::

    python -m experiments.wnts.run --data-dir D:/repository/wnts_hourly
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sciml.core.derivatives import savgol, savgol_derivative
from sciml.core.plotting import set_paper_style
from sciml.methods.sindy import PolynomialLibrary, SINDy

from .data import (
    P_COLS,
    Q_COLS,
    SINK,
    SOURCES,
    Scaler,
    block_mean,
    clean_segments,
    linepack_proxy,
    load_wide,
)

P_NAMES = [c.lower() for c in P_COLS]
Q_NAMES = [c.lower() for c in Q_COLS]

HORIZONS_H = [6, 12, 24, 48, 72, 120, 168]

# (Z_state, Z_input or None, Z_state_absolute) -- fluctuation, input and
# absolute (uncentred, same scale) state arrays for one segment.
SegData = Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]


# --------------------------------------------------------------------------
# generic helpers
# --------------------------------------------------------------------------
def r2_per_column(Y: np.ndarray, Yhat: np.ndarray) -> np.ndarray:
    """Coefficient of determination per column."""
    ss_res = np.sum((Y - Yhat) ** 2, axis=0)
    ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2, axis=0)
    return 1.0 - ss_res / np.where(ss_tot == 0, 1.0, ss_tot)


def interp_u(U: np.ndarray, dt: float) -> Callable[[float], np.ndarray]:
    """Linear interpolant of the input matrix ``U`` sampled every ``dt`` h."""
    tt = np.arange(len(U), dtype=float) * dt
    return lambda t: np.array([np.interp(t, tt, U[:, j]) for j in range(U.shape[1])])


def rollout(
    f: Callable[[float, np.ndarray], np.ndarray],
    x0: np.ndarray,
    n_steps: int,
    t0: float,
    dt: float,
    integrator: str,
    blowup: float = 20.0,
) -> Tuple[np.ndarray, bool]:
    """Integrate the identified model with a divergence guard.

    Returns ``(X, diverged)``: trajectory ``(n_steps + 1, d)`` in z-units
    (NaN after a blow-up) and whether the guard tripped.
    """
    x = np.asarray(x0, dtype=float).copy()
    X = np.full((n_steps + 1, len(x)), np.nan)
    X[0] = x
    for k in range(n_steps):
        t = t0 + k * dt
        if integrator == "euler":
            x = x + dt * f(t, x)
        else:
            k1 = f(t, x)
            k2 = f(t + dt / 2, x + dt / 2 * k1)
            k3 = f(t + dt / 2, x + dt / 2 * k2)
            k4 = f(t + dt, x + dt * k3)
            x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > blowup:
            return X, True
        X[k + 1] = x
    return X, False


# --------------------------------------------------------------------------
# model specification
# --------------------------------------------------------------------------
class ModelSpec:
    """One identification experiment: state/input choice + fitted SINDy."""

    def __init__(
        self,
        key: str,
        label: str,
        state_names: List[str],
        input_names: List[str],
        clip: float = 0.0,
        n_pressure: Optional[int] = None,
    ):
        self.key = key
        self.label = label
        self.state_names = state_names
        self.input_names = input_names
        self.clip = clip
        # Number of leading state columns that are pressures (metrics are
        # computed on these; excludes latent states like L).
        self.n_p = (
            n_pressure
            if n_pressure is not None
            else sum(1 for n in state_names if n.startswith("p"))
        )
        self.model: Optional[SINDy] = None

    @property
    def var_names(self) -> List[str]:
        return self.state_names + self.input_names

    def _aug(self, Zs: np.ndarray, Zu: Optional[np.ndarray]) -> np.ndarray:
        aug = Zs if Zu is None else np.hstack([Zs, Zu])
        if self.clip > 0:
            # Saturate library inputs at the training envelope so the
            # polynomial features stay in their calibrated range.
            aug = np.clip(aug, -self.clip, self.clip)
        return aug

    def _pairs(
        self, seg_data: List[SegData], fit_mode: str, dt: float, window: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stacked (features, targets) over segments for the chosen mode.

        Features are the (centred) fluctuation states; targets are
        increments of the *absolute* states, matching what a rollout in a
        frozen operating-point frame accumulates.
        """
        Xs, Ys = [], []
        for Zs, Zu, Zabs in seg_data:
            aug = self._aug(Zs, Zu)
            if fit_mode == "discrete":
                Xs.append(aug[:-1])
                Ys.append((Zabs[1:] - Zabs[:-1]) / dt)
            else:
                t = np.arange(len(Zs), dtype=float) * dt
                Xs.append(aug)
                Ys.append(
                    np.column_stack(
                        [
                            savgol_derivative(Zabs[:, j], t, window=window)
                            for j in range(Zabs.shape[1])
                        ]
                    )
                )
        return np.vstack(Xs), np.vstack(Ys)

    def fit(
        self,
        seg_data: List[SegData],
        fit_mode: str,
        dt: float,
        window: int,
        threshold: float,
        alpha: float,
        degree: int,
        library=None,
    ) -> "ModelSpec":
        X, Y = self._pairs(seg_data, fit_mode, dt, window)
        lib = library if library is not None else PolynomialLibrary(degree=degree)
        self.model = SINDy(lib, threshold=threshold, alpha=alpha).fit(
            X, Y, input_names=self.var_names
        )
        return self

    def r2_pressures(
        self, seg_data: List[SegData], fit_mode: str, dt: float, window: int
    ) -> np.ndarray:
        X, Y = self._pairs(seg_data, fit_mode, dt, window)
        return r2_per_column(Y, self.model.predict(X))[: self.n_p]

    def rhs(self, u_of_t: Optional[Callable[[float], np.ndarray]]):
        if not self.input_names:
            return lambda t, x: self.model.predict(x[None, :])[0]
        return lambda t, x: self.model.predict(np.concatenate([x, u_of_t(t)])[None, :])[0]

    def coef_dict(self) -> Dict[str, Dict[str, float]]:
        """Non-zero identified coefficients per state equation."""
        names = self.model.feature_names_
        coef = self.model.coef_
        return {
            st: {names[i]: float(coef[i, j]) for i in range(len(names)) if abs(coef[i, j]) > 0}
            for j, st in enumerate(self.state_names)
        }


# --------------------------------------------------------------------------
# evaluation protocol
# --------------------------------------------------------------------------
def rollout_metrics(
    spec: ModelSpec,
    seg_data: List[SegData],
    dt: float,
    integrator: str,
    stride_h: int,
    blowup: float,
    warmup_h: float,
    horizons_h: Optional[List[int]] = None,
) -> dict:
    """Multi-IC, multi-horizon forecast rollouts (z-score units).

    Truth is expressed in the frame frozen at each forecast start
    (operating point held at its value at the IC), so causal centering
    stays causal over the horizon. NRMSE is over the pressure states,
    normalized per column by the pooled test fluctuation std.
    """
    horizons_h = horizons_h or HORIZONS_H
    n_p = spec.n_p
    cps = {h: max(1, int(round(h / dt))) for h in horizons_h}
    steps = max(cps.values())
    stride = max(1, int(round(stride_h / dt)))
    warm = int(round(warmup_h / dt))
    pooled = np.vstack([Zs[:, :n_p] for Zs, _, _ in seg_data])
    col_std = pooled.std(axis=0)
    col_std[col_std == 0] = 1.0
    acc = {h: [] for h in horizons_h}
    div = {h: 0 for h in horizons_h}
    n_total = 0
    for Zs, Zu, Zabs in seg_data:
        u_of_t = interp_u(Zu, dt) if Zu is not None and spec.input_names else None
        for k0 in range(warm, len(Zs) - steps - 1, stride):
            n_total += 1
            X, _ = rollout(
                spec.rhs(u_of_t),
                Zs[k0],
                steps,
                t0=k0 * dt,
                dt=dt,
                integrator=integrator,
                blowup=blowup,
            )
            # Truth in the frame frozen at the IC.
            T = Zabs[k0 + 1 : k0 + steps + 1] - Zabs[k0] + Zs[k0]
            for h, s in cps.items():
                pred = X[1 : s + 1, :n_p]
                if np.isnan(pred).any():
                    div[h] += 1
                    continue
                err = pred - T[:s, :n_p]
                acc[h].append(float(np.mean(np.sqrt(np.mean(err**2, axis=0)) / col_std)))
    return {
        "n_rollouts": n_total,
        "horizons": {
            h: {
                "nrmse": float(np.median(acc[h])) if acc[h] else float("inf"),
                "diverged_frac": div[h] / max(n_total, 1),
            }
            for h in horizons_h
        },
    }


def baseline_metrics(
    seg_pairs: List[Tuple[np.ndarray, np.ndarray]],
    dt: float,
    stride_h: int,
    warmup_h: float,
    horizons_h: Optional[List[int]] = None,
) -> dict:
    """Trivial forecast baselines on the rollout protocol.

    ``persistence`` holds the state at the IC; ``climatology`` predicts
    the operating point (zero in the frozen frame); ``daily_repeat``
    repeats the preceding 24 h pattern. ``seg_pairs`` holds
    ``(Z_fluct, Z_abs)`` per test segment.
    """
    horizons_h = horizons_h or HORIZONS_H
    cps = {h: max(1, int(round(h / dt))) for h in horizons_h}
    steps = max(cps.values())
    stride = max(1, int(round(stride_h / dt)))
    warm = max(int(round(warmup_h / dt)), int(round(24 / dt)))
    s24 = max(1, int(round(24 / dt)))
    pooled = np.vstack([Z for Z, _ in seg_pairs])
    col_std = pooled.std(axis=0)
    col_std[col_std == 0] = 1.0
    names = ("persistence", "climatology", "daily_repeat")
    acc = {b: {h: [] for h in horizons_h} for b in names}
    for Zs, Zabs in seg_pairs:
        for k0 in range(warm, len(Zs) - steps - 1, stride):
            T = Zabs[k0 + 1 : k0 + steps + 1] - Zabs[k0] + Zs[k0]
            reps = int(np.ceil(steps / s24))
            prev_day = Zabs[k0 - s24 + 1 : k0 + 1] - Zabs[k0] + Zs[k0]
            preds = {
                "persistence": np.tile(Zs[k0], (steps, 1)),
                "climatology": np.zeros_like(T),
                "daily_repeat": np.vstack([prev_day] * reps)[:steps],
            }
            for b, pred in preds.items():
                err = pred - T
                for h, s in cps.items():
                    acc[b][h].append(
                        float(np.mean(np.sqrt(np.mean(err[:s] ** 2, axis=0)) / col_std))
                    )
    return {
        b: {h: float(np.median(v[h])) if v[h] else float("inf") for h in horizons_h}
        for b, v in acc.items()
    }


# --------------------------------------------------------------------------
# data preparation
# --------------------------------------------------------------------------
def prepare_data(args) -> dict:
    """Load, clean, split, centre and scale the WNTS data.

    Returns a dict with the segment DataFrames, raw per-segment arrays
    (``P, q, CP, Cq, L``), fitted scalers, per-segment z-arrays for the
    standard model specs, and protocol constants.
    """
    dt = float(args.dt_hours)

    def load_segments(years):
        wide = load_wide(args.data_dir, years)
        segs = sorted(
            clean_segments(wide, min_len=args.min_seg_days * 24),
            key=lambda s: s.index[0],
        )
        return [block_mean(s, args.dt_hours) for s in segs]

    if getattr(args, "test_years", None):
        train_segs = load_segments(args.years)
        test_segs = load_segments(args.test_years)
    else:
        segs = load_segments(args.years)
        total = sum(len(s) for s in segs)
        if args.split == "interleave":
            test_segs = segs[3::4]
            train_segs = [s for i, s in enumerate(segs) if (i - 3) % 4 != 0]
        else:
            train_segs = []
            acc = 0
            for s in segs:
                if acc < args.train_frac * total or len(segs) - len(train_segs) <= 1:
                    train_segs.append(s)
                    acc += len(s)
            test_segs = segs[len(train_segs) :]
        if not test_segs:
            train_segs, test_segs = segs[:-1], segs[-1:]
    if not train_segs or not test_segs:
        raise RuntimeError("No usable train/test segments found.")

    # Fuel/shrinkage bias pooled over train segments.
    imb_train = pd.concat(
        [s[[f"q_{n}" for n in SOURCES]].sum(axis=1) - s[f"q_{SINK}"] for s in train_segs]
    )
    bias = float(imb_train.mean())

    w_op = max(1, int(round(args.op_window / dt)))

    def op_point(A: np.ndarray) -> np.ndarray:
        if args.center == "causal":
            return pd.DataFrame(A).rolling(w_op, min_periods=1).mean().to_numpy()
        return np.tile(A.mean(axis=0), (len(A), 1))

    def raw_arrays(seg: pd.DataFrame) -> dict:
        L, _ = linepack_proxy(seg, bias=bias, dt=dt)
        P, q = seg[P_COLS].to_numpy(), seg[Q_COLS].to_numpy()
        return {"P": P, "q": q, "CP": op_point(P), "Cq": op_point(q), "L": L[:, None]}

    def trim(arrs: List[dict]) -> List[dict]:
        # Drop the trailing-window spin-up: the first w_op samples of a
        # segment have a partially-formed (expanding) operating point,
        # which would pollute training features and scalers.
        if args.center != "causal":
            return arrs
        return [
            {k: v[w_op:] for k, v in d.items()} for d in arrs if len(next(iter(d.values()))) > w_op
        ]

    train_raw = trim([raw_arrays(s) for s in train_segs])
    test_raw = trim([raw_arrays(s) for s in test_segs])
    if not train_raw or not test_raw:
        raise RuntimeError("No segments survive the operating-point spin-up trim.")

    def reduce_P(P: np.ndarray) -> np.ndarray:
        """Collapse the 4 collinear source pressures into one upstream
        pool pressure; keep the ORF (delivery) pressure."""
        return np.column_stack([P[:, :4].mean(axis=1), P[:, 4]])

    sc = {
        "P": Scaler().fit(np.vstack([r["P"] - r["CP"] for r in train_raw])),
        "q": Scaler().fit(np.vstack([r["q"] - r["Cq"] for r in train_raw])),
        "L": Scaler().fit(np.vstack([r["L"] for r in train_raw])),
        "P2": Scaler().fit(np.vstack([reduce_P(r["P"] - r["CP"]) for r in train_raw])),
    }

    def smooth(Z: np.ndarray) -> np.ndarray:
        if args.smooth_window <= 1:
            return Z
        return np.column_stack(
            [savgol(Z[:, j], window=args.smooth_window) for j in range(Z.shape[1])]
        )

    def z_arrays(r: dict) -> dict:
        zL = smooth(sc["L"].transform(r["L"]))
        return {
            "P": smooth(sc["P"].transform(r["P"] - r["CP"])),
            "Pabs": smooth(sc["P"].transform(r["P"])),
            "q": smooth(sc["q"].transform(r["q"] - r["Cq"])),
            "qabs": smooth(sc["q"].transform(r["q"])),
            "L": zL,
            "Labs": zL,
            "P2": smooth(sc["P2"].transform(reduce_P(r["P"] - r["CP"]))),
            "P2abs": smooth(sc["P2"].transform(reduce_P(r["P"]))),
        }

    return {
        "dt": dt,
        "integrator": "euler" if args.fit == "discrete" else "rk4",
        "warmup_h": 24.0,
        "bias": bias,
        "train_segs": train_segs,
        "test_segs": test_segs,
        "train_raw": train_raw,
        "test_raw": test_raw,
        "scalers": sc,
        "train_z": [z_arrays(r) for r in train_raw],
        "test_z": [z_arrays(r) for r in test_raw],
    }


def seg_data(zsegs: List[dict], key: str) -> List[SegData]:
    """Assemble (state, input, absolute-state) triples for a model key."""
    out = []
    for z in zsegs:
        if key == "sindy_P":
            out.append((z["P"], None, z["Pabs"]))
        elif key == "sindy_Pq":
            out.append((np.hstack([z["P"], z["q"]]), None, np.hstack([z["Pabs"], z["qabs"]])))
        elif key == "sindyc":
            out.append((z["P"], z["q"], z["Pabs"]))
        elif key == "sindyc_L":
            out.append((np.hstack([z["P"], z["L"]]), z["q"], np.hstack([z["Pabs"], z["Labs"]])))
        elif key == "sindyc_red":
            out.append((z["P2"], z["q"], z["P2abs"]))
        elif key == "sindyc_redL":
            out.append((np.hstack([z["P2"], z["L"]]), z["q"], np.hstack([z["P2abs"], z["Labs"]])))
        else:
            raise KeyError(key)
    return out


def default_specs(clip: float) -> List[ModelSpec]:
    """The standard six-model ladder."""
    return [
        ModelSpec("sindy_P", "SINDy (states: P)", P_NAMES, [], clip=clip),
        ModelSpec("sindy_Pq", "SINDy (states: P, q)", P_NAMES + Q_NAMES, [], clip=clip),
        ModelSpec("sindyc", "SINDYc (states: P; inputs: q)", P_NAMES, Q_NAMES, clip=clip),
        ModelSpec(
            "sindyc_L", "SINDYc+L (states: P, L; inputs: q)", P_NAMES + ["L"], Q_NAMES, clip=clip
        ),
        ModelSpec(
            "sindyc_red",
            "SINDYc-2 (states: p_up, p_orf; inputs: q)",
            ["p_up", "p_orf"],
            Q_NAMES,
            clip=clip,
        ),
        ModelSpec(
            "sindyc_redL",
            "SINDYc-2+L (states: p_up, p_orf, L; inputs: q)",
            ["p_up", "p_orf", "L"],
            Q_NAMES,
            clip=clip,
        ),
    ]


def evaluate_spec(spec: ModelSpec, data: dict, args) -> dict:
    """Fit one spec on the training segments and evaluate everything."""
    d_tr = seg_data(data["train_z"], spec.key)
    d_te = seg_data(data["test_z"], spec.key)
    spec.fit(
        d_tr, args.fit, data["dt"], args.savgol_window, args.threshold, args.alpha, args.degree
    )
    r2_tr = spec.r2_pressures(d_tr, args.fit, data["dt"], args.savgol_window)
    r2_te = spec.r2_pressures(d_te, args.fit, data["dt"], args.savgol_window)
    roll = rollout_metrics(
        spec,
        d_te,
        data["dt"],
        data["integrator"],
        args.ic_stride,
        args.blowup,
        data["warmup_h"],
    )
    hz = roll["horizons"]
    return {
        "label": spec.label,
        "r2_dPdt_train_mean": float(np.mean(r2_tr)),
        "r2_dPdt_test_mean": float(np.mean(r2_te)),
        "r2_dPdt_test_per_state": {
            n: float(v) for n, v in zip(spec.state_names[: spec.n_p], r2_te)
        },
        "n_rollouts": roll["n_rollouts"],
        "diverged_frac": hz[72]["diverged_frac"],
        "nrmse_24h": hz[24]["nrmse"],
        "nrmse_72h": hz[72]["nrmse"],
        "horizons": {str(h): m for h, m in hz.items()},
        "coefficients": spec.coef_dict(),
    }


# --------------------------------------------------------------------------
# experiment driver
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="D:/repository/wnts_hourly")
    ap.add_argument("--years", type=int, nargs="+", default=[2019])
    ap.add_argument(
        "--test-years",
        type=int,
        nargs="+",
        default=None,
        help="train on --years, test on these contract years",
    )
    ap.add_argument("--train-frac", type=float, default=0.75)
    ap.add_argument("--split", choices=["chrono", "interleave"], default="chrono")
    ap.add_argument(
        "--center",
        choices=["causal", "segment"],
        default="causal",
        help="causal: trailing operating point (honest "
        "forecasting); segment: per-segment mean (oracle)",
    )
    ap.add_argument(
        "--op-window",
        type=int,
        default=72,
        help="trailing operating-point window [h] for --center causal",
    )
    ap.add_argument("--fit", choices=["discrete", "savgol"], default="discrete")
    ap.add_argument("--dt-hours", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.02)
    ap.add_argument("--alpha", type=float, default=1e-2)
    ap.add_argument("--degree", type=int, default=2)
    ap.add_argument("--smooth-window", type=int, default=0)
    ap.add_argument("--savgol-window", type=int, default=9)
    ap.add_argument("--min-seg-days", type=int, default=10)
    ap.add_argument(
        "--horizon", type=int, default=72, help="horizon of the example-forecast figure"
    )
    ap.add_argument("--ic-stride", type=int, default=48)
    ap.add_argument("--clip", type=float, default=3.0)
    ap.add_argument("--blowup", type=float, default=20.0)
    ap.add_argument("--out", default="outputs/wnts")
    ap.add_argument("--no-figures", action="store_true")
    return ap


def run_experiment(args) -> dict:
    """Run the full ladder + baselines; write outputs; return the summary."""
    os.makedirs(args.out, exist_ok=True)
    data = prepare_data(args)
    dt = data["dt"]

    print(
        f"Segments: train {len(data['train_segs'])} "
        f"({sum(len(s) for s in data['train_segs']) * dt / 24:.0f} d), "
        f"test {len(data['test_segs'])} "
        f"({sum(len(s) for s in data['test_segs']) * dt / 24:.0f} d) "
        f"@ dt={args.dt_hours}h, center={args.center}"
    )
    print(f"Train: {data['train_segs'][0].index[0]} -> {data['train_segs'][-1].index[-1]}")
    print(f"Test : {data['test_segs'][0].index[0]} -> {data['test_segs'][-1].index[-1]}")
    print(f"Line-pack bias (fuel/shrinkage estimate): {data['bias']:.2f}")

    base = baseline_metrics(
        [(z["P2"], z["P2abs"]) for z in data["test_z"]], dt, args.ic_stride, data["warmup_h"]
    )
    print("\n[trivial baselines on (p_up, p_orf)]")
    for name, m in base.items():
        print(f"  {name:13s} NRMSE 24h {m[24]:.3f}, 72h {m[72]:.3f}, 168h {m[168]:.3f}")

    specs = default_specs(args.clip)
    summary, eq_lines = {}, []
    for spec in specs:
        entry = evaluate_spec(spec, data, args)
        summary[spec.key] = entry
        eq_lines.append(f"### {spec.label}")
        eq_lines += spec.model.equations([f"d/dt {n}" for n in spec.state_names])
        eq_lines.append("")
        print(f"\n[{spec.label}]")
        print(
            f"  dP/dt fit R^2  train {entry['r2_dPdt_train_mean']:+.3f}   "
            f"test {entry['r2_dPdt_test_mean']:+.3f}"
        )
        print(
            f"  rollouts ({entry['n_rollouts']}): "
            f"diverged {entry['diverged_frac']:.0%}, "
            f"NRMSE 24h {entry['nrmse_24h']:.3f}, 72h {entry['nrmse_72h']:.3f}"
        )

    with open(os.path.join(args.out, "equations.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(eq_lines))
    result = {
        "args": vars(args),
        "train_start": str(data["train_segs"][0].index[0]),
        "train_end": str(data["train_segs"][-1].index[-1]),
        "test_start": str(data["test_segs"][0].index[0]),
        "test_end": str(data["test_segs"][-1].index[-1]),
        "linepack_bias": data["bias"],
        "baselines": {b: {str(h): v for h, v in m.items()} for b, m in base.items()},
        "models": summary,
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    if not args.no_figures:
        make_figures(specs, data, base, summary, args)
    print(f"\nSaved equations, summary and figures to {args.out}")
    return result


def make_figures(specs, data, base, summary, args) -> None:
    """Example-forecast, R^2 bar and horizon-skill figures."""
    set_paper_style(font_size=10)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dt = data["dt"]
    sc = data["scalers"]

    # -- example forecast on the largest test segment ---------------------
    i_fig = max(range(len(data["test_segs"])), key=lambda i: len(data["test_segs"][i]))
    ztest = data["test_z"][i_fig]
    steps = max(1, int(round(args.horizon / dt)))
    k0 = int(round(data["warmup_h"] / dt))
    hours = np.arange(steps + 1) * dt
    # per-spec: (state idx of plotted pressure, scaler key, scaler col)
    views = {
        "orf": {
            k: (P_COLS.index("P_orf"), "P", P_COLS.index("P_orf"))
            for k in ("sindy_P", "sindy_Pq", "sindyc", "sindyc_L")
        },
        "up": {k: (0, "P", 0) for k in ("sindy_P", "sindy_Pq", "sindyc", "sindyc_L")},
    }
    for k in ("sindyc_red", "sindyc_redL"):
        views["orf"][k] = (1, "P2", 1)
        views["up"][k] = (0, "P2", 0)
    styles = ["C3--", "C1-.", "C4:", "C0-", "C5--", "C2-"]
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True)
    for ax, vkey, (abs_key, i_true), ttl in [
        (axes[0], "orf", ("Pabs", P_COLS.index("P_orf")), "ORF (sink)"),
        (axes[1], "up", ("Pabs", 0), "Anoa (source)"),
    ]:
        truth = (
            ztest[abs_key][k0 : k0 + steps + 1, i_true] * sc["P"].std_[i_true]
            + sc["P"].mean_[i_true]
        )
        ax.plot(hours, truth, "k-", lw=1.6, label="measured")
        for spec, style in zip(specs, styles):
            Zs, Zu, Zabs = seg_data([ztest], spec.key)[0]
            u_of_t = interp_u(Zu, dt) if Zu is not None and spec.input_names else None
            X, div = rollout(
                spec.rhs(u_of_t),
                Zs[k0],
                steps,
                t0=k0 * dt,
                dt=dt,
                integrator=data["integrator"],
                blowup=args.blowup,
            )
            i_state, sck, j = views[vkey][spec.key]
            zabs_traj = X[:, i_state] - Zs[k0, i_state] + Zabs[k0, i_state]
            traj = zabs_traj * sc[sck].std_[j] + sc[sck].mean_[j]
            lbl = spec.label + (" (diverged)" if div else "")
            ax.plot(hours, traj, style, lw=1.1, label=lbl)
        ax.set_ylabel(f"P {ttl} [psig]")
        pad = 6 * np.std(truth)
        ax.set_ylim(truth.min() - pad, truth.max() + pad)
    axes[0].legend(fontsize=6.5, ncol=2)
    axes[1].set_xlabel("forecast horizon [h]")
    fig.suptitle("WNTS test-segment forecast: autonomous SINDy vs SINDYc")
    plt.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_rollout.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # -- R^2 bar chart -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 3.2))
    xs = np.arange(len(specs))
    ax.bar(xs - 0.18, [summary[s.key]["r2_dPdt_train_mean"] for s in specs], 0.36, label="train")
    ax.bar(xs + 0.18, [summary[s.key]["r2_dPdt_test_mean"] for s in specs], 0.36, label="test")
    ax.set_xticks(xs)
    ax.set_xticklabels([s.key for s in specs], fontsize=8)
    ax.set_ylabel(r"mean $R^2$ of $\dot{P}$ fit")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylim(min(-0.5, ax.get_ylim()[0]), 1.0)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_r2.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # -- horizon-skill curves (A4) ------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for spec, style in zip(specs, styles):
        hz = summary[spec.key]["horizons"]
        ys = [
            hz[str(h)]["nrmse"] if np.isfinite(hz[str(h)]["nrmse"]) else np.nan for h in HORIZONS_H
        ]
        ax.plot(HORIZONS_H, ys, style, marker="o", ms=3, lw=1.2, label=spec.label)
    for bname, bstyle in [("persistence", "k:"), ("climatology", "k--"), ("daily_repeat", "k-.")]:
        ax.plot(HORIZONS_H, [base[bname][h] for h in HORIZONS_H], bstyle, lw=1.0, label=bname)
    ax.set_xlabel("forecast horizon [h]")
    ax.set_ylabel("median NRMSE (pressures)")
    ax.set_xscale("log")
    ax.set_xticks(HORIZONS_H)
    ax.set_xticklabels([str(h) for h in HORIZONS_H])
    ax.set_ylim(0, 2.0)
    ax.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_horizon.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    run_experiment(build_parser().parse_args())


if __name__ == "__main__":
    main()
