"""System identification on segmented time series: one call, full protocol.

Generalizes the evaluation pipeline developed for the WNTS gas-network
study (``experiments/wnts``) to any :class:`~sciml.data.datasets.TimeSeriesData`:

1. choose state and input channels by name;
2. reference every signal to an operating point (``causal`` trailing mean
   by default -- no future information -- or per-segment ``segment`` mean);
3. z-score with training statistics and saturate library inputs at the
   training envelope;
4. fit discrete-time SINDy / SINDYc / DMDc (forward-difference targets,
   consistent forward-Euler rollouts) or classic Savitzky-Golay SINDy;
5. evaluate one-step R^2 and multi-IC, multi-horizon forecast rollouts on
   held-out segments, against persistence / climatology / daily-repeat
   baselines.

Example::

    from sciml.data.datasets import load
    from sciml.tasks import sysid

    data = load("wnts", years=[2019])
    res = sysid.run(data, states=["P_up", "P_orf"],
                    inputs=[c for c in data.channels if c.startswith("q_")])
    print(res.equations, res.metrics["horizons"][24])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.derivatives import savgol_derivative
from ..data.datasets.base import TimeSeriesData
from ..methods.sindy import FeatureLibrary, PolynomialLibrary, SINDy

#: (state fluctuations, input fluctuations or None, absolute states) per segment.
SegData = Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]

#: Estimator presets: method name -> (degree, threshold, alpha).
PRESETS: Dict[str, Tuple[int, float, float]] = {
    "sindy": (2, 0.02, 1e-2),
    "sindyc": (2, 0.02, 1e-2),
    "dmdc": (1, 0.0, 0.0),
}


# --------------------------------------------------------------------------
# small numeric helpers
# --------------------------------------------------------------------------
class Scaler:
    """Column-wise z-score scaler fitted on training data."""

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


def trailing_mean(A: np.ndarray, window: int) -> np.ndarray:
    """Causal trailing mean per column (expanding over the first samples).

    Parameters
    ----------
    A : np.ndarray
        Data matrix of shape ``(n, d)``.
    window : int
        Trailing-window length in samples.

    Returns
    -------
    np.ndarray
        Row ``i`` holds the mean of rows ``max(0, i - window + 1) .. i``.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    n = len(A)
    padded = np.vstack([np.zeros((1, A.shape[1])), np.cumsum(A, axis=0)])
    hi = np.arange(1, n + 1)
    lo = np.maximum(0, hi - window)
    return (padded[hi] - padded[lo]) / (hi - lo)[:, None]


def r2_per_column(Y: np.ndarray, Yhat: np.ndarray) -> np.ndarray:
    """Coefficient of determination per column.

    Parameters
    ----------
    Y : np.ndarray
        True values ``(m, k)``.
    Yhat : np.ndarray
        Predicted values ``(m, k)``.

    Returns
    -------
    np.ndarray
        One R^2 value per column.
    """
    ss_res = np.sum((Y - Yhat) ** 2, axis=0)
    ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2, axis=0)
    return 1.0 - ss_res / np.where(ss_tot == 0, 1.0, ss_tot)


def _interp_u(U: np.ndarray, dt: float) -> Callable[[float], np.ndarray]:
    """Linear interpolant of an input matrix sampled every ``dt`` hours.

    Parameters
    ----------
    U : np.ndarray
        Input matrix of shape ``(n, m)``.
    dt : float
        Sample spacing in hours.

    Returns
    -------
    Callable[[float], np.ndarray]
        Function mapping time [h] to the interpolated input row.
    """
    tt = np.arange(len(U), dtype=float) * dt
    return lambda t: np.array([np.interp(t, tt, U[:, j]) for j in range(U.shape[1])])


def rollout(f: Callable[[float, np.ndarray], np.ndarray], x0: np.ndarray, n_steps: int,
            t0: float, dt: float, integrator: str = "euler",
            blowup: float = 20.0) -> Tuple[np.ndarray, bool]:
    """Integrate an identified model with a divergence guard (z-units).

    Parameters
    ----------
    f : Callable[[float, np.ndarray], np.ndarray]
        Right-hand side ``f(t, x)``.
    x0 : np.ndarray
        Initial state.
    n_steps : int
        Number of steps of size ``dt``.
    t0 : float
        Start time in hours.
    dt : float
        Step size in hours.
    integrator : str
        ``"euler"`` (consistent with discrete-time fits) or ``"rk4"``.
    blowup : float
        Guard on ``max |x|``.

    Returns
    -------
    Tuple[np.ndarray, bool]
        Trajectory ``(n_steps + 1, d)`` (NaN after a blow-up) and whether
        the guard tripped.
    """
    x = np.asarray(x0, dtype=float).copy()
    X = np.full((n_steps + 1, len(x)), np.nan)
    X[0] = x
    for k in range(n_steps):
        t = t0 + k * dt
        if integrator == "euler":
            x = x + dt * f(t, x)
        else:
            k1 = f(t, x); k2 = f(t + dt / 2, x + dt / 2 * k1)
            k3 = f(t + dt / 2, x + dt / 2 * k2); k4 = f(t + dt, x + dt * k3)
            x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > blowup:
            return X, True
        X[k + 1] = x
    return X, False


# --------------------------------------------------------------------------
# result container
# --------------------------------------------------------------------------
@dataclass
class SysIdResult:
    """Everything produced by one :func:`run` call."""

    model: SINDy                                   #: the fitted sparse estimator
    states: List[str]                              #: state channel names
    inputs: List[str]                              #: input channel names (may be empty)
    equations: List[str]                           #: human-readable identified equations
    coefficients: Dict[str, Dict[str, float]]      #: non-zero terms per state equation
    r2_train: Dict[str, float]                     #: one-step-fit R^2 per state (train)
    r2_test: Dict[str, float]                      #: one-step-fit R^2 per state (test)
    metrics: Dict[str, Any]                        #: rollout metrics (per horizon)
    baselines: Dict[str, Dict[int, float]]         #: trivial-baseline NRMSE per horizon
    details: Dict[str, Any] = field(default_factory=dict)  #: split/protocol information

    def summary(self) -> str:
        """Compact multi-line text summary of the result.

        Returns
        -------
        str
            Human-readable summary (fit quality, rollouts, baselines).
        """
        hz = self.metrics["horizons"]
        lines = [f"states={self.states} inputs={self.inputs}",
                 f"one-step R^2: train {np.mean(list(self.r2_train.values())):+.3f}, "
                 f"test {np.mean(list(self.r2_test.values())):+.3f}",
                 f"rollouts ({self.metrics['n_rollouts']}):"]
        for h, m in hz.items():
            best = min(v[h] for v in self.baselines.values()) if self.baselines else float("nan")
            lines.append(f"  {h:>4d} h: NRMSE {m['nrmse']:.3f} "
                         f"(best baseline {best:.3f}), diverged {m['diverged_frac']:.0%}")
        lines += ["equations:"] + [f"  {e}" for e in self.equations]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# pipeline internals
# --------------------------------------------------------------------------
def _clipped(aug: np.ndarray, clip: float) -> np.ndarray:
    """Saturate library inputs at the training envelope.

    Parameters
    ----------
    aug : np.ndarray
        Augmented (state + input) matrix in z-units.
    clip : float
        Saturation bound in train stds (0 disables).

    Returns
    -------
    np.ndarray
        The (possibly) clipped matrix.
    """
    return np.clip(aug, -clip, clip) if clip > 0 else aug


def _pairs(seg_data: List[SegData], fit: str, dt: float, clip: float,
           savgol_window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Stacked (features, targets) over segments for the chosen fit mode.

    Parameters
    ----------
    seg_data : List[SegData]
        Per-segment (states, inputs, absolute-states) triples.
    fit : str
        ``"discrete"`` (forward differences) or ``"savgol"`` (smoothed
        derivatives).
    dt : float
        Sample spacing in hours.
    clip : float
        Library-input saturation bound.
    savgol_window : int
        Savitzky-Golay window for ``fit="savgol"``.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Stacked feature and target matrices.
    """
    Xs, Ys = [], []
    for Zs, Zu, Zabs in seg_data:
        aug = _clipped(Zs if Zu is None else np.hstack([Zs, Zu]), clip)
        if fit == "discrete":
            Xs.append(aug[:-1])
            Ys.append((Zabs[1:] - Zabs[:-1]) / dt)
        else:
            t = np.arange(len(Zs), dtype=float) * dt
            Xs.append(aug)
            Ys.append(np.column_stack(
                [savgol_derivative(Zabs[:, j], t, window=savgol_window)
                 for j in range(Zabs.shape[1])]))
    return np.vstack(Xs), np.vstack(Ys)


def _rollout_metrics(model: SINDy, seg_data: List[SegData], dt: float, integrator: str,
                     horizons: Sequence[int], ic_stride_h: float, warmup_h: float,
                     clip: float, blowup: float) -> Dict[str, Any]:
    """Multi-IC, multi-horizon forecast rollouts on the test segments.

    Truth is expressed in the frame frozen at each forecast start, so
    causal centering stays causal over the horizon. NRMSE is normalized
    per state by the pooled test fluctuation std.

    Parameters
    ----------
    model : SINDy
        The fitted estimator.
    seg_data : List[SegData]
        Test-segment triples.
    dt : float
        Sample spacing in hours.
    integrator : str
        ``"euler"`` or ``"rk4"``.
    horizons : Sequence[int]
        Forecast horizons in hours.
    ic_stride_h : float
        Spacing between rollout initial conditions in hours.
    warmup_h : float
        Hours skipped at each segment start before the first IC.
    clip : float
        Library-input saturation bound.
    blowup : float
        Divergence guard in z-units.

    Returns
    -------
    Dict[str, Any]
        ``{"n_rollouts": int, "horizons": {h: {"nrmse", "diverged_frac"}}}``.
    """
    cps = {int(h): max(1, int(round(h / dt))) for h in horizons}
    steps = max(cps.values())
    stride = max(1, int(round(ic_stride_h / dt)))
    warm = int(round(warmup_h / dt))
    pooled = np.vstack([Zs for Zs, _, _ in seg_data])
    col_std = pooled.std(axis=0)
    col_std[col_std == 0] = 1.0
    acc = {h: [] for h in cps}
    div = {h: 0 for h in cps}
    n_total = 0
    for Zs, Zu, Zabs in seg_data:
        u_of_t = _interp_u(Zu, dt) if Zu is not None else None
        if u_of_t is None:
            f = lambda t, x: model.predict(_clipped(x[None, :], clip))[0]
        else:
            f = lambda t, x: model.predict(
                _clipped(np.concatenate([x, u_of_t(t)])[None, :], clip))[0]
        for k0 in range(warm, len(Zs) - steps - 1, stride):
            n_total += 1
            X, _ = rollout(f, Zs[k0], steps, t0=k0 * dt, dt=dt,
                           integrator=integrator, blowup=blowup)
            T = Zabs[k0 + 1:k0 + steps + 1] - Zabs[k0] + Zs[k0]
            for h, s in cps.items():
                pred = X[1:s + 1]
                if np.isnan(pred).any():
                    div[h] += 1
                    continue
                err = pred - T[:s]
                acc[h].append(float(np.mean(np.sqrt(np.mean(err**2, axis=0)) / col_std)))
    return {"n_rollouts": n_total,
            "horizons": {h: {"nrmse": float(np.median(acc[h])) if acc[h] else float("inf"),
                             "diverged_frac": div[h] / max(n_total, 1)}
                         for h in cps}}


def _baseline_metrics(seg_data: List[SegData], dt: float, horizons: Sequence[int],
                      ic_stride_h: float, warmup_h: float) -> Dict[str, Dict[int, float]]:
    """Persistence / climatology / daily-repeat baselines, same protocol.

    Parameters
    ----------
    seg_data : List[SegData]
        Test-segment triples.
    dt : float
        Sample spacing in hours.
    horizons : Sequence[int]
        Forecast horizons in hours.
    ic_stride_h : float
        Spacing between rollout initial conditions in hours.
    warmup_h : float
        Hours skipped at each segment start before the first IC.

    Returns
    -------
    Dict[str, Dict[int, float]]
        Median NRMSE per baseline and horizon (``daily_repeat`` only when
        ``dt`` divides a day into at least one sample).
    """
    cps = {int(h): max(1, int(round(h / dt))) for h in horizons}
    steps = max(cps.values())
    stride = max(1, int(round(ic_stride_h / dt)))
    s24 = int(round(24 / dt))
    warm = max(int(round(warmup_h / dt)), s24 if s24 >= 1 else 0)
    pooled = np.vstack([Zs for Zs, _, _ in seg_data])
    col_std = pooled.std(axis=0)
    col_std[col_std == 0] = 1.0
    names = ["persistence", "climatology"] + (["daily_repeat"] if s24 >= 1 else [])
    acc: Dict[str, Dict[int, List[float]]] = {b: {h: [] for h in cps} for b in names}
    for Zs, _, Zabs in seg_data:
        for k0 in range(warm, len(Zs) - steps - 1, stride):
            T = Zabs[k0 + 1:k0 + steps + 1] - Zabs[k0] + Zs[k0]
            preds = {"persistence": np.tile(Zs[k0], (steps, 1)),
                     "climatology": np.zeros_like(T)}
            if "daily_repeat" in acc:
                prev = Zabs[k0 - s24 + 1:k0 + 1] - Zabs[k0] + Zs[k0]
                preds["daily_repeat"] = np.vstack(
                    [prev] * int(np.ceil(steps / s24)))[:steps]
            for b, pred in preds.items():
                err = pred - T
                for h, s in cps.items():
                    acc[b][h].append(float(np.mean(
                        np.sqrt(np.mean(err[:s] ** 2, axis=0)) / col_std)))
    return {b: {h: float(np.median(v[h])) if v[h] else float("inf") for h in cps}
            for b, v in acc.items()}


# --------------------------------------------------------------------------
# the one call
# --------------------------------------------------------------------------
def run(data: TimeSeriesData, states: Sequence[str], inputs: Sequence[str] = (),
        method: str = "sindyc", *, test_data: Optional[TimeSeriesData] = None,
        split: str = "chrono", train_frac: float = 0.75, center: str = "causal",
        op_window_h: float = 72.0, fit: str = "discrete",
        degree: Optional[int] = None, threshold: Optional[float] = None,
        alpha: Optional[float] = None, clip: float = 3.0,
        library: Optional[FeatureLibrary] = None,
        horizons: Sequence[int] = (6, 12, 24, 48, 72),
        ic_stride_h: float = 24.0, blowup: float = 20.0,
        savgol_window: int = 9) -> SysIdResult:
    """Identify and evaluate a dynamical model on a segmented time series.

    Parameters
    ----------
    data : TimeSeriesData
        The dataset (training + test segments, unless ``test_data`` is
        given).
    states : Sequence[str]
        Channel names used as dynamical states.
    inputs : Sequence[str]
        Channel names used as exogenous control inputs (ignored for
        ``method="sindy"``).
    method : str
        ``"sindyc"`` (sparse, with inputs), ``"sindy"`` (sparse,
        autonomous) or ``"dmdc"`` (dense linear least squares with inputs).
    test_data : Optional[TimeSeriesData]
        Explicit held-out dataset; when given, all of ``data`` trains and
        all of ``test_data`` tests (e.g. cross-period transfer).
    split : str
        ``"chrono"`` (test = last segments) or ``"interleave"`` (every 4th
        segment tests); used only when ``test_data`` is None.
    train_frac : float
        Approximate fraction of samples assigned to training
        (``split="chrono"``).
    center : str
        ``"causal"`` (trailing operating point, no future information) or
        ``"segment"`` (per-segment mean; oracle).
    op_window_h : float
        Trailing operating-point window in hours (``center="causal"``).
    fit : str
        ``"discrete"`` (forward-difference targets + Euler rollouts) or
        ``"savgol"`` (smoothed derivatives + RK4 rollouts).
    degree : Optional[int]
        Polynomial library degree; None uses the method preset.
    threshold : Optional[float]
        STRidge threshold; None uses the method preset.
    alpha : Optional[float]
        Ridge penalty; None uses the method preset.
    clip : float
        Library-input saturation bound in train stds (0 disables).
    library : Optional[FeatureLibrary]
        Custom candidate-term library (overrides ``degree``).
    horizons : Sequence[int]
        Forecast horizons in hours for the rollout metrics.
    ic_stride_h : float
        Spacing between rollout initial conditions in hours.
    blowup : float
        Rollout divergence guard in z-units.
    savgol_window : int
        Savitzky-Golay window for ``fit="savgol"``.

    Returns
    -------
    SysIdResult
        Fitted model, identified equations and the full evaluation.

    Raises
    ------
    ValueError
        If ``method`` is unknown or no test segments are available.
    """
    if method not in PRESETS:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(PRESETS)}")
    p_deg, p_th, p_al = PRESETS[method]
    degree = p_deg if degree is None else degree
    threshold = p_th if threshold is None else threshold
    alpha = p_al if alpha is None else alpha
    states = list(states)
    inputs = [] if method == "sindy" else list(inputs)
    dt = float(data.dt_hours)
    integrator = "euler" if fit == "discrete" else "rk4"

    # ---- split ----------------------------------------------------------
    if test_data is not None:
        train_segs, test_segs = data.segments, test_data.segments
    elif split == "interleave":
        test_segs = data.segments[3::4]
        train_segs = [s for i, s in enumerate(data.segments) if (i - 3) % 4 != 0]
    else:
        total = data.n_samples
        train_segs, acc = [], 0
        for s in data.segments:
            if acc < train_frac * total or len(data.segments) - len(train_segs) <= 1:
                train_segs.append(s)
                acc += len(s)
        test_segs = data.segments[len(train_segs):]
    if not test_segs:
        train_segs, test_segs = train_segs[:-1], train_segs[-1:]
    if not train_segs or not test_segs:
        raise ValueError("not enough segments for a train/test split")

    # ---- centering + spin-up trim ---------------------------------------
    s_cols = data.columns(states)
    u_cols = data.columns(inputs) if inputs else []
    w_op = max(1, int(round(op_window_h / dt)))

    def center_of(A: np.ndarray) -> np.ndarray:
        """Operating point of one segment's raw array.

        Parameters
        ----------
        A : np.ndarray
            Segment array ``(n, d)``.

        Returns
        -------
        np.ndarray
            The (trailing or constant) operating point, same shape.
        """
        if center == "causal":
            return trailing_mean(A, w_op)
        return np.tile(A.mean(axis=0), (len(A), 1))

    def prep(segs: List[np.ndarray]) -> List[Tuple[np.ndarray, ...]]:
        """Raw arrays -> (state fluct, state abs, input fluct) per segment.

        Parameters
        ----------
        segs : List[np.ndarray]
            Raw segment arrays over all channels.

        Returns
        -------
        List[Tuple[np.ndarray, ...]]
            Per-segment unscaled (Xs_fluct, Xs_abs, Xu_fluct-or-None),
            with the causal spin-up trimmed off.
        """
        out = []
        trim = w_op if center == "causal" else 0
        for seg in segs:
            if len(seg) <= trim + 2:
                continue
            Xs, Xu = seg[:, s_cols], (seg[:, u_cols] if u_cols else None)
            fl_s = Xs - center_of(Xs)
            fl_u = None if Xu is None else Xu - center_of(Xu)
            out.append((fl_s[trim:], Xs[trim:],
                        None if fl_u is None else fl_u[trim:]))
        return out

    raw_tr, raw_te = prep(train_segs), prep(test_segs)
    if not raw_tr or not raw_te:
        raise ValueError("no segments survive the operating-point spin-up trim")
    sc_s = Scaler().fit(np.vstack([r[0] for r in raw_tr]))
    sc_u = (Scaler().fit(np.vstack([r[2] for r in raw_tr]))
            if u_cols else None)

    def z(raw: List[Tuple[np.ndarray, ...]]) -> List[SegData]:
        """Scale prepared raw arrays into SegData triples.

        Parameters
        ----------
        raw : List[Tuple[np.ndarray, ...]]
            Output of ``prep``.

        Returns
        -------
        List[SegData]
            Scaled (states, inputs, absolute states) per segment.
        """
        return [(sc_s.transform(fl_s),
                 None if fl_u is None else sc_u.transform(fl_u),
                 (ab - sc_s.mean_) / sc_s.std_) for fl_s, ab, fl_u in raw]

    d_tr, d_te = z(raw_tr), z(raw_te)

    # ---- fit + evaluate ---------------------------------------------------
    lib = library if library is not None else PolynomialLibrary(degree=degree)
    X, Y = _pairs(d_tr, fit, dt, clip, savgol_window)
    model = SINDy(lib, threshold=threshold, alpha=alpha).fit(
        X, Y, input_names=states + inputs)
    Xte, Yte = _pairs(d_te, fit, dt, clip, savgol_window)
    r2_tr = r2_per_column(Y, model.predict(X))
    r2_te = r2_per_column(Yte, model.predict(Xte))
    warmup_h = 24.0 if dt <= 24 else dt
    metrics = _rollout_metrics(model, d_te, dt, integrator, horizons, ic_stride_h,
                               warmup_h, clip, blowup)
    baselines = _baseline_metrics(d_te, dt, horizons, ic_stride_h, warmup_h)

    names = model.feature_names_
    coeffs = {st: {names[i]: float(model.coef_[i, j])
                   for i in range(len(names)) if abs(model.coef_[i, j]) > 0}
              for j, st in enumerate(states)}
    return SysIdResult(
        model=model, states=states, inputs=inputs,
        equations=model.equations([f"d/dt {s}" for s in states]),
        coefficients=coeffs,
        r2_train={s: float(v) for s, v in zip(states, r2_tr)},
        r2_test={s: float(v) for s, v in zip(states, r2_te)},
        metrics=metrics, baselines=baselines,
        details={"method": method, "fit": fit, "center": center,
                 "op_window_h": op_window_h, "degree": degree,
                 "threshold": threshold, "alpha": alpha, "clip": clip,
                 "dt_hours": dt, "n_train_segments": len(d_tr),
                 "n_test_segments": len(d_te),
                 "n_train_samples": int(sum(len(s[0]) for s in d_tr)),
                 "n_test_samples": int(sum(len(s[0]) for s in d_te))})
