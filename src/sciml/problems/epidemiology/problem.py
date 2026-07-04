""":class:`EpiProblem` -- dengue beta(t) identification wired to the SINDy engine."""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from ...core.derivatives import savgol
from ...solvers.compartmental import simulate_compartmental
from ..base import Problem
from . import estimators as est
from . import reconstruction as rec
from .config import EpiConfig


class EpiProblem(Problem):
    """Dengue beta(t) identification: load/simulate data, reconstruct S(t), estimate beta."""

    name = "epidemiology"

    def __init__(self, config: Optional[EpiConfig] = None):
        """Initialize the problem with an optional configuration.

        Parameters
        ----------
        config : Optional[EpiConfig]
            Problem configuration; a default :class:`EpiConfig` is used if ``None``.
        """
        super().__init__(config or EpiConfig())
        self.raw: Optional[Dict] = None
        self.data: Optional[Dict] = None

    def beta_true(self) -> Callable[[float], float]:
        """Return the seasonal ground-truth beta(t) callable (used for simulation).

        Returns
        -------
        Callable[[float], float]
            A function mapping time to the true seasonal transmission rate.
        """
        d = self.config.data
        return lambda t: d.beta_base + d.beta_amp * np.sin(
            2 * np.pi * (np.asarray(t) - d.beta_phase) / d.beta_period)

    # -- data -------------------------------------------------------------
    def load_or_simulate(self, *, rng: Optional[np.random.Generator] = None) -> Dict:
        """Load a real weekly series or simulate one; store and return the raw data.

        Parameters
        ----------
        rng : Optional[np.random.Generator]
            Random generator used when simulating; system default if ``None``.

        Returns
        -------
        Dict
            The raw data dictionary (also stored on ``self.raw``).
        """
        m, d = self.config.model, self.config.data
        if d.use_real:
            self.raw = self._load_real()
        else:
            sim = simulate_compartmental(
                m.model, N=m.N, I0=d.sim_i0, n_weeks=d.sim_weeks, beta_fn=self.beta_true(),
                gamma=m.gamma, mu=m.mu, omega=m.omega, noise_std=d.sim_noise, rng=rng)
            sim["I_raw"] = sim["I"].copy()
            sim["year_ticks"] = {i * 52: str(i + 1) for i in range(d.sim_weeks // 52)}
            self.raw = sim
        return self.raw

    def _load_real(self) -> Dict:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Loading real data requires pandas "
                              "(`pip install pandas openpyxl`).") from exc
        d, m = self.config.data, self.config.model
        if not d.path:
            raise ValueError("data.use_real=True but data.path is not set.")
        ext = d.path.rsplit(".", 1)[-1].lower()
        df = pd.read_excel(d.path) if ext in ("xlsx", "xls") else pd.read_csv(d.path)
        sort_cols = [c for c in (d.year_col, d.week_col) if c and c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)
        t = np.arange(len(df), dtype=float)
        I = np.clip(df[d.infected_col].values.astype(float) / m.reporting, 0, m.N)
        I = np.clip(savgol(I, window=d.sg_window, poly=d.sg_poly), 0.5, m.N)
        ticks = {}
        if d.year_col in df.columns:
            for yr in sorted(df[d.year_col].unique()):
                ticks[int(df[df[d.year_col] == yr].index[0])] = str(int(yr))
        return {"t": t, "I": I, "N": m.N, "model": m.model, "year_ticks": ticks,
                "I_raw": df[d.infected_col].values.astype(float)}

    def reconstruct(self) -> Dict:
        """Reconstruct S(t) (and R) via the configured method; store and return data.

        Returns
        -------
        Dict
            The reconstructed data dictionary (also stored on ``self.data``).

        Raises
        ------
        ValueError
            If the configured ``s_recon`` method is unknown.
        """
        if self.raw is None:
            self.load_or_simulate()
        m, e = self.config.model, self.config.estim
        t, I = self.raw["t"], self.raw["I"]
        beta_ekf = None
        if m.model == "SI":
            S, R = m.N - I, np.zeros_like(I)
        elif e.s_recon == "cumulative":
            S, R = rec.reconstruct_S_cumulative(t, I, m.N, m.gamma)
        elif e.s_recon == "ode":
            S, R = rec.reconstruct_S_ode(t, I, m.N, m.gamma, m.mu, m.omega, m.model,
                                         self.config.data.sg_window, self.config.data.sg_poly,
                                         e.mask_eps)
        elif e.s_recon == "ekf":
            S, R, beta_ekf = rec.reconstruct_S_ekf(t, I, m.N, m.gamma, m.mu, m.omega, m.model)
        else:
            raise ValueError(f"Unknown s_recon {e.s_recon!r}")
        self.data = {"t": t, "S": S, "I": I, "R": R, "N": m.N, "model": m.model}
        if "beta_true" in self.raw:
            self.data["beta_true"] = self.raw["beta_true"]
        if beta_ekf is not None:
            self.data["beta_ekf"] = beta_ekf
        return self.data

    # -- estimation -------------------------------------------------------
    def estimate(self) -> Dict:
        """Run the configured local + global beta(t) estimators; return their results.

        Returns
        -------
        Dict
            A dictionary with ``"local"`` per-method estimates and an optional
            ``"global"`` time-basis fit.
        """
        if self.data is None:
            self.reconstruct()
        m, e = self.config.model, self.config.estim
        sgw, sgp = self.config.data.sg_window, self.config.data.sg_poly
        out: Dict = {"local": {}, "global": None}

        if "direct" in e.local_methods:
            td, bd = est.estimate_beta_direct(self.data, mu=m.mu, omega=m.omega,
                                              mask_eps=e.mask_eps, sg_window=sgw, sg_poly=sgp)
            out["local"]["direct"] = {"t": td, "beta": bd}
        if "windowed" in e.local_methods:
            tw, bw = est.estimate_beta_windowed(
                self.data, mu=m.mu, omega=m.omega, window=e.window_size, step=e.window_step,
                threshold=e.str_thresh, alpha=e.str_alpha, mask_eps=e.mask_eps,
                sg_window=sgw, sg_poly=sgp)
            out["local"]["windowed"] = {"t": tw, "beta": bw}
        if "beta_ekf" in self.data:
            out["local"]["ekf"] = {"t": self.data["t"], "beta": self.data["beta_ekf"]}

        if e.global_basis == "time" and out["local"]:
            src = out["local"].get("windowed") or next(iter(out["local"].values()))
            beta_global, xi, names = est.fit_global_time_basis(
                src["t"], src["beta"], self.data["t"], poly_degree=e.poly_degree,
                n_fourier=e.n_fourier, fourier_period=e.fourier_period,
                method=e.sparse_method, alpha=e.str_alpha, lasso_alpha=e.lasso_alpha)
            active = np.abs(xi) > 1e-6
            out["global"] = {"t": self.data["t"], "beta": beta_global,
                             "terms": [(n, float(c)) for n, c in zip(np.array(names)[active], xi[active])]}
        return out

    def reference(self) -> Optional[np.ndarray]:
        """The true beta(t) when data is simulated, else ``None``.

        Returns
        -------
        Optional[np.ndarray]
            The ground-truth ``beta(t)`` array, or ``None`` if unavailable.
        """
        if self.data is not None:
            return self.data.get("beta_true")
        return None
