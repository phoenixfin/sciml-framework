""":class:`SWEProblem` -- the SWE task wired to the DeepONet engine.

Encapsulates the GP training pool, supervised reference snapshots and training
steps as one reusable object. All training-step variants share a uniform
9-tensor batch (from :meth:`sample_batch`) so they are interchangeable from the
:class:`~sciml.methods.deeponet.Trainer`'s point of view.
"""

from __future__ import annotations

import time
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from ...data.gp import PeriodicGPSampler
from ...solvers.swe_lax_friedrichs import lax_friedrichs_swe
from ...tf_utils import grid_interp
from ..base import Problem
from . import cases as swe_cases
from .config import SWEConfig
from .model import VARIANTS, warmup


class SWEProblem(Problem):
    """SWE task wired to the DeepONet engine (GP pool, dataset, training steps)."""

    name = "swe"

    def __init__(self, config: Optional[SWEConfig] = None):
        """Initialize the problem, GP samplers and derived grids from a config.

        Parameters
        ----------
        config : Optional[SWEConfig]
            Problem configuration; a default :class:`SWEConfig` is used if ``None``.
        """
        super().__init__(config or SWEConfig())
        c = self.config
        self.L = float(c.domain.length)
        self.T = float(c.domain.t_final)
        self.g = float(c.domain.gravity)
        self.M = c.model.n_sensors
        self.GRID = c.data.grid
        self.t_snaps: List[float] = list(c.data.t_snaps)
        self.n_snaps = len(self.t_snaps)
        self._L_tf = tf.constant(self.L, tf.float32)
        self._lam_hu = tf.constant(c.train.lam_hu, tf.float32)

        self.x_sensors = np.linspace(0, self.L, self.M, dtype=np.float32)
        self.x_grid = np.linspace(0, self.L, self.GRID, dtype=np.float32)

        self.h0_sampler = PeriodicGPSampler(
            period=self.L, length_scale=c.data.h0_length_scale,
            amplitude=c.data.h0_amp, mean=c.data.h0_mean, clip_min=c.data.h0_clip_min)
        self.bath_sampler = PeriodicGPSampler(
            period=self.L, length_scale=c.data.bath_length_scale,
            amplitude=c.data.bath_amp, mean=0.0, clip_min=0.0)

        self.H0_s = self.B_s = self.H0_grid = self.B_grid = None
        self.idx_c1 = self.idx_c2 = None
        self.data_idx: List[int] = []
        self._dataset_ready = False

    # -- data preparation -------------------------------------------------
    def prepare(self, seed: Optional[int] = None) -> "SWEProblem":
        """Build the GP training pool (+ C1/C2 anchors) and precompute grid interpolations.

        Parameters
        ----------
        seed : Optional[int]
            Random seed for reproducible GP sampling; unseeded if ``None``.

        Returns
        -------
        SWEProblem
            This problem instance (for chaining).
        """
        c = self.config
        if seed is not None:
            np.random.seed(seed)
        n_train = c.data.n_train
        H0 = self.h0_sampler.sample(self.x_sensors, n_train)
        B = self.bath_sampler.sample(self.x_sensors, n_train)
        h0_anchor = swe_cases.h0_gaussian(self.x_sensors).reshape(1, -1)
        b_flat = swe_cases.b_flat(self.x_sensors).reshape(1, -1)
        b_bump = swe_cases.b_bump(self.x_sensors).reshape(1, -1)
        H0 = np.vstack([H0, h0_anchor, h0_anchor])
        B = np.vstack([B, b_flat, b_bump])
        self.H0_s, self.B_s = H0, B
        self.n_total = len(H0)
        self.idx_c1 = self.n_total - 2
        self.idx_c2 = self.n_total - 1
        self.H0_grid = np.array([np.interp(self.x_grid, self.x_sensors, H0[i])
                                 for i in range(self.n_total)], dtype=np.float32)
        self.B_grid = np.array([np.interp(self.x_grid, self.x_sensors, B[i])
                                for i in range(self.n_total)], dtype=np.float32)
        return self

    def boundary_gap(self) -> float:
        """Mean ``|h0(0) - h0(L)|`` over the GP pool (periodicity diagnostic).

        Returns
        -------
        float
            The mean absolute boundary mismatch across the GP training pool.
        """
        n_train = self.config.data.n_train
        return float(np.abs(self.H0_s[:n_train, 0] - self.H0_s[:n_train, -1]).mean())

    def generate_dataset(self, n_data_gp: Optional[int] = None,
                         verbose: bool = True) -> "SWEProblem":
        """Run the reference solver on the supervised subset and stage TF tensors.

        Parameters
        ----------
        n_data_gp : Optional[int]
            Number of GP-pool trajectories to solve; config default if ``None``.
        verbose : bool
            Whether to print progress during dataset generation.

        Returns
        -------
        SWEProblem
            This problem instance (for chaining).
        """
        if self.H0_s is None:
            self.prepare()
        c = self.config
        n_data_gp = n_data_gp if n_data_gp is not None else c.data.n_data_gp
        n_train = c.data.n_train
        gp_idx = np.round(np.linspace(0, n_train - 1, n_data_gp)).astype(int).tolist()
        self.data_idx = gp_idx + [self.idx_c1, self.idx_c2]
        n_data = len(self.data_idx)
        D_h = [np.zeros((n_data, self.GRID), np.float32) for _ in self.t_snaps]
        D_hu = [np.zeros((n_data, self.GRID), np.float32) for _ in self.t_snaps]
        t0 = time.time()
        for k, i in enumerate(self.data_idx):
            xr, sn = self._solve_index(i)
            for s in range(self.n_snaps):
                D_h[s][k] = np.interp(self.x_grid, xr, sn["h"][s])
                D_hu[s][k] = np.interp(self.x_grid, xr, sn["hu"][s])
            if verbose and (k + 1) % 25 == 0:
                print(f"  {k + 1}/{n_data} ({time.time() - t0:.0f}s)")
        self.xt_snaps = [
            tf.constant(np.stack([self.x_grid, np.full(self.GRID, t)], 1).astype(np.float32))
            for t in self.t_snaps]
        self.D_h_tf = [tf.constant(D_h[s]) for s in range(self.n_snaps)]
        self.D_hu_tf = [tf.constant(D_hu[s]) for s in range(self.n_snaps)]
        self.D_H0s = tf.constant(self.H0_s[self.data_idx])
        self.D_Bs = tf.constant(self.B_s[self.data_idx])
        self.D_H0g = tf.constant(self.H0_grid[self.data_idx])
        self.D_Bg = tf.constant(self.B_grid[self.data_idx])
        self._dataset_ready = True
        if verbose:
            print(f"Dataset ready: {n_data} trajectories x {self.n_snaps} snapshots "
                  f"in {time.time() - t0:.1f}s")
        return self

    def _solve_index(self, i: int):
        c = self.config
        h0fn = lambda x, i=i: np.interp(x, self.x_sensors, self.H0_s[i])
        bfn = lambda x, i=i: np.interp(x, self.x_sensors, self.B_s[i])
        return lax_friedrichs_swe(h0fn, bfn, length=self.L, t_final=self.T,
                                  gravity=self.g, nx=c.data.solver_nx,
                                  nt=c.data.solver_nt, t_out=self.t_snaps)

    def reference(self, h0_fn: Callable, b_fn: Callable, **kw: object) -> Tuple:
        """Lax-Friedrichs reference solution for arbitrary IC/bathymetry callables.

        Parameters
        ----------
        h0_fn : Callable
            Initial-depth profile as a function of ``x``.
        b_fn : Callable
            Bathymetry profile as a function of ``x``.
        **kw : object
            Solver overrides (``length``, ``t_final``, ``gravity``, ``nx``, ``nt``,
            ``t_out``); defaults come from the config.

        Returns
        -------
        Tuple
            The reference spatial grid and the snapshot dictionary.
        """
        c = self.config
        kw.setdefault("length", self.L); kw.setdefault("t_final", self.T)
        kw.setdefault("gravity", self.g); kw.setdefault("nx", c.data.solver_nx)
        kw.setdefault("nt", c.data.solver_nt); kw.setdefault("t_out", self.t_snaps)
        return lax_friedrichs_swe(h0_fn, b_fn, **kw)

    # -- model + training step -------------------------------------------
    def build_model(self, variant: str = "full") -> tf.keras.Model:
        """Build and warm up a model variant (``full`` / ``shared_branch`` / ``no_ic_shortcut``).

        Parameters
        ----------
        variant : str
            Model variant key to build.

        Returns
        -------
        tf.keras.Model
            The warmed-up model instance.

        Raises
        ------
        ValueError
            If ``variant`` is not a known model variant.
        """
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}; choose {list(VARIANTS)}")
        c = self.config
        kwargs = dict(n_sensors=self.M, width=c.model.width,
                      hidden=tuple(c.model.hidden), out_std=c.model.out_std)
        if variant != "no_ic_shortcut":
            kwargs.update(h_min=c.model.h_min, eps=c.model.eps)
        return warmup(VARIANTS[variant](**kwargs), self.M)

    @property
    def component_names(self) -> Sequence[str]:
        """Names of the extra scalars returned by the training step (data/BC/grad-norm)."""
        return ["Ld", "Lb", "gnorm"]

    def make_step(self, model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer,
                  variant: str = "full", lam_ic: float = 10.0) -> Callable[..., Tuple]:
        """Build the ``@tf.function`` data+BC training step (adds an IC loss for A2).

        Parameters
        ----------
        model : tf.keras.Model
            The model to train.
        optimizer : tf.keras.optimizers.Optimizer
            Optimizer applying the gradient updates.
        variant : str
            Model variant key; ``no_ic_shortcut`` adds an extra IC loss term.
        lam_ic : float
            Weight of the initial-condition loss term.

        Returns
        -------
        Callable[..., Tuple]
            The compiled training-step function returning ``(loss, Ld, Lb, gnorm)``.

        Raises
        ------
        RuntimeError
            If :meth:`generate_dataset` has not been called first.
        """
        if not self._dataset_ready:
            raise RuntimeError("Call generate_dataset() before make_step().")
        c = self.config
        L = self._L_tf
        lam_bc = tf.constant(c.train.lam_bc, tf.float32)
        lam_hu = self._lam_hu
        clip = c.train.grad_clip
        n_snaps = self.n_snaps
        use_ic_loss = (variant == "no_ic_shortcut")
        lic = tf.constant(lam_ic, tf.float32)
        xt_ic = tf.constant(np.stack([self.x_grid, np.zeros(self.GRID)], 1).astype(np.float32))

        @tf.function(reduce_retracing=True)
        def step(h0_b, b_b, h0_g, b_g, t_bc, h0_ic_s, b_ic_s, h0_ic_g, b_ic_g):
            x0 = tf.zeros_like(t_bc)
            xL = tf.fill(tf.shape(t_bc), L)
            h0_bcl = grid_interp(h0_g, x0, L); h0_bcr = grid_interp(h0_g, xL, L)
            b_bcl = grid_interp(b_g, x0, L); b_bcr = grid_interp(b_g, xL, L)
            with tf.GradientTape() as tape:
                Ld = tf.constant(0.0)
                for s in range(n_snaps):
                    hp, hup = model(self.D_H0s, self.D_Bs, self.xt_snaps[s],
                                    self.D_H0g, self.D_Bg)
                    Ld = Ld + (tf.reduce_mean((hp - self.D_h_tf[s]) ** 2)
                               + lam_hu * tf.reduce_mean((hup - self.D_hu_tf[s]) ** 2))
                Ld = Ld / tf.cast(n_snaps, tf.float32)
                hl, hul = model(h0_b, b_b, tf.stack([x0, t_bc], 1), h0_bcl, b_bcl)
                hr, hur = model(h0_b, b_b, tf.stack([xL, t_bc], 1), h0_bcr, b_bcr)
                Lb = tf.reduce_mean((hl - hr) ** 2 + (hul - hur) ** 2)
                loss = Ld + lam_bc * Lb
                if use_ic_loss:
                    hp_ic, hup_ic = model(h0_ic_s, b_ic_s, xt_ic, h0_ic_g, b_ic_g)
                    Lic = tf.reduce_mean((hp_ic - h0_ic_g) ** 2 + hup_ic ** 2)
                    loss = loss + lic * Lic
            gs = tape.gradient(loss, model.trainable_variables)
            gs, gn = tf.clip_by_global_norm(gs, clip)
            optimizer.apply_gradients(zip(gs, model.trainable_variables))
            return loss, Ld, Lb, gn

        return step

    def sample_batch(self, iteration: int) -> Tuple:
        """Draw one uniform 9-tensor training batch (BC mini-batch + IC mini-batch).

        Parameters
        ----------
        iteration : int
            Current training iteration (unused; kept for the Trainer interface).

        Returns
        -------
        Tuple
            The nine input tensors consumed by the training step.
        """
        c = self.config
        batch = c.train.batch
        idx = np.random.choice(self.n_total, batch, replace=False)
        tbc = np.random.uniform(0, self.T, c.train.n_bc).astype(np.float32)
        n_data = len(self.data_idx)
        ic = np.random.choice(n_data, batch, replace=False)
        di = np.asarray(self.data_idx)
        return (
            tf.constant(self.H0_s[idx]), tf.constant(self.B_s[idx]),
            tf.constant(self.H0_grid[idx]), tf.constant(self.B_grid[idx]),
            tf.constant(tbc),
            tf.constant(self.H0_s[di[ic]]), tf.constant(self.B_s[di[ic]]),
            tf.constant(self.H0_grid[di[ic]]), tf.constant(self.B_grid[di[ic]]),
        )

    # -- inference --------------------------------------------------------
    def predict_grid(self, model: tf.keras.Model, h0_fn: Callable, b_fn: Callable,
                     nx: int = 200, nt: int = 100) -> Tuple:
        """Predict ``(xs, ts, h, hu)`` on an ``nx x nt`` space-time grid.

        Parameters
        ----------
        model : tf.keras.Model
            The trained model to evaluate.
        h0_fn : Callable
            Initial-depth profile as a function of ``x``.
        b_fn : Callable
            Bathymetry profile as a function of ``x``.
        nx : int
            Number of spatial grid points.
        nt : int
            Number of time grid points.

        Returns
        -------
        Tuple
            The spatial grid, time grid, predicted depth and momentum fields.
        """
        h0_s = h0_fn(self.x_sensors).reshape(1, -1).astype(np.float32)
        b_s = b_fn(self.x_sensors).reshape(1, -1).astype(np.float32)
        xs = np.linspace(0, self.L, nx, dtype=np.float32)
        ts = np.linspace(0, self.T, nt, dtype=np.float32)
        XX, TT = np.meshgrid(xs, ts)
        xt = tf.constant(np.stack([XX.ravel(), TT.ravel()], 1))
        h0_at = tf.constant(h0_fn(XX.ravel()).reshape(1, -1).astype(np.float32))
        b_at = tf.constant(b_fn(XX.ravel()).reshape(1, -1).astype(np.float32))
        hp, hup = model(tf.constant(h0_s), tf.constant(b_s), xt, h0_at, b_at)
        return xs, ts, hp.numpy()[0].reshape(nt, nx), hup.numpy()[0].reshape(nt, nx)
