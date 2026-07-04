"""Configuration for the SWE / DeepONet example (defaults reproduce the notebook)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from ...core.config import ConfigBase


@dataclass
class SWEDomain(ConfigBase):
    """Physical domain and constants for the shallow-water problem."""

    length: float = 10.0        #: spatial domain length L
    t_final: float = 1.0        #: final simulation time T
    gravity: float = 9.81       #: gravitational acceleration g


@dataclass
class ModelConfig(ConfigBase):
    """DeepONet architecture hyper-parameters."""

    n_sensors: int = 100        #: M -- branch input dimension (sensor count)
    width: int = 64             #: P -- latent / basis dimension
    hidden: List[int] = field(default_factory=lambda: [128, 128, 128])  #: hidden widths
    out_std: float = 0.1        #: std of the trunk output-layer init
    h_min: float = 0.05         #: minimum water depth (IC-shortcut floor)
    eps: float = 1e-4           #: positivity epsilon added to the depth


@dataclass
class DataConfig(ConfigBase):
    """GP sampling and supervised-dataset parameters."""

    grid: int = 500             #: fine grid for supervised reference fields
    n_train: int = 500          #: GP training-pool size
    n_test: int = 100           #: unseen test pairs for generalization
    n_data_gp: int = 150        #: supervised GP samples used for the data loss
    t_snaps: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0])  #: snapshot times
    solver_nx: int = 400        #: Lax-Friedrichs spatial resolution
    solver_nt: int = 4000       #: Lax-Friedrichs temporal resolution
    h0_length_scale: float = 2.0  #: GP length scale for initial depth
    h0_amp: float = 0.4         #: GP amplitude for initial depth
    h0_mean: float = 1.0        #: GP mean for initial depth
    h0_clip_min: float = 0.3    #: minimum sampled initial depth
    bath_length_scale: float = 3.0  #: GP length scale for bathymetry
    bath_amp: float = 0.12      #: GP amplitude for bathymetry


@dataclass
class TrainConfig(ConfigBase):
    """Optimization / training-loop parameters."""

    n_iter: int = 40000         #: number of training iterations
    batch: int = 8              #: mini-batch size (functions per step)
    n_bc: int = 200             #: boundary collocation points per step
    lr: float = 1e-3            #: initial Adam learning rate
    lr_decay_steps: int = 10000  #: exponential-decay step interval
    lr_decay_rate: float = 0.5  #: exponential-decay rate
    grad_clip: float = 1.0      #: global-norm gradient clip
    lam_bc: float = 5.0         #: boundary-loss weight
    lam_hu: float = 5.0         #: momentum data-loss upweight
    ckpt_every: int = 2000      #: checkpoint interval
    log_every: int = 1000       #: logging interval
    seed: int = 42              #: RNG seed


@dataclass
class SWEConfig(ConfigBase):
    """Top-level configuration for the SWE / DeepONet example."""

    name: str = "swe_v6"
    domain: SWEDomain = field(default_factory=SWEDomain)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output_dir: str = "outputs/swe"
