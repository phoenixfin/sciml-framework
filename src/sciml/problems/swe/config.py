"""Configuration for the SWE / DeepONet example (defaults reproduce the notebook)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from ...core.config import ConfigBase


@dataclass
class SWEDomain(ConfigBase):
    length: float = 10.0
    t_final: float = 1.0
    gravity: float = 9.81


@dataclass
class ModelConfig(ConfigBase):
    n_sensors: int = 100          # M: branch input dimension
    width: int = 64               # P: latent/basis dimension
    hidden: List[int] = field(default_factory=lambda: [128, 128, 128])
    out_std: float = 0.1
    h_min: float = 0.05
    eps: float = 1e-4


@dataclass
class DataConfig(ConfigBase):
    grid: int = 500
    n_train: int = 500
    n_test: int = 100
    n_data_gp: int = 150
    t_snaps: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0])
    solver_nx: int = 400
    solver_nt: int = 4000
    h0_length_scale: float = 2.0
    h0_amp: float = 0.4
    h0_mean: float = 1.0
    h0_clip_min: float = 0.3
    bath_length_scale: float = 3.0
    bath_amp: float = 0.12


@dataclass
class TrainConfig(ConfigBase):
    n_iter: int = 40000
    batch: int = 8
    n_bc: int = 200
    lr: float = 1e-3
    lr_decay_steps: int = 10000
    lr_decay_rate: float = 0.5
    grad_clip: float = 1.0
    lam_bc: float = 5.0
    lam_hu: float = 5.0
    ckpt_every: int = 2000
    log_every: int = 1000
    seed: int = 42


@dataclass
class SWEConfig(ConfigBase):
    name: str = "swe_v6"
    domain: SWEDomain = field(default_factory=SWEDomain)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output_dir: str = "outputs/swe"
