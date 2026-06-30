"""Configuration for the moving-boundary wave PINN example (notebook v4 defaults)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from ...core.config import ConfigBase


@dataclass
class WaveParams(ConfigBase):
    """Physical problem parameters."""

    b: float = 0.3            # obstacle parameter
    a_bc: float = 0.0         # fixed-end boundary value
    eps: float = 0.1          # small parameter
    delta: float = 0.04       # initial-perturbation amplitude
    t_final: float = 6.0      # final time T


@dataclass
class NetConfig(ConfigBase):
    nu_freq: int = 64         # Fourier features for Nu
    nu_hidden: int = 5
    nu_width: int = 128
    ns_hidden: int = 5
    ns_width: int = 96
    s_lo: float = 0.05        # Ns output lower bound
    s_hi_margin: float = 0.01  # Ns upper bound = b - margin


@dataclass
class LossConfig(ConfigBase):
    weights: Dict[str, float] = field(default_factory=lambda: {
        "pde": 1.0, "excl": 1.0, "bcd": 20.0, "bcn": 80.0, "bcr": 10.0,
        "icu": 15.0, "icv": 5.0, "s0": 50.0, "sv": 30.0})
    mask_eps: float = 0.008
    n_pde: int = 5000
    n_bc: int = 600
    n_bcr: int = 200
    n_ic: int = 300
    n_sv: int = 400


@dataclass
class PhaseConfig(ConfigBase):
    name: str = "phase"
    steps: int = 3000
    lr: float = 1e-3
    eps_c_start: float = 5.0
    eps_c_end: float = 2.0
    rar_every: int = 0


def _default_phases() -> List[PhaseConfig]:
    return [
        PhaseConfig("1a", 3000, 1e-3, 5.0, 2.0, 0),
        PhaseConfig("1b", 3000, 1e-3, 2.0, 0.5, 0),
        PhaseConfig("2", 4000, 3e-4, 0.5, 0.1, 500),
        PhaseConfig("2b", 3000, 1e-4, 0.1, 0.0, 500),
        PhaseConfig("2c", 2000, 5e-5, 0.0, 0.0, 0),
    ]


@dataclass
class WaveTrainConfig(ConfigBase):
    phases: List[PhaseConfig] = field(default_factory=_default_phases)
    lbfgs_maxiter: int = 6000
    lbfgs_restart_maxiter: int = 3000
    fdm_nx: int = 300
    seed: int = 42


@dataclass
class WaveObstacleConfig(ConfigBase):
    name: str = "wave_obstacle_v4"
    params: WaveParams = field(default_factory=WaveParams)
    net: NetConfig = field(default_factory=NetConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: WaveTrainConfig = field(default_factory=WaveTrainConfig)
    output_dir: str = "outputs/wave_obstacle"
