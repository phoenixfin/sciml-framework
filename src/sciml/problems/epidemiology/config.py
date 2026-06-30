"""Configuration for the dengue beta(t) SINDy example.

Defaults reproduce the notebook's simulated-data path so the example runs with
no external downloads. Set ``data.use_real=True`` and a ``data.path`` to use a
real weekly series.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ...core.config import ConfigBase


@dataclass
class ModelConfig(ConfigBase):
    model: str = "SIRS"            # 'SI' | 'SIR' | 'SIRS'
    N: int = 100_000              # catchment population
    reporting: float = 0.1        # reporting fraction rho
    gamma: float = 1.0 / 1.4      # recovery rate (per week)
    mu: float = 1.0 / (52 * 70)   # birth = death rate
    omega: float = 1.0 / (52 * 3)  # waning-immunity rate (0 = permanent)


@dataclass
class DataConfig(ConfigBase):
    use_real: bool = False
    path: Optional[str] = None
    year_col: str = "Tahun"
    week_col: str = "Week"
    infected_col: str = "Cases"
    sim_weeks: int = 520
    sim_i0: int = 50
    sim_noise: float = 0.002
    # Seasonal "true" beta for simulation: base + amp*sin(2 pi (t - phase)/period)
    beta_base: float = 0.25
    beta_amp: float = 0.12
    beta_phase: float = 10.0
    beta_period: float = 52.0
    sg_window: int = 11
    sg_poly: int = 3


@dataclass
class EstimatorConfig(ConfigBase):
    s_recon: str = "cumulative"   # 'cumulative' | 'ode' | 'ekf'
    local_methods: List[str] = field(default_factory=lambda: ["direct", "windowed"])
    global_basis: str = "time"    # 'time' | 'none'
    mask_eps: float = 1e-4
    window_size: int = 8
    window_step: int = 1
    str_thresh: float = 0.01
    str_alpha: float = 0.01
    poly_degree: int = 3
    n_fourier: int = 5
    fourier_period: float = 52.0
    lasso_alpha: float = 1e-5
    sparse_method: str = "stridge"  # 'stridge' (numpy) | 'lasso' (needs sklearn)


@dataclass
class EpiConfig(ConfigBase):
    name: str = "dengue_beta"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    estim: EstimatorConfig = field(default_factory=EstimatorConfig)
    output_dir: str = "outputs/epidemiology"
    seed: int = 42
