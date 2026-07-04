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
    """Compartmental-model choice and epidemiological rate constants."""

    model: str = "SIRS"           #: 'SI' | 'SIR' | 'SIRS'
    N: int = 100_000              #: catchment population
    reporting: float = 0.1        #: reporting fraction rho
    gamma: float = 1.0 / 1.4      #: recovery rate (per week)
    mu: float = 1.0 / (52 * 70)   #: birth = death rate
    omega: float = 1.0 / (52 * 3)  #: waning-immunity rate (0 = permanent)


@dataclass
class DataConfig(ConfigBase):
    """Data source (real file vs. simulation) and preprocessing."""

    use_real: bool = False
    path: Optional[str] = None
    year_col: str = "Tahun"
    week_col: str = "Week"
    infected_col: str = "Cases"
    sim_weeks: int = 520          #: simulated series length (weeks)
    sim_i0: int = 50              #: simulated initial infected count
    sim_noise: float = 0.002      #: observation-noise std (fraction of N)
    beta_base: float = 0.25       #: seasonal true-beta baseline
    beta_amp: float = 0.12        #: seasonal true-beta amplitude
    beta_phase: float = 10.0      #: seasonal true-beta phase (weeks)
    beta_period: float = 52.0     #: seasonal period (weeks)
    sg_window: int = 11           #: Savitzky-Golay window
    sg_poly: int = 3              #: Savitzky-Golay polynomial order


@dataclass
class EstimatorConfig(ConfigBase):
    """S(t)-reconstruction and beta(t)-estimation settings."""

    s_recon: str = "cumulative"   #: 'cumulative' | 'ode' | 'ekf'
    local_methods: List[str] = field(default_factory=lambda: ["direct", "windowed"])
    global_basis: str = "time"    #: 'time' | 'none'
    mask_eps: float = 1e-4        #: near-zero-denominator guard (fraction of N)
    window_size: int = 8          #: windowed-SINDy window (weeks)
    window_step: int = 1          #: windowed-SINDy step
    str_thresh: float = 0.01      #: STRidge sparsity threshold
    str_alpha: float = 0.01       #: STRidge ridge penalty
    poly_degree: int = 3          #: global time-basis polynomial degree
    n_fourier: int = 5            #: global time-basis Fourier harmonics
    fourier_period: float = 52.0  #: global time-basis period (weeks)
    lasso_alpha: float = 1e-5     #: LASSO penalty (sparse_method='lasso')
    sparse_method: str = "stridge"  #: 'stridge' (numpy) | 'lasso' (needs sklearn)


@dataclass
class EpiConfig(ConfigBase):
    """Top-level configuration for the dengue beta(t) example."""

    name: str = "dengue_beta"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    estim: EstimatorConfig = field(default_factory=EstimatorConfig)
    output_dir: str = "outputs/epidemiology"
    seed: int = 42
