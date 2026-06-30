"""Shared, backend-light core utilities."""

from .config import ConfigBase, DomainConfig
from .metrics import rel_l2, abs_error, rel_l2_batch
from .plotting import set_paper_style
from .seeding import seed_everything
from .logging import get_logger
from .io import save_json, load_json
from .derivatives import savgol, savgol_derivative

__all__ = [
    "ConfigBase", "DomainConfig",
    "rel_l2", "abs_error", "rel_l2_batch",
    "set_paper_style", "seed_everything", "get_logger",
    "save_json", "load_json",
    "savgol", "savgol_derivative",
]
