"""Shared, backend-light core utilities."""

from .config import ConfigBase, DomainConfig
from .derivatives import savgol, savgol_derivative
from .io import load_json, save_json
from .logging import get_logger
from .metrics import abs_error, rel_l2, rel_l2_batch
from .plotting import set_paper_style
from .seeding import seed_everything

__all__ = [
    "ConfigBase", "DomainConfig",
    "rel_l2", "abs_error", "rel_l2_batch",
    "set_paper_style", "seed_everything", "get_logger",
    "save_json", "load_json",
    "savgol", "savgol_derivative",
]
