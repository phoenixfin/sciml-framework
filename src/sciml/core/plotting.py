"""Matplotlib styling helpers."""

from __future__ import annotations


def set_paper_style(font_size: int = 8) -> None:
    """Apply the compact serif paper style used across the examples.

    Parameters
    ----------
    font_size : int
        Base font size; tick/legend sizes are derived from it.

    Returns
    -------
    None
    """
    import matplotlib
    matplotlib.rcParams.update({
        "font.family": "serif", "font.size": font_size,
        "axes.labelsize": font_size, "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1, "legend.fontsize": font_size - 1,
        "figure.dpi": 150, "text.usetex": False,
        "axes.spines.top": False, "axes.spines.right": False,
    })
