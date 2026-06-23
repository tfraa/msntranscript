"""
Shared matplotlib theme for msnpip figures.
Phase 4, Task T4.1.

Group colours use the Okabe–Ito colourblind-safe palette: case = orange,
control = blue.  These set the look of every figure msnpip draws itself (the
engine draws its own transcriptomics plots with its own style).
"""
from __future__ import annotations

import matplotlib as mpl

# Okabe–Ito colourblind-safe qualitative palette.
OKABE_ITO: tuple[str, ...] = (
    "#E69F00",  # orange   — case (default group 1)
    "#0072B2",  # blue     — control (default group 2)
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#CC79A7",  # purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
)
CASE_COLOR = OKABE_ITO[0]
CONTROL_COLOR = OKABE_ITO[1]


def configure_theme() -> None:
    """Apply msnpip's matplotlib rcParams (idempotent)."""
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
        }
    )


def group_colors(labels) -> dict:
    """Map group labels to Okabe–Ito colours in order (group 1 = orange, …)."""
    return {str(g): OKABE_ITO[i % len(OKABE_ITO)] for i, g in enumerate(labels)}


def significance_stars(p: float) -> str:
    """APA-style significance stars; 'ns' when not significant or undefined."""
    if p is None or not (p == p):  # NaN
        return "ns"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def format_p(p: float) -> str:
    """Compact p-value label, e.g. 'p = 0.013' or 'p < 0.001'."""
    if p is None or not (p == p):
        return "p = n/a"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"
