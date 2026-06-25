"""
Regional plots: per-region contrast bar charts and group mean similarity matrices.
Phase 4 (added per user request).
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np

from msnpip.viz.theme import configure_theme

logger = logging.getLogger("msnpip.viz.regional")

_POS = "#b2182b"  # RdBu_r extremes
_NEG = "#2166ac"


def plot_contrast_bars(
    regional_stat,
    region_labels,
    *,
    stat_type: str,
    title: str,
    output_path,
    subtitle: str | None = None,
):
    """Horizontal bar chart of the per-region case-vs-control contrast statistic.

    Bars are sorted by value and coloured by sign (red positive, blue negative).
    The statistic shown is whatever the contrast used (``--contrast-stat``); pass
    ``t`` for t-value bars.
    """
    configure_theme()
    values = np.asarray(regional_stat, dtype=float)
    labels = list(region_labels)
    order = np.argsort(np.nan_to_num(values))
    values, labels = values[order], [labels[i] for i in order]

    height = max(3.0, 0.16 * len(labels) + 1.0)
    fig, ax = plt.subplots(figsize=(7.0, height))
    colors = [_POS if v >= 0 else _NEG for v in values]
    ax.barh(range(len(values)), values, color=colors, edgecolor="none")
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.axvline(0.0, color="#444444", linewidth=0.8)
    ax.set_xlabel(f"contrast {stat_type}")
    ax.set_ylim(-1, len(values))
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    if subtitle:
        ax.text(
            0.0, 1.005, subtitle, transform=ax.transAxes, fontsize=8.5, va="bottom", color="#555555"
        )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("plot_contrast_bars: wrote %s (%d regions)", output_path, len(values))
    return output_path


def plot_msn_matrix(
    matrix,
    region_labels,
    *,
    title: str,
    output_path,
    subtitle: str | None = None,
):
    """Heatmap of a region×region morphometric similarity matrix (NaN diagonal)."""
    configure_theme()
    mat = np.asarray(matrix, dtype=float)
    labels = list(region_labels)

    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    im = ax.imshow(mat, cmap="magma", aspect="equal", interpolation="nearest")
    # Sparse ticks to keep ~68 labels legible.
    step = max(1, len(labels) // 20)
    idx = list(range(0, len(labels), step))
    ax.set_xticks(idx)
    ax.set_xticklabels([labels[i] for i in idx], rotation=90, fontsize=5.5)
    ax.set_yticks(idx)
    ax.set_yticklabels([labels[i] for i in idx], fontsize=5.5)
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    if subtitle:
        ax.text(
            0.0, 1.01, subtitle, transform=ax.transAxes, fontsize=8.5, va="bottom", color="#555555"
        )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("similarity", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("plot_msn_matrix: wrote %s (%dx%d)", output_path, mat.shape[0], mat.shape[1])
    return output_path
