"""
Regional plots: per-region contrast bar charts and group mean similarity matrices.
Phase 4 (added per user request).
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np

from msnpip.viz.theme import configure_theme, significance_stars

logger = logging.getLogger("msnpip.viz.regional")

_POS = "#b2182b"  # RdBu_r extremes
_NEG = "#2166ac"


def plot_contrast_bars(
    regional_stat,
    region_labels,
    *,
    stat_type: str = "t",
    title: str,
    output_path,
    subtitle: str | None = None,
    significance=None,
    alpha: float = 0.05,
    sig_label: str = "FDR",
):
    """Horizontal bar chart of the per-region case-vs-control contrast statistic.

    Bars are sorted by value and coloured by sign (red = higher in case, blue =
    higher in control).  When *significance* (a per-region p/FDR array aligned to
    *region_labels*) is given, regions surviving ``< alpha`` are drawn at full
    opacity and annotated with APA significance stars (``*`` ``**`` ``***``);
    non-significant regions are faded.  The statistic is typically the t-value
    (``stat_type="t"``).
    """
    configure_theme()
    values = np.asarray(regional_stat, dtype=float)
    labels = list(region_labels)
    sig = (
        np.asarray(significance, dtype=float)
        if significance is not None
        else np.full(len(values), np.nan)
    )

    order = np.argsort(np.nan_to_num(values))
    values = values[order]
    labels = [labels[i] for i in order]
    sig = sig[order]
    is_sig = np.isfinite(sig) & (sig < alpha)

    height = max(3.0, 0.18 * len(labels) + 1.2)
    fig, ax = plt.subplots(figsize=(7.6, height))
    have_sig = bool(np.isfinite(sig).any())
    for i, v in enumerate(values):
        color = _POS if v >= 0 else _NEG
        # Fade non-significant bars only when we actually have significance info.
        opacity = 1.0 if (is_sig[i] or not have_sig) else 0.38
        ax.barh(i, v, color=color, edgecolor="none", alpha=opacity)

    # Asterisks just beyond the tip of each significant bar.
    if have_sig:
        span = float(np.nanmax(np.abs(values))) or 1.0
        pad = 0.012 * span
        for i, v in enumerate(values):
            if not is_sig[i]:
                continue
            stars = significance_stars(float(sig[i]))
            if stars == "ns":
                continue
            if v >= 0:
                ax.text(v + pad, i, stars, va="center", ha="left", fontsize=9, color="#222222")
            else:
                ax.text(v - pad, i, stars, va="center", ha="right", fontsize=9, color="#222222")

    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.axvline(0.0, color="#444444", linewidth=0.8)
    ax.set_xlabel(f"node-strength contrast ({stat_type})")
    ax.set_ylim(-1, len(values))
    # Widen x so the stars are not clipped.
    if have_sig and np.isfinite(values).any():
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
        margin = 0.10 * (max(abs(lo), abs(hi)) or 1.0)
        ax.set_xlim(min(0.0, lo) - margin, max(0.0, hi) + margin)
    sub = subtitle
    if have_sig:
        star_note = f"* {sig_label} < {alpha}   ** < 0.01   *** < 0.001"
        sub = f"{subtitle}\n{star_note}" if subtitle else star_note
    n_sub = (sub.count("\n") + 1) if sub else 0
    # Reserve headroom (inches → figure fraction) so the title/subtitle never
    # overlap the top bars, independent of region count.
    headroom = 0.34 + 0.18 * n_sub + 0.16
    top_frac = max(0.5, 1.0 - headroom / height)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, top_frac))
    pos = ax.get_position()
    fig.text(
        pos.x0, 0.985, title, va="top", ha="left", fontsize=12.5, fontweight="bold", color="#1f2933"
    )
    if sub:
        fig.text(
            pos.x0, 0.985 - 0.34 / height, sub, va="top", ha="left", fontsize=8.5, color="#555555"
        )
    fig.savefig(output_path)
    plt.close(fig)
    logger.info(
        "plot_contrast_bars: wrote %s (%d regions, %d significant)",
        output_path,
        len(values),
        int(is_sig.sum()),
    )
    return output_path


def plot_strength_bars(
    values,
    region_labels,
    *,
    title: str,
    output_path,
    subtitle: str | None = None,
    cmap: str = "viridis",
):
    """Horizontal bar chart of per-region mean node strength.

    Bars are sorted by value and coloured by magnitude on a sequential
    (``viridis``) colormap — a within-group view of which regions carry the
    strongest morphometric similarity hubs.
    """
    configure_theme()
    values = np.asarray(values, dtype=float)
    labels = list(region_labels)
    order = np.argsort(np.nan_to_num(values))
    values, labels = values[order], [labels[i] for i in order]

    finite = values[np.isfinite(values)]
    vmin = float(finite.min()) if finite.size else 0.0
    vmax = float(finite.max()) if finite.size else 1.0
    norm = plt.Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1.0)
    cmap_obj = plt.get_cmap(cmap)
    colors = [cmap_obj(norm(v)) for v in np.nan_to_num(values, nan=vmin)]

    height = max(3.0, 0.16 * len(labels) + 1.0)
    fig, ax = plt.subplots(figsize=(7.0, height))
    ax.barh(range(len(values)), values, color=colors, edgecolor="none")
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("mean node strength")
    ax.set_ylim(-1, len(values))
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    if subtitle:
        ax.text(
            0.0, 1.005, subtitle, transform=ax.transAxes, fontsize=8.5, va="bottom", color="#555555"
        )
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("mean node strength", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("plot_strength_bars: wrote %s (%d regions)", output_path, len(values))
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
    im = ax.imshow(mat, cmap="viridis", aspect="equal", interpolation="nearest")
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
