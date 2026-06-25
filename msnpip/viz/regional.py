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
_INK = "#1f2933"
_MUTED = "#555555"
_HEMI = (("lh", "Left Hemisphere"), ("rh", "Right Hemisphere"))


def _split_hemispheres(region_labels, values, sig):
    """Group ``{hemi: {aparc_label: (value, sig)}}`` and return shared label order.

    Region labels use the ``{lh|rh}_{aparc_label}`` format.  The aparc labels are
    returned in their default alphabetical order (shared by both panels) so the
    bars read like an anatomical table rather than a value ranking.
    """
    data: dict[str, dict[str, tuple[float, float]]] = {"lh": {}, "rh": {}}
    for lab, v, s in zip(region_labels, values, sig):
        hemi, _, name = str(lab).partition("_")
        data.setdefault(hemi, {})[name] = (float(v), float(s))
    names = sorted({n for hemi in data.values() for n in hemi})
    return data, names


def plot_hemisphere_bars(
    regional_stat,
    region_labels,
    *,
    value_label: str,
    title: str,
    output_path,
    subtitle: str | None = None,
    color_mode: str = "sign",
    significance=None,
    alpha: float = 0.05,
    sig_label: str = "FDR",
    cmap: str = "viridis",
):
    """Two-panel horizontal bar chart split by hemisphere (left | right).

    All regions are shown in their default alphabetical order (NOT sorted by
    value), with the same row order in both panels so the two hemispheres line
    up region-by-region — the layout used in the thesis figures.

    ``color_mode``:

    - ``"sign"`` (contrasts): red = higher in case, blue = higher in control.
      When *significance* (a per-region p/FDR array) is given, regions with
      ``< alpha`` are drawn at full opacity and annotated with APA stars
      (``*`` ``**`` ``***``); the rest are faded.  The x-axis is symmetric.
    - ``"sequential"`` (node strength): bars coloured by magnitude on a
      sequential colormap (``viridis``) with a shared colourbar.
    """
    configure_theme()
    values = np.asarray(regional_stat, dtype=float)
    sig = (
        np.asarray(significance, dtype=float)
        if significance is not None
        else np.full(len(values), np.nan)
    )
    data, names = _split_hemispheres(region_labels, values, sig)
    n = len(names)
    y = np.arange(n)

    finite = values[np.isfinite(values)]
    norm = cmap_obj = None
    if color_mode == "sign":
        xmax = (float(np.max(np.abs(finite))) if finite.size else 1.0) or 1.0
        xlim = (-1.15 * xmax, 1.15 * xmax)
    else:
        vmin = float(finite.min()) if finite.size else 0.0
        vmax = float(finite.max()) if finite.size else 1.0
        if vmax <= vmin:
            vmax = vmin + 1.0
        pad = 0.05 * (vmax - vmin)
        # Truncate to the data range so bar lengths show between-region
        # differences (node strength is bounded well away from 0); the colourbar
        # still reports absolute values.
        xlim = (vmin - pad, vmax + pad)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = plt.get_cmap(cmap)

    height = max(4.0, 0.22 * n + 1.8)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, height), sharey=True)
    for ax, (hemi, htitle) in zip(axes, _HEMI):
        cells = data.get(hemi, {})
        vals = np.array([cells.get(nm, (np.nan, np.nan))[0] for nm in names])
        ssig = np.array([cells.get(nm, (np.nan, np.nan))[1] for nm in names])
        is_sig = np.isfinite(ssig) & (ssig < alpha)
        have_sig = bool(np.isfinite(ssig).any())

        if color_mode == "sign":
            for yi, v in zip(y, vals):
                if not np.isfinite(v):
                    continue
                opacity = 1.0 if (is_sig[yi] or not have_sig) else 0.4
                ax.barh(yi, v, color=_POS if v >= 0 else _NEG, alpha=opacity, edgecolor="none")
            if have_sig:
                pad = 0.02 * xmax
                for yi, v in zip(y, vals):
                    if not (np.isfinite(v) and is_sig[yi]):
                        continue
                    stars = significance_stars(float(ssig[yi]))
                    if stars == "ns":
                        continue
                    ha, dx = ("left", pad) if v >= 0 else ("right", -pad)
                    ax.text(v + dx, yi, stars, va="center", ha=ha, fontsize=8, color="#222222")
        else:
            colors = [
                cmap_obj(norm(v)) if np.isfinite(v) else (0.85, 0.85, 0.85, 1.0) for v in vals
            ]
            ax.barh(y, np.nan_to_num(vals), color=colors, edgecolor="none")

        ax.axvline(0.0, color="#444444", linewidth=0.8)
        ax.set_xlim(*xlim)
        ax.set_xlabel(value_label)
        ax.set_title(htitle, fontsize=11, color="#334155")
        ax.grid(axis="x", color="#e3e8ef", linewidth=0.7)
        ax.set_axisbelow(True)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontsize=7)
    axes[0].set_ylim(-1, n)
    axes[0].invert_yaxis()  # alphabetical, top-to-bottom

    sub = subtitle
    if color_mode == "sign" and np.isfinite(sig).any():
        star_note = f"* {sig_label} < {alpha}   ** < 0.01   *** < 0.001"
        sub = f"{subtitle}\n{star_note}" if subtitle else star_note
    n_sub = (sub.count("\n") + 1) if sub else 0
    headroom = 0.55 + 0.18 * n_sub + 0.2
    top_frac = max(0.5, 1.0 - headroom / height)
    right = 0.9 if color_mode != "sign" else 1.0
    fig.tight_layout(rect=(0.0, 0.0, right, top_frac))

    if color_mode != "sign":
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        cbar_ax = fig.add_axes([0.92, 0.18, 0.015, 0.55])
        fig.colorbar(sm, cax=cbar_ax).set_label(value_label, fontsize=9)

    fig.text(0.06, 0.985, title, va="top", ha="left", fontsize=14, fontweight="bold", color=_INK)
    if sub:
        fig.text(0.06, 0.985 - 0.42 / height, sub, va="top", ha="left", fontsize=9, color=_MUTED)

    fig.savefig(output_path)
    plt.close(fig)
    logger.info("plot_hemisphere_bars: wrote %s (%d regions, mode=%s)", output_path, n, color_mode)
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
