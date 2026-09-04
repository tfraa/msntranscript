"""plot_strength_violin — violin + box + jitter of node strength per group."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from msnpip.stats.glm import normalize_group_value
from msnpip.viz.theme import configure_theme, format_p, group_colors, significance_stars

logger = logging.getLogger("msnpip.viz.distributions")


def _two_group_p(a: np.ndarray, b: np.ndarray, test: str) -> float:
    """Two-sided p-value between two groups."""
    if test == "mannwhitney":
        return float(sp_stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    if test == "ttest":
        return float(sp_stats.ttest_ind(a, b, equal_var=False).pvalue)
    raise ValueError(f"test must be 'mannwhitney'/'ttest', got {test!r}")


def _draw_sig_bracket(
    ax, x1: float, x2: float, data: list[np.ndarray], p: float, label_prefix: str = "p"
) -> None:
    """Draw a significance bracket spanning two violins with exact p + stars."""
    top = max(float(np.max(v)) for v in data)
    span = top - min(float(np.min(v)) for v in data)
    pad = 0.06 * span if span > 0 else 0.05
    y = top + pad
    h = 0.4 * pad
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.0, color="#444444")
    # format_p renders 'p = ...'; relabel the leading 'p' when a different
    # statistic is shown (e.g. 'FDR = ...').
    p_text = format_p(p).replace("p", label_prefix, 1) if label_prefix != "p" else format_p(p)
    label = f"{p_text} {significance_stars(p)}".strip()
    ax.text((x1 + x2) / 2, y + h, label, ha="center", va="bottom", fontsize=10)
    ax.set_ylim(top=y + h + 2.5 * pad)


def plot_strength_violin(
    strength_maps,
    df: pd.DataFrame,
    schema,
    *,
    region: str | None = None,
    group_col: str | None = None,
    group_labels=None,
    test: str = "mannwhitney",
    pvalue: float | None = None,
    pvalue_label: str = "p",
    ax=None,
):
    """Violin + box + jittered points of node strength, grouped by diagnosis.

    Raises
    ------
    ValueError
        If the group column is missing or a requested group is empty.

    """
    configure_theme()

    if region is None:
        values = np.asarray(strength_maps.global_strength, dtype=float)
        ylabel = "global node strength"
    else:
        labels = list(strength_maps.region_labels)
        if region not in labels:
            raise ValueError(f"region {region!r} not in region_labels")
        values = np.asarray(strength_maps.strength[:, labels.index(region)], dtype=float)
        ylabel = f"node strength · {region}"

    gcol = group_col or getattr(schema, "group_col", None)
    if gcol is None or gcol not in df.columns:
        raise ValueError(f"group column {gcol!r} not found")

    aligned = df.set_index(df[schema.id_col].astype(str)).loc[strength_maps.subject_ids]
    # Normalize group values so 1 / 1.0 / "1" match the requested labels (issue 5).
    groups = aligned[gcol].map(normalize_group_value).to_numpy()

    order = (
        list(pd.unique(groups))
        if group_labels is None
        else [normalize_group_value(g) for g in group_labels]
    )

    data: list[np.ndarray] = []
    for g in order:
        v = values[groups == g]
        v = v[np.isfinite(v)]
        if v.size == 0:
            raise ValueError(f"group {g!r} has no subjects with finite strength")
        data.append(v)

    if ax is None:
        fig, ax = plt.subplots(figsize=(1.8 + 1.4 * len(order), 4.0))
    else:
        fig = ax.figure

    positions = np.arange(1, len(order) + 1)
    colors = group_colors(order)

    parts = ax.violinplot(data, positions=positions, showextrema=False)
    for body, g in zip(parts["bodies"], order):
        body.set_facecolor(colors[g])
        body.set_alpha(0.25)
        body.set_edgecolor(colors[g])
        body.set_linewidth(1.2)

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.12,
        showfliers=False,
        patch_artist=True,
        medianprops=dict(color="#222222", linewidth=1.4),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("white")
        patch.set_alpha(0.85)
        patch.set_edgecolor("#444444")

    rng = np.random.default_rng(0)
    for i, (v, g) in enumerate(zip(data, order)):
        jitter = rng.uniform(-0.07, 0.07, size=v.size)
        ax.scatter(
            positions[i] + jitter, v, s=16, color=colors[g], alpha=0.8, edgecolors="none", zorder=3
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{g}\n(n={len(v)})" for g, v in zip(order, data)])
    ax.set_ylabel(ylabel)

    if len(order) == 2:
        # Use an externally supplied p (e.g. the covariate-adjusted GLM/FDR value)
        # when given, so the bracket matches the reported inference; otherwise fall
        # back to a descriptive two-group test on the raw strengths.
        p = pvalue if pvalue is not None else _two_group_p(data[0], data[1], test)
        _draw_sig_bracket(ax, positions[0], positions[1], data, p, label_prefix=pvalue_label)
        logger.info("plot_strength_violin: %s vs %s, %s", order[0], order[1], format_p(p))

    fig.tight_layout()
    return fig
