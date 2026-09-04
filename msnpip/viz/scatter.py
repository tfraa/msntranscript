"""plot_demographic_correlation — scatter with fit line, r, p, n annotation."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

from msnpip.viz.theme import CONTROL_COLOR, configure_theme

logger = logging.getLogger("msnpip.viz.scatter")


def plot_demographic_correlation(
    corr_result,
    *,
    x=None,
    y=None,
    annotate: bool = True,
    color: str = CONTROL_COLOR,
    ax=None,
):
    """Scatter of node strength vs a demographic variable, with fit + 95% CI.

    Consumes a global-scope :class:`DemographicCorrelationResult` (which carries
    the underlying ``x_values``/``y_values``).  For results without stored data
    (e.g. regional scope), pass *x* and *y* explicitly.

    Raises
    ------
    ValueError
        If no raw data is available to plot.

    """
    configure_theme()

    xv = corr_result.x_values if x is None else x
    yv = corr_result.y_values if y is None else y
    if xv is None or yv is None:
        raise ValueError(
            "No raw data to scatter — global-scope results carry x_values/y_values; "
            "for regional results pass x= and y= explicitly."
        )
    xv = np.asarray(xv, dtype=float)
    yv = np.asarray(yv, dtype=float)
    mask = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[mask], yv[mask]
    if xv.size < 3:
        raise ValueError("Need at least 3 finite points to plot a correlation.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(4.6, 4.0))
    else:
        fig = ax.figure

    ax.scatter(xv, yv, s=26, color=color, alpha=0.8, edgecolors="none", zorder=3)

    # OLS fit + 95% mean-response confidence band.
    n = xv.size
    X = np.column_stack([np.ones(n), xv])
    beta, _, _, _ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ beta
    dof = n - 2
    s2 = float(resid @ resid) / dof
    x_mean = xv.mean()
    sxx = float(np.sum((xv - x_mean) ** 2))

    xs = np.linspace(xv.min(), xv.max(), 100)
    ys = beta[0] + beta[1] * xs
    se = np.sqrt(s2 * (1.0 / n + (xs - x_mean) ** 2 / sxx))
    tcrit = sp_stats.t.ppf(0.975, dof)
    band = tcrit * se

    ax.fill_between(xs, ys - band, ys + band, color=color, alpha=0.15, zorder=1)
    ax.plot(xs, ys, color=color, lw=1.8, zorder=2)

    ax.set_xlabel(getattr(corr_result, "variable", "variable"))
    ax.set_ylabel("global node strength")

    if annotate:
        r = float(np.atleast_1d(corr_result.r)[0])
        p = float(np.atleast_1d(corr_result.p)[0])
        method = getattr(corr_result, "method", "spearman").capitalize()
        lines = [f"r = {r:.2f} ({method})", f"p = {p:.3f} · n = {corr_result.n}"]
        if getattr(corr_result, "group", None):
            lines.append(f"within-group: {corr_result.group}")
        ax.text(
            0.97,
            0.97,
            "\n".join(lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", linewidth=0.8
            ),
        )
        logger.info(
            "plot_demographic_correlation: %s r=%.3f p=%.3f n=%d",
            getattr(corr_result, "variable", "?"),
            r,
            p,
            corr_result.n,
        )

    fig.tight_layout()
    return fig
