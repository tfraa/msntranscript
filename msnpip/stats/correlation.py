"""
correlate_strength_with_demographic — Spearman, within-group, no spatial null.
Phase 2, Task T2.5.

This is the *behavioural*/demographic correlation (Layer 0): does node strength
track a continuous variable such as age?  It is engine-independent and uses no
spatial null — significance is the ordinary correlation p-value, with
Benjamini–Hochberg FDR across regions for the regional scope.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.stats.multitest import multipletests

from msnpip.errors import SchemaError

logger = logging.getLogger("msnpip.stats.correlation")


@dataclass
class DemographicCorrelationResult:
    """Correlation between node strength and a demographic variable."""

    variable: str
    scope: str                 # "global" | "regional"
    method: str                # "spearman" | "pearson"
    r: np.ndarray              # scalar-in-array (global) or (n_regions,)
    p: np.ndarray              # matching shape
    n: int
    fdr: np.ndarray | None = None      # regional only (BH across regions)
    region_labels: list[str] | None = None
    group: str | None = None           # group value if within_group used
    x_values: np.ndarray | None = None  # variable values (global scope) — for scatter plot
    y_values: np.ndarray | None = None  # global_strength values (global scope)


def _corr(x: np.ndarray, y: np.ndarray, method: str) -> tuple[float, float]:
    """Return (r, p) for the chosen method, NaN-safe on degenerate input."""
    mask = ~(np.isnan(x) | np.isnan(y))
    xv, yv = x[mask], y[mask]
    if xv.size < 3 or np.all(xv == xv[0]) or np.all(yv == yv[0]):
        return np.nan, np.nan
    if method == "spearman":
        res = sp_stats.spearmanr(xv, yv)
        return float(res.statistic), float(res.pvalue)
    if method == "pearson":
        res = sp_stats.pearsonr(xv, yv)
        return float(res.statistic), float(res.pvalue)
    raise ValueError(f"method must be 'spearman'/'pearson', got {method!r}")


def correlate_strength_with_demographic(
    strength_maps,
    df: pd.DataFrame,
    schema,
    *,
    variable: str,
    scope: str = "regional",
    within_group=None,
    group_col: str | None = None,
    method: str = "spearman",
) -> DemographicCorrelationResult:
    """Correlate node strength with a demographic variable.

    Parameters
    ----------
    strength_maps
        :class:`msnpip.msn.construct.StrengthMaps`.
    df
        DataFrame containing *variable* (and the group column if filtering).
        Rows are aligned to ``strength_maps.subject_ids`` by ``schema.id_col``.
    schema
        Column schema.
    variable
        Numeric demographic column to correlate against.
    scope
        ``"global"`` → correlate ``global_strength`` with *variable*;
        ``"regional"`` → per-region correlation + BH FDR across regions.
    within_group
        If given, restrict to subjects whose group column equals this value.
    group_col
        Group column for ``within_group``; defaults to ``schema.group_col``.
    method
        ``"spearman"`` (default) or ``"pearson"``.

    Returns
    -------
    DemographicCorrelationResult

    Raises
    ------
    SchemaError
        If *variable* is missing or non-numeric, or no subjects remain.
    """
    if scope not in ("global", "regional"):
        raise ValueError(f"scope must be 'global'/'regional', got {scope!r}")
    if variable not in df.columns:
        raise SchemaError(f"Correlation variable {variable!r} not found in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[variable]):
        raise SchemaError(
            f"Correlation variable {variable!r} is not numeric "
            f"(dtype={df[variable].dtype})."
        )

    id_col = schema.id_col
    df_idx = df.set_index(df[id_col].astype(str))
    try:
        aligned = df_idx.loc[strength_maps.subject_ids]
    except KeyError as exc:
        raise SchemaError(
            "Some strength-map subject IDs are absent from the DataFrame."
        ) from exc

    mask = np.ones(len(aligned), dtype=bool)
    group = None
    if within_group is not None:
        group_col = group_col or getattr(schema, "group_col", None)
        if group_col is None or group_col not in aligned.columns:
            raise SchemaError(
                f"within_group requested but group column {group_col!r} not found."
            )
        mask = (aligned[group_col] == within_group).to_numpy()
        group = str(within_group)
        if mask.sum() == 0:
            raise SchemaError(
                f"No subjects in group {within_group!r} (column {group_col!r})."
            )

    var_vals = pd.to_numeric(aligned[variable], errors="coerce").to_numpy()[mask]

    if scope == "global":
        gs = strength_maps.global_strength[mask]
        r, p = _corr(gs, var_vals, method)
        result = DemographicCorrelationResult(
            variable=variable,
            scope="global",
            method=method,
            r=np.array([r]),
            p=np.array([p]),
            n=int(np.sum(~np.isnan(gs) & ~np.isnan(var_vals))),
            group=group,
            x_values=var_vals,
            y_values=gs,
        )
    else:
        strength = strength_maps.strength[mask]
        n_regions = strength.shape[1]
        r_arr = np.full(n_regions, np.nan)
        p_arr = np.full(n_regions, np.nan)
        for reg in range(n_regions):
            r_arr[reg], p_arr[reg] = _corr(strength[:, reg], var_vals, method)

        fdr = np.full(n_regions, np.nan)
        valid = ~np.isnan(p_arr)
        if valid.any():
            fdr[valid] = multipletests(p_arr[valid], method="fdr_bh")[1]

        result = DemographicCorrelationResult(
            variable=variable,
            scope="regional",
            method=method,
            r=r_arr,
            p=p_arr,
            n=int(mask.sum()),
            fdr=fdr,
            region_labels=list(strength_maps.region_labels),
            group=group,
        )

    logger.info(
        "correlate_strength_with_demographic: var=%s scope=%s method=%s n=%d group=%s",
        variable, scope, method, result.n, group,
    )
    return result
