"""
covariate_exclusion_contrast: full vs reduced contrast maps + Spearman rank-corr.
Phase 2, Task T2.6.

A robustness check: re-run the group contrast with a covariate (or several)
dropped and quantify how much the regional map moves.  A high Spearman rank
correlation between the full and reduced maps means the contrast is robust to
that covariate's inclusion.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from msnpip.stats.glm import GroupContrastResult, regional_group_contrast

logger = logging.getLogger("msnpip.stats.sensitivity")


@dataclass
class SensitivityResult:
    """Full vs reduced contrast maps and their rank agreement."""

    full: GroupContrastResult
    reduced: GroupContrastResult
    dropped: list[str]
    rank_corr: float           # Spearman r between full and reduced regional maps
    rank_corr_p: float
    stat_type: str
    region_labels: list[str] = field(default_factory=list)


def covariate_exclusion_contrast(
    strength_maps,
    df: pd.DataFrame,
    schema,
    *,
    case_label,
    control_label,
    full_covariates,
    drop,
    group_col: str | None = None,
    stat: str = "beta",
) -> SensitivityResult:
    """Compare a full-covariate contrast against one with covariates removed.

    Parameters
    ----------
    strength_maps
        :class:`msnpip.msn.construct.StrengthMaps`.
    df, schema
        Cohort DataFrame and column schema.
    case_label, control_label
        Group arm values.
    full_covariates
        The full covariate set.
    drop
        Covariate(s) to exclude in the reduced model — a single name or an
        iterable of names.
    group_col
        Group column; defaults to ``schema.group_col``.
    stat
        Contrast statistic (``"beta"`` / ``"t"`` / ``"cohen_d"``).

    Returns
    -------
    SensitivityResult
    """
    full_covariates = list(full_covariates)
    drop_list = [drop] if isinstance(drop, str) else list(drop)
    reduced_covariates = [c for c in full_covariates if c not in drop_list]

    common = dict(
        case_label=case_label,
        control_label=control_label,
        group_col=group_col,
        stat=stat,
    )
    full = regional_group_contrast(
        strength_maps, df, schema, covariates=full_covariates, **common
    )
    reduced = regional_group_contrast(
        strength_maps, df, schema, covariates=reduced_covariates, **common
    )

    a = full.regional_stat
    b = reduced.regional_stat
    valid = ~(np.isnan(a) | np.isnan(b))
    if valid.sum() >= 3:
        res = sp_stats.spearmanr(a[valid], b[valid])
        rank_corr, rank_corr_p = float(res.statistic), float(res.pvalue)
    else:
        rank_corr, rank_corr_p = np.nan, np.nan

    logger.info(
        "covariate_exclusion_contrast: dropped=%s stat=%s rank_corr=%.4f (p=%.3g)",
        drop_list, stat, rank_corr, rank_corr_p,
    )

    return SensitivityResult(
        full=full,
        reduced=reduced,
        dropped=drop_list,
        rank_corr=rank_corr,
        rank_corr_p=rank_corr_p,
        stat_type=stat,
        region_labels=list(strength_maps.region_labels),
    )
