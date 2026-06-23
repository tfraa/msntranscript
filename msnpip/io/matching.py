"""
ID matching and subject-level merge with match-rate threshold.
Phase 1, Task T1.5.
"""
from __future__ import annotations

import logging

import pandas as pd

from msnpip.errors import IDMatchError

logger = logging.getLogger("msnpip.io.matching")


# ---------------------------------------------------------------------------
# T1.5 — normalize_ids
# ---------------------------------------------------------------------------

def normalize_ids(ids: pd.Series) -> pd.Series:
    """Strip invisible leading/trailing whitespace from subject IDs.

    This is the **only** transformation applied.  IDs are otherwise kept
    exactly as they appear in the file — ``sub-001`` stays ``sub-001`` and
    will only match another ``sub-001``, not ``sub-1`` or ``SUB-001``.

    Parameters
    ----------
    ids
        Series of raw ID strings.

    Returns
    -------
    pd.Series
        IDs with whitespace stripped, same index as input.
    """
    return ids.astype(str).str.strip()


# ---------------------------------------------------------------------------
# T1.5 — merge_features_demographics
# ---------------------------------------------------------------------------

def merge_features_demographics(
    features: pd.DataFrame,
    demographics: pd.DataFrame,
    *,
    feat_id_col: str,
    dem_id_col: str,
    min_match_rate: float = 0.95,
) -> pd.DataFrame:
    """Inner-join *features* and *demographics* on subject IDs (exact match).

    IDs are matched exactly after stripping invisible whitespace.  No other
    transformation is applied — ``sub-001`` and ``sub-1`` are treated as
    distinct IDs and will not match.

    Parameters
    ----------
    features
        Wide-format morphometric feature DataFrame (from
        ``read_freesurfer_subjects``).
    demographics
        Demographics DataFrame (from ``read_table``).
    feat_id_col
        Name of the ID column in *features*.
    dem_id_col
        Name of the ID column in *demographics*.
    min_match_rate
        Fraction of feature subjects that must have a matching demographic
        row.  Raises ``IDMatchError`` if the match rate falls below this.
        Computed as ``n_matched / n_features``.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame (inner join).  IDs are preserved exactly as they
        appear in *features*.

    Raises
    ------
    IDMatchError
        If ``n_matched / n_features < min_match_rate``.
    """
    if feat_id_col not in features.columns:
        raise ValueError(f"Feature ID column '{feat_id_col}' not in features DataFrame.")
    if dem_id_col not in demographics.columns:
        raise ValueError(f"Demographics ID column '{dem_id_col}' not in demographics DataFrame.")

    # Build normalized-ID lookup columns (do not modify originals)
    feat_norm = normalize_ids(features[feat_id_col])
    dem_norm = normalize_ids(demographics[dem_id_col])

    feat_work = features.copy()
    dem_work = demographics.copy()

    _NORM_KEY = "__norm_id__"
    feat_work[_NORM_KEY] = feat_norm.values
    dem_work[_NORM_KEY] = dem_norm.values

    merged = feat_work.merge(dem_work, on=_NORM_KEY, how="inner", suffixes=("", "_dem"))
    merged = merged.drop(columns=[_NORM_KEY])

    # Drop duplicate ID column from demographics if present
    if dem_id_col in merged.columns and dem_id_col != feat_id_col:
        merged = merged.drop(columns=[dem_id_col])

    n_feat = len(features)
    n_matched = len(merged)
    match_rate = n_matched / n_feat if n_feat > 0 else 0.0

    unmatched_feat = sorted(
        set(feat_norm.tolist()) - set(dem_norm.tolist())
    )
    unmatched_dem = sorted(
        set(dem_norm.tolist()) - set(feat_norm.tolist())
    )

    logger.info(
        "merge: %d features × %d demographics → %d matched (%.1f%%). "
        "Unmatched in features: %d; in demographics: %d.",
        n_feat, len(demographics), n_matched, match_rate * 100,
        len(unmatched_feat), len(unmatched_dem),
    )

    if unmatched_feat:
        logger.warning("IDs in features not found in demographics: %s", unmatched_feat[:10])
    if unmatched_dem:
        logger.warning("IDs in demographics not found in features: %s", unmatched_dem[:10])

    if match_rate < min_match_rate:
        raise IDMatchError(
            f"ID match rate {match_rate:.1%} < threshold {min_match_rate:.1%}. "
            f"{len(unmatched_feat)} feature subject(s) have no demographic row. "
            "Check that both files use compatible subject IDs.",
            unmatched=unmatched_feat,
        )

    return merged
