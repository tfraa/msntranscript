"""MSN construction: per-subject similarity matrices and node-strength maps.

Each metric is normalised across regions within the subject with a modified z-score
``M = 0.6745*(x - median)/MAD``.  Edges are then either the Euclidean distance over
the normalised metrics converted to a similarity ``S = 1/(1 + d/n_metrics)``, bounded
(0, 1] (Tomasella et al.), or the Pearson correlation between the two regions'
metric vectors.  Node strength is the mean (default) or sum of a region's edges.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from msnpip.errors import MSNInputError

# 0.6745 = 0.75 quantile of the standard normal.
_MAD_CONSTANT = 0.6745

logger = logging.getLogger("msnpip.msn.construct")

# Must match MSNConfig defaults and the synthetic fixture.
DEFAULT_METRICS: tuple[str, ...] = ("SurfArea", "GrayVol", "ThickAvg", "MeanCurv", "GausCurv")

_HEMI_FOR: dict[str, tuple[str, ...]] = {
    "left": ("lh",),
    "right": ("rh",),
    "both": ("lh", "rh"),
}


def _modified_zscore(features: np.ndarray) -> np.ndarray:
    """Robust within-subject normalization per metric: 0.6745·(x − median)/MAD.

    Zero-MAD columns contribute 0 rather than inf/NaN.
    """
    median = np.nanmedian(features, axis=0)
    mad = np.nanmedian(np.abs(features - median), axis=0)
    out = np.zeros_like(features, dtype=float)
    nonzero = mad > 0
    out[:, nonzero] = _MAD_CONSTANT * (features[:, nonzero] - median[nonzero]) / mad[nonzero]
    return out


def _build_one(features: np.ndarray, similarity: str = "distance") -> np.ndarray:
    """Build one subject's region×region morphometric similarity matrix.

    ``"distance"`` gives ``1/(1 + d/n_metrics)`` ∈ (0, 1]; ``"correlation"`` gives the
    Pearson correlation ∈ [-1, 1], which can be negative.  Diagonal is NaN.
    """
    all_nan = np.all(np.isnan(features), axis=1)
    if all_nan.any():
        bad = np.flatnonzero(all_nan).tolist()
        raise MSNInputError(
            f"{all_nan.sum()} region(s) have all-NaN features (rows {bad[:10]}). "
            "MSN construction never imputes — drop incomplete subjects upstream "
            "(see compute_strength_maps drop_threshold)."
        )

    n_metrics = features.shape[1]
    normalized = _modified_zscore(features)
    if similarity == "correlation":
        sim = np.corrcoef(normalized)
    elif similarity == "distance":
        distance = cdist(normalized, normalized, metric="euclidean")
        sim = 1.0 / (1.0 + distance / float(n_metrics))
    else:
        raise MSNInputError(
            f"unknown similarity {similarity!r}; expected 'distance' or 'correlation'."
        )
    sim = np.atleast_2d(sim)
    np.fill_diagonal(sim, np.nan)
    return sim


def build_msn(subject_features: np.ndarray, similarity: str = "distance") -> np.ndarray:
    """Construct per-subject MSNs from ``(n_subjects, n_regions, n_metrics)``.

    A single subject may be passed as 2-D, in which case a 2-D matrix is returned.
    Raises ``MSNInputError`` if a region is all-NaN or a metric is constant.
    """
    arr = np.asarray(subject_features, dtype=float)
    single = arr.ndim == 2
    if single:
        arr = arr[None]
    if arr.ndim != 3:
        raise MSNInputError(f"subject_features must be 2-D or 3-D, got shape {arr.shape}.")

    n_subjects, n_regions, _ = arr.shape
    out = np.empty((n_subjects, n_regions, n_regions), dtype=float)
    for s in range(n_subjects):
        out[s] = _build_one(arr[s], similarity=similarity)

    return out[0] if single else out


def node_strength(msn: np.ndarray, *, agg: str = "mean") -> np.ndarray:
    """Per-region node strength: the aggregate of a region's edges, excluding the NaN
    diagonal.

    ``agg`` is ``"mean"`` (default) or ``"sum"``; on a complete network they differ
    only by a constant scale.
    """
    if agg not in ("sum", "mean"):
        raise ValueError(f"agg must be 'sum'/'mean', got {agg!r}")

    arr = np.asarray(msn, dtype=float)
    single = arr.ndim == 2
    if single:
        arr = arr[None]

    out = np.nansum(arr, axis=2) if agg == "sum" else np.nanmean(arr, axis=2)
    return out[0] if single else out


@dataclass
class StrengthMaps:
    """Per-subject MSN matrices and node-strength maps.

    ``region_labels`` use ``"{hemi}_{aparc_label}"`` so the strength vectors feed
    :func:`msnpip.atlas_align.align_strength_to_atlas` directly.
    """

    matrix: np.ndarray  # (n_subjects, n_regions, n_regions)
    strength: np.ndarray  # (n_subjects, n_regions)
    subject_ids: list[str]
    region_labels: list[str]  # ["lh_bankssts", ...]
    atlas: str
    features: list[str]  # metric names, in column order
    global_strength: np.ndarray  # (n_subjects,) — mean over regions
    hemisphere: str = "both"
    regions: str = "cort"
    agg: str = "sum"
    dropped_subjects: list[str] = field(default_factory=list)

    @property
    def n_subjects(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_regions(self) -> int:
        return self.matrix.shape[1]


def _parse_feature_col(col: str) -> tuple[str, str, str] | None:
    """Split ``"{hemi}_{region}_{metric}"`` → (hemi, region, metric).

    First token = hemi, last = metric, middle = region, so multi-word region names
    are safe.  ``None`` if the column is not a hemi-prefixed feature.
    """
    parts = col.split("_")
    if len(parts) < 3 or parts[0] not in ("lh", "rh"):
        return None
    hemi = parts[0]
    metric = parts[-1]
    region = "_".join(parts[1:-1])
    return hemi, region, metric


def compute_strength_maps(
    df: pd.DataFrame,
    schema,
    *,
    atlas: str = "dk",
    hemisphere: str = "both",
    regions: str = "cort",
    drop_threshold: float = 0.0,
    agg: str = "mean",
    similarity: str = "distance",
    metrics: tuple[str, ...] = DEFAULT_METRICS,
) -> StrengthMaps:
    """Build MSNs and node-strength maps for a cohort.

    Subjects whose fraction of missing features exceeds *drop_threshold* are dropped,
    never imputed; the default ``0.0`` drops any subject with a missing feature.

    *hemisphere* defaults to ``"both"`` because the MSN is a whole-cortex network.
    Which hemisphere reaches the transcriptomics engine is decided later, at the
    :func:`msnpip.atlas_align.align_strength_to_atlas` boundary.

    Raises ``MSNInputError`` if no feature column matches, or no subject survives.
    """
    if hemisphere not in _HEMI_FOR:
        raise MSNInputError(f"hemisphere must be one of {sorted(_HEMI_FOR)}, got {hemisphere!r}")
    wanted_hemis = _HEMI_FOR[hemisphere]
    metric_index = {m: i for i, m in enumerate(metrics)}

    # First-seen region order, i.e. input column order rather than a fixed aparc one.
    # Safe: standardization is per metric and engine alignment is by label, not
    # position; only raw matrix plots see this order.
    region_order: list[str] = []
    seen_regions: set[str] = set()
    grid: dict[tuple[str, str, str], str] = {}  # (hemi, region, metric) → column
    for col in schema.feature_cols:
        parsed = _parse_feature_col(col)
        if parsed is None:
            continue
        hemi, region, metric = parsed
        if hemi not in wanted_hemis or metric not in metric_index:
            continue
        grid[(hemi, region, metric)] = col
        if region not in seen_regions:
            seen_regions.add(region)
            region_order.append(region)

    if not region_order:
        raise MSNInputError(
            f"No feature columns matched hemisphere={hemisphere!r} and "
            f"metrics={metrics}. Check schema.feature_cols and column naming "
            "('{hemi}_{region}_{metric}', e.g. 'lh_bankssts_SurfArea')."
        )

    region_labels = [f"{hemi}_{region}" for hemi in wanted_hemis for region in region_order]
    n_regions = len(region_labels)
    n_metrics = len(metrics)

    if schema.id_col is None or schema.id_col not in df.columns:
        raise MSNInputError("schema.id_col is required to label subjects.")
    all_ids = df[schema.id_col].astype(str).tolist()

    n_total = len(df)
    tensor = np.full((n_total, n_regions, n_metrics), np.nan, dtype=float)
    for ri, label in enumerate(region_labels):
        hemi, region = label.split("_", 1)
        for mi, metric in enumerate(metrics):
            col = grid.get((hemi, region, metric))
            if col is not None:
                tensor[:, ri, mi] = pd.to_numeric(df[col], errors="coerce").to_numpy()

    n_cells = n_regions * n_metrics
    missing_frac = np.isnan(tensor).reshape(n_total, n_cells).mean(axis=1)
    keep_mask = missing_frac <= drop_threshold
    dropped_subjects = [all_ids[i] for i in np.flatnonzero(~keep_mask)]
    if dropped_subjects:
        logger.warning(
            "compute_strength_maps: dropping %d/%d subject(s) over drop_threshold=%.3f "
            "missing-feature fraction: %s",
            len(dropped_subjects),
            n_total,
            drop_threshold,
            dropped_subjects[:10] + (["…"] if len(dropped_subjects) > 10 else []),
        )

    kept_idx = np.flatnonzero(keep_mask)
    if kept_idx.size == 0:
        raise MSNInputError(
            f"All {n_total} subjects dropped at drop_threshold={drop_threshold}. "
            "No complete subjects to build MSNs from."
        )

    subject_ids = [all_ids[i] for i in kept_idx]
    kept_tensor = tensor[kept_idx]

    matrix = build_msn(kept_tensor, similarity=similarity)
    strength = node_strength(matrix, agg=agg)
    global_strength = np.nanmean(strength, axis=1)

    logger.info(
        "compute_strength_maps: atlas=%s hemi=%s regions=%s → %d subjects × %d regions",
        atlas,
        hemisphere,
        regions,
        len(subject_ids),
        n_regions,
    )

    return StrengthMaps(
        matrix=matrix,
        strength=strength,
        subject_ids=subject_ids,
        region_labels=region_labels,
        atlas=atlas,
        features=list(metrics),
        global_strength=global_strength,
        hemisphere=hemisphere,
        regions=regions,
        agg=agg,
        dropped_subjects=dropped_subjects,
    )
