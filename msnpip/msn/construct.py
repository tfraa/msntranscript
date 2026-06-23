"""
MSN construction: build_msn, node_strength, compute_strength_maps → StrengthMaps.
Phase 2, Tasks T2.1–T2.2.

The Morphometric Similarity Network (MSN) for a subject is built from a
``(n_regions, n_metrics)`` feature matrix (the locked default is 5 metrics:
SurfArea, GrayVol, ThickAvg, MeanCurv, GausCurv).  Each metric is z-scored
*across regions within the subject* and the inter-regional similarity is the
Pearson correlation between the two regions' standardized feature vectors.
Node strength summarizes each region's connectivity profile (signed mean by
default).
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from msnpip.errors import MSNInputError

logger = logging.getLogger("msnpip.msn.construct")

# Locked MSN feature set (must match MSNConfig defaults / synthetic fixture).
DEFAULT_METRICS: tuple[str, ...] = ("SurfArea", "GrayVol", "ThickAvg", "MeanCurv", "GausCurv")

_HEMI_FOR: dict[str, tuple[str, ...]] = {
    "left": ("lh",),
    "right": ("rh",),
    "both": ("lh", "rh"),
}


# ---------------------------------------------------------------------------
# T2.1 — build_msn
# ---------------------------------------------------------------------------

def _build_one(features: np.ndarray) -> np.ndarray:
    """Build one subject's region×region MSN from a (n_regions, n_metrics) matrix.

    Steps: z-score each metric across regions, then Pearson-correlate regions
    across the standardized metric axis.  The diagonal is set to NaN.
    """
    n_regions = features.shape[0]

    all_nan = np.all(np.isnan(features), axis=1)
    if all_nan.any():
        bad = np.flatnonzero(all_nan).tolist()
        raise MSNInputError(
            f"{all_nan.sum()} region(s) have all-NaN features (rows {bad[:10]}). "
            "MSN construction never imputes — drop incomplete subjects upstream "
            "(see compute_strength_maps drop_threshold)."
        )

    mean = np.nanmean(features, axis=0)
    std = np.nanstd(features, axis=0, ddof=0)
    if np.any(std == 0):
        bad = np.flatnonzero(std == 0).tolist()
        raise MSNInputError(
            f"Metric column(s) {bad} are constant across regions (zero variance); "
            "morphometric similarity is undefined."
        )

    z = (features - mean) / std

    # Pearson correlation between regions across the standardized metric axis.
    msn = np.corrcoef(z)
    # corrcoef collapses to a scalar for a single region — normalize shape.
    msn = np.atleast_2d(msn)
    np.fill_diagonal(msn, np.nan)
    return msn


def build_msn(subject_features: np.ndarray) -> np.ndarray:
    """Construct per-subject MSNs.

    Parameters
    ----------
    subject_features
        Array of shape ``(n_subjects, n_regions, n_metrics)``.  A single
        subject may be passed as ``(n_regions, n_metrics)``.

    Returns
    -------
    np.ndarray
        ``(n_subjects, n_regions, n_regions)`` similarity matrices with a
        NaN diagonal.  If a single 2-D matrix was passed, a 2-D matrix is
        returned.

    Raises
    ------
    MSNInputError
        If any region is all-NaN or any metric is constant across regions.
    """
    arr = np.asarray(subject_features, dtype=float)
    single = arr.ndim == 2
    if single:
        arr = arr[None]
    if arr.ndim != 3:
        raise MSNInputError(
            f"subject_features must be 2-D or 3-D, got shape {arr.shape}."
        )

    n_subjects, n_regions, _ = arr.shape
    out = np.empty((n_subjects, n_regions, n_regions), dtype=float)
    for s in range(n_subjects):
        out[s] = _build_one(arr[s])

    return out[0] if single else out


# ---------------------------------------------------------------------------
# T2.1 — node_strength
# ---------------------------------------------------------------------------

def node_strength(msn: np.ndarray, *, sign: str = "signed") -> np.ndarray:
    """Compute per-region node strength from MSN matrices.

    Parameters
    ----------
    msn
        ``(n_subjects, n_regions, n_regions)`` (or a single 2-D matrix) with a
        NaN diagonal.
    sign
        - ``"signed"`` (default): ``(pos_mean + neg_mean) / 2`` — the mean of
          positive edges and the mean of negative edges, averaged.  A region
          with no positive (or negative) edges contributes 0 for that side.
        - ``"positive"``: mean of positive edges only.
        - ``"absolute"``: mean of absolute edge weights.

    Returns
    -------
    np.ndarray
        ``(n_subjects, n_regions)`` node strengths (or 1-D for a single matrix).
    """
    if sign not in ("signed", "positive", "absolute"):
        raise ValueError(f"sign must be 'signed'/'positive'/'absolute', got {sign!r}")

    arr = np.asarray(msn, dtype=float)
    single = arr.ndim == 2
    if single:
        arr = arr[None]

    with warnings.catch_warnings():
        # nanmean over an all-NaN slice (region with no pos/neg edges) → warning;
        # we deliberately map that to 0 below.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if sign == "signed":
            pos = np.where(arr > 0, arr, np.nan)
            neg = np.where(arr < 0, arr, np.nan)
            pos_mean = np.nan_to_num(np.nanmean(pos, axis=2), nan=0.0)
            neg_mean = np.nan_to_num(np.nanmean(neg, axis=2), nan=0.0)
            out = (pos_mean + neg_mean) / 2.0
        elif sign == "positive":
            pos = np.where(arr > 0, arr, np.nan)
            out = np.nan_to_num(np.nanmean(pos, axis=2), nan=0.0)
        else:  # absolute
            out = np.nanmean(np.abs(arr), axis=2)

    return out[0] if single else out


# ---------------------------------------------------------------------------
# T2.2 — StrengthMaps + compute_strength_maps
# ---------------------------------------------------------------------------

@dataclass
class StrengthMaps:
    """Container for per-subject MSN matrices and node-strength maps.

    ``region_labels`` use the ``"{hemi}_{aparc_label}"`` format
    (e.g. ``"lh_bankssts"``) so the strength vectors feed
    :func:`msnpip.atlas_align.align_strength_to_atlas` directly.
    """

    matrix: np.ndarray              # (n_subjects, n_regions, n_regions)
    strength: np.ndarray           # (n_subjects, n_regions)
    subject_ids: list[str]
    region_labels: list[str]       # ["lh_bankssts", ...]
    atlas: str
    features: list[str]            # metric names, in column order
    global_strength: np.ndarray    # (n_subjects,) — mean over regions
    hemisphere: str = "both"
    regions: str = "cort"
    sign: str = "signed"
    dropped_subjects: list[str] = field(default_factory=list)

    @property
    def n_subjects(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_regions(self) -> int:
        return self.matrix.shape[1]


def _parse_feature_col(col: str) -> tuple[str, str, str] | None:
    """Split ``"{hemi}_{region}_{metric}"`` → (hemi, region, metric).

    Region names may themselves contain no underscore in the DK atlas, but the
    split is robust to that: first token = hemi, last token = metric, middle =
    region.  Returns ``None`` if the column is not a hemi-prefixed feature.
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
    sign: str = "signed",
    metrics: tuple[str, ...] = DEFAULT_METRICS,
) -> StrengthMaps:
    """Build MSNs and node-strength maps for a cohort.

    Parses ``schema.feature_cols`` into ``(hemi, region, metric)``, assembles a
    ``(n_subjects, n_regions, n_metrics)`` tensor for the requested
    *hemisphere*, drops (never imputes) subjects whose fraction of missing
    features exceeds *drop_threshold*, then builds the MSN and node strength.

    The MSN is a whole-cortex network, so *hemisphere* defaults to ``"both"``:
    every region's node strength reflects its similarity to all other regions
    across both hemispheres, and group differences are available for the left
    and right cortex.  The choice of which hemisphere(s) to feed the
    transcriptomics engine is made later (``EngineConfig.hemisphere``), at the
    :func:`msnpip.atlas_align.align_strength_to_atlas` boundary — not here.

    Parameters
    ----------
    df
        Merged feature + demographics DataFrame.
    schema
        :class:`msnpip.io.schema.ColumnSchema` describing the columns.
    atlas, hemisphere, regions
        Atlas selection.  ``hemisphere`` ∈ {``left``, ``right``, ``both``};
        defaults to ``"both"`` (whole-cortex MSN).
    drop_threshold
        A subject is dropped if its proportion of missing (NaN) selected
        feature values is **greater than** this threshold.  The default
        ``0.0`` drops any subject with one or more missing features.
    sign
        Node-strength sign policy (see :func:`node_strength`).
    metrics
        Metric names and the order of the metric axis.

    Returns
    -------
    StrengthMaps

    Raises
    ------
    MSNInputError
        If no feature columns match the requested hemisphere/metrics, or if no
        subjects survive the drop step.
    """
    if hemisphere not in _HEMI_FOR:
        raise MSNInputError(
            f"hemisphere must be one of {sorted(_HEMI_FOR)}, got {hemisphere!r}"
        )
    wanted_hemis = _HEMI_FOR[hemisphere]
    metric_index = {m: i for i, m in enumerate(metrics)}

    # Discover the (hemi, region) grid present for the requested hemisphere/metrics,
    # preserving first-seen region order for determinism.
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

    # Subject IDs (preserve df row order).
    if schema.id_col is None or schema.id_col not in df.columns:
        raise MSNInputError("schema.id_col is required to label subjects.")
    all_ids = df[schema.id_col].astype(str).tolist()

    # Assemble the per-subject tensor; NaN for any missing column.
    n_total = len(df)
    tensor = np.full((n_total, n_regions, n_metrics), np.nan, dtype=float)
    for ri, label in enumerate(region_labels):
        hemi, region = label.split("_", 1)
        for mi, metric in enumerate(metrics):
            col = grid.get((hemi, region, metric))
            if col is not None:
                tensor[:, ri, mi] = pd.to_numeric(df[col], errors="coerce").to_numpy()

    # Drop-and-report incomplete subjects.
    n_cells = n_regions * n_metrics
    missing_frac = np.isnan(tensor).reshape(n_total, n_cells).mean(axis=1)
    keep_mask = missing_frac <= drop_threshold
    dropped_subjects = [all_ids[i] for i in np.flatnonzero(~keep_mask)]
    if dropped_subjects:
        logger.warning(
            "compute_strength_maps: dropping %d/%d subject(s) over drop_threshold=%.3f "
            "missing-feature fraction: %s",
            len(dropped_subjects), n_total, drop_threshold,
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

    matrix = build_msn(kept_tensor)
    strength = node_strength(matrix, sign=sign)
    global_strength = np.nanmean(strength, axis=1)

    logger.info(
        "compute_strength_maps: atlas=%s hemi=%s regions=%s → %d subjects × %d regions",
        atlas, hemisphere, regions, len(subject_ids), n_regions,
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
        sign=sign,
        dropped_subjects=dropped_subjects,
    )
