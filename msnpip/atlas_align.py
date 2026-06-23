"""
Alignment layer: reorder MSN regional values to the engine atlas label order.
Phase 1, Task T1.6.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

import imaging_transcriptomics as imt

from msnpip.errors import AtlasAlignmentError

logger = logging.getLogger("msnpip.atlas_align")

# Maps msnpip hemisphere prefix → engine hemisphere code
_HEMI_CODE: dict[str, str] = {"lh": "L", "rh": "R"}


# ---------------------------------------------------------------------------
# T1.6 — engine_region_order
# ---------------------------------------------------------------------------

def engine_region_order(atlas: str, hemisphere: str, regions: str) -> pd.DataFrame:
    """Return the canonical atlas label DataFrame from the engine.

    Parameters
    ----------
    atlas
        Atlas identifier (e.g. ``"dk"``).
    hemisphere
        ``"left"`` or ``"both"``.
    regions
        ``"cort"`` / ``"default"`` or ``"cort+sub"`` / ``"all"``.

    Returns
    -------
    pd.DataFrame
        ``id, label, hemisphere, structure`` — the authoritative region order
        the engine expects for its input vector.
    """
    sel = imt.select_atlas_data(atlas=atlas, hemisphere=hemisphere, regions=regions)
    return sel.labels.reset_index(drop=True)


# ---------------------------------------------------------------------------
# T1.6 — align_strength_to_atlas
# ---------------------------------------------------------------------------

def align_strength_to_atlas(
    values: np.ndarray,
    region_labels: Sequence[str],
    *,
    atlas: str,
    hemisphere: str,
    regions: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Reorder *values* to the engine atlas label order.

    Parameters
    ----------
    values
        1-D numpy array of length ``len(region_labels)`` — one value per
        region (e.g. the group-contrast beta map).
    region_labels
        Region identifiers in the format ``"{hemi}_{aparc_label}"``
        (e.g. ``"lh_bankssts"``, ``"rh_insula"``).  The hemisphere prefix
        ``lh`` / ``rh`` is mapped to the engine codes ``L`` / ``R``.
    atlas, hemisphere, regions
        Engine atlas parameters forwarded to :func:`engine_region_order`.

    Returns
    -------
    aligned_vector : np.ndarray
        Values reordered to match the engine label order.
    labels_df : pd.DataFrame
        Engine labels DataFrame (id, label, hemisphere, structure) — the
        same row order as *aligned_vector*.

    Raises
    ------
    AtlasAlignmentError
        If any engine-expected (hemisphere, label) pair is absent from
        *region_labels*.  Never silently zero-fills.
    ValueError
        If ``len(values) != len(region_labels)``.
    """
    values = np.asarray(values, dtype=float)
    region_labels = list(region_labels)

    if len(values) != len(region_labels):
        raise ValueError(
            f"len(values)={len(values)} != len(region_labels)={len(region_labels)}"
        )

    # Build lookup: (engine_hemi_code, aparc_label) → value
    lookup: dict[tuple[str, str], float] = {}
    for label_str, val in zip(region_labels, values):
        parts = label_str.split("_", 1)
        if len(parts) != 2 or parts[0] not in _HEMI_CODE:
            raise AtlasAlignmentError(
                f"Region label '{label_str}' does not match expected format "
                "'{{lh|rh}}_{{aparc_label}}' (e.g. 'lh_bankssts')."
            )
        hemi_code = _HEMI_CODE[parts[0]]
        lookup[(hemi_code, parts[1])] = val

    labels_df = engine_region_order(atlas, hemisphere, regions)

    missing: list[str] = []
    aligned: list[float] = []

    for _, row in labels_df.iterrows():
        key = (row["hemisphere"], row["label"])
        if key not in lookup:
            missing.append(f"{row['hemisphere']}:{row['label']}")
        else:
            aligned.append(lookup[key])

    if missing:
        raise AtlasAlignmentError(
            f"{len(missing)} engine region(s) have no matching MSN value:\n"
            + "  " + ", ".join(missing[:20])
            + ("\n  …" if len(missing) > 20 else "")
            + "\nCheck that the MSN was built from the correct atlas and hemisphere. "
            "Region names and hemisphere prefixes must match exactly."
        )

    logger.debug(
        "align_strength_to_atlas: atlas=%s hemi=%s regions=%s → %d values aligned",
        atlas, hemisphere, regions, len(aligned),
    )
    return np.array(aligned, dtype=float), labels_df.copy()


# ---------------------------------------------------------------------------
# T1.6 — to_region_table
# ---------------------------------------------------------------------------

def to_region_table(
    values: np.ndarray,
    labels_df: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    """Build the region table expected by the engine's plotting functions.

    The engine's ``plotting.*`` functions require a DataFrame with columns
    ``id, label, hemisphere, structure, <value_column>``.  This function
    builds that table from the aligned output of
    :func:`align_strength_to_atlas`.

    Parameters
    ----------
    values
        1-D numpy array aligned to *labels_df* row order.
    labels_df
        Engine labels DataFrame (output of :func:`engine_region_order` or
        the second element of :func:`align_strength_to_atlas`'s return).
    value_column
        Name for the value column in the result (e.g. ``"beta"``).

    Returns
    -------
    pd.DataFrame
        Columns: ``id, label, hemisphere, structure, <value_column>``.
    """
    values = np.asarray(values, dtype=float)
    if len(values) != len(labels_df):
        raise ValueError(
            f"len(values)={len(values)} != len(labels_df)={len(labels_df)}"
        )
    table = labels_df[["id", "label", "hemisphere", "structure"]].copy()
    table[value_column] = values
    return table.reset_index(drop=True)
