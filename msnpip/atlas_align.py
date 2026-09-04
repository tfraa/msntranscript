"""Alignment layer: reorder MSN regional values to the engine atlas label order."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import imaging_transcriptomics as imt
import numpy as np
import pandas as pd

from msnpip.errors import AtlasAlignmentError

logger = logging.getLogger("msnpip.atlas_align")

# Maps msnpip hemisphere prefix → engine hemisphere code
_HEMI_CODE: dict[str, str] = {"lh": "L", "rh": "R"}

# ``hemisphere="right"`` is a homotopic relabel, not a right-hemisphere atlas
# request: the engine's DK expression is left-hemisphere (AHBA samples only 2 of
# 6 donors on the right), so the right arm keeps the LEFT label order — and with
# it the left expression matrix — and takes its *values* from the right
# hemisphere of the contrast map.  See EngineConfig.hemisphere.
_RELABEL_HEMI = "right"
_RELABEL_SOURCE_CODE = "R"


# ---------------------------------------------------------------------------
# T1.6 — engine_region_order
# ---------------------------------------------------------------------------


def engine_region_order(atlas: str, hemisphere: str, regions: str) -> pd.DataFrame:
    """Return the canonical atlas label DataFrame from the engine."""
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
        raise ValueError(f"len(values)={len(values)} != len(region_labels)={len(region_labels)}")

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

    # The right arm borrows the LEFT label order (so the engine pairs the map
    # with its left-hemisphere expression) and reads values from the right.
    relabel = hemisphere == _RELABEL_HEMI
    labels_df = engine_region_order(atlas, "left" if relabel else hemisphere, regions)

    missing: list[str] = []
    aligned: list[float] = []

    for _, row in labels_df.iterrows():
        key = (_RELABEL_SOURCE_CODE if relabel else row["hemisphere"], row["label"])
        if key not in lookup:
            missing.append(f"{key[0]}:{key[1]}")
        else:
            aligned.append(lookup[key])

    if missing:
        raise AtlasAlignmentError(
            f"{len(missing)} engine region(s) have no matching MSN value:\n"
            + "  "
            + ", ".join(missing[:20])
            + ("\n  …" if len(missing) > 20 else "")
            + "\nCheck that the MSN was built from the correct atlas and hemisphere. "
            "Region names and hemisphere prefixes must match exactly."
        )

    logger.debug(
        "align_strength_to_atlas: atlas=%s hemi=%s regions=%s → %d values aligned",
        atlas,
        hemisphere,
        regions,
        len(aligned),
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
    """
    values = np.asarray(values, dtype=float)
    if len(values) != len(labels_df):
        raise ValueError(f"len(values)={len(values)} != len(labels_df)={len(labels_df)}")
    table = labels_df[["id", "label", "hemisphere", "structure"]].copy()
    table[value_column] = values
    return table.reset_index(drop=True)
