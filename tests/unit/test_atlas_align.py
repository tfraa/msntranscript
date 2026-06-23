"""Unit tests for msnpip.atlas_align — T1.6."""

from __future__ import annotations

import numpy as np
import pytest

from msnpip.atlas_align import (
    align_strength_to_atlas,
    engine_region_order,
    to_region_table,
)
from msnpip.errors import AtlasAlignmentError
from tests.fixtures.synthetic import DK_REGIONS

# ---------------------------------------------------------------------------
# engine_region_order
# ---------------------------------------------------------------------------


class TestEngineRegionOrder:
    def test_dk_left_cort_shape(self):
        labels = engine_region_order("dk", "left", "cort")
        assert len(labels) == 34
        assert list(labels.columns) == ["id", "label", "hemisphere", "structure"]

    def test_dk_left_cort_hemisphere_codes(self):
        labels = engine_region_order("dk", "left", "cort")
        assert set(labels["hemisphere"].unique()) == {"L"}

    def test_dk_both_cort_shape(self):
        labels = engine_region_order("dk", "both", "cort")
        assert len(labels) == 68
        assert set(labels["hemisphere"].unique()) == {"L", "R"}

    def test_dk_labels_match_dk_regions(self):
        """Every label in the engine order must be in our canonical DK_REGIONS list."""
        labels = engine_region_order("dk", "left", "cort")
        engine_names = set(labels["label"].tolist())
        fixture_names = set(DK_REGIONS)
        assert engine_names == fixture_names, (
            f"Unexpected labels: {engine_names - fixture_names}; "
            f"Missing: {fixture_names - engine_names}"
        )


# ---------------------------------------------------------------------------
# align_strength_to_atlas
# ---------------------------------------------------------------------------


def _make_input(hemisphere: str = "left") -> tuple[np.ndarray, list[str]]:
    """Build a synthetic (values, region_labels) pair."""
    hemis = ["lh"] if hemisphere == "left" else ["lh", "rh"]
    labels = []
    for hemi in hemis:
        for region in DK_REGIONS:
            labels.append(f"{hemi}_{region}")
    values = np.arange(len(labels), dtype=float)
    return values, labels


class TestAlignStrengthToAtlas:
    def test_left_returns_correct_length(self):
        values, labels = _make_input("left")
        aligned, labels_df = align_strength_to_atlas(
            values, labels, atlas="dk", hemisphere="left", regions="cort"
        )
        assert len(aligned) == 34
        assert len(labels_df) == 34

    def test_both_returns_correct_length(self):
        values, labels = _make_input("both")
        aligned, labels_df = align_strength_to_atlas(
            values, labels, atlas="dk", hemisphere="both", regions="cort"
        )
        assert len(aligned) == 68

    def test_alignment_preserves_values(self):
        """Values must be reordered but not lost or duplicated."""
        values, labels = _make_input("left")
        aligned, _ = align_strength_to_atlas(
            values, labels, atlas="dk", hemisphere="left", regions="cort"
        )
        # The same set of values should appear (just reordered)
        assert set(aligned.tolist()) == set(values.tolist())

    def test_engine_order_is_respected(self):
        """After alignment, the first label in the engine order maps to the first value."""
        expected_order = engine_region_order("dk", "left", "cort")
        # Build lookup: region → value = index in DK_REGIONS (arbitrary but deterministic)
        values, labels = _make_input("left")
        aligned, _ = align_strength_to_atlas(
            values, labels, atlas="dk", hemisphere="left", regions="cort"
        )
        # The first engine region is whatever engine_region_order says
        first_label = expected_order.iloc[0]["label"]
        idx_in_dk = DK_REGIONS.index(first_label)
        # That label was at index idx_in_dk in our lh_* list → value == idx_in_dk
        assert aligned[0] == float(idx_in_dk)

    def test_name_mismatch_raises(self):
        """A typo in a region label must raise AtlasAlignmentError."""
        values, labels = _make_input("left")
        labels[0] = "lh_TYPO_REGION"  # bad label
        with pytest.raises(AtlasAlignmentError, match="no matching MSN value"):
            align_strength_to_atlas(values, labels, atlas="dk", hemisphere="left", regions="cort")

    def test_wrong_hemi_prefix_raises(self):
        values, labels = _make_input("left")
        labels[0] = "xx_bankssts"  # unknown hemi prefix
        with pytest.raises(AtlasAlignmentError, match="format"):
            align_strength_to_atlas(values, labels, atlas="dk", hemisphere="left", regions="cort")

    def test_length_mismatch_raises(self):
        values = np.zeros(10)
        labels = [f"lh_{r}" for r in DK_REGIONS]  # 34 labels, 10 values
        with pytest.raises(ValueError, match="len\\(values\\)"):
            align_strength_to_atlas(values, labels, atlas="dk", hemisphere="left", regions="cort")


# ---------------------------------------------------------------------------
# to_region_table
# ---------------------------------------------------------------------------


class TestToRegionTable:
    def test_output_columns(self):
        values, labels = _make_input("left")
        aligned, labels_df = align_strength_to_atlas(
            values, labels, atlas="dk", hemisphere="left", regions="cort"
        )
        table = to_region_table(aligned, labels_df, "beta")
        assert list(table.columns) == ["id", "label", "hemisphere", "structure", "beta"]

    def test_output_length(self):
        values, labels = _make_input("left")
        aligned, labels_df = align_strength_to_atlas(
            values, labels, atlas="dk", hemisphere="left", regions="cort"
        )
        table = to_region_table(aligned, labels_df, "t_stat")
        assert len(table) == 34

    def test_values_preserved(self):
        values, labels = _make_input("left")
        aligned, labels_df = align_strength_to_atlas(
            values, labels, atlas="dk", hemisphere="left", regions="cort"
        )
        table = to_region_table(aligned, labels_df, "value")
        assert np.allclose(table["value"].values, aligned)
