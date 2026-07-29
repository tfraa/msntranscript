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


# ---------------------------------------------------------------------------
# hemisphere="right" — homotopic relabel (left expression, right phenotype)
# ---------------------------------------------------------------------------


class TestRightHemisphereRelabel:
    """``hemisphere="right"`` keeps the LEFT label order (so the engine pairs the
    map with its left-hemisphere AHBA expression) and fills it from ``rh_*``."""

    @staticmethod
    def _lateralised() -> tuple[np.ndarray, list[str]]:
        """lh_* values are 0.0; each rh_* value is a distinct positive number."""
        labels = [f"lh_{r}" for r in DK_REGIONS] + [f"rh_{r}" for r in DK_REGIONS]
        values = np.concatenate(
            [np.zeros(len(DK_REGIONS)), np.arange(1, len(DK_REGIONS) + 1, dtype=float)]
        )
        return values, labels

    def test_returns_left_labels(self):
        values, labels = self._lateralised()
        aligned, labels_df = align_strength_to_atlas(
            values, labels, atlas="dk", hemisphere="right", regions="cort"
        )
        assert len(aligned) == 34
        # The engine must see a left-hemisphere run: that is what pairs the map
        # with the left expression matrix.
        assert set(labels_df["hemisphere"].unique()) == {"L"}

    def test_takes_values_from_the_right_hemisphere(self):
        values, labels = self._lateralised()
        aligned, _ = align_strength_to_atlas(
            values, labels, atlas="dk", hemisphere="right", regions="cort"
        )
        # Every lh_* value was 0.0; picking any of them would show up here.
        assert (aligned > 0).all()
        assert set(aligned.tolist()) == set(range(1, len(DK_REGIONS) + 1))

    def test_differs_from_the_left_arm_on_a_lateralised_map(self):
        values, labels = self._lateralised()
        left, left_labels = align_strength_to_atlas(
            values, labels, atlas="dk", hemisphere="left", regions="cort"
        )
        right, right_labels = align_strength_to_atlas(
            values, labels, atlas="dk", hemisphere="right", regions="cort"
        )
        assert (left == 0).all()
        assert not np.array_equal(left, right)
        # Same region order in both arms, so the two results are comparable
        # region-by-region — the only thing that changed is the phenotype.
        assert left_labels["label"].tolist() == right_labels["label"].tolist()

    def test_region_order_matches_the_left_arm_homotopically(self):
        values, labels = self._lateralised()
        aligned, labels_df = align_strength_to_atlas(
            values, labels, atlas="dk", hemisphere="right", regions="cort"
        )
        # Slot i holds rh_<label i>, i.e. the homotopic partner of lh_<label i>.
        for i, label in enumerate(labels_df["label"].tolist()):
            assert aligned[i] == float(DK_REGIONS.index(label) + 1)

    def test_missing_right_region_raises_with_r_prefix(self):
        values, labels = self._lateralised()
        labels[len(DK_REGIONS)] = "rh_TYPO_REGION"  # break one rh_* label
        with pytest.raises(AtlasAlignmentError, match=r"R:"):
            align_strength_to_atlas(values, labels, atlas="dk", hemisphere="right", regions="cort")
