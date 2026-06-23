"""Unit tests for msnpip.io.matching — T1.5."""
from __future__ import annotations

import pandas as pd
import pytest

from msnpip.errors import IDMatchError
from msnpip.io.matching import normalize_ids, merge_features_demographics


# ---------------------------------------------------------------------------
# normalize_ids — strips whitespace only, nothing else
# ---------------------------------------------------------------------------

class TestNormalizeIds:
    def test_strips_trailing_whitespace(self):
        s = pd.Series(["sub-001 ", " sub-002", "sub-003"])
        n = normalize_ids(s)
        assert n.tolist() == ["sub-001", "sub-002", "sub-003"]

    def test_zero_padding_preserved(self):
        """sub-001 must NOT be collapsed to sub-1."""
        s = pd.Series(["sub-001", "sub-010", "sub-100"])
        n = normalize_ids(s)
        assert n.tolist() == ["sub-001", "sub-010", "sub-100"]

    def test_case_preserved(self):
        s = pd.Series(["Sub-001", "SUB-002"])
        n = normalize_ids(s)
        assert n.tolist() == ["Sub-001", "SUB-002"]

    def test_no_digits_unchanged(self):
        s = pd.Series(["alice", "bob"])
        n = normalize_ids(s)
        assert n.tolist() == ["alice", "bob"]

    def test_idempotent(self):
        s = pd.Series(["sub-001", "sub-002"])
        assert normalize_ids(s).tolist() == normalize_ids(normalize_ids(s)).tolist()


# ---------------------------------------------------------------------------
# merge_features_demographics
# ---------------------------------------------------------------------------

def _feat(ids):
    return pd.DataFrame({"subject_id": ids, "lh_bankssts_SurfArea": range(len(ids))})


def _dem(ids, group=None):
    g = group or (["HC"] * len(ids))
    return pd.DataFrame({"subject_id": ids, "group": g, "age": [60.0] * len(ids)})


class TestMergeFeaturesDemographics:
    def test_exact_match(self):
        ids = ["sub-001", "sub-002", "sub-003"]
        merged = merge_features_demographics(
            _feat(ids), _dem(ids),
            feat_id_col="subject_id", dem_id_col="subject_id",
        )
        assert len(merged) == 3
        assert "group" in merged.columns
        assert "age" in merged.columns

    def test_ids_preserved_exactly(self):
        """IDs in the merged output must match the features source exactly."""
        ids = ["sub-001", "sub-002"]
        merged = merge_features_demographics(
            _feat(ids), _dem(ids),
            feat_id_col="subject_id", dem_id_col="subject_id",
        )
        assert merged["subject_id"].tolist() == ids

    def test_whitespace_trimmed_for_matching(self):
        """Trailing space in the demographics file should still match."""
        feat_ids = ["sub-001", "sub-002"]
        dem_ids = ["sub-001 ", "sub-002"]   # trailing space — stripped before join
        merged = merge_features_demographics(
            _feat(feat_ids), _dem(dem_ids),
            feat_id_col="subject_id", dem_id_col="subject_id",
        )
        assert len(merged) == 2

    def test_zero_padding_difference_does_not_match(self):
        """sub-001 and sub-1 are different IDs — must not match."""
        feat_ids = ["sub-001", "sub-002"]
        dem_ids = ["sub-1", "sub-2"]
        with pytest.raises(IDMatchError):
            merge_features_demographics(
                _feat(feat_ids), _dem(dem_ids),
                feat_id_col="subject_id", dem_id_col="subject_id",
                min_match_rate=0.5,
            )

    def test_low_match_rate_raises(self):
        feat_ids = ["sub-001", "sub-002", "sub-003", "sub-004"]
        dem_ids = ["sub-001"]  # only 1 of 4 matches
        with pytest.raises(IDMatchError):
            merge_features_demographics(
                _feat(feat_ids), _dem(dem_ids),
                feat_id_col="subject_id", dem_id_col="subject_id",
                min_match_rate=0.95,
            )

    def test_idmatch_error_lists_unmatched(self):
        feat_ids = ["sub-001", "sub-002", "sub-003", "sub-004"]
        dem_ids = ["sub-001"]
        with pytest.raises(IDMatchError) as exc_info:
            merge_features_demographics(
                _feat(feat_ids), _dem(dem_ids),
                feat_id_col="subject_id", dem_id_col="subject_id",
                min_match_rate=0.5,
            )
        assert exc_info.value.unmatched

    def test_perfect_match_passes_at_threshold(self):
        ids = ["sub-001"]
        merged = merge_features_demographics(
            _feat(ids), _dem(ids),
            feat_id_col="subject_id", dem_id_col="subject_id",
            min_match_rate=1.0,
        )
        assert len(merged) == 1
