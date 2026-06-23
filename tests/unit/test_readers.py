"""Unit tests for msnpip.io.readers — T1.1 and T1.2."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from msnpip.errors import AmbiguousFormatError, MsnpipIOError
from msnpip.io.readers import read_table, read_freesurfer_subjects
from tests.fixtures.synthetic import DK_REGIONS, MSN_METRICS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# T1.1 read_table — delimiter detection
# ---------------------------------------------------------------------------

class TestReadTableDelimiters:
    def test_comma_sep(self, tmp_path):
        p = _write(tmp_path, "data.csv", "a,b,c\n1,2,3\n4,5,6\n")
        df = read_table(p)
        assert list(df.columns) == ["a", "b", "c"]
        assert df.shape == (2, 3)

    def test_semicolon_sep(self, tmp_path):
        p = _write(tmp_path, "data.csv", "a;b;c\n1;2;3\n4;5;6\n")
        df = read_table(p)
        assert list(df.columns) == ["a", "b", "c"]
        assert df.shape == (2, 3)

    def test_tab_sep_tsv(self, tmp_path):
        p = _write(tmp_path, "data.tsv", "a\tb\tc\n1\t2\t3\n")
        df = read_table(p)
        assert list(df.columns) == ["a", "b", "c"]

    def test_explicit_sep_overrides_sniff(self, tmp_path):
        # File looks comma-separated but user passes sep=';'
        p = _write(tmp_path, "data.csv", "a,b\n1,2\n")
        df = read_table(p, sep=",")
        assert df.shape == (1, 2)

    def test_ambiguous_raises(self, tmp_path):
        # A file with no recognizable delimiter
        p = _write(tmp_path, "weird.csv", "abcdef\nghijkl\n")
        with pytest.raises(AmbiguousFormatError):
            read_table(p)


class TestReadTableDecimal:
    def test_dot_decimal_default(self, tmp_path):
        p = _write(tmp_path, "data.csv", "id,value\n1,3.14\n2,2.71\n")
        df = read_table(p)
        assert pd.api.types.is_numeric_dtype(df["value"])
        assert abs(df["value"][0] - 3.14) < 1e-6

    def test_comma_decimal_semicolon_sep(self, tmp_path):
        p = _write(tmp_path, "data.csv", "id;value\n1;3,14\n2;2,71\n")
        df = read_table(p)
        assert pd.api.types.is_numeric_dtype(df["value"]), (
            f"value column dtype={df['value'].dtype}, values={df['value'].tolist()}"
        )
        assert abs(df["value"][0] - 3.14) < 1e-6

    def test_explicit_decimal_overrides(self, tmp_path):
        p = _write(tmp_path, "data.csv", "id;value\n1;3,14\n", )
        df = read_table(p, sep=";", decimal=",")
        assert abs(df["value"][0] - 3.14) < 1e-6

    def test_locale_quirks_cohort(self, synthetic_cohort_locale):
        """Reads the locale-quirk fixture end-to-end with auto-detection."""
        c = synthetic_cohort_locale
        df = read_table(c["merged_path"])
        assert df.shape[0] == c["n_subjects"]
        # At least one feature column must be numeric
        feat_cols = [col for col in df.columns if "_SurfArea" in col]
        assert feat_cols, "No SurfArea columns found"
        assert pd.api.types.is_numeric_dtype(df[feat_cols[0]])


# ---------------------------------------------------------------------------
# T1.2 read_freesurfer_subjects
# ---------------------------------------------------------------------------

class TestReadFreeSurferSubjects:
    def test_standard_cohort(self, synthetic_cohort):
        c = synthetic_cohort
        df = read_freesurfer_subjects(c["fs_dir"], expected_regions=DK_REGIONS)
        assert df.shape[0] == c["n_subjects"]
        # Should have subject_id + 2 hemis × 34 regions × 5 metrics
        expected_feat_cols = 2 * len(DK_REGIONS) * len(MSN_METRICS)
        assert len(df.columns) == 1 + expected_feat_cols, (
            f"Expected {1 + expected_feat_cols} columns, got {len(df.columns)}"
        )

    def test_all_values_finite(self, synthetic_cohort):
        c = synthetic_cohort
        df = read_freesurfer_subjects(c["fs_dir"], expected_regions=DK_REGIONS)
        feat_cols = [c for c in df.columns if c != "subject_id"]
        assert not df[feat_cols].isnull().any().any(), "Unexpected NaN in synthetic data"

    def test_missing_region_filled_with_nan(self, tmp_path):
        """If a region is in expected_regions but not in the file, it gets NaN."""
        stats_dir = tmp_path / "sub-001" / "stats"
        stats_dir.mkdir(parents=True)
        # Write a file with only ONE region
        content = (
            "# Table of FreeSurfer cortical parcellation anatomical statistics\n"
            "# ColHeaders  StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd\n"
            "bankssts 1000 500 2000 2.5 0.3 0.1 0.01 20 1.0\n"
        )
        (stats_dir / "lh.aparc.stats").write_text(content)
        (stats_dir / "rh.aparc.stats").write_text(content)

        regions = ["bankssts", "cuneus"]  # cuneus is absent from the file
        df = read_freesurfer_subjects(tmp_path, expected_regions=regions)
        assert math.isnan(df["lh_cuneus_SurfArea"].iloc[0])
        assert not math.isnan(df["lh_bankssts_SurfArea"].iloc[0])

    def test_issues_recorded_in_attrs(self, tmp_path):
        """Missing aparc.stats file records an issue in df.attrs['issues']."""
        stats_dir = tmp_path / "sub-001" / "stats"
        stats_dir.mkdir(parents=True)
        content = (
            "# ColHeaders  StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd\n"
            "bankssts 1000 500 2000 2.5 0.3 0.1 0.01 20 1.0\n"
        )
        (stats_dir / "lh.aparc.stats").write_text(content)
        # rh is missing

        df = read_freesurfer_subjects(tmp_path, expected_regions=["bankssts"])
        assert df.attrs.get("issues"), "Expected issues to be recorded"
        assert any("rh" in str(i) for i in df.attrs["issues"][0]["issues"])

    def test_no_subjects_raises(self, tmp_path):
        with pytest.raises(MsnpipIOError, match="No subject directories"):
            read_freesurfer_subjects(tmp_path, expected_regions=DK_REGIONS)
