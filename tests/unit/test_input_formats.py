"""Tests for flexible input (issues 1-5): feature tables, aparc location,
id-column detection, numeric group matching."""

from __future__ import annotations

import pandas as pd

from msnpip.io.readers import (
    detect_input_kind,
    read_feature_tables,
    read_freesurfer_subjects,
)
from msnpip.io.schema import detect_id_column
from msnpip.stats.glm import group_mask, normalize_group_value


def _write(path, text):
    path.write_text(text, encoding="utf-8")


# --- issue 1: per-feature tables -------------------------------------------


def _make_feature_dir(root):
    root.mkdir(parents=True, exist_ok=True)
    # Semicolon-sep, comma-decimal, metric token 'gauscurv'; first col = id; embedded demo.
    _write(
        root / "aparc_gausscurv_lh.csv",
        "lh.aparc.gauscurv;Group;Diagnosis;lh_bankssts_gauscurv;lh_insula_gauscurv;eTIV\n"
        "CLM_1;1;BPD;0,025;0,028;1338742\n"
        "CTRL_1;0;na;0,015;0,026;1486608\n",
    )
    _write(
        root / "aparc_thickness_lh.csv",
        "lh.aparc.thickness;lh_bankssts_thickness;lh_insula_thickness\n"
        "CLM_1;2,28;2,74\n"
        "CTRL_1;2,26;2,70\n",
    )
    return root


def test_read_feature_tables_merges_and_canonicalizes(tmp_path):
    d = _make_feature_dir(tmp_path / "feats")
    df = read_feature_tables(d)
    assert set(df["subject_id"]) == {"CLM_1", "CTRL_1"}
    # metric tokens canonicalized (gauscurv -> GausCurv, thickness -> ThickAvg)
    assert "lh_bankssts_GausCurv" in df.columns
    assert "lh_insula_ThickAvg" in df.columns
    # embedded demographics carried through; eTIV -> tiv
    assert "Group" in df.columns
    assert "tiv" in df.columns
    assert df.loc[df.subject_id == "CLM_1", "lh_bankssts_GausCurv"].iloc[0] == 0.025


def test_detect_input_kind(tmp_path):
    feats = _make_feature_dir(tmp_path / "feats")
    assert detect_input_kind(feats) == "feature_tables"

    subj = tmp_path / "subjects" / "sub-001" / "stats"
    subj.mkdir(parents=True)
    _write(subj / "lh.aparc.stats", "# ColHeaders StructName\nbankssts\n")
    assert detect_input_kind(tmp_path / "subjects") == "subjects"


# --- issue 3: aparc.stats directly in the subject folder --------------------


def _aparc(region_vals):
    header = "# ColHeaders StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd\n"
    lines = [f"{r} 1000 800 2000 2.4 0.4 0.1 0.02 10 1.0" for r in region_vals]
    return header + "\n".join(lines) + "\n"


def test_aparc_stats_found_directly_in_subject_folder(tmp_path):
    root = tmp_path / "fs"
    subj = root / "sub-001"
    subj.mkdir(parents=True)
    # No stats/ subfolder — files directly in the subject dir.
    _write(subj / "lh.aparc.stats", _aparc(["bankssts", "insula"]))
    _write(subj / "rh.aparc.stats", _aparc(["bankssts", "insula"]))
    df = read_freesurfer_subjects(root, expected_regions=["bankssts", "insula"])
    assert len(df) == 1
    assert df["lh_bankssts_ThickAvg"].iloc[0] == 2.4


# --- issue 2: id column name detection -------------------------------------


def test_detect_id_column(tmp_path):
    assert detect_id_column(pd.DataFrame(columns=["subject_id", "age"])) == "subject_id"
    assert detect_id_column(pd.DataFrame(columns=["ID", "age"])) == "ID"
    # no alias match -> first column
    assert detect_id_column(pd.DataFrame(columns=["Codice", "age"])) == "Codice"
    # explicit override wins
    assert detect_id_column(pd.DataFrame(columns=["a", "b"]), override="b") == "b"


# --- issue 5: numeric / string group matching ------------------------------


def test_normalize_group_value():
    assert normalize_group_value(1) == "1"
    assert normalize_group_value(1.0) == "1"
    assert normalize_group_value("1") == "1"
    assert normalize_group_value("1.0") == "1"
    assert normalize_group_value("BPD") == "BPD"


def test_group_mask_matches_across_types():
    s = pd.Series([1, 0, 1, 0])
    assert group_mask(s, "1").tolist() == [True, False, True, False]
    assert group_mask(s, 0).tolist() == [False, True, False, True]
    s2 = pd.Series([1.0, 0.0])
    assert group_mask(s2, "1").tolist() == [True, False]
