"""Fast pipeline unit tests (no transcriptomics engine call) — T5.2."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from msnpip.config import EngineConfig, GLMConfig, IOConfig, PipelineConfig
from msnpip.pipeline import Pipeline, run_pipeline
from tests.fixtures.synthetic import make_synthetic_cohort


@pytest.fixture
def df_cfg(tmp_path):
    info = make_synthetic_cohort(tmp_path / "data", n_case=8, n_control=8, seed=3)
    out = tmp_path / "out"
    cfg = PipelineConfig(
        io=IOConfig(dataframe=Path(info["merged_path"])),
        output=out,
        group_col="group",
        case="FTD",
        control="HC",
        glm=GLMConfig(predictors=("age", "sex")),
        engine=EngineConfig(methods=("pls",), n_permutations=10),
    )
    return cfg, out


def test_stop_at_msn_builds_strength(df_cfg):
    cfg, out = df_cfg
    run_pipeline(cfg, stop_stage="MSN")
    assert (out / "merged_dataset.csv").exists()
    assert (out / "strength_maps.csv").exists()
    assert (out / "mean_msn_per_group.csv").exists()
    # contrast/transcriptomics not reached
    assert not (out / "case_control_difference_maps.csv").exists()


def test_stop_at_contrast_writes_difference_maps(df_cfg):
    cfg, out = df_cfg
    run_pipeline(cfg, stop_stage="CONTRAST")
    path = out / "case_control_difference_maps.csv"
    assert path.exists()
    df = pd.read_csv(path)
    assert "region" in df.columns and "FTD_vs_HC_beta" in df.columns
    assert len(df) == 68  # both-hemisphere MSN


def test_referenced_groups_from_contrasts(tmp_path):
    cfg = PipelineConfig(
        io=IOConfig(dataframe=Path("x.csv")),
        output=tmp_path,
        group_col="GROUP",
        contrasts=(("1", "0"), ("2", "0"), ("3", "0")),
    )
    assert Pipeline(cfg)._referenced_groups() == {"0", "1", "2", "3"}


def test_referenced_groups_none_for_rest_arm(tmp_path):
    cfg = PipelineConfig(
        io=IOConfig(dataframe=Path("x.csv")), output=tmp_path, group_col="GROUP", case="1"
    )
    # control defaults to 'rest' → needs all subjects → no restriction.
    assert Pipeline(cfg)._referenced_groups() is None


def test_scope_restricts_to_contrast_groups(tmp_path):
    info = make_synthetic_cohort(tmp_path / "data", n_case=8, n_control=8, seed=5)
    df = pd.read_csv(info["merged_path"])
    df.loc[df.index[:4], "group"] = "OTHER"  # a third group, not in the contrast
    relabeled = tmp_path / "data" / "relabeled.csv"
    df.to_csv(relabeled, index=False)
    out = tmp_path / "out"
    cfg = PipelineConfig(
        io=IOConfig(dataframe=relabeled),
        output=out,
        group_col="group",
        contrasts=(("FTD", "HC"),),
        glm=GLMConfig(predictors=("age", "sex")),
        engine=EngineConfig(methods=("pls",), n_permutations=10),
    )
    run_pipeline(cfg, stop_stage="MSN")
    sm = pd.read_csv(out / "strength_maps.csv")
    assert len(sm) == 12  # 16 subjects − 4 relabelled OTHER
    mean_grp = pd.read_csv(out / "mean_msn_per_group.csv")
    assert not any("OTHER" in c for c in mean_grp.columns)


def test_resume_from_contrast_hydrates(df_cfg):
    cfg, out = df_cfg
    run_pipeline(cfg, stop_stage="MSN")
    # New pipeline instance resumes using persisted strength_maps.csv.
    ctx = Pipeline(cfg).run(start_stage="CONTRAST", stop_stage="CONTRAST")
    assert "strength_maps" in ctx
    assert (out / "case_control_difference_maps.csv").exists()
