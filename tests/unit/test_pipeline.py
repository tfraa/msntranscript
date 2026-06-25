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


def test_resume_from_contrast_hydrates(df_cfg):
    cfg, out = df_cfg
    run_pipeline(cfg, stop_stage="MSN")
    # New pipeline instance resumes using persisted strength_maps.csv.
    ctx = Pipeline(cfg).run(start_stage="CONTRAST", stop_stage="CONTRAST")
    assert "strength_maps" in ctx
    assert (out / "case_control_difference_maps.csv").exists()
