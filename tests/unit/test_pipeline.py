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
    assert (out / "00_inputs" / "merged_data.csv").exists()
    assert (out / "01_msn" / "strength_maps.csv").exists()
    # transcriptomics not reached
    assert not (out / "03_transcriptomics").exists()


def test_stop_at_contrast_writes_contrast_table(df_cfg):
    cfg, out = df_cfg
    run_pipeline(cfg, stop_stage="CONTRAST")
    assert (out / "02_stats" / "contrasts" / "FTD_vs_HC_contrast.csv").exists()
    df = pd.read_csv(out / "02_stats" / "contrasts" / "FTD_vs_HC_contrast.csv")
    assert "region" in df.columns and "beta" in df.columns
    assert len(df) == 68  # both-hemisphere MSN


def test_resume_from_contrast_hydrates(df_cfg):
    cfg, out = df_cfg
    run_pipeline(cfg, stop_stage="MSN")
    # New pipeline instance resumes using persisted strength_maps.csv.
    ctx = Pipeline(cfg).run(start_stage="CONTRAST", stop_stage="CONTRAST")
    assert "strength_maps" in ctx
    assert (out / "02_stats" / "contrasts" / "FTD_vs_HC_contrast.csv").exists()
