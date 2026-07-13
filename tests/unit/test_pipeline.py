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


def _fake_run_transcriptomics(vec, labels_df, eng_cfg, base, tag):
    for method in eng_cfg.methods:
        (Path(base) / tag / method).mkdir(parents=True, exist_ok=True)
    return {m: {} for m in eng_cfg.methods}


def test_overview_violin_covers_all_inscope_groups(tmp_path, monkeypatch):
    import msnpip.pipeline as pipeline_mod

    info = make_synthetic_cohort(tmp_path / "data", n_case=8, n_control=8, seed=6)
    df = pd.read_csv(info["merged_path"])
    df.loc[df.index[:4], "group"] = "OTHER"  # third in-scope group
    three = tmp_path / "data" / "three.csv"
    df.to_csv(three, index=False)
    out = tmp_path / "out"
    cfg = PipelineConfig(
        io=IOConfig(dataframe=three),
        output=out,
        group_col="group",
        contrasts=(("FTD", "HC"), ("OTHER", "HC")),
        glm=GLMConfig(predictors=("age", "sex")),
        engine=EngineConfig(methods=("pls",), n_permutations=10, enrichment_methods=("ensemble",)),
    )
    monkeypatch.setattr(pipeline_mod.engine_mod, "run_transcriptomics", _fake_run_transcriptomics)
    run_pipeline(cfg, stop_stage="FIGURES")
    assert (out / "plots" / "overview_violin.png").exists()


def test_per_region_violins_generated(tmp_path, monkeypatch):
    import msnpip.pipeline as pipeline_mod

    info = make_synthetic_cohort(tmp_path / "data", n_case=10, n_control=10, seed=8)
    out = tmp_path / "out"
    cfg = PipelineConfig(
        io=IOConfig(dataframe=Path(info["merged_path"])),
        output=out,
        group_col="group",
        case="FTD",
        control="HC",
        glm=GLMConfig(predictors=("age", "sex")),
        engine=EngineConfig(methods=("pls",), n_permutations=10, enrichment_methods=("ensemble",)),
    )
    monkeypatch.setattr(pipeline_mod.engine_mod, "run_transcriptomics", _fake_run_transcriptomics)
    run_pipeline(cfg, stop_stage="FIGURES")
    # at least one per-region violin exists (fallback top-5 when nothing is FDR-sig)
    assert list((out / "plots").glob("FTD_vs_HC_region-*_violin.png"))


def test_pooled_cases_contrast_runs_alongside_per_contrast(tmp_path, monkeypatch):
    import msnpip.pipeline as pipeline_mod

    info = make_synthetic_cohort(tmp_path / "data", n_case=12, n_control=6, seed=9)
    df = pd.read_csv(info["merged_path"])
    df["group"] = ["1"] * 4 + ["2"] * 4 + ["3"] * 4 + ["0"] * 6  # cases 1/2/3, control 0
    multi = tmp_path / "data" / "multi.csv"
    df.to_csv(multi, index=False)
    out = tmp_path / "out"
    cfg = PipelineConfig(
        io=IOConfig(dataframe=multi),
        output=out,
        group_col="group",
        contrasts=(("1", "0"), ("2", "0"), ("3", "0")),
        glm=GLMConfig(predictors=("age", "sex")),
        engine=EngineConfig(methods=("pls",), n_permutations=10, pool_cases=True),
    )
    monkeypatch.setattr(pipeline_mod.engine_mod, "run_transcriptomics", _fake_run_transcriptomics)
    run_pipeline(cfg, stop_stage="CONTRAST")

    diff = pd.read_csv(out / "case_control_difference_maps.csv")
    cols = " ".join(diff.columns)
    assert "1_vs_0" in cols  # per-contrast (primary) still present
    assert "1+2+3_vs_0" in cols  # pooled supplementary present


def test_pooled_pairs_helper(tmp_path):
    cfg = PipelineConfig(
        io=IOConfig(dataframe=Path("x.csv")),
        output=tmp_path,
        group_col="g",
        engine=EngineConfig(pool_cases=True),
    )
    p = Pipeline(cfg)
    pooled = p._pooled_pairs([("1", "0"), ("2", "0"), ("3", "0")])
    assert pooled == [(("1", "2", "3"), "0")]
    # no pooling when only one case per control
    assert p._pooled_pairs([("1", "0")]) == []


def test_resume_from_contrast_hydrates(df_cfg):
    cfg, out = df_cfg
    run_pipeline(cfg, stop_stage="MSN")
    # New pipeline instance resumes using persisted strength_maps.csv.
    ctx = Pipeline(cfg).run(start_stage="CONTRAST", stop_stage="CONTRAST")
    assert "strength_maps" in ctx
    assert (out / "case_control_difference_maps.csv").exists()
