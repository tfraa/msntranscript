"""Integration test: full pipeline on the synthetic cohort — T5.6.

The transcriptomics engine is monkeypatched (writes a fake bundle) so this runs
fast in CI; atlas alignment and all msnpip stages run for real.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import msnpip.pipeline as pipeline_mod
from msnpip.config import (
    CorrelationConfig,
    EngineConfig,
    GLMConfig,
    IOConfig,
    PipelineConfig,
)
from msnpip.pipeline import run_pipeline
from tests.fixtures.synthetic import make_synthetic_cohort


def _fake_run_transcriptomics(vec, labels_df, eng_cfg, base, tag):
    """Write a minimal engine-like bundle (pls gene table + enrichment + plot)."""
    results = {}
    for method in eng_cfg.methods:
        d = Path(base) / tag / method
        d.mkdir(parents=True, exist_ok=True)
        if method == "pls":
            (d / "pls_component_1.tsv").write_text(
                "gene\tweight\tp\tfdr\nGENE1\t0.12\t0.01\t0.2\n", encoding="utf-8"
            )
            (d / "ensemble_lake_results.tsv").write_text(
                "Term\tp_val\tfdr\nEx1\t0.02\t0.4\n", encoding="utf-8"
            )
        fig = plt.figure(figsize=(3, 2))
        plt.plot([0, 1], [0, 1])
        fig.savefig(d / f"{method}_plot.png")
        plt.close(fig)
        results[method] = {"fake": True}
    return results


@pytest.fixture
def full_cfg(tmp_path):
    info = make_synthetic_cohort(tmp_path / "data", n_case=10, n_control=10, seed=9)
    out = tmp_path / "out"
    cfg = PipelineConfig(
        io=IOConfig(dataframe=Path(info["merged_path"])),
        output=out,
        group_col="group",
        case="FTD",
        control="HC",
        glm=GLMConfig(predictors=("age", "sex", "tiv")),
        correlation=CorrelationConfig(variables=("age",), scope="global"),
        engine=EngineConfig(methods=("pls",), n_permutations=10, enrichment_methods=("ensemble",)),
    )
    return cfg, out


def test_full_pipeline_builds_curated_outputs(full_cfg, monkeypatch):
    cfg, out = full_cfg
    monkeypatch.setattr(pipeline_mod.engine_mod, "run_transcriptomics", _fake_run_transcriptomics)

    run_pipeline(cfg)

    # Curated CSV set (issue 7)
    assert (out / "merged_dataset.csv").exists()
    assert (out / "strength_maps.csv").exists()
    assert (out / "mean_msn_per_group.csv").exists()
    assert (out / "case_control_difference_maps.csv").exists()
    assert (out / "FTD_vs_HC_pls.csv").exists()
    assert (out / "FTD_vs_HC_enrichment.csv").exists()

    # Plots
    assert (out / "plots" / "FTD_vs_HC_violin.png").exists()
    assert (out / "plots" / "age_scatter.png").exists()
    assert (out / "plots" / "FTD_vs_HC_tvalue_bars.png").exists()  # per-region t-value bars
    assert (out / "plots" / "FTD_mean_msn_matrix.png").exists()  # per-group similarity matrix
    assert (out / "plots" / "HC_mean_msn_matrix.png").exists()

    # corr method not requested → no corr table
    assert not (out / "FTD_vs_HC_corr.csv").exists()

    # Report (kept as an output; layout TBD)
    assert (out / "report.pdf").exists()

    # Verbose engine staging removed; no manifest / report / staged tree
    assert not (out / ".engine").exists()
    assert not (out / "manifest.json").exists()
    assert not (out / "03_transcriptomics").exists()


def _fake_degraded_null(vec, labels_df, eng_cfg, base, tag):
    """Engine bundle whose result reports a degraded (non-spin) resolved null."""
    import types

    d = Path(base) / tag / "pls"
    d.mkdir(parents=True, exist_ok=True)
    (d / "pls_component_1.tsv").write_text(
        "gene\tweight\tp\tfdr\nG1\t0.1\t0.01\t0.2\n", encoding="utf-8"
    )
    (d / "pls_summary.tsv").write_text(
        "component\tvar\tcumvar\tp\n1\t0.4\t0.4\t0.03\n", encoding="utf-8"
    )
    (d / "ensemble_lake_results.tsv").write_text(
        "Term\tz_score\tp_val\tfdr\nT1\t1.0\t0.02\t0.3\n", encoding="utf-8"
    )
    return {"pls": types.SimpleNamespace(metadata=types.SimpleNamespace(null_method="random"))}


def test_resolved_null_is_recorded_in_curated_outputs(tmp_path, monkeypatch):
    info = make_synthetic_cohort(tmp_path / "data", n_case=10, n_control=10, seed=3)
    out = tmp_path / "out"
    cfg = PipelineConfig(
        io=IOConfig(dataframe=Path(info["merged_path"])),
        output=out,
        group_col="group",
        case="FTD",
        control="HC",
        glm=GLMConfig(predictors=("age",)),
        engine=EngineConfig(methods=("pls",), n_permutations=10),
    )
    monkeypatch.setattr(pipeline_mod.engine_mod, "run_transcriptomics", _fake_degraded_null)
    run_pipeline(cfg, stop_stage="TRANSCRIPTOMICS")

    # A degraded (non-spin) null must be visible in the curated CSVs, not just logs.
    enr = pd.read_csv(out / "FTD_vs_HC_enrichment.csv")
    assert "null_method" in enr.columns
    assert (enr["null_method"] == "random").all()
    pls = pd.read_csv(out / "FTD_vs_HC_pls.csv")
    assert (pls["null_method"] == "random").all()


def test_no_pickle_anywhere(full_cfg, monkeypatch):
    cfg, out = full_cfg
    monkeypatch.setattr(pipeline_mod.engine_mod, "run_transcriptomics", _fake_run_transcriptomics)
    run_pipeline(cfg)
    assert not list(out.rglob("*.pkl"))
    assert not list(out.rglob("*.pickle"))
