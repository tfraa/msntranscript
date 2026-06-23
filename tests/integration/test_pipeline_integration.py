"""Integration test: full LOAD→REPORT on the synthetic cohort — T5.6.

The transcriptomics engine is monkeypatched (writes a fake bundle) so this runs
fast in CI; atlas alignment and all msnpip stages run for real.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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
    """Write a minimal engine-like bundle and return a stand-in result dict."""
    results = {}
    for method in eng_cfg.methods:
        d = Path(base) / tag / method
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{method}_summary.tsv").write_text("gene\tweight\nGENE1\t0.12\n", encoding="utf-8")
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
        glm=GLMConfig(predictors=("age", "sex", "tiv"), exclude_covariates=("age",)),
        correlation=CorrelationConfig(variables=("age",), scope="global"),
        engine=EngineConfig(methods=("pls",), n_permutations=10, enrichment_methods=("ensemble",)),
    )
    return cfg, out


def test_full_pipeline_builds_tree_and_report(full_cfg, monkeypatch):
    cfg, out = full_cfg
    monkeypatch.setattr(pipeline_mod.engine_mod, "run_transcriptomics", _fake_run_transcriptomics)

    run_pipeline(cfg)

    # Output tree
    assert (out / "00_inputs" / "merged_data.csv").exists()
    assert (out / "00_inputs" / "schema.json").exists()
    assert (out / "00_inputs" / "resolved_config.yaml").exists()
    assert (out / "01_msn" / "strength_maps.csv").exists()
    assert (out / "01_msn" / "global_strength.csv").exists()
    assert (out / "02_stats" / "contrasts" / "FTD_vs_HC_contrast.csv").exists()
    assert (out / "02_stats" / "correlation" / "age__global.csv").exists()
    assert (out / "02_stats" / "sensitivity" / "FTD_vs_HC__drop_age.csv").exists()
    assert (out / "03_transcriptomics" / "FTD_vs_HC" / "pls" / "pls_plot.png").exists()
    assert (out / "05_report" / "Report.pdf").exists()
    assert (out / "05_report" / "run_log.txt").exists()

    # Manifest provenance
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["seed"] == 1234
    assert manifest["engine_commit"].startswith("e6a2c237")
    assert len(manifest["artifacts"]) > 5
    assert (out / "05_report" / "Report.pdf").stat().st_size > 0


def test_no_pickle_anywhere(full_cfg, monkeypatch):
    cfg, out = full_cfg
    monkeypatch.setattr(pipeline_mod.engine_mod, "run_transcriptomics", _fake_run_transcriptomics)
    run_pipeline(cfg)
    assert not list(out.rglob("*.pkl"))
    assert not list(out.rglob("*.pickle"))
