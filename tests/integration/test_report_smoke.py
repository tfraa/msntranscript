"""Slow end-to-end smoke test: real engine → report builds.

Runs the whole pipeline on a synthetic cohort against the REAL
imaging-transcriptomics engine with the full transcriptomics surface that tends
to break on engine / gseapy drift: PLS, *both* the ensemble and GSEA enrichment
backends, and *multiple* gene sets (fit once, enrich many).  The point is to
catch contract drift that the mocked integration test cannot — if the engine
runs end-to-end and the report builds, the whole stack is wired correctly.

Deselect with ``-m 'not slow'``.  Skips gracefully (rather than failing) when the
engine's data assets are unavailable — e.g. CI without the AHBA expression data
or neuromaps surface-null assets.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

from msnpip.config import EngineConfig, GLMConfig, IOConfig, PipelineConfig
from msnpip.errors import MsnpipEngineError, MsnpipSurfaceNullError
from msnpip.pipeline import run_pipeline
from tests.fixtures.synthetic import make_synthetic_cohort

pytestmark = pytest.mark.slow


def test_full_pipeline_real_engine_builds_report(tmp_path):
    info = make_synthetic_cohort(tmp_path / "data", n_case=10, n_control=10, seed=11)
    out = tmp_path / "out"
    gene_sets = ("KEGG_2021_H", "GO_Biological_Process_2025")
    cfg = PipelineConfig(
        io=IOConfig(dataframe=Path(info["merged_path"])),
        output=out,
        group_col="group",
        case="FTD",
        control="HC",
        glm=GLMConfig(predictors=("age", "sex")),
        engine=EngineConfig(
            methods=("pls",),
            n_components=1,
            n_permutations=50,  # tiny — smoke test, not a publication run
            enrichment_methods=("ensemble", "gsea"),
            gene_sets=gene_sets,
        ),
    )

    try:
        run_pipeline(cfg)
    except MsnpipSurfaceNullError as exc:
        pytest.skip(f"Surface-null assets unavailable: {exc}")
    except MsnpipEngineError as exc:
        pytest.skip(f"Engine unavailable / assets missing: {exc}")

    # The curated PLS gene table was produced (raw engine staging is discarded
    # after curation, so assert against the curated CSV set, not the bundle tree).
    assert (out / "FTD_vs_HC_pls.csv").exists()

    # Fit-once / enrich-many drove BOTH backends across BOTH gene sets through the
    # real engine — the curated enrichment table records the backend and gene set.
    enr = pd.read_csv(out / "FTD_vs_HC_enrichment.csv")
    assert {"enrichment", "geneset"} <= set(enr.columns)
    assert {"ensemble", "gsea"} <= set(enr["enrichment"]), (
        f"expected both ensemble and gsea rows, got {sorted(set(enr['enrichment']))} — "
        "the GSEA compatibility shim or an enrichment backend may have drifted"
    )
    assert set(gene_sets) <= set(enr["geneset"]), (
        f"expected both gene sets enriched, got {sorted(set(enr['geneset']))}"
    )

    # The report assembled over the real engine output.
    report = out / "report.pdf"
    assert report.exists()
    assert report.stat().st_size > 1024  # a real multi-page PDF, not an empty stub
