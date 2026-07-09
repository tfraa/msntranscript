"""The enrichment section must render one table per backend, correctly labelled.

Pins P1a: when both ensemble and GSEA run, the report emits a distinct table for
each (ensemble → z_score, gsea → nes), with a backend-specific caption (so an
ensemble table is never described as GSEA) and the BH-FDR denominator stated (P3).
"""

from __future__ import annotations

import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from msnpip.config import EngineConfig, IOConfig, PipelineConfig
from msnpip.report.builder import ReportBuilder


def test_contrast_section_emits_coupled_and_per_region_violins(tmp_path):
    out = tmp_path / "out"
    (out / "plots").mkdir(parents=True)
    for name in (
        "1_vs_0_violin.png",
        "1_vs_0_region-lh_bankssts_violin.png",
        "1_vs_0_region-lh_cuneus_violin.png",
    ):
        (out / "plots" / name).write_bytes(b"\x89PNG\r\n")

    cfg = PipelineConfig(io=IOConfig(dataframe=Path("x.csv")), output=out, engine=EngineConfig())
    rb = ReportBuilder(out, cfg)

    seen: list[str] = []
    rb._figure_page = lambda pdf, png, **kw: seen.append(Path(png).name) or True  # type: ignore[assignment]
    rb._open_page = lambda pdf: plt.figure()  # type: ignore[assignment]
    rb._heading = lambda *a, **k: 0.9  # type: ignore[assignment]
    rb._paragraphs = lambda *a, **k: None  # type: ignore[assignment]
    rb._close_page = lambda pdf, fig: plt.close(fig)  # type: ignore[assignment]
    rb._toc_mark = lambda *a, **k: None  # type: ignore[assignment]
    for m in (
        "_significant_regions_page",
        "_region_stats_pages",
        "_pls_parameters_page",
        "_top_genes_page",
        "_enrichment_section",
    ):
        setattr(rb, m, lambda *a, **k: None)

    res = types.SimpleNamespace(
        stat_type="beta",
        covariates=[],
        n_case=5,
        n_control=5,
        region_labels=["lh_bankssts", "lh_cuneus"],
    )
    rb._contrast_section(None, {}, "1_vs_0", res, "1", "0")

    assert "1_vs_0_violin.png" in seen  # coupled per-contrast violin
    assert "1_vs_0_region-lh_bankssts_violin.png" in seen  # per-region violins (correct dir)
    assert "1_vs_0_region-lh_cuneus_violin.png" in seen


def _curated_both_backends() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "method": ["pls"] * 4,
            "enrichment": ["ensemble", "ensemble", "gsea", "gsea"],
            "geneset": ["lake"] * 4,
            "component": [1, 1, 1, 1],
            "Term": ["Astro", "Micro", "Astro", "Micro"],
            "category_score": [0.5, -0.3, None, None],
            "z_score": [2.1, -1.8, None, None],
            "es": [None, None, 0.6, -0.4],
            "nes": [None, None, 1.9, -1.5],
            "p_val": [0.01, 0.03, 0.02, 0.04],
            "fdr": [0.05, 0.08, 0.06, 0.09],
            "null_method": ["vasa"] * 4,
        }
    )


def _render_tables(tmp_path):
    out = tmp_path / "out"
    out.mkdir(parents=True)
    _curated_both_backends().to_csv(out / "TAG_enrichment.csv", index=False)
    cfg = PipelineConfig(io=IOConfig(dataframe=Path("x.csv")), output=out, engine=EngineConfig())
    rb = ReportBuilder(out, cfg)

    calls: list[dict] = []
    rb._table_page = lambda pdf, title, df, **kw: calls.append(  # type: ignore[assignment]
        {"title": title, "cols": list(df.columns), "intro": kw.get("intro"), "df": df}
    )
    rb._figure_page = lambda *a, **k: False  # type: ignore[assignment]
    rb._enrichment_section(None, "TAG", "kick", "pretty")
    return calls


def test_both_backends_get_a_table(tmp_path):
    calls = _render_tables(tmp_path)
    titles = [c["title"] for c in calls]
    assert any("(ensemble)" in t for t in titles), titles
    assert any("(gsea)" in t for t in titles), titles


def test_backend_tables_use_correct_effect_columns(tmp_path):
    calls = _render_tables(tmp_path)
    by_backend = {"ensemble": None, "gsea": None}
    for c in calls:
        for b in by_backend:
            if f"({b})" in c["title"]:
                by_backend[b] = c
    # ensemble table carries z_score, not nes/es
    ens_cols = by_backend["ensemble"]["cols"]
    assert "z_score" in ens_cols and "nes" not in ens_cols and "es" not in ens_cols
    # gsea table carries nes/es, not z_score
    gsea_cols = by_backend["gsea"]["cols"]
    assert "nes" in gsea_cols and "z_score" not in gsea_cols


def test_caption_is_backend_specific_and_reports_fdr_denominator(tmp_path):
    calls = _render_tables(tmp_path)
    for c in calls:
        intro_text = " ".join(str(x) for x in (c["intro"] or []))
        assert "BH-FDR denominator: 2 categories tested" in intro_text
        if "(ensemble)" in c["title"]:
            assert "z_score is the enrichment effect" in intro_text
            assert "GSEA" not in intro_text  # not mislabelled
        if "(gsea)" in c["title"]:
            assert "re-ranked per spin surrogate" in intro_text
