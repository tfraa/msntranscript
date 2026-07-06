"""Tests for null-method stability comparison (vasa vs moran)."""

from __future__ import annotations

import pandas as pd

from msnpip.null_sensitivity import compare_stability, significant_terms


def _df(rows):
    return pd.DataFrame(rows)


def test_significant_terms_grouped_by_backend_and_geneset():
    df = _df(
        [
            {"enrichment": "gsea", "geneset": "lake", "Term": "Astro", "fdr": 0.01},
            {"enrichment": "gsea", "geneset": "lake", "Term": "Micro", "fdr": 0.20},
            {"enrichment": "ensemble", "geneset": "lake", "Term": "Astro", "fdr": 0.30},
        ]
    )
    sig = significant_terms(df, alpha=0.05)
    assert sig[("gsea", "lake")] == {"Astro"}
    assert sig[("ensemble", "lake")] == set()


def test_compare_stability_reports_overlap_and_flips():
    a = _df(
        [
            {"enrichment": "gsea", "geneset": "lake", "Term": "Astro", "fdr": 0.01},
            {"enrichment": "gsea", "geneset": "lake", "Term": "Micro", "fdr": 0.02},
        ]
    )
    b = _df(
        [
            {"enrichment": "gsea", "geneset": "lake", "Term": "Astro", "fdr": 0.03},
            {"enrichment": "gsea", "geneset": "lake", "Term": "Micro", "fdr": 0.40},
        ]
    )
    out = compare_stability(a, b, alpha=0.05, label_a="vasa", label_b="moran")
    row = out.iloc[0]
    assert row["n_sig_vasa"] == 2 and row["n_sig_moran"] == 1
    assert row["n_sig_both"] == 1
    assert row["jaccard"] == 0.5
    assert row["only_vasa"] == "Micro" and row["only_moran"] == ""


def test_empty_significance_is_trivially_stable():
    a = _df([{"enrichment": "gsea", "geneset": "lake", "Term": "Astro", "fdr": 0.9}])
    b = _df([{"enrichment": "gsea", "geneset": "lake", "Term": "Astro", "fdr": 0.8}])
    out = compare_stability(a, b)
    assert float(out.iloc[0]["jaccard"]) == 1.0
