"""Engine-contract test — guard against imaging-transcriptomics layout drift.

``msnpip.engine`` (and the CLI) reach into the *pinned* engine's internal API so
PLS can be fit once and enriched per gene set, and so the spatial-null / GSEA
compatibility shims can monkeypatch the right targets.  ``run_transcriptomics``
guards these with a clear ``MsnpipEngineError`` at call time, but that only fires
during an (expensive) run.

This fast, deterministic test asserts every symbol the engine wrapper depends on
still exists at the expected location.  If the pinned engine is ever bumped and
its layout moves, this fails immediately with the exact missing name(s) — re-pin
to commit ``e6a2c237`` or update ``msnpip.engine`` to the new layout.
"""

from __future__ import annotations

import importlib

import pytest

# (module path, attribute) pairs that msnpip.engine / cli import or monkeypatch.
# Mirror the guarded import block in engine._run_pls_fit_once_enrich_many plus
# the shim targets and the imt.* dispatch functions.
_CONTRACT: list[tuple[str, str]] = [
    # fit-once / enrich-many PLS workflow primitives
    ("imaging_transcriptomics.models", "PLSResult"),
    ("imaging_transcriptomics.nulls", "permute_scan_values"),
    ("imaging_transcriptomics.pls", "PLSAnalysis"),
    ("imaging_transcriptomics.scan", "regional_values_frame"),
    ("imaging_transcriptomics.serialization", "write_result_bundle"),
    ("imaging_transcriptomics.workflows.shared", "pls_components"),
    ("imaging_transcriptomics.workflows.shared", "prepare_analysis_inputs"),
    ("imaging_transcriptomics.workflows.shared", "result_metadata"),
    # top-level dispatch / introspection used by engine + cli
    ("imaging_transcriptomics", "run_pls"),
    ("imaging_transcriptomics", "run_corr"),
    ("imaging_transcriptomics", "build_run_config"),
    ("imaging_transcriptomics", "atlas_table"),
    ("imaging_transcriptomics", "list_atlases"),
    ("imaging_transcriptomics", "get_atlas"),
    # shim targets and the engine exception the null policy matches on
    ("imaging_transcriptomics.gene_stats.pls", "result_column"),
    ("imaging_transcriptomics.gene_stats.pls", "PLSGenes"),
    ("imaging_transcriptomics.exceptions", "NullModelError"),
    ("neuromaps.nulls.spins", "load_gifti"),
]


@pytest.mark.parametrize(("module_path", "attr"), _CONTRACT)
def test_engine_symbol_present(module_path: str, attr: str) -> None:
    try:
        mod = importlib.import_module(module_path)
    except ImportError as exc:  # pragma: no cover - missing dependency surfaces clearly
        pytest.fail(f"engine contract: cannot import {module_path!r} ({exc})")
    assert hasattr(mod, attr), (
        f"engine contract broken: {module_path}.{attr} is missing. "
        "The pinned imaging-transcriptomics API that msnpip.engine relies on has "
        "drifted — re-pin to commit e6a2c237 or update msnpip.engine."
    )


def test_pls_analysis_has_boot_pls() -> None:
    """PLS is fit via ``PLSAnalysis(...).boot_pls(...)`` in the fit-once path."""
    from imaging_transcriptomics.pls import PLSAnalysis

    assert hasattr(PLSAnalysis, "boot_pls"), (
        "engine contract broken: PLSAnalysis.boot_pls is missing — "
        "engine._run_pls_fit_once_enrich_many drives PLS through it."
    )


def test_pls_genes_exposes_enrichment_backends() -> None:
    """Per-gene-set enrichment is called on ``analysis.gene_results.results``."""
    from imaging_transcriptomics.gene_stats.pls import PLSGenes

    missing = [m for m in ("ensemble", "gsea", "ora", "compute") if not hasattr(PLSGenes, m)]
    assert not missing, (
        f"engine contract broken: PLSGenes is missing {missing} — these are the "
        "enrichment backends msnpip drives per gene set after a single PLS fit."
    )
