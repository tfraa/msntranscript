"""Slow integration test for msnpip.engine against the REAL engine — T3.1.

Deselect with ``-m 'not slow'``.  Skips gracefully if the engine's surface-null
assets are unavailable (e.g. CI without neuromaps data) rather than failing.
"""

from __future__ import annotations

import numpy as np
import pytest

from msnpip.atlas_align import engine_region_order
from msnpip.config import EngineConfig
from msnpip.engine import run_transcriptomics
from msnpip.errors import MsnpipEngineError, MsnpipSurfaceNullError

pytestmark = pytest.mark.slow


def test_run_pls_on_synthetic_aligned_vector(tmp_path):
    # Canonical DK left-cortex label order straight from the engine.
    labels = engine_region_order("dk", "left", "cort")
    rng = np.random.default_rng(0)
    regional_map = rng.normal(size=len(labels))

    cfg = EngineConfig(
        methods=("pls",),
        n_permutations=100,  # tiny — this is a smoke test, not a publication run
        enrichment_methods=("ensemble",),
        gene_sets=("GO_Biological_Process_2025",),
    )

    try:
        results = run_transcriptomics(regional_map, labels, cfg, tmp_path, "synthetic")
    except MsnpipSurfaceNullError as exc:
        pytest.skip(f"Surface-null assets unavailable: {exc}")
    except MsnpipEngineError as exc:
        pytest.skip(f"Engine unavailable / assets missing: {exc}")

    assert set(results) == {"pls"}
    # Engine wrote its own bundle under the method directory.
    assert any((tmp_path / "synthetic" / "pls").iterdir())
    # Surface null must have been honoured (not a silent shuffle).
    assert results["pls"].metadata.null_method in ("vasa", "alexander_bloch", "moran")


def test_run_corr_on_synthetic_aligned_vector(tmp_path):
    # Same smoke test for the mass-univariate correlation backend: it must run the
    # spatial null, write the corr bundle, and produce the corrected enrichment.
    labels = engine_region_order("dk", "left", "cort")
    rng = np.random.default_rng(1)
    regional_map = rng.normal(size=len(labels))

    cfg = EngineConfig(
        methods=("corr",),
        n_permutations=100,  # tiny smoke test
        enrichment_methods=("ensemble", "gsea", "ora"),
        gene_sets=("GO_Biological_Process_2025",),
    )

    try:
        results = run_transcriptomics(regional_map, labels, cfg, tmp_path, "synthetic")
    except MsnpipSurfaceNullError as exc:
        pytest.skip(f"Surface-null assets unavailable: {exc}")
    except MsnpipEngineError as exc:
        pytest.skip(f"Engine unavailable / assets missing: {exc}")

    assert set(results) == {"corr"}
    bundle = tmp_path / "synthetic" / "corr"
    assert (bundle / "corr_genes.tsv").exists()
    assert results["corr"].metadata.null_method in ("vasa", "alexander_bloch", "moran")
    # Corrected enrichment landed under the pls1-named scheme the curation reads.
    enr = bundle / "enrichment" / "GO_Biological_Process_2025"
    assert (enr / "ensemble_pls1_results.tsv").exists()
    assert (enr / "gsea_pls1_results.tsv").exists()
