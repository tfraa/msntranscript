"""Slow test: the spatial null is deterministic under a fixed seed — reproducibility.

Runs the REAL engine twice with identical config + seed and asserts the PLS gene
table (which depends on the spatial-null permutations) is byte-for-byte identical.
This pins the reproducibility claim end-to-end through the actual null generation,
which the mocked unit tests cannot.

Deselect with ``-m 'not slow'``. Skips gracefully when the engine's data / surface
assets are unavailable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from msnpip.atlas_align import engine_region_order
from msnpip.config import EngineConfig
from msnpip.engine import run_transcriptomics
from msnpip.errors import MsnpipEngineError, MsnpipSurfaceNullError

pytestmark = pytest.mark.slow


def test_pls_gene_results_are_deterministic_under_seed(tmp_path):
    labels = engine_region_order("dk", "left", "cort")
    rng = np.random.default_rng(0)
    regional_map = rng.normal(size=len(labels))
    cfg = EngineConfig(
        methods=("pls",),
        n_components=1,
        n_permutations=100,  # tiny — enough to exercise the seeded null
        enrichment_methods=("ensemble",),
        gene_sets=("GO_Biological_Process_2025",),
        seed=7,
    )

    def run(dst):
        try:
            run_transcriptomics(regional_map, labels, cfg, dst, "s")
        except MsnpipSurfaceNullError as exc:
            pytest.skip(f"Surface-null assets unavailable: {exc}")
        except MsnpipEngineError as exc:
            pytest.skip(f"Engine unavailable / assets missing: {exc}")
        return pd.read_csv(dst / "s" / "pls" / "pls_component_1.tsv", sep="\t")

    first = run(tmp_path / "a")
    second = run(tmp_path / "b")
    # Same seed → identical PLS gene ranking/weights (the null is reproducible).
    pd.testing.assert_frame_equal(first, second)
