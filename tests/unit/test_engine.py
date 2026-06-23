"""Unit tests for msnpip.engine — T3.1. Engine is monkeypatched.

The real engine is exercised once in test_engine_integration.py (slow).
"""
from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

import msnpip.engine as engine
from msnpip.config import EngineConfig
from msnpip.engine import _primary_enrichment, run_transcriptomics
from msnpip.errors import MsnpipEngineError, MsnpipSurfaceNullError


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

def _fake_result(null_method: str = "vasa"):
    """A minimal stand-in for PLSResult/CorrelationResult."""
    return types.SimpleNamespace(metadata=types.SimpleNamespace(null_method=null_method))


def _labels_df(n_left: int = 34, n_right: int = 0) -> pd.DataFrame:
    rows = [{"id": i, "label": f"r{i}", "hemisphere": "L", "structure": "cort"} for i in range(n_left)]
    rows += [
        {"id": n_left + i, "label": f"r{i}", "hemisphere": "R", "structure": "cort"}
        for i in range(n_right)
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def patched_engine(monkeypatch):
    """Patch imt.run_pls / run_corr to record calls and return fake results."""
    calls: list[dict] = []

    def fake_run_pls(data, **kwargs):
        calls.append({"method": "pls", "data": np.asarray(data), **kwargs})
        return _fake_result(kwargs.get("null_method", "vasa"))

    def fake_run_corr(data, **kwargs):
        calls.append({"method": "corr", "data": np.asarray(data), **kwargs})
        return _fake_result(kwargs.get("null_method", "vasa"))

    monkeypatch.setattr(engine.imt, "run_pls", fake_run_pls, raising=False)
    monkeypatch.setattr(engine.imt, "run_corr", fake_run_corr, raising=False)
    return calls


# ---------------------------------------------------------------------------
# _primary_enrichment
# ---------------------------------------------------------------------------

class TestPrimaryEnrichment:
    def test_ensemble_is_primary_over_gsea(self):
        assert _primary_enrichment(("ensemble", "gsea")) == "ensemble"

    def test_gsea_only_yields_none(self):
        assert _primary_enrichment(("gsea",)) == "none"

    def test_ora_primary(self):
        assert _primary_enrichment(("ora", "gsea")) == "ora"


# ---------------------------------------------------------------------------
# run_transcriptomics — happy path
# ---------------------------------------------------------------------------

class TestRunTranscriptomics:
    def test_runs_all_methods(self, patched_engine, tmp_path):
        cfg = EngineConfig(n_permutations=10)
        rmap = np.arange(34, dtype=float)
        out = run_transcriptomics(rmap, _labels_df(34), cfg, tmp_path, "FTD_vs_HC")
        assert set(out) == {"pls", "corr"}
        assert [c["method"] for c in patched_engine] == ["pls", "corr"]

    def test_forwards_engine_kwargs(self, patched_engine, tmp_path):
        cfg = EngineConfig(n_permutations=10)
        run_transcriptomics(np.arange(34.0), _labels_df(34), cfg, tmp_path, "tag")
        pls = next(c for c in patched_engine if c["method"] == "pls")
        assert pls["atlas"] == "dk"
        assert pls["null_method"] == "vasa"
        assert pls["enrichment_method"] == "ensemble"
        assert pls["run_gsea"] is True
        assert pls["gene_set"] == cfg.gene_sets
        assert pls["n_components"] == 1
        # corr must NOT receive PLS-only kwargs
        corr = next(c for c in patched_engine if c["method"] == "corr")
        assert "n_components" not in corr

    def test_output_dir_layout(self, patched_engine, tmp_path):
        cfg = EngineConfig(n_permutations=10)
        run_transcriptomics(np.arange(34.0), _labels_df(34), cfg, tmp_path, "FTD_vs_HC")
        assert (tmp_path / "FTD_vs_HC" / "pls").is_dir()
        assert (tmp_path / "FTD_vs_HC" / "corr").is_dir()
        pls = next(c for c in patched_engine if c["method"] == "pls")
        assert pls["output_dir"] == tmp_path / "FTD_vs_HC" / "pls"

    def test_single_method(self, patched_engine, tmp_path):
        cfg = EngineConfig(methods=("pls",), n_permutations=10)
        out = run_transcriptomics(np.arange(34.0), _labels_df(34), cfg, tmp_path, "tag")
        assert set(out) == {"pls"}

    def test_both_hemisphere_splits_input(self, patched_engine, tmp_path):
        cfg = EngineConfig(hemisphere="both", n_permutations=10)
        rmap = np.arange(68, dtype=float)  # 34 L then 34 R
        run_transcriptomics(rmap, _labels_df(34, 34), cfg, tmp_path, "tag")
        pls = next(c for c in patched_engine if c["method"] == "pls")
        assert pls["data"].shape == (34,)
        assert pls["input_rh"].shape == (34,)
        np.testing.assert_array_equal(pls["data"], np.arange(34.0))
        np.testing.assert_array_equal(pls["input_rh"], np.arange(34.0, 68.0))


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

class TestErrors:
    def test_length_mismatch_raises(self, patched_engine, tmp_path):
        cfg = EngineConfig(n_permutations=10)
        with pytest.raises(MsnpipEngineError, match="length"):
            run_transcriptomics(np.arange(10.0), _labels_df(34), cfg, tmp_path, "tag")

    def test_surface_null_fallback_raises(self, monkeypatch, tmp_path):
        def fallback_pls(data, **kwargs):
            return _fake_result(null_method="random")
        monkeypatch.setattr(engine.imt, "run_pls", fallback_pls, raising=False)
        cfg = EngineConfig(methods=("pls",), n_permutations=10)
        with pytest.raises(MsnpipSurfaceNullError, match="random"):
            run_transcriptomics(np.arange(34.0), _labels_df(34), cfg, tmp_path, "tag")

    def test_require_surface_null_false_allows_random(self, monkeypatch, tmp_path):
        def fallback_pls(data, **kwargs):
            return _fake_result(null_method="random")
        monkeypatch.setattr(engine.imt, "run_pls", fallback_pls, raising=False)
        cfg = EngineConfig(methods=("pls",), require_surface_null=False, n_permutations=10)
        out = run_transcriptomics(np.arange(34.0), _labels_df(34), cfg, tmp_path, "tag")
        assert "pls" in out

    def test_engine_exception_wrapped(self, monkeypatch, tmp_path):
        def boom(data, **kwargs):
            raise ValueError("engine exploded")
        monkeypatch.setattr(engine.imt, "run_pls", boom, raising=False)
        cfg = EngineConfig(methods=("pls",), n_permutations=10)
        with pytest.raises(MsnpipEngineError, match="exploded") as excinfo:
            run_transcriptomics(np.arange(34.0), _labels_df(34), cfg, tmp_path, "tag")
        assert isinstance(excinfo.value.__cause__, ValueError)
