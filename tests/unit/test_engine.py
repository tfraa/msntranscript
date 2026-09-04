"""Unit tests for msnpip.engine — T3.1.

Both the PLS and corr paths fit once then enrich each gene set via the engine's
workflow primitives; those primitives are monkeypatched here.  The corr path
drives the engine's ``CorrAnalysis`` (not the removed ``imt.run_corr``) and runs
the *same* corrected enrichment as PLS.  The real engine is exercised once in
test_engine_integration.py (slow).
"""

from __future__ import annotations

import logging
import types
from pathlib import Path

import imaging_transcriptomics.corr as _corr_mod
import imaging_transcriptomics.nulls as _nulls
import imaging_transcriptomics.pls as _pls_mod
import imaging_transcriptomics.scan as _scan_mod
import imaging_transcriptomics.serialization as _ser
import imaging_transcriptomics.workflows.shared as _shared
import numpy as np
import pandas as pd
import pytest
from imaging_transcriptomics.exceptions import NullModelError

import msnpip.engine as engine
from msnpip.config import EngineConfig
from msnpip.engine import _primary_enrichment, _resolve_geneset, run_transcriptomics
from msnpip.errors import MsnpipEngineError, MsnpipSurfaceNullError

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _fake_result(null_method: str = "vasa"):
    """A minimal stand-in for CorrelationResult."""
    return types.SimpleNamespace(metadata=types.SimpleNamespace(null_method=null_method))


def _labels_df(n_left: int = 34, n_right: int = 0) -> pd.DataFrame:
    rows = [
        {"id": i, "label": f"r{i}", "hemisphere": "L", "structure": "cort"} for i in range(n_left)
    ]
    rows += [
        {"id": n_left + i, "label": f"r{i}", "hemisphere": "R", "structure": "cort"}
        for i in range(n_right)
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def patched(monkeypatch):
    """Patch the PLS + corr workflow primitives; record what happened."""
    rec: dict = {
        "fit_count": 0,
        "permute_null": [],
        "permute_resolved": "vasa",
        "permute_raises_on": None,
        "enrich": [],
        "bundle_dirs": [],
        "corr": [],
        "corr_fit_count": 0,
        "ensemble_n_iter": None,
        "prepare_data": None,
        "prepare_input_rh": "unset",
    }

    class FakeResults:
        def compute(self, n_jobs: int = 1):
            pass

        def _write(self, backend, gene_set, outdir):
            rec["enrich"].append({"backend": backend, "gene_set": gene_set, "outdir": str(outdir)})
            Path(outdir, f"{backend}_pls1_results.tsv").write_text(
                "Term\tz_score\tp_val\tfdr\nT1\t1.0\t0.1\t0.2\n", encoding="utf-8"
            )

        def ensemble(self, gene_set, outdir, n_iter=1000, geneset_organism="Human", **k):
            rec["ensemble_n_iter"] = n_iter  # engine default is 1000 — must be overridden
            self._write("ensemble", gene_set, outdir)

        def gsea(self, gene_set, outdir, geneset_organism="Human", **k):
            self._write("gsea", gene_set, outdir)

        def ora(self, gene_set, outdir, p_threshold=None, geneset_organism="Human", **k):
            # Mirror the toolbox: one table per direction, and no direction column
            # (engine._run_toolbox_ora is what adds it and merges them).
            rec["enrich"].append({"backend": "ora", "gene_set": gene_set, "outdir": str(outdir)})
            for direction in ("up", "down"):
                Path(outdir, f"ora_pls1_{direction}.tsv").write_text(
                    "Term\tselected_size\todds_ratio\tp_value\tfdr\nT1\t10\t2.0\t0.01\t0.02\n",
                    encoding="utf-8",
                )

    class FakePLSAnalysis:
        def __init__(self, imaging, gene_exp, n_components=1, var=None, n_iter=10, n_jobs=1):
            rec["fit_count"] += 1
            self.n_components = n_components
            self.components_var = np.array([0.4])
            self.p_val = np.array([0.01])
            self.gene_results = types.SimpleNamespace(results=FakeResults())

        def boot_pls(self, *a, **k):
            pass

    class FakeCorrGenes:
        # Minimal stand-in for the engine's CorrGenes (what the adapter reads).
        # `pval` feeds ORA's `p` tail; CorrGenes.sort_genes() keeps genes/corr/
        # pval/boot_corr in one shared order, so these are element-wise aligned.
        genes = np.array([["g0"], ["g1"], ["g2"]], dtype=object)
        corr = np.array([[0.5, -0.2, 0.1]])
        pval = np.array([[0.01, 0.02, 0.90]])
        boot_corr = np.zeros((3, 4), dtype=float)

    class FakeCorrAnalysis:
        def __init__(self, n_iterations=10, n_genes=None, store_boot_corr=True, n_jobs=1):
            rec["corr_fit_count"] += 1
            self.gene_results = types.SimpleNamespace(results=FakeCorrGenes())

        def bootstrap_correlation(self, *a, **k):
            pass

        def ensemble(self, gene_set, outdir=None, n_perm=1000, geneset_organism="Human", **k):
            rec["enrich"].append(
                {"backend": "ensemble", "gene_set": gene_set, "outdir": str(outdir)}
            )
            return pd.DataFrame({"Term": ["T1"], "z_score": [1.0], "p_val": [0.1], "fdr": [0.2]})

        def ora(self, gene_set, outdir=None, p_threshold=0.05, geneset_organism="Human", **k):
            # The toolbox's correlation ORA lives on CorrAnalysis, not the adapter.
            rec["enrich"].append({"backend": "ora", "gene_set": gene_set, "outdir": str(outdir)})
            for direction in ("up", "down"):
                Path(outdir, f"ora_corr_{direction}.tsv").write_text(
                    "Term\tselected_size\todds_ratio\tp_value\tfdr\nT1\t10\t2.0\t0.01\t0.02\n",
                    encoding="utf-8",
                )

    def fake_prepare(data, config, input_rh=None):
        rec["prepare_data"] = np.asarray(data)
        rec["prepare_input_rh"] = None if input_rh is None else np.asarray(input_rh)
        extracted = types.SimpleNamespace(values=np.asarray(data))
        gene_labels = np.array(["g0", "g1", "g2"], dtype=object)  # needs a .shape for corr
        return extracted, object(), gene_labels, np.asarray(data)

    def fake_permute(extracted, n_permutations, null_method, seed):
        rec["permute_null"].append(null_method)
        if rec["permute_raises_on"] == null_method:
            raise NullModelError(f"cannot generate nulls with {null_method!r}")
        return object(), rec["permute_resolved"]

    def fake_build_run_config(method, **kw):
        return types.SimpleNamespace(method=method, **kw)

    # GSEA runs through msnpip's corrected backend; ORA is the TOOLBOX's own
    # res_obj.ora / analysis.ora, staged and merged by engine._run_toolbox_ora.
    def fake_corrected_gsea(res_obj, gene_set, outdir, geneset_organism="Human", **k):
        rec["enrich"].append({"backend": "gsea", "gene_set": gene_set, "outdir": str(outdir)})
        Path(outdir, "gsea_pls1_results.tsv").write_text("Term\tp_val\n1\t0.1\n", encoding="utf-8")

    monkeypatch.setattr(_shared, "prepare_analysis_inputs", fake_prepare, raising=True)
    monkeypatch.setattr(_shared, "pls_components", lambda a, g, e, o: (), raising=True)
    monkeypatch.setattr(
        _shared,
        "result_metadata",
        lambda extracted, config, null_method, n_components=None: types.SimpleNamespace(
            null_method=null_method
        ),
        raising=True,
    )
    monkeypatch.setattr(_nulls, "permute_scan_values", fake_permute, raising=True)
    monkeypatch.setattr(_pls_mod, "PLSAnalysis", FakePLSAnalysis, raising=True)
    monkeypatch.setattr(_scan_mod, "regional_values_frame", lambda e: None, raising=True)
    monkeypatch.setattr(
        _ser,
        "write_result_bundle",
        lambda result, out_dir: rec["bundle_dirs"].append(str(out_dir)),
        raising=True,
    )
    monkeypatch.setattr(engine.imt, "build_run_config", fake_build_run_config, raising=False)
    monkeypatch.setattr(_corr_mod, "CorrAnalysis", FakeCorrAnalysis, raising=True)
    monkeypatch.setattr(
        _shared,
        "corr_gene_table",
        lambda analysis: pd.DataFrame({"gene": ["g0"], "score": [0.5], "p": [0.1], "fdr": [0.2]}),
        raising=True,
    )
    monkeypatch.setattr(engine, "run_corrected_gsea", fake_corrected_gsea, raising=True)
    return rec


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
# Gene-set resolution to bundled .gmt files
# ---------------------------------------------------------------------------


class TestGenesetResolution:
    def test_bundled_names_resolve_to_gmt(self):
        for name in ("GO_Biological_Process_2025", "KEGG_2021_H", "DisGeNET"):
            resolved = _resolve_geneset(name)
            assert resolved.endswith(".gmt")
            assert Path(resolved).exists()

    def test_kegg_alias(self):
        assert _resolve_geneset("KEGG_2021_Human").endswith("KEGG_2021_H.gmt")

    def test_unknown_passes_through(self):
        # lake/pooled are resolved by the engine itself, not bundled here.
        assert _resolve_geneset("lake") == "lake"


# ---------------------------------------------------------------------------
# run_transcriptomics — happy path
# ---------------------------------------------------------------------------


class TestRunTranscriptomics:
    def test_pls_only_runs(self, patched, tmp_path):
        cfg = EngineConfig(methods=("pls",), n_permutations=10)
        out = run_transcriptomics(np.arange(34.0), _labels_df(34), cfg, tmp_path, "FTD_vs_HC")
        assert set(out) == {"pls"}
        assert patched["fit_count"] == 1

    def test_corr_only_runs(self, patched, tmp_path):
        cfg = EngineConfig(methods=("corr",), n_permutations=10, enrichment_methods=("ensemble",))
        out = run_transcriptomics(np.arange(34.0), _labels_df(34), cfg, tmp_path, "FTD_vs_HC")
        assert set(out) == {"corr"}
        assert patched["corr_fit_count"] == 1
        # PLS path was not touched, and the corr bundle + enrichment landed under corr/.
        assert patched["fit_count"] == 0
        assert (tmp_path / "FTD_vs_HC" / "corr").is_dir()
        assert {e["backend"] for e in patched["enrich"]} == {"ensemble"}

    def test_corr_and_pls_both_run(self, patched, tmp_path):
        cfg = EngineConfig(
            methods=("pls", "corr"), n_permutations=10, enrichment_methods=("ensemble",)
        )
        out = run_transcriptomics(np.arange(34.0), _labels_df(34), cfg, tmp_path, "tag")
        assert set(out) == {"pls", "corr"}
        assert patched["fit_count"] == 1 and patched["corr_fit_count"] == 1

    def test_unsupported_method_raises(self, patched, tmp_path):
        cfg = EngineConfig(methods=("gedar",), n_permutations=10)  # type: ignore[arg-type]
        with pytest.raises(MsnpipEngineError, match="expected 'pls' or 'corr'"):
            run_transcriptomics(np.arange(34.0), _labels_df(34), cfg, tmp_path, "tag")

    def test_gcea_uses_all_configured_permutations(self, patched, tmp_path):
        """GCEA must run on every surrogate, not the engine's 1000-surrogate default.

        The engine's ``ensemble(n_iter=1000)`` default silently caps the empirical
        p at 1/1001, which alone can make larger gene sets unable to reach BH
        significance — so the configured count has to be passed through.
        """
        cfg = EngineConfig(methods=("pls",), n_permutations=20000, enrichment_methods=("ensemble",))
        run_transcriptomics(np.arange(34.0), _labels_df(34), cfg, tmp_path, "tag")
        assert patched["ensemble_n_iter"] == 20000

    def test_skipped_enrichment_backend_is_warned(self, patched, tmp_path, caplog):
        """A backend left out of --enrichment must be called out, not silently dropped."""
        cfg = EngineConfig(
            methods=("pls",), n_permutations=10, enrichment_methods=("ensemble", "ora")
        )
        with caplog.at_level(logging.WARNING, logger="msnpip.engine"):
            run_transcriptomics(np.arange(34.0), _labels_df(34), cfg, tmp_path, "tag")
        assert "NOT requested" in caplog.text and "gsea" in caplog.text

    def test_full_enrichment_warns_nothing(self, patched, tmp_path, caplog):
        cfg = EngineConfig(
            methods=("pls",), n_permutations=10, enrichment_methods=("ensemble", "gsea", "ora")
        )
        with caplog.at_level(logging.WARNING, logger="msnpip.engine"):
            run_transcriptomics(np.arange(34.0), _labels_df(34), cfg, tmp_path, "tag")
        assert "NOT requested" not in caplog.text

    def test_pls_fits_once_enriches_each_geneset(self, patched, tmp_path):
        cfg = EngineConfig(
            methods=("pls",),
            n_permutations=10,
            enrichment_methods=("ensemble",),
            gene_sets=("lake", "GO_Biological_Process_2025"),
        )
        run_transcriptomics(np.arange(34.0), _labels_df(34), cfg, tmp_path, "tag")
        # Fit happened once; enrichment ran for both gene sets.
        assert patched["fit_count"] == 1
        labels = {Path(e["outdir"]).name for e in patched["enrich"]}
        assert labels == {"lake", "GO_Biological_Process_2025"}
        # Each enrichment table is written under enrichment/<label>/.
        assert (tmp_path / "tag" / "pls" / "enrichment" / "lake").is_dir()
        assert (tmp_path / "tag" / "pls" / "enrichment" / "GO_Biological_Process_2025").is_dir()

    def test_enrichment_uses_resolved_gmt_path(self, patched, tmp_path):
        cfg = EngineConfig(
            methods=("pls",),
            n_permutations=10,
            enrichment_methods=("ensemble",),
            gene_sets=("GO_Biological_Process_2025",),
        )
        run_transcriptomics(np.arange(34.0), _labels_df(34), cfg, tmp_path, "tag")
        passed = patched["enrich"][0]["gene_set"]
        assert passed.endswith("GO_Biological_Process_2025.gmt")
        assert Path(passed).exists()

    def test_multiple_backends_per_geneset(self, patched, tmp_path):
        cfg = EngineConfig(
            methods=("pls",),
            n_permutations=10,
            enrichment_methods=("ensemble", "ora"),
            gene_sets=("lake",),
        )
        run_transcriptomics(np.arange(34.0), _labels_df(34), cfg, tmp_path, "tag")
        backends = {e["backend"] for e in patched["enrich"]}
        assert backends == {"ensemble", "ora"}

    def test_pls_bundle_written_once(self, patched, tmp_path):
        cfg = EngineConfig(methods=("pls",), n_permutations=10, gene_sets=("lake", "pooled"))
        run_transcriptomics(np.arange(34.0), _labels_df(34), cfg, tmp_path, "tag")
        assert patched["bundle_dirs"] == [str(tmp_path / "tag" / "pls")]

    def test_output_dir_layout(self, patched, tmp_path):
        cfg = EngineConfig(methods=("pls",), n_permutations=10)
        run_transcriptomics(np.arange(34.0), _labels_df(34), cfg, tmp_path, "FTD_vs_HC")
        assert (tmp_path / "FTD_vs_HC" / "pls").is_dir()

    def test_single_method(self, patched, tmp_path):
        cfg = EngineConfig(methods=("pls",), n_permutations=10)
        out = run_transcriptomics(np.arange(34.0), _labels_df(34), cfg, tmp_path, "tag")
        assert set(out) == {"pls"}

    def test_both_hemisphere_splits_input(self, patched, tmp_path):
        cfg = EngineConfig(hemisphere="both", n_permutations=10, methods=("pls",))
        rmap = np.arange(68, dtype=float)  # 34 L then 34 R
        run_transcriptomics(rmap, _labels_df(34, 34), cfg, tmp_path, "tag")
        np.testing.assert_array_equal(patched["prepare_data"], np.arange(34.0))
        np.testing.assert_array_equal(patched["prepare_input_rh"], np.arange(34.0, 68.0))


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestErrors:
    def test_length_mismatch_raises(self, patched, tmp_path):
        cfg = EngineConfig(n_permutations=10)
        with pytest.raises(MsnpipEngineError, match="length"):
            run_transcriptomics(np.arange(10.0), _labels_df(34), cfg, tmp_path, "tag")

    def test_surface_null_fallback_raises_when_fallback_disabled(self, patched, tmp_path):
        patched["permute_resolved"] = "random"  # degraded null
        cfg = EngineConfig(methods=("pls",), n_permutations=10, allow_null_fallback=False)
        with pytest.raises(MsnpipSurfaceNullError, match="random"):
            run_transcriptomics(np.arange(34.0), _labels_df(34), cfg, tmp_path, "tag")

    def test_degraded_null_allowed_with_fallback(self, patched, tmp_path):
        patched["permute_resolved"] = "random"
        cfg = EngineConfig(methods=("pls",), n_permutations=10)  # fallback on by default
        out = run_transcriptomics(np.arange(34.0), _labels_df(34), cfg, tmp_path, "tag")
        assert "pls" in out

    def test_null_error_triggers_auto_retry(self, patched, tmp_path):
        patched["permute_raises_on"] = "vasa"  # vasa fails → retry 'auto'
        cfg = EngineConfig(methods=("pls",), n_permutations=10)
        out = run_transcriptomics(np.arange(34.0), _labels_df(34), cfg, tmp_path, "tag")
        assert "pls" in out
        assert patched["permute_null"] == ["vasa", "auto"]

    def test_enable_annot_shim_idempotent(self):
        engine.enable_annot_surface_nulls()
        engine.enable_annot_surface_nulls()  # no crash on second call

    def test_pls_exception_wrapped(self, patched, tmp_path, monkeypatch):
        def boom(data, config, input_rh=None):
            raise ValueError("prep exploded")

        monkeypatch.setattr(_shared, "prepare_analysis_inputs", boom, raising=True)
        cfg = EngineConfig(methods=("pls",), n_permutations=10)
        with pytest.raises(MsnpipEngineError, match="exploded") as excinfo:
            run_transcriptomics(np.arange(34.0), _labels_df(34), cfg, tmp_path, "tag")
        assert isinstance(excinfo.value.__cause__, ValueError)


class TestEngineHemisphere:
    """``hemisphere="right"`` must reach the engine as a LEFT-hemisphere run.

    The right arm swaps only the phenotype (atlas_align puts the rh_* values into
    the left label order); the AHBA expression stays left-hemisphere, so telling
    the engine "right" would either fail or silently pair the map with a
    2-donor right-hemisphere expression matrix.
    """

    def test_right_is_reported_to_the_engine_as_left(self):
        from msnpip.engine import _engine_hemisphere

        assert _engine_hemisphere(EngineConfig(hemisphere="right")) == "left"

    def test_left_and_both_pass_through(self):
        from msnpip.engine import _engine_hemisphere

        assert _engine_hemisphere(EngineConfig(hemisphere="left")) == "left"
        assert _engine_hemisphere(EngineConfig(hemisphere="both")) == "both"


class TestEnrichmentPlanLogging:
    """The plan log must never name a backend that was not requested.

    A line reading "surrogates used by the spin-null enrichment backends:
    ensemble=20000" during an ORA-only run reads as though GCEA is running, and
    cost a real 20k-surrogate run to be killed by hand.
    """

    def _plan(self, caplog, backends):
        from msnpip.engine import _log_enrichment_plan

        caplog.clear()
        with caplog.at_level(logging.INFO, logger="msnpip.engine"):
            _log_enrichment_plan("pls", backends, ("lake",), 20000)
        return caplog.text

    def test_ora_only_never_mentions_ensemble_or_gsea_as_running(self, caplog):
        text = self._plan(caplog, ["ora"])
        assert "surrogates used by the spin-null" not in text
        # The "not requested" warning may name them; the surrogate line may not.
        for line in text.splitlines():
            if "surrogates" in line:
                assert "ensemble" not in line and "gsea" not in line

    def test_requested_spin_backends_are_named_with_the_count(self, caplog):
        text = self._plan(caplog, ["ensemble", "ora"])
        assert "surrogates used by the spin-null enrichment backend(s) ensemble: 20000" in text
        assert "gsea" not in text.split("surrogates used by")[1].split("\n")[0]
