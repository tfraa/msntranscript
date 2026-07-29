"""Dispatch + labelling of the corrected vs engine (frozen-rank) GSEA backends."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from msnpip import engine as eng
from msnpip.config import EngineConfig
from msnpip.errors import MsnpipEngineError


class FakeRunner:
    """Stands in for PLSGenes / CorrAnalysis: writes what the engine writes."""

    def __init__(self, filename="gsea_pls1_results.tsv"):
        self.filename = filename
        self.calls: list[dict] = []

    def gsea(self, **kwargs):
        self.calls.append(kwargs)
        outdir = Path(kwargs["outdir"])
        pd.DataFrame(
            {"Term": ["A"], "es": [0.4], "nes": [1.1], "p_val": [0.2], "fdr": [0.5]}
        ).to_csv(outdir / self.filename, index=False, sep="\t")


def test_engine_gsea_is_relabelled_never_pooled_with_corrected(tmp_path):
    runner = FakeRunner()
    eng._run_engine_gsea(runner, "gs.gmt", tmp_path, EngineConfig(), kind="pls")

    written = [p.name for p in tmp_path.glob("*.tsv")]
    assert written == ["gseafrozen_pls1_results.tsv"]
    # The curation step derives `enrichment` from the filename prefix — that is the
    # mechanism keeping the invalid table out of the corrected backend's rows.
    assert written[0].partition("_")[0] == "gseafrozen" != "gsea"


def test_engine_gsea_defaults_to_the_engines_own_1000_surrogates(tmp_path):
    runner = FakeRunner()
    eng._run_engine_gsea(runner, "gs.gmt", tmp_path, EngineConfig(n_permutations=20000), kind="pls")
    assert runner.calls[0]["n_iter"] == 1000  # not 20000


def test_engine_gsea_n_iter_is_overridable(tmp_path):
    runner = FakeRunner()
    cfg = EngineConfig(n_permutations=20000, gsea_engine_n_iter=20000)
    eng._run_engine_gsea(runner, "gs.gmt", tmp_path, cfg, kind="pls")
    assert runner.calls[0]["n_iter"] == 20000


def test_corr_path_uses_n_perm_not_n_iter(tmp_path):
    runner = FakeRunner(filename="gsea_corr_results.tsv")
    eng._run_engine_gsea(runner, "gs.gmt", tmp_path, EngineConfig(), kind="corr")
    assert "n_perm" in runner.calls[0] and "n_iter" not in runner.calls[0]
    # No pls<N> in the engine's corr filename → component defaults to 1.
    assert [p.name for p in tmp_path.glob("*.tsv")] == ["gseafrozen_pls1_results.tsv"]


def test_staging_dir_is_always_cleaned_up(tmp_path):
    runner = FakeRunner()
    eng._run_engine_gsea(runner, "gs.gmt", tmp_path, EngineConfig(), kind="pls")
    assert not [p for p in tmp_path.iterdir() if p.is_dir()]


def test_silent_engine_failure_is_an_error_not_a_missing_file(tmp_path):
    class Silent:
        def gsea(self, **kwargs):  # writes nothing
            pass

    with pytest.raises(MsnpipEngineError, match="wrote no table"):
        eng._run_engine_gsea(Silent(), "gs.gmt", tmp_path, EngineConfig(), kind="pls")
    assert not [p for p in tmp_path.iterdir() if p.is_dir()]


@pytest.mark.parametrize(
    ("backend", "corrected", "frozen"),
    [("corrected", True, False), ("engine", False, True), ("both", True, True)],
)
def test_backend_selector_semantics(backend, corrected, frozen):
    cfg = EngineConfig(gsea_backend=backend)
    assert (cfg.gsea_backend in ("corrected", "both")) is corrected
    assert (cfg.gsea_backend in ("engine", "both")) is frozen


def test_default_is_the_corrected_backend():
    assert EngineConfig().gsea_backend == "corrected"
    assert EngineConfig().gsea_engine_n_iter is None
