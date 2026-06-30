"""Unit tests for msnpip.cli — T5.5."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt

import msnpip.pipeline as pipeline_mod
from msnpip.cli import build_parser, main
from tests.fixtures.synthetic import make_synthetic_cohort


def _fake_run_transcriptomics(vec, labels_df, eng_cfg, base, tag):
    results = {}
    for method in eng_cfg.methods:
        d = Path(base) / tag / method
        d.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(2, 2))
        plt.plot([0, 1], [1, 0])
        fig.savefig(d / f"{method}.png")
        plt.close(fig)
        results[method] = {"fake": True}
    return results


class TestSimpleSubcommands:
    def test_list_genesets(self, capsys):
        assert main(["list-genesets"]) == 0
        assert "lake" in capsys.readouterr().out

    def test_list_atlases(self, capsys):
        assert main(["list-atlases"]) == 0
        assert "dk" in capsys.readouterr().out


class TestArgParsing:
    def test_method_and_enrichment_append(self):
        args = build_parser().parse_args(
            [
                "full",
                "--dataframe",
                "m.csv",
                "--output",
                "o",
                "--method",
                "pls",
                "--method",
                "corr",
                "--enrichment",
                "ensemble",
                "--enrichment",
                "gsea",
            ]
        )
        assert args.method == ["pls", "corr"]
        assert args.enrichment == ["ensemble", "gsea"]


class TestFullRun:
    def test_full_run_writes_report(self, tmp_path, monkeypatch):
        info = make_synthetic_cohort(tmp_path / "data", n_case=8, n_control=8, seed=4)
        out = tmp_path / "out"
        monkeypatch.setattr(
            pipeline_mod.engine_mod, "run_transcriptomics", _fake_run_transcriptomics
        )
        rc = main(
            [
                "full",
                "--dataframe",
                str(info["merged_path"]),
                "--output",
                str(out),
                "--group-col",
                "group",
                "--case",
                "FTD",
                "--control",
                "HC",
                "--predictors",
                "age",
                "sex",
                "--method",
                "pls",
                "--ncomp",
                "1",
                "--enrichment",
                "ensemble",
                "--n-perm",
                "10",
            ]
        )
        assert rc == 0
        assert (out / "merged_dataset.csv").exists()
        assert (out / "strength_maps.csv").exists()
