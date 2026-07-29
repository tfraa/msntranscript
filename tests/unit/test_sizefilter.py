"""Category-size filtering (msnpip.genes.sizefilter)."""

from __future__ import annotations

from pathlib import Path

import pytest

from msnpip.genes.sizefilter import (
    apply_size_filter,
    matched_sizes,
    size_report,
    write_filtered_gmt,
)

# term -> members; the universe below matches only a subset of each.
GENESETS = {
    "tiny": ("A", "B"),  # 2 matched
    "small": ("A", "B", "C", "D"),  # 4 matched
    "mid": tuple(f"G{i}" for i in range(20)),  # 20 matched
    "big": (*(f"G{i}" for i in range(20)), "A", "B", "C"),  # 23 matched
    "absent": ("Z1", "Z2"),  # 0 matched
}
UNIVERSE = ["A", "B", "C", "D"] + [f"G{i}" for i in range(20)]


def test_matched_sizes_counts_overlap_not_gmt_size():
    sizes = matched_sizes(GENESETS, UNIVERSE)
    assert sizes == {"tiny": 2, "small": 4, "mid": 20, "big": 23, "absent": 0}


def test_size_report_partitions_every_term():
    r = size_report(GENESETS, UNIVERSE, min_size=4, max_size=20)
    assert (r.n_below, r.n_above, r.n_unmatched) == (1, 1, 1)  # tiny / big / absent
    assert r.n_terms_out == 2  # small, mid
    assert r.n_terms_in == r.n_terms_out + r.n_below + r.n_above + r.n_unmatched
    assert r.applied


def test_no_window_reports_but_does_not_filter():
    r = size_report(GENESETS, UNIVERSE)
    assert not r.applied
    assert r.n_terms_out == 4  # every matched term kept; only 'absent' excluded
    assert r.n_below == 0 and r.n_above == 0


def test_written_gmt_keeps_full_member_lists(tmp_path):
    path, r = write_filtered_gmt(GENESETS, UNIVERSE, tmp_path / "out.gmt", min_size=4, max_size=20)
    lines = [ln.split("\t") for ln in Path(path).read_text(encoding="utf-8").splitlines()]
    kept = {ln[0]: ln[2:] for ln in lines}
    assert set(kept) == {"small", "mid"}
    # Members are NOT restricted to the universe — each backend recomputes overlap.
    assert kept["small"] == list(GENESETS["small"])
    assert r.n_terms_out == 2


def test_apply_size_filter_passthrough_when_no_window(tmp_path):
    resource, r = apply_size_filter(GENESETS, UNIVERSE, outdir=tmp_path, label="x")
    assert resource is GENESETS  # untouched → run stays bit-reproducible
    assert not r.applied
    assert not list(tmp_path.glob("*.gmt"))


def test_apply_size_filter_writes_auditable_gmt(tmp_path):
    resource, r = apply_size_filter(
        GENESETS, UNIVERSE, min_size=4, max_size=20, outdir=tmp_path, label="x"
    )
    assert resource == str(tmp_path / "x_filtered.gmt")
    assert (tmp_path / "x_filtered.gmt").exists()
    assert r.n_terms_out == 2


def test_apply_size_filter_refuses_to_empty_the_geneset(tmp_path):
    with pytest.raises(ValueError, match="removed every term"):
        apply_size_filter(GENESETS, UNIVERSE, min_size=999, outdir=tmp_path, label="x")


def test_filtered_gmt_round_trips_through_the_engine_parser(tmp_path):
    from imaging_transcriptomics.genesets import as_geneset_mapping

    path, _ = write_filtered_gmt(GENESETS, UNIVERSE, tmp_path / "o.gmt", min_size=4, max_size=20)
    assert as_geneset_mapping(path) == {"small": GENESETS["small"], "mid": GENESETS["mid"]}
