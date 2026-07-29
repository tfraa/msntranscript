"""Tests for the template-style ORA (Fisher over-representation, 3 tail rules)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from msnpip.genes.ora_mainstyle import (
    ORA_TAILS,
    P_THRESHOLD,
    TAIL_BACKENDS,
    TOP_N,
    Z_CUT,
    ora_table,
    run_ora,
    select_tails,
)


def _mapping(**terms):
    return {k: tuple(v) for k, v in terms.items()}


# ---------------------------------------------------------------------------
# z tail (|z| >= 3)
# ---------------------------------------------------------------------------


def test_positive_tail_enriched_term_is_flagged():
    n = 400
    genes = [f"g{i}" for i in range(n)]
    scores = np.zeros(n)
    scores[:20] = 5.0  # positive tail = g0..g19 (|z|>=3)
    # a term made mostly of the positive-tail genes must be enriched (low p, OR>1)
    enriched = genes[:15] + genes[100:105]
    random_term = genes[200:220]
    mapping = _mapping(ENR=enriched, RND=random_term)

    df = ora_table(genes, scores, mapping, tail="z")
    pos = df[df["direction"] == "positive"].set_index("Term")
    assert pos.loc["ENR", "p_val"] < 0.001
    assert pos.loc["ENR", "odds_ratio"] > 1
    assert pos.loc["ENR", "overlap"] == 15
    # a random term of tail-free genes is not enriched
    assert pos.loc["RND", "overlap"] == 0
    assert pos.loc["RND", "p_val"] > 0.05


def test_z_tail_uses_weight_cut_not_significance():
    # scores are the standardized loadings; only |z|>=cut define the tails
    genes = [f"g{i}" for i in range(100)]
    scores = np.linspace(-2, 2, 100)  # nothing reaches |z|>=3
    df = ora_table(genes, scores, _mapping(T=genes[:10]), tail="z")
    assert df.empty  # no genes in either tail -> no rows


def test_negative_tail_direction():
    n = 300
    genes = [f"g{i}" for i in range(n)]
    scores = np.zeros(n)
    scores[:15] = -4.0  # negative tail
    df = ora_table(genes, scores, _mapping(NEG=genes[:12]), tail="z")
    neg = df[df["direction"] == "negative"].set_index("Term")
    assert neg.loc["NEG", "overlap"] == 12
    assert neg.loc["NEG", "p_val"] < 0.001
    # no positive-tail rows exist
    assert (df["direction"] == "positive").sum() == 0


def test_z_cut_boundary_is_inclusive():
    genes = ["a", "b", "c"]
    scores = np.array([Z_CUT, -Z_CUT, 0.0])
    pos, neg = select_tails(genes, scores, tail="z")
    assert pos == {"a"} and neg == {"b"}


# ---------------------------------------------------------------------------
# p tail (nominal spin p <= 0.05)
# ---------------------------------------------------------------------------


def test_p_tail_selects_by_pvalue_and_splits_by_score_sign():
    genes = ["a", "b", "c", "d"]
    scores = np.array([2.0, -2.0, 5.0, -5.0])
    pvals = np.array([0.01, 0.01, 0.20, 0.20])
    pos, neg = select_tails(genes, scores, tail="p", pvals=pvals)
    # c/d have the largest |score| but fail the p threshold
    assert pos == {"a"} and neg == {"b"}


def test_p_tail_boundary_is_inclusive():
    genes = ["a", "b"]
    scores = np.array([1.0, 1.0])
    pvals = np.array([P_THRESHOLD, P_THRESHOLD + 1e-9])
    pos, _ = select_tails(genes, scores, tail="p", pvals=pvals)
    assert pos == {"a"}


def test_p_tail_requires_pvalues():
    with pytest.raises(ValueError, match="needs per-gene p-values"):
        select_tails(["a"], np.array([1.0]), tail="p")


def test_zero_score_is_in_neither_p_tail():
    genes = ["a"]
    pos, neg = select_tails(genes, np.array([0.0]), tail="p", pvals=np.array([0.001]))
    assert pos == set() and neg == set()


# ---------------------------------------------------------------------------
# topn tail (top/bottom 500)
# ---------------------------------------------------------------------------


def test_topn_selects_fixed_tail_sizes():
    n = 3 * TOP_N
    genes = [f"g{i}" for i in range(n)]
    scores = np.linspace(-10, 10, n)
    pos, neg = select_tails(genes, scores, tail="topn")
    assert len(pos) == TOP_N and len(neg) == TOP_N
    # highest scores are the last elements of the linspace
    assert f"g{n - 1}" in pos and "g0" in neg
    assert pos.isdisjoint(neg)


def test_topn_sign_guard_drops_wrong_signed_genes():
    # only 3 genes are positive, so the positive tail cannot reach TOP_N
    n = TOP_N + 10
    genes = [f"g{i}" for i in range(n)]
    scores = np.full(n, -1.0)
    scores[:3] = 1.0
    pos, _ = select_tails(genes, scores, tail="topn")
    assert pos == {"g0", "g1", "g2"}


def test_topn_ties_break_deterministically_by_gene_name():
    n = TOP_N + 50
    genes = [f"g{i:04d}" for i in range(n)]
    scores = np.ones(n)  # every score identical -> pure tie
    first, _ = select_tails(genes, scores, tail="topn")
    second, _ = select_tails(list(reversed(genes)), scores, tail="topn")
    assert first == second


# ---------------------------------------------------------------------------
# table plumbing
# ---------------------------------------------------------------------------


def test_table_records_the_tail_rule_and_size():
    n = 300
    genes = [f"g{i}" for i in range(n)]
    scores = np.zeros(n)
    scores[:20] = 5.0
    df = ora_table(genes, scores, _mapping(T=genes[:10]), tail="z")
    assert set(df["ora_tail"]) == {"z"}
    assert set(df["tail_size"]) == {20}


def test_unknown_tail_rule_raises():
    with pytest.raises(ValueError, match="Unknown ORA tail rule"):
        select_tails(["a"], np.array([1.0]), tail="nope")


# ---------------------------------------------------------------------------
# run_ora: namespace pairing + per-tail output files
# ---------------------------------------------------------------------------


def _res_obj(n=300):
    """A result object whose orig/boot namespaces are in DIFFERENT gene orders.

    This mirrors the pinned engine's PLS layout (orig sorted by weight, boot
    sorted by z-score) so a test fails loudly if the tail rules ever pair arrays
    across the two namespaces.
    """
    genes = np.array([[f"g{i}" for i in range(n)]], dtype=object)
    zscored = np.zeros((1, n))
    zscored[0, :20] = 5.0  # orig tail: g0..g19

    boot_genes = np.array([[f"g{n - 1 - i}" for i in range(n)]], dtype=object)
    boot_z = np.zeros((1, n))
    boot_z[0, :10] = 4.0  # boot tail: g299..g290
    boot_p = np.ones((1, n))
    boot_p[0, :10] = 0.001

    return SimpleNamespace(
        n_components=1,
        orig=SimpleNamespace(genes=genes, zscored=zscored),
        boot=SimpleNamespace(genes=boot_genes, z_score=boot_z, pval=boot_p),
    )


def _gmt(tmp_path, **terms) -> str:
    """Write a minimal .gmt; run_ora resolves gene sets through the engine, which
    accepts a local .gmt path but not an in-memory mapping."""
    path = tmp_path / "terms.gmt"
    path.write_text(
        "\n".join("\t".join([term, "", *members]) for term, members in terms.items()) + "\n",
        encoding="utf-8",
    )
    return str(path)


def test_run_ora_writes_one_file_per_tail(tmp_path):
    res = _res_obj()
    run_ora(res, gene_set=_gmt(tmp_path, T=[f"g{i}" for i in range(10)]), outdir=tmp_path)
    written = {p.name for p in tmp_path.glob("*.tsv")}
    assert written == {f"{TAIL_BACKENDS[t]}_pls1_results.tsv" for t in ORA_TAILS}


def test_run_ora_p_tail_reads_the_boot_namespace(tmp_path):
    """The p tail must use boot.genes, not orig.genes, when the orders differ."""
    res = _res_obj(n=300)
    out = run_ora(res, gene_set=_gmt(tmp_path, T=["g299", "g298", "g297"]), outdir=tmp_path)
    per_tail = out[0]
    # boot's significant genes are g299.. so a term of those overlaps fully;
    # pairing boot.pval with orig.genes would instead select g0.. and give 0.
    pos = per_tail["p"].query("direction == 'positive'").set_index("Term")
    assert pos.loc["T", "overlap"] == 3
    # the z tail reads orig, whose tail is g0..g19 -> no overlap with g297..g299
    z_pos = per_tail["z"].query("direction == 'positive'").set_index("Term")
    assert z_pos.loc["T", "overlap"] == 0


def test_run_ora_skips_p_tail_without_pvalues(tmp_path):
    res = _res_obj()
    res.boot = SimpleNamespace(weights=None)  # no pval available
    out = run_ora(res, gene_set=_gmt(tmp_path, T=[f"g{i}" for i in range(10)]), outdir=tmp_path)
    assert "p" not in out[0]
    assert not (tmp_path / f"{TAIL_BACKENDS['p']}_pls1_results.tsv").exists()
    assert (tmp_path / f"{TAIL_BACKENDS['z']}_pls1_results.tsv").exists()
