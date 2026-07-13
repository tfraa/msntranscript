"""Tests for the template-style weight-ranked ORA (Fisher over-representation)."""

from __future__ import annotations

import numpy as np

from msnpip.genes.ora_mainstyle import ora_table


def _mapping(**terms):
    return {k: tuple(v) for k, v in terms.items()}


def test_positive_tail_enriched_term_is_flagged():
    n = 400
    genes = [f"g{i}" for i in range(n)]
    scores = np.zeros(n)
    scores[:20] = 5.0  # positive tail = g0..g19 (|z|>=3)
    # a term made mostly of the positive-tail genes must be enriched (low p, OR>1)
    enriched = genes[:15] + genes[100:105]
    random_term = genes[200:220]
    mapping = _mapping(ENR=enriched, RND=random_term)

    df = ora_table(genes, scores, mapping, z_cut=3.0)
    pos = df[df["direction"] == "positive"].set_index("Term")
    assert pos.loc["ENR", "p_val"] < 0.001
    assert pos.loc["ENR", "odds_ratio"] > 1
    assert pos.loc["ENR", "overlap"] == 15
    # a random term of tail-free genes is not enriched
    assert pos.loc["RND", "overlap"] == 0
    assert pos.loc["RND", "p_val"] > 0.05


def test_tails_use_weight_cut_not_significance():
    # scores are the standardized loadings; only |z|>=cut define the tails
    genes = [f"g{i}" for i in range(100)]
    scores = np.linspace(-2, 2, 100)  # nothing reaches |z|>=3
    df = ora_table(genes, scores, _mapping(T=genes[:10]), z_cut=3.0)
    assert df.empty  # no genes in either tail -> no rows


def test_negative_tail_direction():
    n = 300
    genes = [f"g{i}" for i in range(n)]
    scores = np.zeros(n)
    scores[:15] = -4.0  # negative tail
    df = ora_table(genes, scores, _mapping(NEG=genes[:12]), z_cut=3.0)
    neg = df[df["direction"] == "negative"].set_index("Term")
    assert neg.loc["NEG", "overlap"] == 12
    assert neg.loc["NEG", "p_val"] < 0.001
    # no positive-tail rows exist
    assert (df["direction"] == "positive").sum() == 0
