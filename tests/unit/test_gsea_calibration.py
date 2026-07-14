"""Correctness tests for the per-surrogate re-ranked GSEA.

These pin two things:

* the enrichment statistic is the engine's own running-sum ES (reused verbatim);
* the p-value is byte-for-byte the engine's ``nominal_pvalues_from_nulls`` (the
  sign-aware one-sided empirical p imaging-transcriptomics v2 uses).  That
  p-value is ~2x anti-conservative by construction (it counts only one tail), so
  a pure-H0 run has FPR ~0.10, **not** 0.05 — matching imt v2 exactly is a
  deliberate choice, not a calibration target.

The correction this module keeps over the engine is the **per-surrogate
re-ranking** of the null: the engine froze gene positions at the observed
ranking, which is grossly anti-conservative (H0 FPR ~0.5).  ``test_rerank_...``
demonstrates that gap on the *same* p-value so the fix cannot silently regress.
"""

from __future__ import annotations

import numpy as np
import pytest
from imaging_transcriptomics.gsea_utils import (
    PreparedPrerankGeneSets,
    enrichment_scores_many,
    nominal_pvalues_from_nulls,
)

from msnpip.genes.gsea_mainstyle import main_style_gsea_table, prepare_over_universe


def _textbook_es(sorted_scores: np.ndarray, hit_mask: np.ndarray) -> float:
    """Reference Subramanian weighted (p=1) running-sum ES for one ranking."""

    n = sorted_scores.size
    nh = int(hit_mask.sum())
    hit_weights = np.abs(sorted_scores) * hit_mask
    norm = hit_weights.sum()
    miss_step = 1.0 / (n - nh)
    running = 0.0
    peak = 0.0
    for i in range(n):
        if hit_mask[i]:
            running += abs(sorted_scores[i]) / norm
        else:
            running -= miss_step
        if abs(running) > abs(peak):
            peak = running
    return peak


def test_es_matches_textbook():
    """The engine ES function (which we reuse) equals the textbook running sum."""

    rng = np.random.default_rng(0)
    scores = np.sort(rng.normal(size=40))[::-1].copy()  # ranked high→low
    members = np.array([2, 5, 6, 9, 20, 21, 30], dtype=np.int32)
    hit_mask = np.zeros(scores.size, dtype=bool)
    hit_mask[members] = True

    prepared = PreparedPrerankGeneSets(terms=("SET",), hit_positions=(members,))
    engine_es = enrichment_scores_many(scores[:, None], prepared)[0, 0]
    reference = _textbook_es(scores, hit_mask)
    assert engine_es == pytest.approx(reference, abs=1e-9)


def test_pvalue_matches_engine_nominal():
    """The reported p-value is exactly imt v2's ``nominal_pvalues_from_nulls``.

    Recomputing the observed/null enrichment scores independently and feeding the
    engine's own nominal-p function must reproduce ``main_style_gsea_table``'s
    ``p_val`` column to the bit — that is the whole point of matching the package.
    """

    rng = np.random.default_rng(3)
    n_genes, n_iter = 150, 120
    genes = [f"g{i}" for i in range(n_genes)]
    members = rng.choice(n_genes, size=25, replace=False)
    mapping = {"SET_A": tuple(genes[i] for i in members[:15]), "SET_B": tuple(genes[i] for i in members[10:])}

    draws = rng.normal(size=(n_genes, n_iter + 1))
    draws = (draws - draws.mean(0)) / draws.std(0, ddof=1)
    order = np.argsort(draws[:, 0], kind="mergesort")[::-1]
    gene_list = [genes[i] for i in order]
    observed_scores = draws[:, 0][order]
    boot_scores = draws[:, 1:][order, :]

    table = main_style_gsea_table(gene_list, observed_scores, boot_scores, mapping)

    # Recompute the null the same way the module does, then the engine's p.
    from msnpip.genes.gsea_mainstyle import enrichment_scores_reranked

    prepared = prepare_over_universe(gene_list, mapping)
    observed_es = enrichment_scores_many(np.asarray(observed_scores)[:, None], prepared)[:, 0]
    null_es = enrichment_scores_reranked(np.asarray(boot_scores), prepared)
    expected = nominal_pvalues_from_nulls(observed_es, null_es)
    np.testing.assert_array_equal(table["p_val"].to_numpy(), expected)


def test_njobs_does_not_change_results():
    """Parallelising the surrogate loop returns identical numbers (n_jobs only
    splits independent columns)."""

    rng = np.random.default_rng(11)
    n_genes, n_iter = 120, 100
    genes = [f"g{i}" for i in range(n_genes)]
    members = rng.choice(n_genes, size=20, replace=False)
    mapping = {"SET": tuple(genes[i] for i in members)}
    draws = rng.normal(size=(n_genes, n_iter + 1))
    order = np.argsort(draws[:, 0], kind="mergesort")[::-1]
    gene_list = [genes[i] for i in order]
    observed_scores = draws[:, 0][order]
    boot_scores = draws[:, 1:][order, :]

    serial = main_style_gsea_table(gene_list, observed_scores, boot_scores, mapping, n_jobs=1)
    parallel = main_style_gsea_table(gene_list, observed_scores, boot_scores, mapping, n_jobs=2)
    np.testing.assert_array_equal(serial["p_val"].to_numpy(), parallel["p_val"].to_numpy())
    np.testing.assert_array_equal(serial["es"].to_numpy(), parallel["es"].to_numpy())


def _h0_pvalues(*, n_genes, set_size, n_iter, trials, seed, reranked):
    """Return p-values from ``trials`` pure-H0 GSEA runs, using the *same*
    (engine nominal) p-value for both the re-ranked and the frozen-position null.

    ``reranked=False`` reproduces the engine's fixed-position bug for comparison:
    the only difference between the two branches is whether the null is re-ranked
    per surrogate, isolating the effect of the correction.
    """

    rng = np.random.default_rng(seed)
    genes = [f"g{i}" for i in range(n_genes)]
    pvals = np.empty(trials, dtype=float)
    for t in range(trials):
        members = rng.choice(n_genes, size=set_size, replace=False)
        mapping = {"SET": tuple(genes[i] for i in members)}

        draws = rng.normal(size=(n_genes, n_iter + 1))
        draws = (draws - draws.mean(0)) / draws.std(0, ddof=1)
        obs_raw = draws[:, 0]
        order = np.argsort(obs_raw, kind="mergesort")[::-1]  # observed ranking
        gene_list = [genes[i] for i in order]
        observed_scores = obs_raw[order]
        boot_scores = draws[:, 1:][order, :]

        if reranked:
            table = main_style_gsea_table(gene_list, observed_scores, boot_scores, mapping)
            pvals[t] = float(table["p_val"].iloc[0])
        else:  # engine's fixed-position null (the bug), same p-value function
            prepared = prepare_over_universe(gene_list, mapping)
            observed_es = enrichment_scores_many(observed_scores[:, None], prepared)[:, 0]
            null_es = enrichment_scores_many(boot_scores, prepared)  # positions frozen
            pvals[t] = float(nominal_pvalues_from_nulls(observed_es, null_es)[0])
    return pvals


def test_rerank_reduces_inflation():
    """Re-ranking the null (the correction) is far less inflated than freezing it.

    With the one-sided nominal p the re-ranked H0 FPR sits around ~0.10-0.12 (the
    ~2x anti-conservative price of matching imt v2), while the frozen-position
    null is catastrophically inflated (~0.5). The gap is the correction.
    """

    kwargs = dict(n_genes=200, set_size=20, n_iter=200, trials=200, seed=7)
    fixed = _h0_pvalues(**kwargs, reranked=False)
    reranked = _h0_pvalues(**kwargs, reranked=True)
    fixed_fpr = float(np.mean(fixed < 0.05))
    reranked_fpr = float(np.mean(reranked < 0.05))
    assert reranked_fpr <= 0.20, f"re-ranked H0 FPR unexpectedly high: {reranked_fpr:.3f}"
    # The frozen-position bug must stay materially worse (guards against regressing).
    assert fixed_fpr > reranked_fpr + 0.20, (
        f"expected fixed-position inflation, got fixed={fixed_fpr:.3f} reranked={reranked_fpr:.3f}"
    )
