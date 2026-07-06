"""Calibration + correctness tests for the per-surrogate re-ranked GSEA.

These pin the P0 fix: the corrected null re-ranks genes on every surrogate, so a
pure-H0 (no-signal) run must be well-calibrated — false-positive rate ~0.05 and
mean p ~0.5.  The engine's original null froze gene positions at the observed
ranking and is grossly anti-conservative; a companion test demonstrates that gap
so the fix cannot silently regress to it.
"""

from __future__ import annotations

import numpy as np
import pytest
from imaging_transcriptomics.gsea_utils import (
    PreparedPrerankGeneSets,
    enrichment_scores_many,
)

from msnpip.genes.gsea_mainstyle import (
    magnitude_two_sided_pvalues,
    main_style_gsea_table,
    prepare_over_universe,
)


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


def _reordered(weights: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Reorder a (n_genes, n_iter) weight matrix rows into ``order``."""

    return weights[order, :]


def _h0_pvalues(*, n_genes, set_size, n_iter, trials, seed, reranked):
    """Return p-values from ``trials`` pure-H0 GSEA runs.

    Every gene-weight vector (observed and all surrogates) is an i.i.d. draw, so
    under a correct null the observed statistic is exchangeable with the null and
    the p-values are uniform.  ``reranked=False`` reproduces the engine's
    fixed-position bug for comparison.
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
        boot_scores = _reordered(draws[:, 1:], order)

        if reranked:
            table = main_style_gsea_table(gene_list, observed_scores, boot_scores, mapping)
            pvals[t] = float(table["p_val"].iloc[0])
        else:  # engine's fixed-position null (the bug)
            prepared = prepare_over_universe(gene_list, mapping)
            observed_es = enrichment_scores_many(observed_scores[:, None], prepared)[:, 0]
            null_es = enrichment_scores_many(boot_scores, prepared)  # positions frozen
            pvals[t] = float(magnitude_two_sided_pvalues(observed_es, null_es)[0])
    return pvals


def test_pure_h0_is_calibrated():
    """Re-ranked null: FPR ≈ 0.05 and mean p ≈ 0.5 under pure H0."""

    pvals = _h0_pvalues(n_genes=200, set_size=20, n_iter=200, trials=200, seed=1, reranked=True)
    fpr = float(np.mean(pvals < 0.05))
    mean_p = float(np.mean(pvals))
    assert fpr <= 0.10, f"H0 false-positive rate too high: {fpr:.3f}"
    assert 0.42 <= mean_p <= 0.58, f"mean p not ~0.5: {mean_p:.3f}"


def test_rerank_fixes_fixed_position_inflation():
    """The fixed-position null is anti-conservative; re-ranking restores calibration."""

    kwargs = dict(n_genes=200, set_size=20, n_iter=200, trials=200, seed=7)
    fixed = _h0_pvalues(**kwargs, reranked=False)
    reranked = _h0_pvalues(**kwargs, reranked=True)
    fixed_fpr = float(np.mean(fixed < 0.05))
    reranked_fpr = float(np.mean(reranked < 0.05))
    assert reranked_fpr <= 0.10
    # The bug must be materially worse than the fix (guards against regressing).
    assert fixed_fpr > reranked_fpr + 0.10, (
        f"expected fixed-position inflation, got fixed={fixed_fpr:.3f} reranked={reranked_fpr:.3f}"
    )
