"""Tests for the gene-set specificity test (orthogonal to the spin null).

A category enriched for high observed weights must be *specific* — score higher
than size-matched random gene sets — so its ``p_specificity`` is small.  Under H0
(observed weights carry no category structure) the specificity p is uniform.
"""

from __future__ import annotations

import numpy as np
from imaging_transcriptomics.gsea_utils import PreparedPrerankGeneSets

from msnpip.genes.gene_specificity import category_specificity


def _prepared(*member_arrays):
    terms = tuple(f"T{i}" for i in range(len(member_arrays)))
    positions = tuple(np.asarray(m, dtype=np.int32) for m in member_arrays)
    return PreparedPrerankGeneSets(terms=terms, hit_positions=positions)


def test_specificity_flags_a_specific_category():
    n_genes = 500
    rng = np.random.default_rng(0)
    scores = rng.normal(size=n_genes)
    # Make genes 0..29 strongly positive -> a category over them is specific.
    specific = np.arange(30)
    scores[specific] += 5.0
    random_set = rng.choice(np.arange(30, n_genes), size=30, replace=False)

    prepared = _prepared(specific, random_set)
    out = category_specificity(scores, prepared, n_random=500, seed=1)

    p_specific = float(out.loc[out["Term"] == "T0", "p_specificity"].iloc[0])
    p_random = float(out.loc[out["Term"] == "T1", "p_specificity"].iloc[0])
    assert p_specific < 0.01, f"specific category not flagged: p={p_specific}"
    assert p_random > 0.05, f"random category wrongly flagged: p={p_random}"


def test_specificity_h0_is_calibrated():
    n_genes, set_size, trials = 400, 25, 200
    rng = np.random.default_rng(3)
    pvals = np.empty(trials)
    for t in range(trials):
        scores = rng.normal(size=n_genes)
        members = rng.choice(n_genes, size=set_size, replace=False)
        out = category_specificity(scores, _prepared(members), n_random=200, seed=t)
        pvals[t] = float(out["p_specificity"].iloc[0])
    fpr = float(np.mean(pvals < 0.05))
    mean_p = float(np.mean(pvals))
    assert fpr <= 0.10, f"H0 specificity FPR too high: {fpr:.3f}"
    assert 0.42 <= mean_p <= 0.58, f"mean specificity p not ~0.5: {mean_p:.3f}"
