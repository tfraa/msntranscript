"""Gene-set specificity test — the axis orthogonal to the spatial-spin null.

Even a category that survives the spin (phenotype) null may not be *specific*: a
random set of brain-expressed genes of the same size might score just as high,
because many genes share the dominant spatial-expression gradients (Wei et al.;
Arnatkeviciute et al. 2023 report only ~3% of associations survive BOTH the
spatial-autocorrelation and gene-specificity corrections).  The spin null and the
specificity test are *orthogonal* — this module supplies the second.

For each category it compares the observed category statistic (the GCEA statistic:
the mean z-scored PLS gene weight over the category's genes — order-independent, so
it is backend-neutral and applies to both the ensemble and GSEA results) against
the distribution of that same statistic over many **size-matched random gene sets**
drawn from the ranked gene universe.  The universe here is the AHBA brain-expressed
gene set, so this is already the brain-expressed-matched variant recommended by the
best-practices reference (a broader universe would only make it more liberal).

``p_specificity`` is a two-sided magnitude empirical p with the Davison–Hinkley
``+1/+1`` correction: the fraction of random sets whose |mean| meets or exceeds the
observed |mean|.  Small p ⇒ the real set is more extreme than random sets of equal
size ⇒ the association is specific to those genes, not a generic-gene-set effect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from imaging_transcriptomics.genesets import resolve_geneset_resource

from .gsea_mainstyle import prepare_over_universe


def category_specificity(
    observed_scores: np.ndarray,
    prepared,
    *,
    n_random: int = 1000,
    seed: int = 1234,
) -> pd.DataFrame:
    """Specificity p-values for prepared categories against size-matched random sets.

    ``observed_scores`` is the per-gene z-scored PLS weight vector (length n_genes)
    in the ranked gene order that ``prepared.hit_positions`` index into.  Returns a
    table with the observed category statistic, the random-set mean/sd, and
    ``p_specificity`` per term.
    """

    scores = np.asarray(observed_scores, dtype=float).reshape(-1)
    n_genes = scores.size
    rng = np.random.default_rng(seed)

    terms = list(prepared.terms)
    sizes = np.array([int(idx.size) for idx in prepared.hit_positions], dtype=int)
    observed_stat = np.array(
        [float(scores[idx].mean()) if idx.size else np.nan for idx in prepared.hit_positions]
    )

    # Draw the random-set means once per distinct category size (vectorised).
    null_mean = np.full(len(terms), np.nan)
    null_sd = np.full(len(terms), np.nan)
    p_spec = np.full(len(terms), np.nan)
    for size in np.unique(sizes):
        if size <= 0 or size >= n_genes:
            continue
        # (n_random, size) independent gene draws without replacement per row.
        draws = np.array([rng.choice(n_genes, size=size, replace=False) for _ in range(n_random)])
        rand_means = scores[draws].mean(axis=1)  # (n_random,)
        abs_rand = np.abs(rand_means)
        which = np.flatnonzero(sizes == size)
        for t in which:
            obs = observed_stat[t]
            exceed = int(np.sum(abs_rand >= abs(obs)))
            p_spec[t] = (exceed + 1.0) / (n_random + 1.0)
            null_mean[t] = float(rand_means.mean())
            null_sd[t] = float(rand_means.std(ddof=1))

    return pd.DataFrame(
        {
            "Term": terms,
            "category_score": observed_stat,
            "rand_mean": null_mean,
            "rand_sd": null_sd,
            "matched_size": sizes,
            "p_specificity": p_spec,
            "n_random": n_random,
        }
    )


def run_gene_specificity(
    res_obj,
    gene_set="lake",
    outdir=None,
    *,
    geneset_organism: str = "Human",
    n_random: int = 1000,
    seed: int = 1234,
    min_overlap: int = 1,
):
    """Write a size-matched-random-set specificity table per PLS component.

    Reads the observed z-scored gene weights (``res_obj.orig``) and writes one
    ``gene_specificity_pls<N>.tsv`` per component into ``outdir``.  Named without the
    ``_results`` suffix so it is curated separately from the enrichment backends.
    """

    resolved = resolve_geneset_resource(gene_set, organism=geneset_organism)
    outputs: list[pd.DataFrame] = []
    for component in range(res_obj.n_components):
        gene_list = list(res_obj.orig.genes[component, :])
        observed_scores = res_obj.orig.zscored[component, :]
        prepared = prepare_over_universe(gene_list, resolved, min_overlap=min_overlap)
        table = category_specificity(observed_scores, prepared, n_random=n_random, seed=seed)
        outputs.append(table)
        if outdir is not None:
            from pathlib import Path

            output_dir = Path(outdir)
            assert output_dir.exists()
            table.to_csv(
                output_dir / f"gene_specificity_pls{component + 1}.tsv", index=False, sep="\t"
            )
    return outputs
