"""Main-style preranked GSEA on the PLS phenotype/spin null.

The pinned ``imaging-transcriptomics`` engine already refits PLS on every spun
phenotype map (``boot_pls`` → ``boot.weights``), so the enrichment null is the
recommended spatially-constrained phenotype null.  The engine's ``PLSGenes.gsea``
however computes each surrogate's enrichment score with the gene hit-positions
**frozen at the observed ranking** — only the weight *magnitudes* vary per
surrogate.  Because the GSEA running-sum statistic depends on gene *order*, that
null is miscalibrated (pure-H0 false-positive rate ~0.7 instead of 0.05).

This module fixes exactly that one defect while staying wired to the engine:

* it **reuses the engine's own enrichment-score function**
  (:func:`imaging_transcriptomics.gsea_utils.enrichment_scores_many`) and geneset
  preparation (:func:`prepare_prerank_genesets`), so the statistic is identical to
  the engine's; and
* it **re-ranks the genes by each surrogate's own weights** before scoring, which
  is the only thing the engine's null was missing.

Observed and null enrichment scores are computed with the *same* function, so
they are directly comparable (the engine mixed a GSEApy-computed observed ES with
a NumPy-computed null ES).  Significance uses a magnitude two-sided empirical
p-value with the Davison–Hinkley ``+1/+1`` correction.

Implementation choices where the hand-off spec was silent (flagged for review):
* FDR is Benjamini–Hochberg across the categories tested within the component
  (matches the ensemble backend and plan item P3), not GSEA's NES-based FDR.
* ``n_iter=None`` uses *all* stored surrogate columns (the engine defaulted to
  1000 even when 10000 permutations were available).
* Tie-handling in the per-surrogate ranking uses a stable ``mergesort`` argsort,
  matching the engine's ``prepare_from_fit`` ordering.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from imaging_transcriptomics.genesets import as_geneset_mapping, resolve_geneset_resource

# Reuse the pinned engine's statistic and geneset machinery verbatim.
from imaging_transcriptomics.gsea_utils import (
    PreparedPrerankGeneSets,
    enrichment_scores_many,
    normalize_enrichment_scores,
    prepare_prerank_genesets,
)
from imaging_transcriptomics.stats_utils import bh_fdr
from scipy.stats import zscore

from ..logging_ import get_logger

logger = get_logger("genes")


def prepare_over_universe(
    gene_list, geneset_resource, *, min_overlap: int = 1
) -> PreparedPrerankGeneSets:
    """Prepare geneset hit-positions over a ranked gene universe.

    Terms that do not overlap the universe (or fall below ``min_overlap``) are
    dropped so :func:`prepare_prerank_genesets` never raises on an empty term.
    ``hit_positions`` are member indices into ``gene_list`` order.
    """

    genes = [str(gene) for gene in np.asarray(gene_list, dtype=object).reshape(-1).tolist()]
    universe = set(genes)
    mapping = as_geneset_mapping(geneset_resource)
    keep_terms = [
        term
        for term, members in mapping.items()
        if sum(1 for gene in members if gene in universe) >= max(1, int(min_overlap))
    ]
    if not keep_terms:
        raise ValueError("No geneset term overlaps the ranked gene universe.")
    return prepare_prerank_genesets(genes, geneset_resource, term_order=keep_terms)


def enrichment_scores_reranked(
    boot_scores: np.ndarray,
    prepared: PreparedPrerankGeneSets,
) -> np.ndarray:
    """Enrichment scores for many surrogate rankings, re-ranked per surrogate.

    ``boot_scores`` is ``(n_genes, n_iter)`` in the observed gene order (the order
    ``prepared.hit_positions`` index into).  For each surrogate column the genes
    are re-sorted by that surrogate's score, the geneset hit-positions are mapped
    into the surrogate's ranking, and the engine's :func:`enrichment_scores_many`
    is applied — so the statistic is identical to the observed one, only the gene
    order differs.  Returns ``(n_terms, n_iter)``.
    """

    scores = np.asarray(boot_scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError("boot_scores must be a 2D (n_genes, n_iter) array.")
    n_genes, n_iter = scores.shape
    member_indices = prepared.hit_positions
    out = np.zeros((len(prepared.terms), n_iter), dtype=float)

    for j in range(n_iter):
        column = scores[:, j]
        order = np.argsort(column, kind="mergesort")[::-1]  # ranked high→low
        rank = np.empty(n_genes, dtype=np.int64)
        rank[order] = np.arange(n_genes)
        positions = tuple(np.sort(rank[idx]).astype(np.int32) for idx in member_indices)
        prepared_j = PreparedPrerankGeneSets(terms=prepared.terms, hit_positions=positions)
        out[:, j] = enrichment_scores_many(column[order][:, None], prepared_j)[:, 0]
    return out


def magnitude_two_sided_pvalues(observed_es: np.ndarray, null_es: np.ndarray) -> np.ndarray:
    """Two-sided magnitude empirical p with Davison–Hinkley ``(+1)/(+1)``."""

    observed = np.abs(np.asarray(observed_es, dtype=float).reshape(-1, 1))
    nulls = np.abs(np.asarray(null_es, dtype=float))
    n_iter = nulls.shape[1]
    exceed = np.sum(nulls >= observed, axis=1)
    return (exceed + 1.0) / (n_iter + 1.0)


def main_style_gsea_table(
    gene_list,
    observed_scores: np.ndarray,
    boot_scores: np.ndarray,
    geneset_resource,
    *,
    min_overlap: int = 1,
) -> pd.DataFrame:
    """Compute the corrected GSEA table for one PLS component.

    Parameters mirror the engine: ``gene_list`` and ``observed_scores`` are in the
    observed ranked order; ``boot_scores`` is ``(n_genes, n_iter)`` in that same
    order (the engine's ``boot.weights[component]`` aligned to ``orig.genes``).
    """

    prepared = prepare_over_universe(gene_list, geneset_resource, min_overlap=min_overlap)
    genes = np.asarray(gene_list, dtype=object).reshape(-1)
    observed = np.asarray(observed_scores, dtype=float).reshape(-1)

    observed_es = enrichment_scores_many(observed[:, None], prepared)[:, 0]
    null_es = enrichment_scores_reranked(boot_scores, prepared)

    nes = normalize_enrichment_scores(observed_es, null_es)
    p_val = magnitude_two_sided_pvalues(observed_es, null_es)
    fdr = bh_fdr(p_val)

    matched_size = [int(idx.size) for idx in prepared.hit_positions]
    matched_genes = [";".join(genes[idx].astype(str).tolist()) for idx in prepared.hit_positions]

    out: "OrderedDict[str, object]" = OrderedDict()
    out["Term"] = list(prepared.terms)
    out["es"] = observed_es
    out["nes"] = nes
    out["p_val"] = p_val
    out["fdr"] = fdr
    out["matched_size"] = matched_size
    out["matched_genes"] = matched_genes
    return pd.DataFrame.from_dict(out)


def run_gsea(
    res_obj,
    gene_set="lake",
    outdir=None,
    *,
    geneset_organism: str = "Human",
    n_iter: int | None = None,
    min_overlap: int = 1,
):
    """Drop-in replacement for ``PLSGenes.gsea`` using the per-surrogate re-rank.

    Reads ``res_obj.orig`` (observed ranking) and ``res_obj.boot.weights``
    (per-surrogate PLS gene weights) from the engine result object and writes one
    ``gsea_pls<N>_results.tsv`` per component into ``outdir`` — same filename and
    key columns the pipeline/report already consume.
    """

    if res_obj.boot.weights is None:
        raise RuntimeError("Corrected GSEA requires stored permutation gene weights.")
    resolved = resolve_geneset_resource(gene_set, organism=geneset_organism)
    logger.info("Performing per-surrogate re-ranked GSEA.")
    outputs: list[pd.DataFrame] = []
    for component in range(res_obj.n_components):
        gene_list = list(res_obj.orig.genes[component, :])
        observed_scores = res_obj.orig.zscored[component, :]
        available = res_obj.boot.weights.shape[2]
        limit = available if n_iter is None else min(int(n_iter), available)
        boot_scores = zscore(
            np.asarray(res_obj.boot.weights[component, :, :limit], dtype=float),
            axis=0,
            ddof=1,
        )
        out_df = main_style_gsea_table(
            gene_list, observed_scores, boot_scores, resolved, min_overlap=min_overlap
        )
        outputs.append(out_df)
        if outdir is not None:
            output_dir = Path(outdir)
            assert output_dir.exists()
            logger.info("Saving corrected GSEA results for PLS component %d.", component + 1)
            out_df.to_csv(
                output_dir / f"gsea_pls{component + 1}_results.tsv", index=False, sep="\t"
            )
    return outputs
