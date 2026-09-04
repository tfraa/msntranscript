"""Preranked GSEA on the PLS spin null, with each surrogate re-ranked by its own weights.

The pinned engine's ``PLSGenes.gsea`` scores every surrogate at the gene hit-positions
of the *observed* ranking.  Enrichment score is a rank-position statistic, so that null
is miscalibrated (pure-H0 false-positive rate ~0.7).  Re-ranking per surrogate is the
only correction here: the statistic and the nominal p-value are the engine's own, so
``p_val`` matches imaging-transcriptomics v2 exactly.  That p is one-sided per observed
sign and therefore ~2x anti-conservative; ``fdr`` is BH across the categories tested.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from imaging_transcriptomics.genesets import as_geneset_mapping, resolve_geneset_resource
from imaging_transcriptomics.gsea_utils import (
    PreparedPrerankGeneSets,
    enrichment_scores_many,
    nominal_pvalues_from_nulls,
    normalize_enrichment_scores,
    prepare_prerank_genesets,
)
from imaging_transcriptomics.stats_utils import bh_fdr
from scipy.stats import zscore

from ..logging_ import get_logger

logger = get_logger("genes")

# Bounds the working set per surrogate block, whatever the permutation count.
_BLOCK_TARGET_BYTES = 256 * 1024 * 1024


def prepare_over_universe(
    gene_list, geneset_resource, *, min_overlap: int = 1
) -> PreparedPrerankGeneSets:
    """Prepare geneset hit-positions over a ranked gene universe.

    Terms below ``min_overlap`` are dropped so ``prepare_prerank_genesets`` never
    raises on an empty term.  ``hit_positions`` index into ``gene_list`` order.
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


def _term_block_scores(
    scores_block: np.ndarray,
    n_terms: int,
    member_indices,
) -> np.ndarray:
    """Enrichment scores for a block of surrogates, vectorised over the block.

    Same statistic as ``gsea_utils.enrichment_scores_many``, evaluated on each
    surrogate's own ranking; only the loop order differs.
    """

    n_genes, n_cols = scores_block.shape
    # Stable mergesort matches the engine's prepare_from_fit ordering.
    orders = np.argsort(scores_block, axis=0, kind="mergesort")[::-1, :]
    sorted_abs = np.abs(np.take_along_axis(scores_block, orders, axis=0))
    # rank[gene, j] = position of that gene in surrogate j's ranking.
    rank = np.empty((n_genes, n_cols), dtype=np.int64)
    rank[orders, np.arange(n_cols)] = np.arange(n_genes)[:, None]
    del orders

    out = np.zeros((n_terms, n_cols), dtype=float)
    zeros_row = np.zeros((1, n_cols), dtype=float)
    for term_index, idx in enumerate(member_indices):
        nh = int(idx.size)
        if nh == 0 or nh >= n_genes:
            continue
        positions = np.sort(rank[idx, :], axis=0)
        pos_f = positions.astype(float)
        hit_weights = np.take_along_axis(sorted_abs, positions, axis=0)

        norm = hit_weights.sum(axis=0)
        norm_safe = np.where(norm == 0, 1.0, norm)
        hit_cumulative = np.cumsum(hit_weights / norm_safe.reshape(1, -1), axis=0)

        miss_scale = 1.0 / float(n_genes - nh)
        hit_order = np.arange(1, nh + 1, dtype=float).reshape(-1, 1)
        rs_after_hits = hit_cumulative - (pos_f + 1.0 - hit_order) * miss_scale

        if nh == 1:
            rs_before_hits = -pos_f * miss_scale
        else:
            misses_before = pos_f - np.arange(nh, dtype=float).reshape(-1, 1)
            rs_before_hits = np.vstack(
                [
                    -pos_f[0:1, :] * miss_scale,
                    hit_cumulative[:-1, :] - misses_before[1:, :] * miss_scale,
                ]
            )

        max_pos = np.max(rs_after_hits, axis=0)
        min_neg = np.min(np.vstack([rs_before_hits, zeros_row]), axis=0)
        out[term_index, :] = np.where(np.abs(max_pos) >= np.abs(min_neg), max_pos, min_neg)
    return out


def enrichment_scores_reranked(
    boot_scores: np.ndarray,
    prepared: PreparedPrerankGeneSets,
    *,
    n_jobs: int = 1,
) -> np.ndarray:
    """Enrichment scores for many surrogate rankings, re-ranked per surrogate.

    ``boot_scores`` is ``(n_genes, n_iter)`` in observed gene order; returns
    ``(n_terms, n_iter)``.  ``n_jobs`` changes speed only — blocks are independent,
    so the numbers match the serial computation.
    """

    scores = np.asarray(boot_scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError("boot_scores must be a 2D (n_genes, n_iter) array.")
    n_genes, n_iter = scores.shape
    member_indices = prepared.hit_positions
    n_terms = len(prepared.terms)

    per_col_bytes = n_genes * 8 * 4
    block = int(np.clip(_BLOCK_TARGET_BYTES // max(per_col_bytes, 1), 1, n_iter))
    starts = range(0, n_iter, block)
    workers = min(max(1, int(n_jobs)), len(starts))

    if workers == 1:
        blocks = [
            _term_block_scores(scores[:, s : s + block], n_terms, member_indices) for s in starts
        ]
    else:
        from joblib import Parallel, delayed

        blocks = Parallel(n_jobs=workers)(
            delayed(_term_block_scores)(scores[:, s : s + block], n_terms, member_indices)
            for s in starts
        )
    return np.concatenate(blocks, axis=1)


def main_style_gsea_table(
    gene_list,
    observed_scores: np.ndarray,
    boot_scores: np.ndarray,
    geneset_resource,
    *,
    min_overlap: int = 1,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Corrected GSEA table for one PLS component.

    ``gene_list`` and ``observed_scores`` are in observed ranked order; ``boot_scores``
    is ``(n_genes, n_iter)`` in that same order.
    """

    prepared = prepare_over_universe(gene_list, geneset_resource, min_overlap=min_overlap)
    genes = np.asarray(gene_list, dtype=object).reshape(-1)
    observed = np.asarray(observed_scores, dtype=float).reshape(-1)

    observed_es = enrichment_scores_many(observed[:, None], prepared)[:, 0]
    null_es = enrichment_scores_reranked(boot_scores, prepared, n_jobs=n_jobs)

    nes = normalize_enrichment_scores(observed_es, null_es)
    # The engine's own sign-aware nominal p; the correction is the null, not this.
    p_val = nominal_pvalues_from_nulls(observed_es, null_es)
    fdr = bh_fdr(p_val)

    matched_size = [int(idx.size) for idx in prepared.hit_positions]
    matched_genes = [";".join(genes[idx].astype(str).tolist()) for idx in prepared.hit_positions]

    out: OrderedDict[str, object] = OrderedDict()
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
    n_jobs: int = 1,
):
    """Drop-in replacement for ``PLSGenes.gsea`` using the per-surrogate re-rank.

    Writes one ``gsea_pls<N>_results.tsv`` per component into ``outdir``.
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
            gene_list,
            observed_scores,
            boot_scores,
            resolved,
            min_overlap=min_overlap,
            n_jobs=n_jobs,
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
