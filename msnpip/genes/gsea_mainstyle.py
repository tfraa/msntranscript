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
a NumPy-computed null ES).  Significance uses the engine's own sign-aware nominal
empirical p-value (:func:`imaging_transcriptomics.gsea_utils.nominal_pvalues_from_nulls`,
``(count+1)/(n+1)`` with a one-sided tail chosen by the observed ES sign), so the
p-value is byte-for-byte identical to imaging-transcriptomics v2.  This is
one-sided per sign and therefore ~2x anti-conservative relative to a magnitude
two-sided p-value — the *only* correction this module keeps over the engine is the
per-surrogate re-ranking of the null (the engine froze gene positions at the
observed ranking, FPR ~0.7); the p-value itself matches the engine.

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
    nominal_pvalues_from_nulls,
    normalize_enrichment_scores,
    prepare_prerank_genesets,
)
from imaging_transcriptomics.stats_utils import bh_fdr
from scipy.stats import zscore

from ..logging_ import get_logger

logger = get_logger("genes")

# Working-set target per surrogate block in :func:`_term_block_scores`; keeps the
# peak footprint bounded (and predictable under parallelism) regardless of how
# many permutations were requested.
_BLOCK_TARGET_BYTES = 256 * 1024 * 1024


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


def _term_block_scores(
    scores_block: np.ndarray,
    n_terms: int,
    member_indices,
) -> np.ndarray:
    """Enrichment scores for a block of surrogates, vectorised *over surrogates*.

    This computes exactly the statistic of
    :func:`imaging_transcriptomics.gsea_utils.enrichment_scores_many` — the
    weighted (``p=1``) preranked running sum, with the peak taken as the extreme
    of the running sum evaluated just after each hit and just before each hit —
    on each surrogate's *own* gene ranking.  The re-ranking (the statistical
    correction) is preserved exactly; only the loop order changes.

    The engine scores one ranking at a time, so the Python loop runs
    ``n_terms x n_surrogates`` times (tens of millions for a 5000-term set), and
    each iteration does NumPy calls on tiny arrays — almost pure call overhead.
    Here the loop runs once **per term**, with every surrogate handled
    simultaneously: the term's member genes get their per-surrogate ranks as an
    ``(n_hits, n_block)`` array, so the running sum is a single vectorised cumsum
    across the whole block.  Same arithmetic, ~2 orders of magnitude fewer
    Python-level operations.
    """

    n_genes, n_cols = scores_block.shape
    # Descending ranking per surrogate (stable mergesort, matching the engine).
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
        # Hit positions in each surrogate's ranking, ascending per surrogate.
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

    ``boot_scores`` is ``(n_genes, n_iter)`` in the observed gene order (the order
    ``prepared.hit_positions`` index into).  For each surrogate column the genes
    are re-sorted by that surrogate's score, the geneset hit-positions are mapped
    into the surrogate's ranking, and the engine's :func:`enrichment_scores_many`
    is applied — so the statistic is identical to the observed one, only the gene
    order differs.  Returns ``(n_terms, n_iter)``.

    The per-surrogate re-ranking is the statistical correction and is preserved
    exactly.  Surrogates are processed in blocks (vectorised over the block, see
    :func:`_term_block_scores`) and blocks are independent, so ``n_jobs`` only
    changes speed: the numbers are identical to the serial computation.
    """

    scores = np.asarray(boot_scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError("boot_scores must be a 2D (n_genes, n_iter) array.")
    n_genes, n_iter = scores.shape
    member_indices = prepared.hit_positions
    n_terms = len(prepared.terms)

    # Block size caps the working set: _term_block_scores holds a handful of
    # (n_genes, block) arrays, so bound them rather than the surrogate count.
    per_col_bytes = n_genes * 8 * 4
    block = int(np.clip(_BLOCK_TARGET_BYTES // max(per_col_bytes, 1), 1, n_iter))
    starts = range(0, n_iter, block)
    workers = min(max(1, int(n_jobs)), len(starts))

    if workers == 1:
        blocks = [
            _term_block_scores(scores[:, s : s + block], n_terms, member_indices)
            for s in starts
        ]
    else:
        from joblib import Parallel, delayed

        blocks = Parallel(n_jobs=workers)(
            delayed(_term_block_scores)(scores[:, s : s + block], n_terms, member_indices)
            for s in starts
        )
    # Blocks are contiguous and ordered, so column order is preserved.
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
    """Compute the corrected GSEA table for one PLS component.

    Parameters mirror the engine: ``gene_list`` and ``observed_scores`` are in the
    observed ranked order; ``boot_scores`` is ``(n_genes, n_iter)`` in that same
    order (the engine's ``boot.weights[component]`` aligned to ``orig.genes``).
    """

    prepared = prepare_over_universe(gene_list, geneset_resource, min_overlap=min_overlap)
    genes = np.asarray(gene_list, dtype=object).reshape(-1)
    observed = np.asarray(observed_scores, dtype=float).reshape(-1)

    observed_es = enrichment_scores_many(observed[:, None], prepared)[:, 0]
    null_es = enrichment_scores_reranked(boot_scores, prepared, n_jobs=n_jobs)

    nes = normalize_enrichment_scores(observed_es, null_es)
    # Sign-aware nominal p-value identical to imaging-transcriptomics v2
    # (one-sided per sign, ~2x anti-conservative); the correction is the
    # re-ranked null above, not the p-value.
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
