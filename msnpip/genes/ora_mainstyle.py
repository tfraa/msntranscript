"""Over-representation analysis (ORA) on the PLS gene tails — template method.

This reproduces the enrichment approach of the source literature (Martins et al.
2022 via ``GeneOverlap``/Fisher; Giacomel et al. 2026 via ToppGene/g:Profiler
over-representation): the PLS1+/PLS1− gene tails are tested for over-representation
of each gene-set term with a **Fisher exact test** against the full gene
background.

Two deliberate choices make this comparable to that literature:

* the tails are defined by the **standardized observed loading** (``orig.zscored``,
  ``|z| ≥ z_cut``, default 3) — i.e. the classic weight-ranking cut. This is
  **null-independent**, so it does not collapse under the (correct but stringent)
  spin null the way a spin-p threshold would. The gene ranking reproduces the
  source analyses (weights correlate ~0.98 with the classic toolbox).
* significance is the plain hypergeometric/Fisher **random-gene null**.

IMPORTANT — interpretation. This is an over-representation test; like the source
papers, its results are **candidate biological mechanisms**, not spatially- or
co-expression-corrected inference. It is reported alongside, and clearly
subordinate to, the spin-null tests (component significance, GCEA). It exists for
comparability with the template and for hypothesis generation — never as the
primary, rigorous result.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from imaging_transcriptomics.genesets import as_geneset_mapping, resolve_geneset_resource
from imaging_transcriptomics.stats_utils import bh_fdr
from scipy.stats import fisher_exact

from ..logging_ import get_logger

logger = get_logger("genes")


def ora_table(
    gene_list,
    scores: np.ndarray,
    geneset_resource,
    *,
    z_cut: float = 3.0,
    min_term_size: int = 1,
) -> pd.DataFrame:
    """Fisher over-representation of the ``|z| ≥ z_cut`` gene tails per term.

    ``scores`` is the standardized observed loading per gene (``orig.zscored``),
    in the same order as ``gene_list``.  Returns one row per (term, direction)
    with the odds ratio, overlap counts, Fisher p and BH-FDR (within direction).
    """

    genes = [str(g) for g in np.asarray(gene_list, dtype=object).reshape(-1).tolist()]
    scores = np.asarray(scores, dtype=float).reshape(-1)
    universe = set(genes)
    n_bg = len(genes)
    mapping = as_geneset_mapping(geneset_resource)

    pos_tail = {g for g, s in zip(genes, scores) if s >= z_cut}
    neg_tail = {g for g, s in zip(genes, scores) if s <= -z_cut}

    rows: list[dict] = []
    for direction, tail in (("positive", pos_tail), ("negative", neg_tail)):
        n_tail = len(tail)
        if n_tail == 0:
            continue
        for term, members in mapping.items():
            term_genes = {m for m in members if m in universe}
            n_term = len(term_genes)
            if n_term < max(1, int(min_term_size)):
                continue
            overlap = tail & term_genes
            k = len(overlap)
            # 2x2 contingency: rows = in-tail / not; cols = in-term / not.
            table = [[k, n_tail - k], [n_term - k, n_bg - n_tail - n_term + k]]
            odds, p = fisher_exact(table, alternative="greater")
            rows.append(
                {
                    "Term": term,
                    "direction": direction,
                    "odds_ratio": float(odds),
                    "overlap": k,
                    "tail_size": n_tail,
                    "term_size": n_term,
                    "p_val": float(p),
                    "matched_genes": ";".join(sorted(overlap)),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["fdr"] = np.nan
    for _direction, idx in df.groupby("direction").groups.items():
        df.loc[idx, "fdr"] = bh_fdr(df.loc[idx, "p_val"].to_numpy(dtype=float))
    order = OrderedDict()  # stable column order
    for c in (
        "Term",
        "direction",
        "odds_ratio",
        "overlap",
        "tail_size",
        "term_size",
        "p_val",
        "fdr",
        "matched_genes",
    ):
        order[c] = df[c]
    return pd.DataFrame(order).sort_values(["direction", "fdr"]).reset_index(drop=True)


def run_ora(
    res_obj,
    gene_set="lake",
    outdir=None,
    *,
    geneset_organism: str = "Human",
    z_cut: float = 3.0,
    min_term_size: int = 1,
):
    """Write a template-style ORA table per PLS component.

    Reads the standardized observed loadings (``res_obj.orig.zscored``) so the
    tails are the weight-ranked PLS1± sets, then Fisher-tests each gene-set term.
    Writes ``ora_pls<N>_results.tsv`` per component into ``outdir``.
    """

    resolved = resolve_geneset_resource(gene_set, organism=geneset_organism)
    logger.info("Performing template-style ORA (weight-ranked tails, Fisher).")
    outputs: list[pd.DataFrame] = []
    for component in range(res_obj.n_components):
        gene_list = list(res_obj.orig.genes[component, :])
        scores = res_obj.orig.zscored[component, :]
        df = ora_table(gene_list, scores, resolved, z_cut=z_cut, min_term_size=min_term_size)
        outputs.append(df)
        if outdir is not None and not df.empty:
            output_dir = Path(outdir)
            assert output_dir.exists()
            df.to_csv(output_dir / f"ora_pls{component + 1}_results.tsv", index=False, sep="\t")
    return outputs
