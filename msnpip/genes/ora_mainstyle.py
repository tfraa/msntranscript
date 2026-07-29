"""Over-representation analysis (ORA) on the gene tails — template method.

This reproduces the enrichment approach of the source literature (Martins et al.
2022 via ``GeneOverlap``/Fisher; Giacomel et al. 2026 via ToppGene/g:Profiler
over-representation): the positive/negative gene tails are tested for
over-representation of each gene-set term with a **Fisher exact test** against
the full gene background.

Three tail definitions run side by side, all with **fixed, pre-specified**
constants (no CLI flags — the cut must not be tunable on the results):

``z``      ``|z| >= 3`` on the standardized observed statistic.  Null-independent,
           so it does not collapse under the (correct but stringent) spin null.
           This is the classic Z>3 cut of the source analyses.  Constructed
           identically for both backends: the engine computes
           ``orig.zscored = zscore(orig.weights)`` across genes for PLS
           (``gene_stats/pls.py:125``), and :func:`msnpip.engine._corr_enrichment_adapter`
           applies the same ``zscore`` to the observed Spearman correlations.
``p``      nominal spin ``p <= 0.05``, uncorrected.  This is what the pinned
           engine's own ORA does (``ora.py:89``).  It is **not** comparable
           across backends: the same threshold selects tens of genes on the PLS
           path and thousands on the corr path, because the PLS gene null is
           sign-folded (see the calibration note in ``docs/statistics.md``).
``topn``   the 500 highest and 500 lowest genes by observed statistic.  Fixed
           tail *size* rather than fixed threshold, so the two backends are
           directly comparable and the Fisher background ratio is held constant.

Every emitted row carries ``ora_tail`` and ``tail_size`` so a table can never be
read without knowing how its gene list was selected.

Category-size filtering is **not** done here: the caller passes an already
size-filtered gene set (see :mod:`msnpip.genes.sizefilter`) so that ORA, GCEA and
GSEA all test an identical term set and each one's BH sees the same ``m``.

IMPORTANT — interpretation. This is an over-representation test; like the source
papers, its significance is the plain hypergeometric/Fisher **random-gene null**.
Its results are **candidate biological mechanisms**, not spatially- or
co-expression-corrected inference.  It is reported alongside, and clearly
subordinate to, the spin-null tests (component significance, GCEA).  It exists
for comparability with the template and for hypothesis generation — never as the
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

# Pre-specified tail constants. Deliberately module-level and not configurable:
# an ORA threshold tuned after seeing the results is not an ORA threshold.
Z_CUT = 3.0
P_THRESHOLD = 0.05
TOP_N = 500

#: Tail rules run on every ORA invocation, in output order.
ORA_TAILS: tuple[str, ...] = ("z", "p", "topn")

#: Curated ``enrichment`` label per tail. Distinct backend names (rather than one
#: shared "ora" label plus a discriminator column) keep the three arms separate
#: in the curated CSV, the per-backend figures and the report tables — the same
#: convention the frozen GSEA uses with ``gseafrozen``.
TAIL_BACKENDS: dict[str, str] = {"z": "oraz", "p": "orap", "topn": "oratopn"}


def _tail_description(tail: str) -> str:
    """Human-readable statement of the selection rule, for logs and provenance."""
    if tail == "z":
        return f"|z| >= {Z_CUT:g}"
    if tail == "p":
        return f"nominal spin p <= {P_THRESHOLD:g}"
    if tail == "topn":
        return f"top/bottom {TOP_N} by observed statistic"
    raise ValueError(f"Unknown ORA tail rule: {tail!r} (expected one of {ORA_TAILS})")


def select_tails(
    genes: list[str],
    scores: np.ndarray,
    *,
    tail: str,
    pvals: np.ndarray | None = None,
) -> tuple[set[str], set[str]]:
    """Return the ``(positive, negative)`` gene tails under one selection rule.

    *genes*, *scores* and *pvals* must all come from the **same** engine
    namespace, so they are element-wise aligned.  This matters: on the PLS path
    ``orig.*`` is sorted by weight while ``boot.*`` is sorted by z-score, so
    mixing them silently mis-assigns statistics to genes.
    """
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if len(genes) != scores.size:
        raise ValueError(f"len(genes)={len(genes)} != len(scores)={scores.size}")

    if tail == "z":
        pos = {g for g, s in zip(genes, scores) if s >= Z_CUT}
        neg = {g for g, s in zip(genes, scores) if s <= -Z_CUT}
        return pos, neg

    if tail == "p":
        if pvals is None:
            raise ValueError("The 'p' tail rule needs per-gene p-values.")
        pvals = np.asarray(pvals, dtype=float).reshape(-1)
        if pvals.size != scores.size:
            raise ValueError(f"len(pvals)={pvals.size} != len(scores)={scores.size}")
        keep = pvals <= P_THRESHOLD
        pos = {g for g, s, k in zip(genes, scores, keep) if k and s > 0}
        neg = {g for g, s, k in zip(genes, scores, keep) if k and s < 0}
        return pos, neg

    if tail == "topn":
        # Sort by descending score; ties broken by gene name so the tail is
        # reproducible across platforms and runs.
        names = np.asarray(genes, dtype=object)
        order = np.lexsort((names, -scores))
        head, foot = order[:TOP_N], order[-TOP_N:]
        # Sign guard: a "positive" tail must not contain negative statistics.
        # With ~15.7k genes and a roughly symmetric score distribution this never
        # bites, but a truncated tail must be visible rather than silent.
        pos = {str(names[i]) for i in head if scores[i] > 0}
        neg = {str(names[i]) for i in foot if scores[i] < 0}
        # Warn only when the sign guard actually truncated a tail (i.e. the head
        # held non-positive scores), not merely when the gene list is shorter
        # than 2*TOP_N — that is expected for small inputs and unit fixtures.
        if len(pos) < len(head):
            logger.warning(
                "ORA topn: %d of the top %d genes have score <= 0 and were dropped.",
                len(head) - len(pos),
                len(head),
            )
        if len(neg) < len(foot):
            logger.warning(
                "ORA topn: %d of the bottom %d genes have score >= 0 and were dropped.",
                len(foot) - len(neg),
                len(foot),
            )
        return pos, neg

    raise ValueError(f"Unknown ORA tail rule: {tail!r} (expected one of {ORA_TAILS})")


def ora_table(
    gene_list,
    scores: np.ndarray,
    geneset_resource,
    *,
    tail: str = "z",
    pvals: np.ndarray | None = None,
    min_term_size: int = 1,
) -> pd.DataFrame:
    """Fisher over-representation of the *tail*-selected gene sets per term.

    ``scores`` is the observed per-gene statistic in the same order as
    *gene_list* (the standardized loading for the ``z``/``topn`` rules, any
    sign-carrying score for ``p``).  Returns one row per (term, direction) with
    the odds ratio, overlap counts, Fisher p and BH-FDR (within direction).
    """
    genes = [str(g) for g in np.asarray(gene_list, dtype=object).reshape(-1).tolist()]
    scores = np.asarray(scores, dtype=float).reshape(-1)
    universe = set(genes)
    n_bg = len(genes)
    mapping = as_geneset_mapping(geneset_resource)

    pos_tail, neg_tail = select_tails(genes, scores, tail=tail, pvals=pvals)

    rows: list[dict] = []
    for direction, selected in (("positive", pos_tail), ("negative", neg_tail)):
        n_tail = len(selected)
        if n_tail == 0:
            continue
        for term, members in mapping.items():
            term_genes = {m for m in members if m in universe}
            n_term = len(term_genes)
            if n_term < max(1, int(min_term_size)):
                continue
            overlap = selected & term_genes
            k = len(overlap)
            # 2x2 contingency: rows = in-tail / not; cols = in-term / not.
            table = [[k, n_tail - k], [n_term - k, n_bg - n_tail - n_term + k]]
            odds, p = fisher_exact(table, alternative="greater")
            rows.append(
                {
                    "Term": term,
                    "direction": direction,
                    "ora_tail": tail,
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
        "ora_tail",
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


def _tail_inputs(res_obj, component: int, tail: str):
    """Return the ``(genes, scores, pvals)`` triple for one component and rule.

    Each rule reads from exactly one engine namespace, never a mix:

    * ``z`` / ``topn`` use ``orig.genes`` + ``orig.zscored`` — the observed
      ranking, sorted by weight on the PLS path;
    * ``p`` uses ``boot.genes`` + ``boot.z_score`` + ``boot.pval`` — sorted by
      z-score on the PLS path, so it is a *different* row order from ``orig``.

    Pairing arrays across the two namespaces is the defect that scrambled the
    v1 gene table; :func:`select_tails` cannot detect it, so it is prevented here.
    """
    if tail in ("z", "topn"):
        genes = [str(g) for g in np.asarray(res_obj.orig.genes[component, :]).reshape(-1)]
        return genes, np.asarray(res_obj.orig.zscored[component, :], dtype=float), None

    if tail == "p":
        boot = getattr(res_obj, "boot", None)
        if boot is None or getattr(boot, "pval", None) is None:
            return None
        genes = [str(g) for g in np.asarray(boot.genes[component, :]).reshape(-1)]
        return (
            genes,
            np.asarray(boot.z_score[component, :], dtype=float),
            np.asarray(boot.pval[component, :], dtype=float),
        )

    raise ValueError(f"Unknown ORA tail rule: {tail!r} (expected one of {ORA_TAILS})")


def _tail_size(df: pd.DataFrame, direction: str) -> int:
    """Tail size for one direction, or 0 when that direction selected no genes."""
    sizes = df.loc[df["direction"] == direction, "tail_size"]
    return int(sizes.iloc[0]) if len(sizes) else 0


def run_ora(
    res_obj,
    gene_set="lake",
    outdir=None,
    *,
    geneset_organism: str = "Human",
    tails: tuple[str, ...] = ORA_TAILS,
    min_term_size: int = 1,
):
    """Write one template-style ORA table per component × tail rule.

    Writes ``<backend>_pls<N>_results.tsv`` into *outdir*, where ``<backend>`` is
    ``oraz`` / ``orap`` / ``oratopn`` (see :data:`TAIL_BACKENDS`) so the three
    arms stay separate all the way through curation and the report.

    Returns a list of ``{tail: DataFrame}`` dicts, one per component.
    """
    resolved = resolve_geneset_resource(gene_set, organism=geneset_organism)
    logger.info(
        "Performing template-style ORA (Fisher, random-gene null) — tails: %s.",
        ", ".join(f"{t} ({_tail_description(t)})" for t in tails),
    )
    outputs: list[dict[str, pd.DataFrame]] = []
    for component in range(res_obj.n_components):
        per_tail: dict[str, pd.DataFrame] = {}
        for tail in tails:
            inputs = _tail_inputs(res_obj, component, tail)
            if inputs is None:
                logger.warning(
                    "ORA tail %r skipped: this result object carries no per-gene "
                    "p-values (boot.pval).",
                    tail,
                )
                continue
            gene_list, scores, pvals = inputs
            df = ora_table(
                gene_list,
                scores,
                resolved,
                tail=tail,
                pvals=pvals,
                min_term_size=min_term_size,
            )
            per_tail[tail] = df
            if df.empty:
                logger.warning(
                    "ORA tail %r (%s) selected no genes for component %d — no table written.",
                    tail,
                    _tail_description(tail),
                    component + 1,
                )
                continue
            logger.info(
                "ORA tail %r (%s): %d positive / %d negative genes, %d terms tested.",
                tail,
                _tail_description(tail),
                _tail_size(df, "positive"),
                _tail_size(df, "negative"),
                len(df),
            )
            if outdir is not None:
                output_dir = Path(outdir)
                assert output_dir.exists()
                name = f"{TAIL_BACKENDS[tail]}_pls{component + 1}_results.tsv"
                df.to_csv(output_dir / name, index=False, sep="\t")
        outputs.append(per_tail)
    return outputs


__all__ = [
    "ORA_TAILS",
    "P_THRESHOLD",
    "TAIL_BACKENDS",
    "TOP_N",
    "Z_CUT",
    "ora_table",
    "run_ora",
    "select_tails",
]
