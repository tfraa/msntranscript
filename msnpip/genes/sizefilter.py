"""Category-size filtering of a gene set, applied uniformly across all backends.

Standard practice in enrichment analysis is to test only categories whose size —
counted **after** intersecting with the ranked gene universe — falls in a
pre-specified window (GSEA's own defaults are 15–500; Fulcher's GCEA toolbox uses
10–200; most GO analyses use 10–500).  Two reasons, both statistical and both
decided *a priori*:

* a category score that is the mean of 3 genes is dominated by sampling noise, so
  such terms land in the empirical tail more often than large ones and inflate the
  BH denominator *and* the BH mass; and
* a 3-gene or 4000-gene term is not biologically interpretable even if it survives.

The filter here is deliberately applied **once, upstream of every backend**, by
materialising a filtered ``.gmt``:

* all three backends (GCEA/``ensemble``, re-ranked GSEA, template ORA) then test
  the *identical* term set, so their results are directly comparable;
* filtering happens before each backend's own BH correction, which is the only
  point at which it can legitimately change ``m``; and
* the filtered ``.gmt`` is written next to the enrichment output, so the exact
  tested term set is auditable rather than implied by a config value.

The filter is **off by default** (``min_size=1``, ``max_size=None``) so existing
runs stay bit-reproducible.  :func:`size_report` is logged regardless, so the
number of degenerate terms is visible even when nothing is filtered.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from imaging_transcriptomics.genesets import as_geneset_mapping

from ..logging_ import get_logger

logger = get_logger("genes")


@dataclass(frozen=True)
class SizeFilterReport:
    """What a size filter did to one gene set (for logs, provenance, methods)."""

    n_terms_in: int
    n_terms_out: int
    n_below: int
    n_above: int
    n_unmatched: int
    min_size: int
    max_size: int | None
    median_matched_size: float

    @property
    def applied(self) -> bool:
        return self.min_size > 1 or self.max_size is not None

    def describe(self) -> str:
        window = f"{self.min_size}-{self.max_size if self.max_size is not None else 'inf'}"
        return (
            f"{self.n_terms_out}/{self.n_terms_in} terms kept (matched size {window}); "
            f"dropped {self.n_below} below, {self.n_above} above, "
            f"{self.n_unmatched} with no gene in the universe; "
            f"median matched size {self.median_matched_size:.0f}"
        )


def matched_sizes(geneset_resource, gene_universe) -> dict[str, int]:
    """Map each term to the number of its genes present in *gene_universe*."""
    universe = {str(g) for g in np.asarray(gene_universe, dtype=object).reshape(-1).tolist()}
    mapping = as_geneset_mapping(geneset_resource)
    return {term: sum(1 for g in members if g in universe) for term, members in mapping.items()}


def size_report(
    geneset_resource,
    gene_universe,
    *,
    min_size: int = 1,
    max_size: int | None = None,
) -> SizeFilterReport:
    """Summarise what a ``[min_size, max_size]`` matched-size window would keep."""
    sizes = matched_sizes(geneset_resource, gene_universe)
    counts = np.asarray(list(sizes.values()), dtype=float)
    matched = counts[counts > 0]
    lo = max(1, int(min_size))
    n_below = int(((counts > 0) & (counts < lo)).sum())
    n_above = 0 if max_size is None else int((counts > int(max_size)).sum())
    return SizeFilterReport(
        n_terms_in=len(sizes),
        n_terms_out=len(sizes) - n_below - n_above - int((counts == 0).sum()),
        n_below=n_below,
        n_above=n_above,
        n_unmatched=int((counts == 0).sum()),
        min_size=lo,
        max_size=None if max_size is None else int(max_size),
        median_matched_size=float(np.median(matched)) if matched.size else 0.0,
    )


def write_filtered_gmt(
    geneset_resource,
    gene_universe,
    destination: str | Path,
    *,
    min_size: int = 1,
    max_size: int | None = None,
) -> tuple[str, SizeFilterReport]:
    """Write a ``.gmt`` holding only terms inside the matched-size window.

    Members are **not** restricted to the universe — the full member list is kept
    so each backend still computes its own overlap exactly as it would have; only
    whole terms are dropped.  Returns the written path and the report.
    """
    sizes = matched_sizes(geneset_resource, gene_universe)
    mapping = as_geneset_mapping(geneset_resource)
    report = size_report(geneset_resource, gene_universe, min_size=min_size, max_size=max_size)

    lo = max(1, int(min_size))
    hi = None if max_size is None else int(max_size)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for term, members in mapping.items():
            n = sizes.get(term, 0)
            if n < lo or (hi is not None and n > hi):
                continue
            handle.write("\t".join([str(term), "", *(str(g) for g in members)]) + "\n")
    return str(path), report


def apply_size_filter(
    geneset_resource,
    gene_universe,
    *,
    min_size: int = 1,
    max_size: int | None = None,
    outdir: str | Path | None = None,
    label: str = "geneset",
):
    """Resolve a gene set to what the backends should actually test.

    Returns ``(resource, report)``.  When no window is set the original
    *geneset_resource* is returned untouched (so runs stay bit-reproducible) and
    the report still describes how many terms *would* have been dropped.
    """
    report = size_report(geneset_resource, gene_universe, min_size=min_size, max_size=max_size)
    if not report.applied:
        logger.info(
            "geneset %r: no size filter applied — %d/%d terms have <10 matched genes "
            "(median matched size %.0f). Consider --geneset-min-size/--geneset-max-size.",
            label,
            sum(1 for n in matched_sizes(geneset_resource, gene_universe).values() if 0 < n < 10),
            report.n_terms_in,
            report.median_matched_size,
        )
        return geneset_resource, report

    if report.n_terms_out == 0:
        raise ValueError(
            f"Size filter [{min_size}, {max_size}] removed every term of gene set "
            f"{label!r} ({report.n_terms_in} terms in). Widen the window."
        )
    if outdir is None:
        raise ValueError("apply_size_filter needs an outdir to write the filtered .gmt into.")
    path, report = write_filtered_gmt(
        geneset_resource,
        gene_universe,
        Path(outdir) / f"{label}_filtered.gmt",
        min_size=min_size,
        max_size=max_size,
    )
    logger.info("geneset %r size filter: %s → %s", label, report.describe(), path)
    return path, report


__all__ = [
    "SizeFilterReport",
    "apply_size_filter",
    "matched_sizes",
    "size_report",
    "write_filtered_gmt",
]
