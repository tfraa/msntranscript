#!/usr/bin/env python
"""Recompute ORA for a whole grid with the PINNED TOOLBOX's own implementation.

ORA never touches the spatial null: it is a hypergeometric over-representation
test on the observed ranking, and the p-values it thresholds are already in the
curated ``<tag>_pls.csv`` / ``<tag>_corr.csv`` of a finished run.  So the whole
ORA layer can be rebuilt from those tables in minutes instead of re-fitting the
20k surrogates.

The test is ``imaging_transcriptomics.ora.ora_from_gene_table`` — the toolbox's
own function, called directly.  msnpip has no ORA of its own, so the output is
exactly what the toolbox produces and can be cited as such:

* tail = ``p <= --p-threshold`` (default 0.05) on the **uncorrected** empirical
  spin p-value, split by the sign of the ranking statistic;
* term test = ``hypergeom.sf``, BH **within direction**;
* **no category-size filter** — the toolbox's loader applies none, and neither
  do we here, so the term set is the full gene set.

Note the toolbox drops terms with zero overlap with the tail before correcting,
so ``m`` is data-dependent and smaller than the full term set.  That is the
reference behaviour; it is reported in the summary as ``m_tested``.

Usage::

    python scripts/ora_toolbox.py <grid-root> [--p-threshold 0.05] [--alpha 0.05]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from imaging_transcriptomics.ora import ora_from_gene_table

from msnpip.engine import _resolve_geneset

GENESETS = ("lake", "pooled", "KEGG_2021_H", "GO_Biological_Process_2025", "DisGeNET")
DIRECTION = {"up": "positive", "down": "negative"}


def cells(root: Path):
    for marker in sorted(root.glob("*/*/*/*/.msnpip_complete")):
        d = marker.parent
        msn, stat, method, cov = d.relative_to(root).parts
        yield msn, stat, method, cov, d


def gene_table(cell_dir: Path, tag: str) -> pd.DataFrame | None:
    """Curated gene table as the toolbox's ORA expects it: gene / score / p_value.

    ``zscore`` (PLS) and ``score`` (corr) are the ranking statistics; only their
    SIGN is used, to split the tail into over- and under-expressed directions.
    """
    for suffix, score in (("pls", "zscore"), ("corr", "score")):
        path = cell_dir / f"{tag}_{suffix}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if score in df.columns and {"gene", "p"} <= set(df.columns):
            return pd.DataFrame({"gene": df["gene"], "score": df[score], "p_value": df["p"]})
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--p-threshold", type=float, default=0.05, help="gene-tail cut")
    ap.add_argument("--alpha", type=float, default=0.05, help="FDR cut for the summary")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out_path = args.out or args.root / "ora_toolbox.csv"

    resolved = {}
    for name in GENESETS:
        try:
            resolved[name] = _resolve_geneset(name)
        except Exception as exc:  # a missing set must not silently vanish
            print(f"  ! gene set {name!r} could not be resolved: {exc}")

    frames = []
    for msn, stat, method, cov, cell_dir in cells(args.root):
        for path in sorted(cell_dir.glob("*_enrichment.csv")):
            tag = path.name[: -len("_enrichment.csv")]
            table = gene_table(cell_dir, tag)
            if table is None:
                print(f"  ! no curated gene table for {msn}/{stat}/{method}/{tag}")
                continue
            for name, resource in resolved.items():
                tails = ora_from_gene_table(
                    table,
                    gene_set=resource,
                    score_column="score",
                    p_threshold=args.p_threshold,
                )
                for key, direction in DIRECTION.items():
                    df = tails.get(key)
                    if df is None or df.empty:
                        continue
                    df = df.copy()
                    df.insert(0, "direction", direction)
                    df.insert(0, "geneset", name)
                    df.insert(0, "contrast", tag)
                    df.insert(0, "method", method)
                    df.insert(0, "cov", cov)
                    df.insert(0, "stat", stat)
                    df.insert(0, "msn", msn)
                    frames.append(df)
            print(f"  {msn}/{stat}/{method}/{tag}: done")

    if not frames:
        print("Nothing computed.")
        return 1
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}  ({len(out)} rows)")

    summary = (
        out.assign(sig=out["fdr"] < args.alpha)
        .groupby(["msn", "stat", "method", "contrast", "geneset", "direction"])
        .agg(
            m_tested=("Term", "nunique"),
            tail_size=("selected_size", "first"),
            n_sig=("sig", "sum"),
            min_fdr=("fdr", "min"),
        )
        .reset_index()
    )
    summary_path = out_path.with_name(out_path.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 500)
    print(f"\nn(FDR<{args.alpha}) per gene set — toolbox ORA, unfiltered term sets\n")
    print(
        summary.pivot_table(
            index=["msn", "stat", "method", "contrast"],
            columns="geneset",
            values="n_sig",
            aggfunc="sum",
        ).to_string()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
