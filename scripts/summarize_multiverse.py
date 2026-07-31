#!/usr/bin/env python
"""Collapse a multiverse grid into comparison tables.

Reads the curated CSVs of every completed cell under a grid root laid out as
``<msn>/<stat>/<method>/<cov>/`` (what ``run_multiverse.py`` writes) and emits
four tables, each also written as a CSV next to the printed output:

``map_gene``     one row per cell x contrast: regional significance, the PLS
                 component-level spin test (the *primary* inference), and
                 gene-level significance under BH and max-T.
``enrichment``   n(FDR<0.05) per backend, summed over gene sets, plus the BH
                 factor that says whether hits were attainable at all.
``by_geneset``   the same counts split per gene set, so a hit count can never be
                 read without knowing which m produced it.
``ora_tails``    how many genes each ORA tail actually selected — an ORA table is
                 uninterpretable without it.

Usage::

    python scripts/summarize_multiverse.py <grid-root> [--alpha 0.05] [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Backend → how to read it. The spin-null backends carry inference; the ORA
# family is a random-gene null (candidate mechanisms only); gseafrozen is the
# engine's invalid frozen-ranking null, kept for the methods comparison.
SPIN_BACKENDS = ("ensemble", "gsea")
ORA_BACKENDS = ("oraz", "orap", "oratopn")
INVALID_BACKENDS = ("gseafrozen",)
BACKEND_ORDER = (*SPIN_BACKENDS, *INVALID_BACKENDS, *ORA_BACKENDS)

PRIMARY_GENESETS = ("lake", "KEGG_2021_H")


def bh_factor(pvals: np.ndarray, alpha: float) -> float:
    """max_t alpha*F(t)/t — BH rejects something iff this is >= 1.

    Reported because ``m/(B+1)`` (the rank-1 adjusted p) is the wrong quantity:
    BH is a step-up procedure, so what matters is the *mass* of small p-values,
    not the single best one.  A factor below 1 means no term could have been
    significant at this alpha no matter how the effects fell.
    """
    p = np.sort(np.asarray(pvals, dtype=float))
    p = p[np.isfinite(p)]
    if p.size == 0:
        return float("nan")
    ranks = np.arange(1, p.size + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.nanmax(alpha * (ranks / p.size) / p))


def _max_abs(series) -> float:
    """max|x| ignoring NaN, and NaN (not a warning) when there is nothing to take."""
    vals = np.abs(np.asarray(series, dtype=float))
    vals = vals[np.isfinite(vals)]
    return float(vals.max()) if vals.size else float("nan")


def cells(root: Path) -> list[tuple[str, str, str, str, Path]]:
    """Every completed cell, in a stable order."""
    found = []
    for marker in sorted(root.glob("*/*/*/*/.msnpip_complete")):
        d = marker.parent
        msn, stat, method, cov = d.relative_to(root).parts
        found.append((msn, stat, method, cov, d))
    return found


def _read(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def contrasts(cell_dir: Path) -> list[str]:
    return sorted(p.name[: -len("_enrichment.csv")] for p in cell_dir.glob("*_enrichment.csv"))


def map_gene_row(msn, stat, method, cov, cell_dir: Path, tag: str, alpha: float) -> dict:
    """Regional + component-level + gene-level summary for one cell x contrast."""
    row = {"msn": msn, "stat": stat, "method": method, "cov": cov, "contrast": tag}

    reg = _read(cell_dir / f"{tag}_region_stats.csv")
    row["n_regions"] = len(reg) if reg is not None else np.nan
    row["reg_fdr_sig"] = int((reg["fdr"] < alpha).sum()) if reg is not None else np.nan

    # PLS component-level spin test — the primary inference. Only the pls cells
    # produce it; the corr backend has no component decomposition.
    summ = _read(cell_dir / f"{tag}_pls_summary.csv")
    if summ is not None and not summ.empty:
        first = summ.iloc[0]
        row["pls1_p"] = float(first.get("p", np.nan))
        row["pls1_ve"] = float(first.get("variance_explained", np.nan))
    else:
        row["pls1_p"] = np.nan
        row["pls1_ve"] = np.nan

    genes = _read(cell_dir / f"{tag}_pls.csv")
    if genes is None:
        genes = _read(cell_dir / f"{tag}_corr.csv")
    if genes is not None and not genes.empty:
        row["n_genes"] = len(genes)
        row["gene_min_p"] = float(genes["p"].min())
        row["gene_min_fdr"] = float(genes["fdr"].min())
        row["gene_fdr_sig"] = int((genes["fdr"] < alpha).sum())
        if "maxT" in genes.columns:
            row["gene_min_maxT"] = float(genes["maxT"].min())
            row["gene_maxT_sig"] = int((genes["maxT"] < alpha).sum())
    return row


def enrichment_rows(msn, stat, method, cov, cell_dir: Path, tag: str, alpha: float):
    """Per-backend and per-(backend, geneset) enrichment summaries."""
    enr = _read(cell_dir / f"{tag}_enrichment.csv")
    if enr is None or enr.empty:
        return [], [], []
    base = {"msn": msn, "stat": stat, "method": method, "cov": cov, "contrast": tag}

    per_backend, per_geneset, ora = [], [], []
    for backend, sub in enr.groupby("enrichment"):
        per_backend.append(
            {
                **base,
                "backend": backend,
                "n_terms": len(sub),
                "n_sig": int((sub["fdr"] < alpha).sum()),
                "min_fdr": float(sub["fdr"].min()),
                # NO pooled bh_factor: each backend BH-corrects WITHIN a gene set
                # (and ORA within direction too), so a factor computed over the
                # pooled p-values describes a correction that never ran. It lives
                # in the by_geneset table, at the m it was actually applied to.
                # Only the GCEA/ensemble table carries a category z; the GSEA and
                # ORA backends leave it empty, so an all-NaN column is expected.
                "max_abs_z": _max_abs(sub["z_score"]) if "z_score" in sub else np.nan,
            }
        )
        for geneset, gsub in sub.groupby("geneset"):
            per_geneset.append(
                {
                    **base,
                    "backend": backend,
                    "geneset": geneset,
                    "m": len(gsub),
                    "n_sig": int((gsub["fdr"] < alpha).sum()),
                    "min_fdr": float(gsub["fdr"].min()),
                    "bh_factor": bh_factor(gsub["p_val"].to_numpy(), alpha),
                }
            )

    if "ora_tail" in enr.columns:
        oenr = enr[enr["ora_tail"].notna()]
        for (backend, tail, direction), sub in oenr.groupby(
            ["enrichment", "ora_tail", "direction"]
        ):
            ora.append(
                {
                    **base,
                    "backend": backend,
                    "ora_tail": tail,
                    "direction": direction,
                    "tail_size": int(sub["tail_size"].iloc[0]),
                    "n_sig": int((sub["fdr"] < alpha).sum()),
                    "min_fdr": float(sub["fdr"].min()),
                }
            )
    return per_backend, per_geneset, ora


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path, help="grid root (contains <msn>/<stat>/<method>/<cov>/)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=None, help="where to write CSVs (default: root)")
    args = ap.parse_args()

    root = args.root
    out_dir = args.out or root
    out_dir.mkdir(parents=True, exist_ok=True)

    found = cells(root)
    if not found:
        print(f"No completed cells (.msnpip_complete) under {root}")
        return 1

    mg, eb, eg, ot = [], [], [], []
    for msn, stat, method, cov, cell_dir in found:
        for tag in contrasts(cell_dir):
            mg.append(map_gene_row(msn, stat, method, cov, cell_dir, tag, args.alpha))
            a, b, c = enrichment_rows(msn, stat, method, cov, cell_dir, tag, args.alpha)
            eb += a
            eg += b
            ot += c

    tables = {
        "map_gene": pd.DataFrame(mg),
        "enrichment": pd.DataFrame(eb),
        "by_geneset": pd.DataFrame(eg),
        "ora_tails": pd.DataFrame(ot),
    }
    for name, df in tables.items():
        if df.empty:
            continue
        path = out_dir / f"summary_{name}.csv"
        df.to_csv(path, index=False)
        print(f"wrote {path}  ({len(df)} rows)")

    pd.set_option("display.width", 240)
    pd.set_option("display.max_rows", 400)

    print("\n" + "=" * 100)
    print(f"MAP-LEVEL + GENE-LEVEL  (alpha = {args.alpha:.2f})")
    print("=" * 100)
    print(tables["map_gene"].to_string(index=False))

    if not tables["enrichment"].empty:
        print("\n" + "=" * 100)
        print("ENRICHMENT — n(FDR<alpha) per backend, all gene sets pooled")
        print("=" * 100)
        wide = tables["enrichment"].pivot_table(
            index=["msn", "stat", "method", "contrast"],
            columns="backend",
            values="n_sig",
            aggfunc="sum",
        )
        cols = [c for c in BACKEND_ORDER if c in wide.columns]
        print(wide[cols].to_string())

    if not tables["by_geneset"].empty:
        print("\n" + "=" * 100)
        print(f"PRIMARY GENE SETS — n(FDR<alpha) [min FDR] for {', '.join(PRIMARY_GENESETS)}")
        print("=" * 100)
        prim = tables["by_geneset"][tables["by_geneset"]["geneset"].isin(PRIMARY_GENESETS)]
        if not prim.empty:
            print(
                prim.pivot_table(
                    index=["msn", "stat", "method", "contrast", "geneset"],
                    columns="backend",
                    values="n_sig",
                    aggfunc="sum",
                ).to_string()
            )

    if not tables["ora_tails"].empty:
        print("\n" + "=" * 100)
        print("ORA TAIL SIZES — how many genes each rule selected")
        print("=" * 100)
        print(
            tables["ora_tails"]
            .pivot_table(
                index=["msn", "stat", "method", "contrast"],
                columns=["ora_tail", "direction"],
                values="tail_size",
                aggfunc="first",
            )
            .to_string()
        )

    print(
        "\nNOTES\n"
        "  * pls1_p is the PRIMARY inference (component-level spin test). The corr\n"
        "    cells have no component decomposition, so pls1_p/pls1_ve are blank there.\n"
        "  * bh_factor = max_t alpha*F(t)/t: BH yields >=1 hit iff it reaches 1.0.\n"
        "    A value below 1 means no term was attainable at this alpha, whatever\n"
        "    the effect sizes — read a zero hit count against it before concluding.\n"
        "  * gseafrozen scores every surrogate at the OBSERVED gene positions\n"
        "    (pure-H0 FPR ~0.7) and its 'fdr' is a NES-ratio q, not BH. Methods\n"
        "    comparison only — never inference.\n"
        "  * the ORA backends use a random-gene (hypergeometric) null and are BH\n"
        "    corrected WITHIN direction. Candidate mechanisms only.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
