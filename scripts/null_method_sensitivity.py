#!/usr/bin/env python
"""Compare enrichment significance across two spatial nulls (e.g. vasa vs moran).

Re-run the pipeline once with ``--null-method vasa`` and once with
``--null-method moran`` (into separate output folders), then point this script at
the two curated ``<TAG>_enrichment.csv`` files.  It reports, per backend × gene
set, how stable the significant categories are between the two nulls.

Usage:
    python scripts/null_method_sensitivity.py VASA_enrichment.csv MORAN_enrichment.csv \
        [--alpha 0.05] [--label-a vasa] [--label-b moran]

Exit code is 0 regardless of the outcome — this is a reporting aid, not a gate.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from msnpip.null_sensitivity import compare_stability


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv_a", help="curated <TAG>_enrichment.csv from the first null (e.g. vasa)")
    ap.add_argument("csv_b", help="curated <TAG>_enrichment.csv from the second null (e.g. moran)")
    ap.add_argument("--alpha", type=float, default=0.05, help="FDR/p significance threshold")
    ap.add_argument("--label-a", default="vasa")
    ap.add_argument("--label-b", default="moran")
    args = ap.parse_args()

    df_a = pd.read_csv(args.csv_a)
    df_b = pd.read_csv(args.csv_b)
    table = compare_stability(
        df_a, df_b, alpha=args.alpha, label_a=args.label_a, label_b=args.label_b
    )
    if table.empty:
        print("No comparable (backend x geneset) groups found in the two CSVs.")
        return 0
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(table.to_string(index=False))
    mean_j = float(table["jaccard"].mean())
    print(f"\nMean Jaccard overlap of significant categories: {mean_j:.3f}")
    print(
        "Interpretation: values near 1.0 mean the significant categories are stable "
        f"between {args.label_a!r} and {args.label_b!r}; low values warrant caution."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
