"""Null-method sensitivity: are the significant categories stable across nulls?

Spin tests are distorted by the spherical projection and do not perfectly control
false positives for strongly autocorrelated maps (Bazinet, Liu & Misic 2025;
Markello & Misic 2021).  Best practice is to re-run the primary result under a
non-spin null (``moran``, no spherical projection) and confirm the significant
categories are stable.

The pipeline already exposes ``--null-method {vasa, alexander_bloch, moran}``.
This module does not re-run the engine; it compares two curated enrichment CSVs
(one per null method) and reports, for each backend × gene set, how much the set
of significant categories overlaps and which terms flip.  High Jaccard overlap ⇒
the result is null-robust; large disagreement ⇒ interpret with caution.
"""

from __future__ import annotations

import pandas as pd

_SIG_COLS = ("fdr", "p_val", "p")


def _sig_column(df: pd.DataFrame) -> str | None:
    return next((c for c in _SIG_COLS if c in df.columns), None)


def significant_terms(df: pd.DataFrame, alpha: float = 0.05) -> dict[tuple[str, str], set[str]]:
    """Map ``(enrichment, geneset) -> {significant Term}`` at ``fdr`` (or p) < alpha."""

    sig_col = _sig_column(df)
    if sig_col is None or "Term" not in df.columns:
        return {}
    group_cols = [c for c in ("enrichment", "geneset") if c in df.columns]
    out: dict[tuple[str, str], set[str]] = {}
    if not group_cols:
        hits = set(df.loc[df[sig_col] < alpha, "Term"].astype(str))
        out[("", "")] = hits
        return out
    for key, sub in df.groupby(group_cols):
        key_t = key if isinstance(key, tuple) else (key,)
        backend = str(key_t[0]) if "enrichment" in group_cols else ""
        geneset = str(key_t[-1]) if "geneset" in group_cols else ""
        out[(backend, geneset)] = set(sub.loc[sub[sig_col] < alpha, "Term"].astype(str))
    return out


def compare_stability(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    *,
    alpha: float = 0.05,
    label_a: str = "a",
    label_b: str = "b",
) -> pd.DataFrame:
    """Per backend × gene set overlap of significant terms between two null runs.

    Returns a table with the counts, the Jaccard overlap, and the terms significant
    under only one of the two nulls.  ``jaccard`` is 1.0 when both sets are empty
    (trivially stable: nothing significant either way).
    """

    a = significant_terms(df_a, alpha)
    b = significant_terms(df_b, alpha)
    rows = []
    for key in sorted(set(a) | set(b)):
        sa, sb = a.get(key, set()), b.get(key, set())
        union = sa | sb
        jaccard = 1.0 if not union else len(sa & sb) / len(union)
        rows.append(
            {
                "enrichment": key[0],
                "geneset": key[1],
                f"n_sig_{label_a}": len(sa),
                f"n_sig_{label_b}": len(sb),
                "n_sig_both": len(sa & sb),
                "jaccard": jaccard,
                f"only_{label_a}": ";".join(sorted(sa - sb)),
                f"only_{label_b}": ";".join(sorted(sb - sa)),
            }
        )
    return pd.DataFrame(rows)
