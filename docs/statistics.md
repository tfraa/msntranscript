# Statistical methods

This is the practitioner summary. The full theory companion is
`msnpip_methods_theory.md`; locked parameters live in `msnpip_refactor_spec.md` §0.0.

## Pipeline of inferences

```
subjects → per-subject MSN → node strength → GROUP CONTRAST (regional map x)
                                                   │
        demographic correlation (Layer 0)          ▼
                                          imaging transcriptomics (Layers A/B)
```

### 1. MSN construction
Per subject, each of the 5 morphometric metrics is standardized across regions with a robust
modified z-score `M = 0.6745 · (x − median) / MAD` (per metric). Inter-regional similarity is
the distance kernel `S = 1 / (1 + d / n_metrics)`, where `d` is the Euclidean distance between
the two regions' standardized 5-metric vectors and `n_metrics = 5` (diagonal NaN). This yields
a symmetric matrix bounded in `(0, 1]` — similarity is strictly positive, so there are no
negative edges. The MSN is whole-cortex (both hemispheres). Node strength is the **sum**
(default) or mean of a region's similarity edges (`strength_agg`, dimensionless).

### 2. Group contrast (subject-level)
Per region, OLS of `strength ~ group + covariates`; the exported regional statistic is the
group coefficient (`beta`, default), its `t`, or `cohen_d` (standardized mean difference on
covariate-residualized strength). Categorical covariates and site/scanner are one-hot encoded
(reference dropped). **This is a subject-level test — no spatial null applies here**; spatial
autocorrelation only matters when correlating two regional maps (see Layer A).

### 3. Demographic correlation (Layer 0)
Spearman (default) of node strength vs a continuous variable, globally or per region (per-region
gets Benjamini–Hochberg FDR across regions), optionally within a single group. Ordinary
correlation p-values; no spatial null.

### 4. Imaging transcriptomics — two null layers
- **Layer A (spatial)** — handled by the engine via the **`vasa` surface spin**. Answers: "is
  this gene/pathway association stronger than under spatially-autocorrelated random brain
  maps?" msnpip fixes the null to `vasa`. By default (`allow_null_fallback=True`), if the
  surface spin is unavailable the engine falls back (`auto` → grouped shuffle) and msnpip
  **warns** rather than aborting; the *resolved* null is recorded in the result metadata and
  flagged with a red banner on the report cover, so a degraded spatial test is visible but not
  silent. Set `allow_null_fallback=False` to hard-fail (`MsnpipSurfaceNullError`) instead.
- **Layer B (sampling)** — subject-level resampling of the contrast map's stability. Documented
  as a future option; not built in v2.

### What the engine reports (consume, don't re-derive)
- **Correlation:** sign-aware empirical `p` with `+1` smoothing, BH `fdr`, FWE `maxT`.
- **PLS:** component `p` is on the *cumulative* variance through component k; gene columns are
  `weight, zscore, p, fdr, maxT` — **`zscore` is a descriptive ranking aid, not significance**.
- **Enrichment:** `ensemble`-GCEA (primary, phenotype-side null) and `gsea` (NES recalibrated
  against the engine's imaging-permutation null — its `fdr` is a NES q-value, **not** BH, and is
  not numerically comparable to ensemble `fdr`).

## Reporting caveats `[PUB]`
- Empirical p-resolution is `1/(B+1)`; with ~15,677 genes (DK) even 10⁴ permutations may not
  yield small adjusted single-gene values — report primarily at the component/category level.
- Cross-run multiplicity (multiple contrasts/components/genesets) is **your** responsibility;
  pre-specify the primary analysis and treat the rest as exploratory.
- Report the exact gene count for your atlas (DK = 15,677) — it is the FDR denominator.
- Hemisphere/region choices change the science; defaults are recorded in `manifest.json` and the
  report. The MSN uses both hemispheres; the engine input hemisphere (default `left`) is the
  selectable part.
