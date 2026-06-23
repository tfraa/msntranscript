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
Per subject, each of the 5 morphometric metrics is z-scored across regions, then inter-regional
similarity is the Pearson correlation between regions' standardized feature vectors (diagonal
NaN). The MSN is whole-cortex (both hemispheres). Node strength is the **signed mean**:
`(mean of positive edges + mean of negative edges) / 2` (`positive` / `absolute` selectable).
The z-score `ddof` does not affect the result — a uniform per-column rescale cancels in the
row-wise Pearson.

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
  maps?" msnpip fixes the null to `vasa` and hard-fails (`MsnpipSurfaceNullError`) if the
  engine falls back to a grouped shuffle, so an invalid spin test can never reach a figure.
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
