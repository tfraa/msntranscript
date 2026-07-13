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
negative edges. The MSN is whole-cortex (both hemispheres). Node strength is the **mean** of a
region's similarity edges (Morgan 2019 / Seidlitz 2018; dimensionless). On this complete network
the mean equals the sum up to a constant scale, so the choice does not affect any downstream
result; msnpip fixes it to the mean.

Edge definition (`msn.similarity`, `--msn-similarity`): `distance` (default, above) or
`correlation` — the **canonical morphometric similarity** (Seidlitz 2018 / Morgan 2019), the
Pearson correlation between the two regions' z-scored metric vectors, `∈ [-1, 1]` (allows
negative edges). Use `correlation` to match that literature; note that with only 5 features the
per-pair correlation is comparatively noisy (Seidlitz used ~10).

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

### Enrichment validity — three orthogonal axes
Category/cell-type inference is only as good as its null. msnpip checks these axes:
1. **Spatial (spin) null** — the `vasa` surface spin above (Layer A). The recommended, most
   stringent phenotype null (Arnatkeviciute et al. 2023). Verify it is real, not a silent
   shuffle, with `python scripts/verify_vasa_null.py` (asserts resolved==`vasa`, 34 cortical
   parcels, seed honoured) before any publication run.
2. **Null-method sensitivity** — spin tests are distorted by the spherical projection. Re-run the
   primary result under a non-spin null (`--null-method moran`) into a second output folder, then
   `python scripts/null_method_sensitivity.py VASA_enrichment.csv MORAN_enrichment.csv` reports
   the Jaccard overlap of significant categories. High overlap ⇒ null-robust.

### Cell-type enrichment (spin null, not Fisher)
The bundled `LAKE_Pooled` set is the Lake et al. snRNA-seq human-cortex cell-type marker set.
Because it runs through `ensemble`/`gsea` on the spin null (not a `GeneOverlap`/Fisher random-gene
null), cell-type results are already tested on the recommended null. AHBA is bulk microarray, so
regional cell-composition can drive apparent cell-type signal: cross-check any headline cell-type
result against an **independent** snRNA-seq marker set by adding its `.gmt` to `engine.gene_sets`
(no code change needed).

### Headline inference is cortical (34 parcels)
The template (Martins et al. 2022) is left-hemisphere cortical only; msnpip's headline enrichment
is the 34 DK cortical parcels that match it and carry the real spin null. (Subcortical regions
would need a non-spatial null and are not part of the headline inference.)

### What the engine reports (consume, don't re-derive)
- **Correlation:** sign-aware empirical `p` with `+1` smoothing, BH `fdr`, FWE `maxT`.
- **PLS:** component `p` is on the *cumulative* variance through component k; gene columns are
  `weight, zscore, p, fdr, maxT` — **`zscore` is a descriptive ranking aid, not significance**.
- **Enrichment:** `ensemble`-GCEA (primary, phenotype/spin null) and the corrected `gsea`
  (secondary cross-check). Both consume the same spin null (weights refit on each spun map via
  `boot_pls`). GSEA is computed by msnpip (`msnpip.genes.gsea_mainstyle`), **not** the engine:
  genes are re-ranked on every surrogate (the engine froze them at the observed ranking, which is
  anti-conservative — pure-H0 FPR ≈0.7); significance is a magnitude two-sided empirical `p` with
  Davison–Hinkley `+1/+1`, and `fdr` is **BH across the categories tested** (same convention as
  ensemble). Only `pls` gene-ranking is supported; the engine's `corr` backend is not used.
- **Two-tier reporting.** The **rigorous, primary** result is the **component-level** spin-null
  test (a significant PLS1/PLS2 means the transcriptomic axis explains the map beyond spatial
  autocorrelation). Enrichment then *characterises* that axis: **GCEA (`ensemble`) is the primary
  spin-null enrichment**, `gsea` a spin-null cross-check, and **`ora` is a template
  over-representation test** (Fisher on the weight-ranked PLS1± tails, `ora_z_cut`) reported as
  **candidate mechanisms only** — the random-gene null used by the source literature (Martins
  2022, Giacomel 2026), *not* spatial-null-corrected, never primary inference.

## Reporting caveats `[PUB]`
- Empirical p-resolution is `1/(B+1)`; with ~15,677 genes (DK) even 10⁴ permutations may not
  yield small adjusted single-gene values — report primarily at the component/category level.
- Cross-run multiplicity (multiple contrasts/components/genesets) is **your** responsibility;
  pre-specify the primary analysis and treat the rest as exploratory.
- Report the exact gene count for your atlas (DK = 15,677) — it is the FDR denominator.
- Hemisphere/region choices change the science; defaults are recorded in `manifest.json` and the
  report. The MSN uses both hemispheres; the engine input hemisphere (default `left`) is the
  selectable part.
