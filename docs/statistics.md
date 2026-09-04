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

### Enrichment validity rests on the null
Category/cell-type inference is only as good as its null. The `vasa` surface spin (Layer A) is
the recommended phenotype null (Arnatkeviciute et al. 2023), but the default policy is
fallback-with-warning, so a run that *finished* has not necessarily *spun*: the resolved null is
recorded in a `null_method` column on every curated table and stated on the report cover — read
it before reporting anything. Set `allow_null_fallback=False` to turn a failed spin into a hard
error instead.

Spin tests are also distorted by the spherical projection, so a result that matters is worth
re-running under a non-spin null (`--null-method moran`) and comparing which categories survive.

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
  grossly anti-conservative — pure-H0 FPR ≈0.7). The **only** difference that remains from the
  engine's GSEA is this re-ranking — the p-value itself is now the engine's own sign-aware nominal
  empirical `p` (`gsea_utils.nominal_pvalues_from_nulls`, one-sided per observed-ES sign, `+1/+1`),
  so the reported `p_val` is byte-for-byte identical to imaging-transcriptomics v2. That p is
  **one-sided per sign, hence ~2× anti-conservative** relative to a magnitude two-sided p (pure-H0
  FPR ≈0.10, not 0.05) — a deliberate choice to match the reference package, not a calibration
  target; interpret `fdr` accordingly. `fdr` is **BH across the categories tested** (same
  convention as ensemble).
- **Gene-ranking method (`--method`).** Two methods are supported and feed the *same* corrected
  enrichment: `pls` (default, multivariate PLS) and `corr` (mass-univariate map↔gene correlation,
  the classic-toolbox alternative). `corr` runs the same spatial-null permutations, writes
  `corr_genes.tsv` (per-gene `score, p, fdr, maxT` from the engine's `CorrAnalysis`), and its
  single ranking is passed through the corrected re-ranked `gsea`, the template `ora`, and GCEA
  (`ensemble`). The engine's own `corr` GSEA is bypassed — it freezes gene positions like the PLS
  one. Pre-specify a single method as primary; the other is a declared sensitivity analysis.
- **Category-size filter (`--geneset-min-size` / `--geneset-max-size`).** Off by default, so a run
  stays bit-reproducible against earlier ones. When set, terms whose size *after* intersecting with
  the ranked gene universe falls outside the window are dropped by materialising a filtered `.gmt`
  next to the enrichment output, so what was tested is auditable rather than implied by a config
  value. It is applied to the **spin-null backends only** — `ensemble` and `gsea` share a term set
  and an `m`; **ORA is deliberately left unfiltered**, because the pinned toolbox's ORA applies no
  size window and the point of that backend is to reproduce the toolbox exactly. So an ORA `m` is
  not comparable with a GCEA `m` in the same run.
  `10–2000` is the conventional window; GSEA's own `15–500` is **wrong for this gene-set mix**
  (`LAKE_Pooled` has a median matched size of 783 and loses 6 of 7 terms). Pre-specify the bounds —
  tuning them on the results is p-hacking. On DK/left this filter *reduces* hits (small categories
  have noisier scores and land in the empirical tail more often, so it strips BH mass faster than
  it strips `m`), which is worth stating in the methods rather than hiding.
- **Reproducing the engine's (invalid) GSEA (`--gsea-backend`).** `corrected` (default) runs the
  re-ranked backend above; `engine` runs the pinned toolbox's own `PLSGenes.gsea`/`CorrAnalysis.gsea`;
  `both` emits each. The engine's output is written as backend **`gseafrozen`**, never `gsea`, so it
  cannot be pooled with or mistaken for the corrected table. It exists to reproduce or exhibit
  published v2 behaviour and is **not reportable as inference**. Two asymmetries make it more than a
  one-variable contrast: the engine routes the observed ranking through `gseapy.prerank`, which
  applies its own `min_size=15`/`max_size=1500` window, and its `fdr` is a GSEA-style NES-ratio
  q-value rather than BH. It also defaults to the engine's hardcoded **1000** surrogates whatever
  `--n-perm` says (`--gsea-engine-n-iter` overrides); at 1000 the empirical `p` granularity is
  `0.001`, which is coarser than the region where the BH crossing usually falls.
- **Two-tier reporting.** The **rigorous, primary** result is the **component-level** spin-null
  test (a significant PLS1/PLS2 means the transcriptomic axis explains the map beyond spatial
  autocorrelation). Enrichment then *characterises* that axis: **GCEA (`ensemble`) is the primary
  spin-null enrichment**, `gsea` a spin-null cross-check, and **ORA is a template
  over-representation test** (Fisher against the gene background) reported as **candidate
  mechanisms only** — the random-gene null used by the source literature (Martins 2022,
  Giacomel 2026), *not* spatial-null-corrected, never primary inference.
- **How ORA selects its genes.** `--enrichment ora` runs the **pinned toolbox's own**
  `imaging_transcriptomics.ora`, not a msnpip reimplementation, so the output is exactly what the
  reference package produces. The tail is `p <= ora_p_threshold` (default `0.05`) on the
  **uncorrected** empirical spin p-value, split by the sign of the ranking statistic; each term
  then gets a hypergeometric test, with BH applied **within direction**. Rows carry a `direction`
  column (`positive`/`negative`).

  Two properties decide how the table may be read. The term test uses the **random-gene**
  (hypergeometric) null — the spin null enters only through *which genes reach the tail*, never
  through the term test itself — so ORA is never spatial-null inference. And the toolbox drops
  terms with zero overlap with the tail before correcting, so `m` is data-dependent and smaller
  than the full term set. The tail is also not comparable across gene-ranking methods: the same
  `p` threshold selects tens of genes on the PLS path and thousands on the `corr` path, because
  the engine's PLS gene null is sign-folded.

## Reporting caveats `[PUB]`
- Empirical p-resolution is `1/(B+1)`; with ~15,677 genes (DK) even 10⁴ permutations may not
  yield small adjusted single-gene values — report primarily at the component/category level.
- Cross-run multiplicity (multiple contrasts/components/genesets) is **your** responsibility;
  pre-specify the primary analysis and treat the rest as exploratory.
- Report the exact gene count for your atlas (DK = 15,677) — it is the FDR denominator.
- Hemisphere choice changes the science; the resolved settings are recorded on the report cover.
  The MSN uses both hemispheres; the engine input hemisphere (default `left`) is the selectable
  part.
