# Output files

`msnpip` writes a deliberately small, curated output set — one flat folder of
CSVs, a `plots/` folder of PNGs, and a single `report.pdf`. The verbose engine
bundle is written to a temporary `.engine/` staging directory, curated into the
files below, and then **deleted**; there is no `manifest.json`, no pickle, and no
numbered stage tree.

Throughout, `<tag>` is a contrast identifier of the form `<case>_vs_<ctrl>`
(e.g. `FTD_vs_HC`), and `<group>` / `<var>` are a group label and a correlation
variable.

```
out/
  merged_dataset.csv
  strength_maps.csv
  mean_msn_per_group.csv
  case_control_difference_maps.csv
  <tag>_region_stats.csv
  <tag>_pls.csv
  <tag>_pls_summary.csv
  <tag>_corr.csv                 # only when the `corr` method runs
  <tag>_enrichment.csv
  report.pdf
  plots/
    <tag>_violin.png
    <tag>_tvalue_bars.png
    <tag>_surface_inflated.png
    <tag>_surface_pial.png
    <tag>_surface_significant.png
    <var>_scatter.png
    <group>_strength_surface.png
    <group>_mean_msn_matrix.png
    <tag>_<engine-plot>.png            # copied PLS / enrichment engine plots
    <tag>_<backend>_<geneset>_enrichment.png
```

## CSV tables

| File | One row per | Key columns |
|---|---|---|
| `merged_dataset.csv` | subject | id, group, covariates, and the FreeSurfer region×metric columns that entered the MSN |
| `strength_maps.csv` | subject | one column per region — that region's node strength (sum of its morphometric-similarity edges; dimensionless) |
| `mean_msn_per_group.csv` | region | `region`, plus `mean_strength_<group>` for each group |
| `case_control_difference_maps.csv` | region | `region`, plus one column per contrast holding the regional contrast statistic (`beta`/`t`/`cohen_d`, per config) |
| `<tag>_region_stats.csv` | region | `region`, `beta`, `t`, `cohen_d`, `p`, `fdr` — the per-region OLS group contrast |
| `<tag>_pls.csv` | gene | `component`, `gene`, PLS `zscore`/`weight`, `p`, `fdr` — gene loadings on the contrast map |
| `<tag>_pls_summary.csv` | PLS component | explained variance, cumulative variance, and the spatial-null p-value per component |
| `<tag>_corr.csv` | gene | gene-wise correlation of expression with the contrast map (written **only** when `--method corr` runs) |
| `<tag>_enrichment.csv` | gene-set term | `method` (pls/corr), `enrichment` (ensemble/gsea/ora backend), `geneset`, `Term`, effect (`nes`/`es` or `z_score`), `p_val`, `fdr` |

Node strength is **dimensionless** (a sum of similarity ratios in (0, 1]), so the
strength and difference maps carry no physical unit.

## Plots (`plots/`)

| File | Content |
|---|---|
| `<tag>_violin.png` | node-strength distribution by group for the contrast |
| `<tag>_tvalue_bars.png` | per-region contrast t-values, left/right panels, FDR asterisks |
| `<tag>_surface_inflated.png`, `<tag>_surface_pial.png` | contrast statistic on the cortical surface (RdBu_r, centred at 0) |
| `<tag>_surface_significant.png` | only FDR-significant regions coloured |
| `<var>_scatter.png` | node strength vs. the correlation variable |
| `<group>_strength_surface.png` | group-mean node strength on the surface (viridis) |
| `<group>_mean_msn_matrix.png` | group-mean region×region similarity matrix |
| `<tag>_<engine-plot>.png` | PLS variance / enrichment dotplots & heatmaps copied from the engine |
| `<tag>_<backend>_<geneset>_enrichment.png` | diverging NES / z-score bar plot per backend × gene set |

## `report.pdf`

A single A4-portrait PDF that walks the analysis in order — cover, a Contents
page, then dataset, MSN construction, node strength, and one section per contrast
(t-value bars, surfaces, significant-region tables, PLS parameters, top genes, and
enrichment). Pages are numbered. The cover carries a red banner if the spatial
null degraded from a surface spin to a shuffle.

See [running_on_real_data.md](running_on_real_data.md) for how to drive a real
run and [statistics.md](statistics.md) for the methods behind these numbers.
