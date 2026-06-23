# Engine contract — imaging-transcriptomics v2.0.0

This document records exactly how msnpip binds to the imaging-transcriptomics
engine.  It is the authoritative reference for `msnpip/engine.py` and for
mocking the engine in unit tests.  Any change to the engine API must be
reflected here before the msnpip code is updated.

---

## Pinned version

| Field | Value |
|---|---|
| Package | `imaging-transcriptomics` |
| Branch | `refactor-v2.0.0` |
| Commit SHA | `e6a2c237fc74a0b2072a6d58efaf9d1c22cc08e1` |
| Install (pyproject.toml) | `imaging-transcriptomics @ git+https://github.com/alegiac95/Imaging-transcriptomics@e6a2c237fc74a0b2072a6d58efaf9d1c22cc08e1` |

### Smoke test (run after `pip install -e .[dev]`)

```python
import imaging_transcriptomics as imt
atlases = imt.list_atlases()
assert any(a.id == "dk" for a in atlases), "DK atlas missing"
print("Engine OK — atlases:", [a.id for a in atlases])
```

---

## Top-level functions

```python
import imaging_transcriptomics as imt

# PLS workflow
result: imt.PLSResult = imt.run_pls(
    data,                          # np.ndarray shape (n_regions,)
    atlas="dk",
    hemisphere="left",
    regions="default",             # "default" == "cort" for most atlases
    source_space=None,
    input_rh=None,                 # only for hemisphere="both"
    n_components=1,                # supply exactly one of n_components or var
    var=None,
    n_permutations=10000,
    null_method="vasa",            # msnpip always passes "vasa"
    output_dir=None,               # Path → engine writes its own bundle
    enrichment_method="ensemble",  # "ensemble"|"gsea"|"ora"|"none"
    run_gsea=True,                 # add GSEA alongside primary enrichment
    gene_set=("lake","pooled","GO_Biological_Process_2025","KEGG_2021_Human","DisGeNET"),
    geneset_organism="Human",
    ora_p_threshold=None,
    seed=1234,
    n_jobs=1,
)

# Correlation workflow (same kwargs minus n_components/var)
result: imt.CorrelationResult = imt.run_corr(data, ...)

# Config-based entry point
config: imt.RunConfig = imt.build_run_config("pls", atlas="dk", ...)
result = imt.run_analysis(data, config)
```

---

## Atlas access

```python
atlases: list[imt.AtlasSpec] = imt.list_atlases()
spec: imt.AtlasSpec = imt.get_atlas("dk")
    # spec.n_regions_left  = 41  (34 cort + 7 sub)
    # spec.n_regions_both  = 83
    # spec.has_subcortex   = True
    # spec.labels_path, spec.default_hemisphere, spec.surface_*

desc: dict = imt.describe_atlas("dk")
df: pd.DataFrame = imt.atlas_table()

sel: imt.AtlasSelection = imt.select_atlas_data(
    atlas="dk", hemisphere="left", regions="default"
)
    # sel.labels      → DataFrame[id, label, hemisphere, structure]  ← CANONICAL ORDER
    # sel.expression  → DataFrame (regions × genes)
    # sel.gene_labels → np.ndarray
    # sel.region_names
```

**DK label order** (what `atlas_align.align_strength_to_atlas` must match):
- Columns `id, label, hemisphere, structure`
- Left hemisphere first, then right; cortex (34/hemi) before subcortex
- `n_regions_left = 41` (34 cort + 7 sub); cortex-only slice = first 34
- Region names are FreeSurfer aparc names (`bankssts`, `superiorfrontal`, …)
- msnpip aligns by `(hemisphere, label)` pair — the `id` column is engine-internal

---

## Result objects (consume these; do not re-derive)

```python
# PLSResult
result.metadata          # AnalysisMetadata (see below)
result.regional_values   # DataFrame: the aligned input map
result.components        # tuple[PLSComponentResult, ...]
result.cumulative_variance  # np.ndarray
result.output_dir        # Path where the engine wrote its bundle

# PLSComponentResult (one per component)
comp.index               # int (1-based)
comp.explained_variance  # float
comp.p_value             # cumulative-variance p against spatial null
comp.gene_table          # DataFrame: weight, zscore, p, fdr, maxT
comp.gsea_table          # DataFrame | None
comp.ensemble_table      # DataFrame | None
comp.ora_tables          # dict{"pos": DataFrame, "neg": DataFrame} | None

# CorrelationResult
result.metadata
result.regional_values
result.gene_table        # DataFrame: r, p, fdr, maxT
result.gsea_table
result.ensemble_table
result.ora_tables
result.output_dir

# AnalysisMetadata
meta.method              # "pls" | "corr"
meta.atlas_id
meta.atlas_label
meta.hemisphere          # "left" | "both"
meta.regions             # "default" / "cort" / "cort+sub"
meta.source
meta.source_kind
meta.source_space
meta.n_permutations
meta.null_method         # msnpip checks this is NOT "random"
meta.enrichment_method
meta.geneset
meta.geneset_organism
meta.ora_p_threshold
meta.n_components
```

---

## Engine-written output files (per `output_dir`)

```
regional_values.tsv
corr_genes.tsv              (correlation mode)
pls_summary.tsv             (PLS mode)
pls_component_{i}.tsv       (one per component)
gsea_*_results.tsv
ensemble_*_results.tsv
ora_*_{pos,neg}.tsv
metadata.json
README
*.png                       (engine-generated plots)
```

msnpip places each engine call in its own subdirectory:
`<output>/03_transcriptomics/<case>_vs_<ctrl>/<method>/`

---

## Exceptions to catch in `engine.py`

| Engine exception | msnpip wraps as |
|---|---|
| `ImagingTranscriptomicsError` (base) | `MsnpipEngineError` |
| `AtlasError`, `AtlasAssetError` | `MsnpipEngineError` |
| `ConfigurationError` | `MsnpipEngineError` |
| `InputDataError`, `InputAlignmentError` | `MsnpipEngineError` |
| `NullModelError` | `MsnpipEngineError` (then check for silent fallback) |
| `PlottingUnavailableError` | logged as WARNING; plotting skipped |

**Surface-null enforcement** — after every engine call, check:
```python
if result.metadata.null_method == "random" and cfg.require_surface_null:
    raise MsnpipSurfaceNullError(
        f"Engine fell back to grouped shuffle (null_method='random') "
        f"after being requested 'vasa'. Surface assets may be missing. "
        f"Run: python -c \"import neuromaps; neuromaps.datasets.fetch_fsaverage()\""
    )
```

---

## Plotting reuse

```python
from imaging_transcriptomics import plotting

plotting.plot_cortical_surface_map(table, atlas_id=..., value_column=..., title=..., output_path=...)
plotting.plot_brain_volume_map(table, ...)
plotting.plot_pls_component(...)
plotting.plot_pls_variance(...)
plotting.plot_correlation_distribution(...)
plotting.plot_correlation_ranking(...)
plotting.plot_ensemble_dotplot(...)
plotting.plot_gsea_dotplot(...)
plotting.plot_ora_heatmap(...)
plotting.save_result_plots(result, output_dir) -> list[Path]

# Low-level surface primitives (used by viz/surface_extra.py for dorsal view)
from imaging_transcriptomics.outputs.brain import (
    surface_view,
    load_surface_mesh,
    load_surface_parcellation,
    vertex_values_for_hemisphere,
    surface_mesh_paths,
)
```

All plotting functions accept a **region table** with columns
`id, label, hemisphere, structure, <value_column>` — exactly what
`atlas_align.to_region_table` produces.

---

## What msnpip adds (not in the engine)

| Capability | msnpip module |
|---|---|
| FreeSurfer aparc.stats reading | `io/readers.py` |
| Locale-aware CSV reading | `io/readers.py` |
| Schema validation | `io/schema.py` |
| ID normalization + match-rate guard | `io/matching.py` |
| Within-subject z-score → MSN matrix | `msn/construct.py` |
| Node-strength aggregation | `msn/construct.py` |
| GLM group contrast (beta/t/cohen_d) | `stats/glm.py` |
| Demographic correlation (Spearman, within-group) | `stats/correlation.py` |
| Covariate exclusion sensitivity | `stats/sensitivity.py` |
| Atlas label alignment | `atlas_align.py` |
| Violin-by-group strength plot | `viz/distributions.py` |
| Dorsal surface view | `viz/surface_extra.py` |
| Demographic scatter plot | `viz/scatter.py` |
| Aggregated PDF report | `report/builder.py` |
| Output tree + sha256 manifest | `io/writers.py` |
| Stage machine + checkpoint/resume | `pipeline.py` |
