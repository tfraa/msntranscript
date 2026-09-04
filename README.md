# msnpip — Morphometric Similarity Networks and imaging transcriptomics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

msnpip builds **Morphometric Similarity Networks (MSN)** from FreeSurfer cortical data,
contrasts regional node strength between groups, and links the resulting regional map to
gene expression from the **Allen Human Brain Atlas** via PLS or mass-univariate correlation,
followed by gene-set enrichment. The transcriptomics work is done by the
[Imaging Transcriptomics Toolbox](https://github.com/alegiac95/Imaging-transcriptomics)
(v2.0.0), pinned to a fixed commit.

The package, CLI and imports are all `msnpip`; the repository is hosted as `msntranscript`.

> **Thesis snapshot.** The version used for the MSc thesis is preserved on branch
> [`v1-april`](https://github.com/tfraa/msntranscript/tree/v1-april) (tag `v1.0-thesis`,
> commit `cc8c776`). `main` is the current v2, which fixes several statistical defects in
> that version — see [docs/statistics.md](docs/statistics.md).

![Pipeline Overview](assets/MSNTRANSCRIPT.png)

---

## What it does

```
LOAD → VALIDATE → MSN → CONTRAST → (CORRELATION) → TRANSCRIPTOMICS → FIGURES → REPORT
```

- **MSN.** Each region's 5 morphometric features (`SurfArea, GrayVol, ThickAvg, MeanCurv,
  GausCurv`) are standardised across regions within subject with a modified z-score
  `M = 0.6745·(x − median)/MAD`. Edges are then either the distance kernel
  `S = 1/(1 + d/n_metrics)` (default, strictly positive, `d` = Euclidean distance between the
  two regions' standardised vectors) or the Pearson correlation between them (the canonical
  Seidlitz/Morgan definition, which allows negative edges). **Node strength is the mean of a
  region's edges.** The MSN always spans both hemispheres.
- **Group contrast.** Per region, OLS of `strength ~ group + covariates`, exporting `beta`
  (default), `t`, or `cohen_d`. Categorical covariates and site/scanner are one-hot encoded.
- **Transcriptomics.** Runs in the pinned engine against the **`vasa` surface-spin null**. By
  default a failed spin falls back with a warning; the null actually resolved is recorded in a
  `null_method` column on every curated table and stated on the report cover. Set
  `allow_null_fallback=False` to make a failed spin an error instead.

---

## Installation

Python **3.12** is the supported and tested version (3.10 is the declared floor). The engine
installs from a pinned git commit, so **git must be available**.

```bash
pip install -e .                    # add [dev] for the test/lint toolchain
pip install -e ".[dev]" -c constraints.txt   # the exact versions the suite is verified against
```

Verify the engine is wired up:

```python
import imaging_transcriptomics as imt
assert any(a.id == "dk" for a in imt.list_atlases())
```

The first run that needs cortical surfaces fetches the neuromaps fsaverage meshes; the Docker
image bakes them in.

---

## Quick start

### From FreeSurfer data

```bash
msnpip full \
    --input /path/to/freesurfer_subjects/ \
    --demographics demographics.csv \
    --output out/ \
    --group-col group --case FTD --control HC \
    --predictors age sex tiv \
    --hemisphere left \
    --method pls --ncomp 1 --n-perm 10000 \
    --enrichment ensemble gsea
```

### From a pre-merged table

```bash
msnpip full \
    --dataframe merged.csv \
    --output out/ \
    --group-col group --case FTD --control HC \
    --predictors age sex tiv \
    --correlate-with age --corr-scope global \
    --method pls --ncomp 1 --n-perm 1000 --enrichment ensemble --seed 1234
```

The atlas is locked to DK and the region scope to cortex.

### Python API

```python
from pathlib import Path
from msnpip.config import IOConfig, GLMConfig, EngineConfig, PipelineConfig
from msnpip.pipeline import run_pipeline

cfg = PipelineConfig(
    io=IOConfig(dataframe=Path("merged.csv")),
    output=Path("out/"),
    group_col="group", case="FTD", control="HC",
    glm=GLMConfig(predictors=("age", "sex", "tiv")),
    engine=EngineConfig(methods=("pls",), n_components=1, n_permutations=10000),
)
run_pipeline(cfg)
```

### Resume / partial runs

```bash
msnpip full ... --stop-stage MSN
msnpip full ... --start-stage TRANSCRIPTOMICS     # reuses persisted earlier stages
msnpip from-strength --output out/ --case FTD --control HC --predictors age sex tiv
```

Helpers: `msnpip list-atlases`, `msnpip list-genesets`.

---

## Input data format

### FreeSurfer directory layout

```
freesurfer_subjects/
├── sub-001/stats/{lh,rh}.aparc.stats
├── sub-002/stats/{lh,rh}.aparc.stats
└── ...
```

Extracted metrics: `SurfArea, GrayVol, ThickAvg, MeanCurv, GausCurv` for the 34 Desikan–Killiany
cortical regions per hemisphere.

### Demographics / merged CSV

Column roles are auto-detected; `--id-col` and `--group-col` override the detection.

| Role | Example column names |
|---|---|
| id | `subject_id`, `participant_id`, `id` |
| group | `group`, `diagnosis`, `dx` |
| age | `age` |
| sex | `sex`, `gender` |
| tiv | `tiv`, `icv` |
| site | `site`, `scanner` |

IDs are matched **exactly after whitespace stripping** — `sub-001` and `sub-1` are distinct.
Feature columns follow `{hemisphere}_{region}_{metric}`.

---

## Outputs

A flat, curated set of CSVs, a `plots/` folder and one `report.pdf` (`<tag>` is
`<case>_vs_<ctrl>`). The verbose engine bundle is staged in a temporary `.engine/` folder,
curated into these files, then deleted.

```
out/
  merged_dataset.csv                  validated, merged input table
  strength_maps.csv                   per-subject node strength per region
  mean_msn_per_group.csv              group-mean node strength per region
  case_control_difference_maps.csv    per-contrast regional contrast map
  <tag>_region_stats.csv              per-region beta/t/cohen_d/p/fdr
  <tag>_pls.csv  <tag>_pls_summary.csv  PLS gene results + component variance
  <tag>_corr.csv                      correlation gene results (only if --method corr)
  <tag>_enrichment.csv                enrichment terms per backend × gene set
  plots/                              violins, t-value bars, surfaces, matrices, enrichment
  report.pdf                          assembled A4-portrait report
```

See [docs/outputs.md](docs/outputs.md) for a column-by-column reference and
[docs/tutorial.md](docs/tutorial.md) for a runnable first run on synthetic data.

---

## Reading the results

The layers do not carry equal weight.

- **The primary inference is the component-level spin test.** A significant PLS component means
  the transcriptomic axis explains the regional map beyond spatial autocorrelation. Report this.
- **`ensemble` (GCEA) is the primary enrichment**, tested against the same spin null.
- **`gsea` is a spin-null cross-check.** msnpip computes it itself, re-ranking genes on every
  surrogate; the engine's own GSEA freezes gene positions at the observed ranking, which is not a
  valid null for a rank-position statistic. If you deliberately re-enable it with
  `--gsea-backend engine`, its output is labelled `gseafrozen` and is not reportable.
- **`ora` is a template over-representation test** against the random-gene (hypergeometric) null,
  as used by the source literature. The spin null enters only through which genes reach the tail,
  never through the term test, so ORA results are **candidate mechanisms, never primary
  inference**.
- **Do not expect per-gene significance.** With 34 cortical parcels and ~15,677 genes, BH across
  the gene table has essentially no power; "component significant, no individual gene significant"
  is the expected outcome, not a bug.

[docs/statistics.md](docs/statistics.md) documents each of these, including the calibration
defects inherited from the pinned engine.

---

## Docker

```bash
docker build -f docker/Dockerfile -t msnpip:2.0 .

docker run --rm -v "$PWD/data:/data:ro" -v "$PWD/out:/out" msnpip:2.0 \
  full --dataframe /data/merged.csv --output /out \
  --group-col group --case FTD --control HC \
  --predictors age sex tiv --method pls --ncomp 1 \
  --n-perm 1000 --enrichment ensemble --seed 1234
```

The image bakes the neuromaps fsaverage cache so cortical plots and the spin null work offline.

---

## Methodological decisions

| Item | Value |
|---|---|
| Null model | `vasa` surface spin, falling back with a warning |
| MSN | 5 features, within-subject modified z-score, distance kernel `1/(1+d/n)` by default, **mean** node strength, both hemispheres |
| Contrast statistic | `beta` (default), `t`, or `cohen_d` |
| Enrichment | `ensemble` (primary) + corrected `gsea` (cross-check) + `ora` (candidate mechanisms) |
| Gene sets | LAKE, pooled, GO_BP_2025, KEGG_2021_H, DisGeNET |
| Defaults | atlas `dk`, engine hemisphere `left`, regions `cort`, n-perm 10,000 |

---

## Documentation

- [docs/tutorial.md](docs/tutorial.md) — a runnable first run on synthetic data
- [docs/running_on_real_data.md](docs/running_on_real_data.md) — real-cohort workflow and sanity checklist
- [docs/outputs.md](docs/outputs.md) — what every output file and column means
- [docs/statistics.md](docs/statistics.md) — methods, nulls, and the known calibration limits
- [docs/engine_contract.md](docs/engine_contract.md) — the pinned engine API msnpip depends on
- [docs/adding_an_atlas.md](docs/adding_an_atlas.md) — extending beyond DK

## License

MIT — see [LICENSE](LICENSE).
