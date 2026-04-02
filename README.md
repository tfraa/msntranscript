# msnpip — Morphometric Similarity Networks and Transcriptomics Pipeline


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A modular Python pipeline for constructing **Morphometric Similarity Networks (MSN)** from FreeSurfer cortical data and linking regional brain patterns to gene expression through **Partial Least Squares (PLS)** and **Gene Set Enrichment Analysis (GSEA)**, using transcriptomic data from the Allen Human Brain Atlas.
---

> ⚠️ **This project is still under active development.** Features may change or break without notice.

![Pipeline Overview](assets/MSNTRANSCRIPT.png)

---

## Installation Notes

> **Please read carefully before installing.**

### 1. Imaging Transcriptomics Toolbox

Follow the official installation instructions for the
[Imaging Transcriptomics Toolbox](https://github.com/alegiac95/Imaging-transcriptomics)
using **Python 3.9**.

> ⚠️ **Known issue (ENIGMA version mismatch):**
> During installation, you may encounter an error where the version
> from the Git repository and the version in the local version file
> do not match. To fix this, manually update the version number in
> the ENIGMA version file to match the Git version, then re-run the
> installation.

### 2. Platform-Specific Issues

> ⚠️ **Windows & macOS (Apple Silicon):**
> The Imaging Transcriptomics Toolbox currently has known issues on
> **Windows** and **macOS with Apple M-series chips**. If you run into
> problems, please check the toolbox's repository for updates or
> workarounds. Try to install those two packages with `--no-deps` first, and then install this package.
---

## Installation

### Prerequisites

- Python 3.9
- [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/) (for data collection; not required at runtime)
---

## Quick Start

### From FreeSurfer data (CLI)

```bash
msnpip full \
    --input  /path/to/freesurfer_subjects/ \
    --demographics demographics.csv \
    --output /path/to/output/ \
    --save-figures
```

### From a pre-merged DataFrame (CLI)

If you already have a CSV with morphometric features and demographics merged together:

```bash
msnpip full \
    --dataframe merged_dataset.csv \
    --output /path/to/output/ \
    --save-figures
```

### From a pre-merged DataFrame (Python API)

```python
import pandas as pd
from msnpip import Pipeline

df = pd.read_csv("merged_dataset.csv")

pipeline = Pipeline(save_all=True, save_figures=True)
pipeline.run_full_pipeline(
    dataframe=df,
    output_pdf="output/",
    groups=[0, 1, 2],
)
```

### Resume from intermediate results

```bash
# From pre-computed strength vectors (saved as strength_maps.pkl by the pipeline)
msnpip from-vectors --vectors strength_maps.pkl --output output/

# From PLS results
msnpip from-pls --pls-results pls_results.pkl --output output/

# From enrichment results (report only)
msnpip from-enrichment --enrichment-results enrichment_results.pkl --output output/
```

---

## Pipeline Stages

| Stage | CLI command | Input | Output |
|---|---|---|---|
| Full pipeline | `full` | FreeSurfer dir + demographics CSV **or** pre-merged DataFrame | PDF report + figures |
| From strength vectors | `from-vectors` | `.csv` of regional strength maps | PDF report |
| From PLS results | `from-pls` | `.pkl` PLS results | PDF report |
| From enrichment | `from-enrichment` | `.pkl` enrichment results | PDF report |

---

## Input Data Format

### FreeSurfer directory layout

```
freesurfer_subjects/
├── subject_001/
│   └── stats/
│       ├── lh.aparc.stats
│       └── rh.aparc.stats
├── subject_002/
│   └── ...
```

The pipeline extracts: `SurfArea`, `GrayVol`, `ThickAvg`, `MeanCurv`, `GausCurv` for all 68 Desikan-Killiany regions.

### Demographics CSV

Required columns (auto-detected, case-insensitive):

| Column | Description |
|---|---|
| `patient_id` / `id` | Subject identifier |
| `age` | Age in years |
| `sex` / `gender` | Biological sex |
| `tiv` / `icv` | Total intracranial volume |
| `group` / `grp` | Group label (0 = healthy controls) |

### Pre-merged DataFrame

When using `--dataframe` or the `dataframe=` API argument, provide a CSV that already contains both morphometric features and demographic columns. The pipeline validates the required columns before starting and raises a clear error if anything is missing.

Expected feature column naming convention: `{hemisphere}_{region}_{metric}`

Example: `lh_superiorfrontal_ThickAvg`, `rh_cuneus_SurfArea`

---

## Output

| File | Description |
|---|---|
| `Report.pdf` | Multi-page report with all figures and statistical summaries |
| `figures/*.png` | Individual figures saved as PNG (when `--save-figures` is used) |
| `merged_data.csv` | Merged input data (when `--save-all` is used) |
| `strength_maps.pkl` | Regional strength vectors per group comparison |
| `pls_results.pkl` | PLS gene results per group comparison |
| `enrichment_results.pkl` | GSEA results per group comparison and gene library |
| `{comparison}_{library}.csv` | Enrichment results as CSV tables |

---

## Gene Libraries

The following gene set libraries are bundled with the package:

| Library | Description |
|---|---|
| `GO_Biological_Process_2025` | Gene Ontology Biological Process terms (2025) |
| `KEGG_2021_H` | KEGG pathway database (2021, human) |
| `DisGeNET` | Disease-gene association database |
| `LAKE_Pooled` | Brain cell-type specific gene sets (Lake et al.) |

---

## CLI Reference

```
msnpip full
  --input PATH            FreeSurfer subjects directory
  --demographics PATH     Demographics CSV file
  --dataframe PATH        Pre-merged CSV (alternative to --input + --demographics)
  --output PATH           Output directory for report and figures  [required]
  --groups INT [INT ...]  Subset of group IDs to analyze
  --save-figures          Save individual figures as PNG files
  --figures-dir PATH      Directory for individual figures (default: ./figures)
  --save-all              Save all intermediate results to disk
  -v, --verbose           Enable debug-level logging
```

> **Gene libraries**: all `.gmt` files bundled in `msnpip/genes/` are used automatically — no flag needed.

---

## Citation

If you use this pipeline in your work, please cite:

```bibtex
@mastersthesis{tomasella2026msn,
  author  = {Tomasella, Francesco},
  title   = {Morphometric Similarity Networks and Imaging Transcriptomics},
  school  = {[Università di Padova]},
  year    = {2026},
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
