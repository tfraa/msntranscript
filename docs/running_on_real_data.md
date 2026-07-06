# Running msnpip on real data

A practical guide for the first real-cohort run, with a results-sanity checklist.
Use the **dev** settings first (fast), confirm everything looks right, then do the
**publication** run.

> Data never goes in the repo. Outputs and inputs are git-ignored — keep them
> outside the project or in an ignored folder.

---

## 0. Pre-flight

```bash
pip install -e .                       # engine installs from the pinned commit (needs git)
python -c "import imaging_transcriptomics as imt; print([a.id for a in imt.list_atlases()])"
msnpip list-atlases                    # sanity: 'dk' should appear
```

Inputs, two supported modes:

- **FreeSurfer mode** — a subjects directory plus a demographics CSV:
  ```
  freesurfer_subjects/<subj>/stats/{lh,rh}.aparc.stats
  demographics.csv      # columns: subject_id, group, age, sex, tiv, site, ...
  ```
- **Merged mode** — a single wide CSV with feature columns
  `{hemi}_{region}_{metric}` (e.g. `lh_superiorfrontal_ThickAvg`) plus the
  demographic columns.

IDs are matched **exactly after whitespace stripping** (`sub-001` ≠ `sub-1`), so
make the IDs in the CSV match the FreeSurfer directory names exactly.

---

## 1. Dev run (fast — do this first)

Use 1,000 permutations and a single gene set so a full run finishes quickly while
you shake out data issues.

```bash
msnpip full \
  --input /path/to/freesurfer_subjects/ \
  --demographics /path/to/demographics.csv \
  --output runs/dev_001/ \
  --group-col group --case PATIENT --control CONTROL \
  --predictors age sex tiv site \
  --correlate-with age \
  --atlas dk --hemisphere left --regions cort \
  --method pls --ncomp 1 --n-perm 1000 \
  --enrichment ensemble \
  --geneset GO_Biological_Process_2025 \
  --seed 1234 -v
```

Merged-CSV equivalent: replace `--input/--demographics` with `--dataframe merged.csv`.
European locale CSV? add `--sep ';' --decimal ','` (auto-detection usually handles it).

---

## 2. Results-sanity checklist

After the dev run, check these — most data problems show up here:

- [ ] **Merge / match rate** — `00_inputs/merge_report.json`: `n_merged` is what you
      expect. A low count means IDs didn't match (see §4).
- [ ] **Dropped subjects** — `01_msn/dropped_subjects.json` is empty or only the
      subjects you know are incomplete (msnpip never imputes).
- [ ] **Group sizes** — run log / console: no "Small group(s)" warning, or you accept
      it. With n<10 per arm, prefer `--contrast-stat cohen_d`.
- [ ] **Schema** — `00_inputs/schema.json`: `id_col`, `group_col`, covariates, and
      `n_feature_cols` (≈ 340 for DK both-hemisphere × 5 metrics) are correct.
- [ ] **Spatial null actually engaged** — in any
      `03_transcriptomics/<contrast>/<method>/metadata.json`, `null_method` is
      `vasa` (**not** `random`). If the surface assets were missing the run would
      have hard-failed with `MsnpipSurfaceNullError`, so a finished run already
      guarantees this — but verify.
- [ ] **Report** — `05_report/Report.pdf` opens and contains the violin, the
      surface map (lateral/medial/dorsal), the demographic scatter, and the engine
      PLS/enrichment plots.
- [ ] **Provenance** — `manifest.json` records the engine commit, seed, and resolved
      config; `resolved_config.yaml` reflects what you intended.

---

## 3. Publication run

Once the dev run is clean, scale permutations and gene sets:

```bash
msnpip full \
  --input /path/to/freesurfer_subjects/ \
  --demographics /path/to/demographics.csv \
  --output runs/pub_001/ \
  --group-col group --case PATIENT --control CONTROL \
  --predictors age sex tiv site \
  --exclude-covariate site \
  --correlate-with age --corr-scope global \
  --atlas dk --hemisphere left --regions cort \
  --method pls --ncomp 1 --n-perm 10000 \
  --enrichment ensemble gsea \
  --geneset lake pooled GO_Biological_Process_2025 KEGG_2021_Human DisGeNET \
  --seed 1234 -v
```

Reproducibility: the same `--seed` gives byte-identical msnpip-side tables. Pin the
exact command (or a `--config run.yaml`) in your methods. With a YAML config, only
the CLI flags you also pass override it — everything else comes from the file.

---

## 4. Common failures and fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| `IDMatchError` / very low `n_merged` | IDs differ between FreeSurfer dirs and CSV | Make them identical (whitespace only is stripped); check leading zeros |
| `SchemaError: ... non-numeric ... feature column(s)` | locale/decimal parse (commas) | pass `--sep`/`--decimal`, or fix the CSV |
| `MsnpipSurfaceNullError` | neuromaps surface assets missing | `python -c "import neuromaps; neuromaps.datasets.fetch_fsaverage()"`, or use the Docker image (assets baked in) |
| `ConfigurationError: Unknown atlas` | atlas not in the engine | `msnpip list-atlases` for valid ids |
| Small-group warning | <10 subjects per arm | accept with caution; prefer `--contrast-stat cohen_d` |

---

## 5. Resume / partial runs

```bash
msnpip full ... --stop-stage MSN                 # build strength maps only
msnpip full ... --start-stage TRANSCRIPTOMICS    # reuse persisted earlier stages
msnpip from-strength --output runs/dev_001/ \
  --case PATIENT --control CONTROL --predictors age sex tiv   # re-run stats→report
```
