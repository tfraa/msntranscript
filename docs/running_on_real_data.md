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
  --hemisphere left \
  --method pls --ncomp 1 --n-perm 1000 \
  --enrichment ensemble \
  --geneset GO_Biological_Process_2025 \
  --seed 1234 -v
```

Merged-CSV equivalent: replace `--input/--demographics` with `--dataframe merged.csv`.
European locale CSV? add `--sep ';' --decimal ','` (auto-detection usually handles it).

The atlas is fixed to DK and the regions to cortex, so there are no `--atlas` or
`--regions` flags.

---

## 2. Results-sanity checklist

After the dev run, check these — most data problems show up here:

- [ ] **Merge / match rate** — the console log reports how many subjects merged.
      A low count means IDs didn't match (see §4).
- [ ] **Dropped subjects** — the MSN stage logs every subject it drops for missing
      features. msnpip never imputes, so this list should be only the subjects you
      already know are incomplete.
- [ ] **Group sizes** — run log: no "Small group(s)" warning, or you accept it.
      With n<10 per arm, prefer `--contrast-stat cohen_d`.
- [ ] **Schema** — the log reports the detected `id_col`, `group_col` and covariates,
      and the feature-column count (≈340 for DK both-hemisphere × 5 metrics). Force a
      role with `--id-col` / `--group-col` if detection picked the wrong column.
- [ ] **Spatial null actually engaged** — every curated `*_pls.csv`, `*_corr.csv` and
      `*_enrichment.csv` carries a `null_method` column with the null that was
      *actually resolved*, and the report cover states it. It must read `vasa`, not
      `random`. The default is fallback-with-warning, so a finished run does **not**
      by itself guarantee a real spin — check the column.
- [ ] **Report** — `report.pdf` opens and contains the violins, the surface maps
      (2×2 lateral/medial per hemisphere), the demographic scatter, and the
      enrichment plots.
- [ ] **Contrast map** — `case_control_difference_maps.csv` and
      `<tag>_region_stats.csv` have one row per region with finite `beta`/`t`. A whole
      column of NaN `t` means a rank-deficient design (too many covariates); the run
      warns about this up front.

Outputs are a flat, curated set — see [outputs.md](outputs.md) for the
column-by-column reference. There is no numbered stage tree and no `manifest.json`.

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
  --correlate-with age --corr-scope global \
  --hemisphere left \
  --msn-similarity distance \
  --method pls --ncomp 2 --n-perm 10000 \
  --enrichment ensemble gsea ora \
  --geneset lake pooled GO_Biological_Process_2025 KEGG_2021_H DisGeNET \
  --null-method vasa \
  --seed 1234 -v
```

Useful flags for the re-analysis:

- `--msn-similarity {distance,correlation}` — edge definition; `correlation` = the canonical
  Seidlitz/Morgan MS (allows negative edges).
- `--ncomp 2` — retain PLS1 **and** PLS2 (each gets its own spin-tested component p and gene tables).
- `--enrichment ensemble gsea ora` — `ensemble` (GCEA) is the primary spin-null test, `gsea` a
  spin-null cross-check, `ora` the template over-representation test (candidate mechanisms only).
  Note this flag *replaces* the default set rather than adding to it.
- `--pool-cases` — with several `--contrast X 0` flags, also runs a supplementary pooled
  `{X…}_vs_0` contrast alongside the per-group ones (which stay primary).
- `--geneset-min-size` / `--geneset-max-size` — category-size window for the spin-null
  backends. Pre-specify it; ORA is deliberately left unfiltered.
- To drop a covariate, run again without it in `--predictors`; there is no exclude flag.
- Node strength is fixed to the **mean** of a region's edges (Morgan/Seidlitz).

Reproducibility: the same `--seed` gives byte-identical msnpip-side tables. Pin the
exact command (or a `--config run.yaml`) in your methods. With a YAML config, only
the CLI flags you also pass override it — everything else comes from the file.

---

## 4. Common failures and fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| `IDMatchError` / very low merged count | IDs differ between FreeSurfer dirs and CSV | Make them identical (whitespace only is stripped); check leading zeros |
| `SchemaError: ... non-numeric ... feature column(s)` | locale/decimal parse (commas) | pass `--sep`/`--decimal`, or fix the CSV |
| `null_method` reads `random` | neuromaps surface assets missing, so the spin fell back | `python -c "import neuromaps; neuromaps.datasets.fetch_fsaverage()"`, or use the Docker image (assets baked in) |
| `MsnpipSurfaceNullError` | as above, with fallback disabled | same fix; this is the hard-fail path |
| `ConfigurationError: Unknown atlas` | engine atlas table unavailable | `msnpip list-atlases` for valid ids |
| Small-group warning | <10 subjects per arm | accept with caution; prefer `--contrast-stat cohen_d` |
| Rank-deficient design warning, NaN `t` | more covariate terms than the sample supports | drop covariates from `--predictors` |

---

## 5. Resume / partial runs

```bash
msnpip full ... --stop-stage MSN                 # build strength maps only
msnpip full ... --start-stage TRANSCRIPTOMICS    # reuse persisted earlier stages
msnpip from-strength --output runs/dev_001/ \
  --case PATIENT --control CONTROL --predictors age sex tiv   # re-run stats→report
```
