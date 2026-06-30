# Tutorial: a first run on synthetic data

This walks through a complete `msnpip` run end to end. It uses the bundled
synthetic-cohort generator so you can run every step from a fresh clone — no real
data required. The MSN + statistics portion runs fully offline; the
transcriptomics portion additionally needs the engine's gene-expression and
surface assets (downloaded on first use).

## 1. Install

```bash
git clone https://github.com/tfraa/msntranscript.git
cd msntranscript
pip install -e ".[dev]" -c constraints.txt   # constraints = the verified version set
```

Check the engine resolved:

```python
import imaging_transcriptomics as imt
assert any(a.id == "dk" for a in imt.list_atlases())
```

## 2. Generate a synthetic cohort

The test fixtures ship a generator that writes a realistic merged table (FTD vs
HC, with age / sex / site / TIV covariates):

```python
from pathlib import Path
from tests.fixtures.synthetic import make_synthetic_cohort

info = make_synthetic_cohort(Path("demo_data"), n_case=12, n_control=12, seed=1)
print("merged table:", info["merged_path"])
```

This creates `demo_data/` with a `merged.csv` you can feed straight to the CLI.

## 3. Run the offline part (MSN → contrast)

Stop before the transcriptomics engine to get the network and group-contrast
outputs without needing any downloaded assets — this completes in seconds:

```bash
msnpip full \
  --dataframe demo_data/merged.csv \
  --output demo_out \
  --group-col group --case FTD --control HC \
  --predictors age sex tiv \
  --stop-stage CONTRAST
```

You now have, under `demo_out/`:

- `merged_dataset.csv`, `strength_maps.csv`, `mean_msn_per_group.csv`
- `case_control_difference_maps.csv` and `FTD_vs_HC_region_stats.csv`

See [outputs.md](outputs.md) for what each column means.

## 4. Full run with transcriptomics

To run PLS + enrichment and build the report, drop `--stop-stage` and request the
engine. The first run fetches the AHBA expression data and fsaverage surfaces:

```bash
msnpip full \
  --dataframe demo_data/merged.csv \
  --output demo_out \
  --group-col group --case FTD --control HC \
  --predictors age sex tiv \
  --method pls --ncomp 1 \
  --enrichment ensemble --geneset KEGG_2021_H \
  --n-perm 1000 --seed 1234
```

This adds `FTD_vs_HC_pls.csv`, `FTD_vs_HC_enrichment.csv`, the `plots/` figures,
and `report.pdf`. (The `vasa` surface spin null is required by default; see
[statistics.md](statistics.md) and [engine_contract.md](engine_contract.md) if
the surface assets are unavailable in your environment.)

> Use a larger `--n-perm` (the default is 10,000) for a real analysis; 1,000 here
> just keeps the demo quick.

## 5. Inspect the results

- Open `demo_out/report.pdf` — the cover, Contents page, dataset, MSN, node
  strength, and the per-contrast sections.
- Browse `demo_out/plots/` for the individual figures.
- Load any CSV in pandas; column meanings are in [outputs.md](outputs.md).

## Where to go next

- [running_on_real_data.md](running_on_real_data.md) — point the same commands at
  a real FreeSurfer cohort.
- [statistics.md](statistics.md) — the methods behind the numbers.
- [adding_an_atlas.md](adding_an_atlas.md) — extend beyond DK.
