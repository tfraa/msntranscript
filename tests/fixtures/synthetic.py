"""
Synthetic cohort fixture for msnpip v2 tests.

``make_synthetic_cohort`` writes a minimal but realistic dataset under a
temporary directory and returns a dict describing what was created.  Tests
toggle locale_quirks and id_quirks to drive the robustness paths in
io/readers.py and io/matching.py.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

# DK atlas: 34 cortical regions per hemisphere (FreeSurfer aparc label names)
DK_REGIONS: list[str] = [
    "bankssts",
    "caudalanteriorcingulate",
    "caudalmiddlefrontal",
    "cuneus",
    "entorhinal",
    "fusiform",
    "inferiorparietal",
    "inferiortemporal",
    "isthmuscingulate",
    "lateraloccipital",
    "lateralorbitofrontal",
    "lingual",
    "medialorbitofrontal",
    "middletemporal",
    "parahippocampal",
    "paracentral",
    "parsopercularis",
    "parsorbitalis",
    "parstriangularis",
    "pericalcarine",
    "postcentral",
    "posteriorcingulate",
    "precentral",
    "precuneus",
    "rostralanteriorcingulate",
    "rostralmiddlefrontal",
    "superiorfrontal",
    "superiorparietal",
    "superiortemporal",
    "supramarginal",
    "frontalpole",
    "temporalpole",
    "transversetemporal",
    "insula",
]
assert len(DK_REGIONS) == 34, "DK atlas must have exactly 34 cortical regions per hemisphere"

# MSN feature columns (must match MSNConfig defaults)
MSN_METRICS: list[str] = ["SurfArea", "GrayVol", "ThickAvg", "MeanCurv", "GausCurv"]

_APARC_HEADER = textwrap.dedent(
    """\
    # Table of FreeSurfer cortical parcellation anatomical statistics
    # ColHeaders  StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd
    """
)


def _aparc_stats_content(rng: np.random.Generator, regions: list[str]) -> str:
    """Return the text content of a synthetic {lh,rh}.aparc.stats file."""
    lines: list[str] = [_APARC_HEADER]
    for region in regions:
        numvert = int(rng.integers(500, 5000))
        surfarea = int(rng.integers(300, 3000))
        grayvol = int(rng.integers(1000, 8000))
        thickavg = round(float(rng.uniform(1.5, 4.0)), 3)
        thickstd = round(float(rng.uniform(0.1, 0.6)), 3)
        meancurv = round(float(rng.uniform(0.05, 0.25)), 3)
        gauscurv = round(float(rng.uniform(0.005, 0.03)), 3)
        foldind = int(rng.integers(5, 50))
        curvind = round(float(rng.uniform(0.5, 3.0)), 1)
        lines.append(
            f"{region} {numvert} {surfarea} {grayvol} "
            f"{thickavg} {thickstd} {meancurv} {gauscurv} "
            f"{foldind} {curvind}"
        )
    return "\n".join(lines) + "\n"


def make_synthetic_cohort(
    root: Path,
    *,
    n_case: int = 12,
    n_control: int = 12,
    case_label: str = "FTD",
    control_label: str = "HC",
    seed: int = 42,
    locale_quirks: bool = False,
    id_quirks: bool = False,
    n_sites: int = 2,
) -> dict:
    """
    Create a minimal synthetic cohort under *root* and return a descriptor dict.

    Directory layout
    ----------------
    root/
      freesurfer/
        <subj_id>/stats/{lh,rh}.aparc.stats    (FreeSurfer input mode)
      demographics.csv                          (paired with freesurfer/)
      merged.csv                                (single-file dataframe mode)

    Parameters
    ----------
    root
        Directory that will be created if absent.
    n_case / n_control
        Group sizes.
    case_label / control_label
        String values written into the ``group`` column.
    seed
        RNG seed — same seed always produces the same files.
    locale_quirks
        If True, CSVs use semicolon separators and comma decimal points,
        exercising ``read_table`` locale detection.
    id_quirks
        If True, the CSV IDs diverge from the on-disk directory names in ways
        that will **not** be resolved automatically (IDs are matched exactly
        after whitespace stripping).  Subject 0 loses its leading zeros
        (``sub-001`` → ``sub-1``) and subject 1 has a trailing space
        (``sub-002 ``).  Use this fixture to test that ``merge_features_demographics``
        raises ``IDMatchError`` when IDs are genuinely inconsistent.
    n_sites
        Number of scanner sites to assign round-robin (``site`` covariate).

    Returns
    -------
    dict
        Keys:
        ``fs_dir``             Path  — root of the FreeSurfer directory tree
        ``demographics_path``  Path  — demographics CSV
        ``merged_path``        Path  — wide-format single-file CSV
        ``subject_ids``        list[str]  — IDs as written in the CSVs
        ``raw_subject_ids``    list[str]  — canonical on-disk IDs
        ``n_subjects``         int
        ``n_case``             int
        ``n_control``          int
        ``case_label``         str
        ``control_label``      str
        ``sep``                str  — CSV field separator used
        ``decimal``            str  — decimal character used
    """
    rng = np.random.default_rng(seed)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    n_total = n_case + n_control
    fs_dir = root / "freesurfer"

    # Canonical on-disk subject IDs
    raw_ids = [f"sub-{i + 1:03d}" for i in range(n_total)]

    # CSV-side IDs (may diverge when id_quirks=True)
    if id_quirks:
        csv_ids = list(raw_ids)
        csv_ids[0] = csv_ids[0].replace("001", "1")  # drop leading zeros
        csv_ids[1] = csv_ids[1] + " "  # trailing space
    else:
        csv_ids = list(raw_ids)

    groups = [case_label] * n_case + [control_label] * n_control
    ages = rng.integers(50, 80, size=n_total).tolist()
    sexes = rng.choice(["M", "F"], size=n_total).tolist()
    tivs = rng.normal(1500.0, 150.0, size=n_total).tolist()
    site_labels = [f"site{(i % n_sites) + 1}" for i in range(n_total)]

    # Write FreeSurfer stats files (always under canonical raw_ids)
    for raw_id in raw_ids:
        stats_dir = fs_dir / raw_id / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        for hemi in ("lh", "rh"):
            content = _aparc_stats_content(rng, DK_REGIONS)
            (stats_dir / f"{hemi}.aparc.stats").write_text(content, encoding="utf-8")

    # Separator and decimal character
    sep = ";" if locale_quirks else ","
    decimal = "," if locale_quirks else "."

    # --- Demographics CSV (paired with FreeSurfer mode) ---
    dem_df = pd.DataFrame(
        {
            "subject_id": csv_ids,
            "group": groups,
            "age": [round(float(a), 1) for a in ages],
            "sex": sexes,
            "tiv": [round(float(t), 2) for t in tivs],
            "site": site_labels,
        }
    )
    dem_path = root / ("demographics_locale.csv" if locale_quirks else "demographics.csv")
    dem_df.to_csv(dem_path, index=False, sep=sep, decimal=decimal)

    # --- Wide merged CSV (single-file dataframe mode) ---
    rows: list[dict] = []
    for csv_id, group, age, sex, tiv, site in zip(csv_ids, groups, ages, sexes, tivs, site_labels):
        row: dict = {
            "subject_id": csv_id,
            "group": group,
            "age": round(float(age), 1),
            "sex": sex,
            "tiv": round(float(tiv), 2),
            "site": site,
        }
        for hemi in ("lh", "rh"):
            for region in DK_REGIONS:
                for metric in MSN_METRICS:
                    if metric == "ThickAvg":
                        val: float | int = round(float(rng.uniform(1.5, 4.0)), 3)
                    elif metric == "MeanCurv":
                        val = round(float(rng.uniform(0.05, 0.25)), 3)
                    elif metric == "GausCurv":
                        val = round(float(rng.uniform(0.005, 0.03)), 3)
                    elif metric == "GrayVol":
                        val = int(rng.integers(1000, 8000))
                    else:  # SurfArea
                        val = int(rng.integers(300, 3000))
                    row[f"{hemi}_{region}_{metric}"] = val
        rows.append(row)

    merged_df = pd.DataFrame(rows)
    merged_path = root / ("merged_locale.csv" if locale_quirks else "merged.csv")
    merged_df.to_csv(merged_path, index=False, sep=sep, decimal=decimal)

    return {
        "fs_dir": fs_dir,
        "demographics_path": dem_path,
        "merged_path": merged_path,
        "subject_ids": csv_ids,
        "raw_subject_ids": raw_ids,
        "n_subjects": n_total,
        "n_case": n_case,
        "n_control": n_control,
        "case_label": case_label,
        "control_label": control_label,
        "sep": sep,
        "decimal": decimal,
    }
