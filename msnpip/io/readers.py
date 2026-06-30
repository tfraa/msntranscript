"""
Locale-aware table reader and FreeSurfer subject loader.
Phase 1, Tasks T1.1–T1.2.
"""

from __future__ import annotations

import csv
import io
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from msnpip.errors import AmbiguousFormatError, MsnpipIOError

logger = logging.getLogger("msnpip.io.readers")

_DEFAULT_METRICS: tuple[str, ...] = ("SurfArea", "GrayVol", "ThickAvg", "MeanCurv", "GausCurv")

# ---------------------------------------------------------------------------
# T1.1 — read_table
# ---------------------------------------------------------------------------


def read_table(
    path: str | Path,
    *,
    sep: str | None = None,
    decimal: str | None = None,
    sheet: str | int | None = 0,
) -> pd.DataFrame:
    """Read a tabular file with locale-aware delimiter and decimal detection.

    Parameters
    ----------
    path
        CSV, TSV, TXT, XLSX, or XLS file.
    sep
        Field separator.  If ``None``, auto-detected via ``csv.Sniffer``.
        TSV files always use ``'\\t'``.
    decimal
        Decimal character.  If ``None``, auto-detected: the character that
        maximises the number of numeric columns is chosen.
    sheet
        Sheet name/index for Excel files.

    Raises
    ------
    AmbiguousFormatError
        If the delimiter cannot be determined automatically.
    MsnpipIOError
        If the file cannot be read or parsed.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".xlsx", ".xls"):
        return _read_excel(path, sheet=sheet)

    raw = _read_raw(path)

    if sep is None:
        sep = "\t" if suffix == ".tsv" else _sniff_sep(raw, path)

    if decimal is None:
        decimal = _detect_decimal(raw, sep)

    try:
        df = pd.read_csv(io.StringIO(raw), sep=sep, decimal=decimal, engine="python")
    except Exception as exc:
        raise MsnpipIOError(f"Failed to parse {path}: {exc}") from exc

    logger.debug("read_table: %s  shape=%s  sep=%r  decimal=%r", path.name, df.shape, sep, decimal)
    return df


def _read_raw(path: Path) -> str:
    """Read file text, trying UTF-8-sig then latin-1 as fallback."""
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise MsnpipIOError(f"Cannot decode {path} with utf-8 or latin-1.")


def _sniff_sep(raw: str, path: Path) -> str:
    sample = raw[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error as exc:
        raise AmbiguousFormatError(
            f"Cannot detect delimiter in '{path.name}'. "
            "Pass sep=',' / sep=';' / sep='\\t' explicitly."
        ) from exc


def _detect_decimal(raw: str, sep: str) -> str:
    """Return ',' if comma-decimal produces more numeric columns, else '.'."""
    if sep == ",":
        return "."  # comma cannot be both separator and decimal

    counts: dict[str, int] = {}
    for dec in (".", ","):
        try:
            df = pd.read_csv(io.StringIO(raw), sep=sep, decimal=dec, engine="python", nrows=100)
            counts[dec] = sum(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)
        except Exception:
            counts[dec] = 0

    return "," if counts.get(",", 0) > counts.get(".", 0) else "."


def _read_excel(path: Path, sheet: str | int | None) -> pd.DataFrame:
    try:
        # sheet_name=None would return a dict of every sheet; default to the first
        # so this always yields a single DataFrame.
        return pd.read_excel(path, sheet_name=0 if sheet is None else sheet, engine="openpyxl")
    except Exception as exc:
        raise MsnpipIOError(f"Failed to read Excel file '{path.name}': {exc}") from exc


# ---------------------------------------------------------------------------
# T1.2 — read_freesurfer_subjects
# ---------------------------------------------------------------------------


def read_freesurfer_subjects(
    root: str | Path,
    expected_regions: Sequence[str] | None = None,
    metrics: tuple[str, ...] = _DEFAULT_METRICS,
) -> pd.DataFrame:
    """Read ``{lh,rh}.aparc.stats`` files for all subjects under *root*.

    Directory layout::

        root/<subject_id>/stats/lh.aparc.stats
        root/<subject_id>/stats/rh.aparc.stats

    Parameters
    ----------
    root
        FreeSurfer subjects directory (parent of per-subject dirs).
    expected_regions
        Cortical region names expected to appear in each file (aparc label
        names without hemisphere prefix).  Regions absent from a file are
        filled with NaN.  If ``None``, whatever the file contains is used.
    metrics
        Subset of aparc.stats columns to extract (default: the 5 MSN
        features).

    Returns
    -------
    pd.DataFrame
        Wide format: one row per subject, columns ``subject_id`` followed by
        ``{hemi}_{region}_{metric}`` for every (hemisphere, region, metric)
        combination.  Missing values are NaN.  Per-subject issues are
        recorded in ``df.attrs["issues"]``.
    """
    root = Path(root)
    issues: list[dict] = []
    rows: list[dict] = []

    subject_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not subject_dirs:
        raise MsnpipIOError(f"No subject directories found under '{root}'.")

    for subj_dir in subject_dirs:
        subj_id = subj_dir.name
        row: dict = {"subject_id": subj_id}
        subj_issues: list[str] = []

        for hemi in ("lh", "rh"):
            # Stats files are usually in <subj>/stats/, but FreeSurfer exports
            # sometimes place them directly in <subj>/ — accept both (issue 3).
            stats_path = subj_dir / "stats" / f"{hemi}.aparc.stats"
            if not stats_path.exists():
                alt = subj_dir / f"{hemi}.aparc.stats"
                if alt.exists():
                    stats_path = alt
            if not stats_path.exists():
                msg = f"{subj_id}: missing {hemi}.aparc.stats"
                subj_issues.append(msg)
                logger.warning(msg)
                # Fill all expected with NaN
                for region in expected_regions or []:
                    for metric in metrics:
                        row[f"{hemi}_{region}_{metric}"] = np.nan
                continue

            try:
                region_data = _parse_aparc_stats(stats_path.read_text(encoding="utf-8"), metrics)
            except Exception as exc:
                msg = f"{subj_id}: error parsing {hemi}.aparc.stats — {exc}"
                subj_issues.append(msg)
                logger.warning(msg)
                region_data = {}

            # Determine the region name set to use
            regions_to_use = (
                list(expected_regions) if expected_regions is not None else list(region_data.keys())
            )

            for region in regions_to_use:
                for metric in metrics:
                    key = f"{hemi}_{region}_{metric}"
                    if region in region_data and metric in region_data[region]:
                        row[key] = region_data[region][metric]
                    else:
                        row[key] = np.nan
                        if region not in region_data:
                            warn = f"{subj_id}/{hemi}: region '{region}' missing"
                            if warn not in subj_issues:
                                subj_issues.append(warn)

        rows.append(row)
        if subj_issues:
            issues.append({"subject_id": subj_id, "issues": subj_issues})

    if not rows:
        raise MsnpipIOError(f"No data could be read from '{root}'.")

    df = pd.DataFrame(rows)
    df.attrs["issues"] = issues
    logger.info(
        "read_freesurfer_subjects: %d subjects, %d columns, %d with issues",
        len(df),
        len(df.columns),
        len(issues),
    )
    return df


# ---------------------------------------------------------------------------
# Per-feature tables (one file per metric × hemisphere, all subjects inside)
# ---------------------------------------------------------------------------

# Map the metric token found in a column suffix → canonical MSN metric name.
_FEATURE_METRIC_ALIASES: dict[str, str] = {
    "surfarea": "SurfArea",
    "area": "SurfArea",
    "volume": "GrayVol",
    "grayvol": "GrayVol",
    "vol": "GrayVol",
    "thickness": "ThickAvg",
    "thickavg": "ThickAvg",
    "thick": "ThickAvg",
    "meancurv": "MeanCurv",
    "gauscurv": "GausCurv",
    "gausscurv": "GausCurv",
}
_DEMO_TABLE_COLS = ("group", "diagnosis", "etiv", "tiv", "icv", "brainsegvolnotvent", "age", "sex")
_TABULAR_SUFFIXES = (".csv", ".tsv", ".txt", ".xlsx", ".xls")


def _parse_feature_column(col: str) -> str | None:
    """Map a region-feature column to canonical ``{hemi}_{region}_{Metric}``.

    Robust to metric-token spelling (``gauscurv``/``gausscurv`` → ``GausCurv``)
    by reading the canonical metric from the column's own suffix, not the file
    name.  Returns ``None`` for non-feature columns.
    """
    parts = col.strip().split("_")
    if len(parts) < 3 or parts[0] not in ("lh", "rh"):
        return None
    metric = _FEATURE_METRIC_ALIASES.get(parts[-1].lower())
    if metric is None:
        return None
    region = "_".join(parts[1:-1])
    return f"{parts[0]}_{region}_{metric}"


def _gather_feature_files(source) -> list[Path]:
    if isinstance(source, (str, Path)):
        root = Path(source)
        files = sorted(p for p in root.glob("aparc_*") if p.suffix.lower() in _TABULAR_SUFFIXES)
        if not files:
            files = sorted(p for p in root.iterdir() if p.suffix.lower() in _TABULAR_SUFFIXES)
    else:
        files = [Path(p) for p in source]
    return _dedupe_by_stem(files)


def _dedupe_by_stem(files: list[Path]) -> list[Path]:
    """Keep one file per stem when the same table ships in several formats
    (e.g. ``aparc_thickness_lh.tsv`` and ``.xlsx``) — prefer text formats."""
    pref = {".csv": 0, ".tsv": 1, ".txt": 2, ".xlsx": 3, ".xls": 4}
    best: dict[str, Path] = {}
    for f in files:
        key = f.stem.lower()
        if key not in best or pref.get(f.suffix.lower(), 9) < pref.get(best[key].suffix.lower(), 9):
            best[key] = f
    return [best[k] for k in sorted(best)]


def read_feature_tables(
    source,
    *,
    sep: str | None = None,
    decimal: str | None = None,
) -> pd.DataFrame:
    """Read per-feature morphometric tables and merge into the canonical matrix.

    Each input file holds one metric × hemisphere for all subjects: the first
    column is the subject ID and the region columns are
    ``{hemi}_{region}_{metric_token}`` (e.g. ``lh_bankssts_gauscurv``).  Files
    are merged on subject ID into the wide ``{hemi}_{region}_{Metric}`` matrix.
    Embedded demographic columns (``Group``, ``Diagnosis``, ``eTIV``→``tiv`` …)
    are carried through once if present.

    Parameters
    ----------
    source
        A directory (globs ``aparc_*`` tabular files, else any tabular file) or
        an explicit list of file paths.
    sep, decimal
        Forwarded to :func:`read_table` (auto-detected when ``None``).

    Returns
    -------
    pd.DataFrame
        ``subject_id`` + canonical feature columns + any demographic columns.
    """
    paths = _gather_feature_files(source)
    if not paths:
        raise MsnpipIOError(f"No tabular feature files found under '{source}'.")

    feature_frames: list[pd.DataFrame] = []
    demo_frame: pd.DataFrame | None = None

    for path in paths:
        df = read_table(path, sep=sep, decimal=decimal)
        if df.shape[1] < 2:
            continue
        df = df.rename(columns={df.columns[0]: "subject_id"})
        df["subject_id"] = df["subject_id"].astype(str).str.strip()

        rename = {c: _parse_feature_column(c) for c in df.columns}
        rename = {c: new for c, new in rename.items() if new is not None}
        if not rename:
            continue

        sub = df[["subject_id"]].copy()
        for old, new in rename.items():
            sub[new] = df[old]
        feature_frames.append(sub)

        if demo_frame is None:
            demo_cols = [c for c in df.columns if c.lower() in _DEMO_TABLE_COLS]
            if demo_cols:
                demo = df[["subject_id", *demo_cols]].copy()
                demo = demo.rename(columns={c: "tiv" for c in demo.columns if c.lower() == "etiv"})
                demo_frame = demo

    if not feature_frames:
        raise MsnpipIOError(
            f"No recognizable '{{hemi}}_{{region}}_{{metric}}' feature columns in {len(paths)} "
            f"file(s) under '{source}'."
        )

    merged = feature_frames[0]
    for frame in feature_frames[1:]:
        merged = merged.merge(frame, on="subject_id", how="outer")
    if demo_frame is not None:
        merged = merged.merge(demo_frame, on="subject_id", how="left")

    logger.info(
        "read_feature_tables: %d file(s) → %d subjects, %d feature columns",
        len(paths),
        len(merged),
        sum(c not in ("subject_id",) for c in merged.columns),
    )
    return merged


def detect_input_kind(path: str | Path) -> str:
    """Classify a FreeSurfer input directory as ``'subjects'`` or ``'feature_tables'``.

    ``'subjects'`` → per-subject ``aparc.stats`` (in ``<subj>/stats/`` or
    ``<subj>/``).  ``'feature_tables'`` → per-metric tables with all subjects.
    """
    root = Path(path)
    if root.is_dir():
        for sub in root.iterdir():
            if sub.is_dir() and any(
                (sub / "stats" / f"{h}.aparc.stats").exists() or (sub / f"{h}.aparc.stats").exists()
                for h in ("lh", "rh")
            ):
                return "subjects"
        for p in root.glob("aparc_*"):
            if p.suffix.lower() in _TABULAR_SUFFIXES:
                return "feature_tables"
        if any(p.suffix.lower() in _TABULAR_SUFFIXES for p in root.iterdir()):
            return "feature_tables"
    return "subjects"


def _parse_aparc_stats(text: str, metrics: tuple[str, ...]) -> dict[str, dict[str, float]]:
    """Parse the text of an aparc.stats file.

    Returns
    -------
    dict
        ``{region_name: {metric_name: float_value}}``
    """
    col_headers: list[str] | None = None
    data: dict[str, dict[str, float]] = {}

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("# ColHeaders"):
            # "# ColHeaders  StructName NumVert SurfArea ..."
            col_headers = line.split()[2:]
            continue

        if line.startswith("#"):
            continue

        if col_headers is None:
            continue

        parts = line.split()
        if len(parts) < len(col_headers):
            continue

        region = parts[0]
        row: dict[str, float] = {}
        for metric in metrics:
            try:
                idx = col_headers.index(metric)
                row[metric] = float(parts[idx])
            except (ValueError, IndexError):
                row[metric] = np.nan
        data[region] = row

    return data
