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
        return pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
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
            stats_path = subj_dir / "stats" / f"{hemi}.aparc.stats"
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
