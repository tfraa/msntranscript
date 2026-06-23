"""
ColumnSchema dataclass, detect_schema, validate_schema.
Phase 1, Tasks T1.3–T1.4.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from msnpip.errors import SchemaError

logger = logging.getLogger("msnpip.io.schema")

# ---------------------------------------------------------------------------
# Column role aliases (case-insensitive substring match)
# ---------------------------------------------------------------------------

_ID_KEYWORDS = ("subject_id", "participant_id", "patient_id", "subjectid",
                 "participantid", "patientid", "sub_id", "id")
_GROUP_KEYWORDS = ("group", "grp", "diagnosis", "dx", "label", "class", "condition")
_AGE_KEYWORDS = ("age",)
_SEX_KEYWORDS = ("sex", "gender")
_TIV_KEYWORDS = ("tiv", "icv", "intracranial")
_SITE_KEYWORDS = ("site", "scanner", "centre", "center", "acquisition")


@dataclass
class ColumnSchema:
    """Describes the role of each column in the merged input DataFrame.

    All column names reference actual columns in the DataFrame.
    ``None`` means the role was not detected.
    """

    id_col: str | None
    group_col: str | None
    age_col: str | None
    sex_col: str | None
    tiv_col: str | None
    site_cols: list[str] = field(default_factory=list)
    feature_cols: list[str] = field(default_factory=list)
    other_cols: list[str] = field(default_factory=list)

    @property
    def demographic_cols(self) -> list[str]:
        """All non-feature, non-id, non-group columns (age, sex, tiv, site, …)."""
        cols = []
        for c in (self.age_col, self.sex_col, self.tiv_col):
            if c is not None:
                cols.append(c)
        cols.extend(self.site_cols)
        return cols


# ---------------------------------------------------------------------------
# T1.3 — detect_schema
# ---------------------------------------------------------------------------

def detect_schema(
    df: pd.DataFrame,
    expected_regions: list[str] | None = None,
    expected_metrics: tuple[str, ...] = ("SurfArea", "GrayVol", "ThickAvg", "MeanCurv", "GausCurv"),
) -> ColumnSchema:
    """Infer the role of each column by keyword matching.

    Parameters
    ----------
    df
        The merged input DataFrame.
    expected_regions
        List of atlas region names (aparc labels).  When provided, feature
        columns are identified by the pattern ``{hemi}_{region}_{metric}``.
        If ``None``, any numeric column not matched to a known role is
        treated as a feature column.
    expected_metrics
        Metric names used to build the feature column pattern.

    Returns
    -------
    ColumnSchema
        Detected roles.  Logs a warning for each role that could not be
        detected.
    """
    cols = list(df.columns)
    assigned: set[str] = set()

    def _find(keywords: tuple[str, ...], label: str, required: bool = False) -> str | None:
        candidates = [
            c for c in cols
            if c not in assigned and any(kw in c.lower() for kw in keywords)
        ]
        if not candidates:
            if required:
                logger.warning("Could not detect '%s' column in %s", label, cols)
            return None
        if len(candidates) > 1:
            logger.warning(
                "Multiple candidates for '%s': %s — using '%s'",
                label, candidates, candidates[0],
            )
        assigned.add(candidates[0])
        return candidates[0]

    id_col = _find(_ID_KEYWORDS, "id")
    if id_col:
        assigned.add(id_col)

    group_col = _find(_GROUP_KEYWORDS, "group")
    age_col = _find(_AGE_KEYWORDS, "age")
    sex_col = _find(_SEX_KEYWORDS, "sex")
    tiv_col = _find(_TIV_KEYWORDS, "tiv")

    # Site columns: may be multiple (multiple scanners one-hot encoded)
    site_cols = [
        c for c in cols
        if c not in assigned and any(kw in c.lower() for kw in _SITE_KEYWORDS)
    ]
    assigned.update(site_cols)

    # Feature columns
    if expected_regions is not None:
        feature_set = {
            f"{hemi}_{region}_{metric}"
            for hemi in ("lh", "rh")
            for region in expected_regions
            for metric in expected_metrics
        }
        feature_cols = [c for c in cols if c in feature_set and c not in assigned]
    else:
        # Fall back: any remaining numeric column is a feature
        feature_cols = [
            c for c in cols
            if c not in assigned and pd.api.types.is_numeric_dtype(df[c])
        ]
    assigned.update(feature_cols)

    other_cols = [c for c in cols if c not in assigned]

    schema = ColumnSchema(
        id_col=id_col,
        group_col=group_col,
        age_col=age_col,
        sex_col=sex_col,
        tiv_col=tiv_col,
        site_cols=site_cols,
        feature_cols=feature_cols,
        other_cols=other_cols,
    )

    logger.debug(
        "detect_schema: id=%s group=%s age=%s sex=%s tiv=%s sites=%s features=%d",
        id_col, group_col, age_col, sex_col, tiv_col, site_cols, len(feature_cols),
    )
    return schema


# ---------------------------------------------------------------------------
# T1.4 — validate_schema
# ---------------------------------------------------------------------------

def validate_schema(
    df: pd.DataFrame,
    schema: ColumnSchema,
    *,
    predictor_cols: tuple[str, ...] = (),
    correlation_cols: tuple[str, ...] = (),
) -> None:
    """Hard-gate validation: raises SchemaError on any violation.

    Checks
    ------
    - ``id_col`` exists and contains unique values (no duplicates).
    - All ``predictor_cols`` exist in the DataFrame.
    - All ``correlation_cols`` exist **and** are numeric.
    - ``feature_cols`` are numeric dtype, not object (guards locale/encoding bugs).
    - At least one feature column is present.

    Parameters
    ----------
    df
        The DataFrame to validate.
    schema
        Detected column roles.
    predictor_cols
        Column names requested as GLM predictors (from ``GLMConfig.predictors``).
    correlation_cols
        Column names requested for demographic correlation.

    Raises
    ------
    SchemaError
        On the first violation found (does not accumulate).
    """
    errors: list[str] = []

    # ID column present and unique
    if schema.id_col is None:
        errors.append("No ID column detected. Add a column named 'subject_id' or 'participant_id'.")
    else:
        if df[schema.id_col].duplicated().any():
            dupes = df[schema.id_col][df[schema.id_col].duplicated()].tolist()
            errors.append(f"Duplicate subject IDs in column '{schema.id_col}': {dupes[:5]}")

    # Feature columns exist and are numeric
    if not schema.feature_cols:
        errors.append("No feature columns detected. Ensure morphometric data is present.")
    else:
        object_feats = [
            c for c in schema.feature_cols
            if df[c].dtype == object
        ]
        if object_feats:
            errors.append(
                f"{len(object_feats)} feature column(s) have object dtype (likely locale issue "
                f"or mixed text/number). First offenders: {object_feats[:5]}. "
                "Check decimal separator (use --decimal=',' for European locale)."
            )

    # Predictor columns exist
    for col in predictor_cols:
        if col not in df.columns:
            errors.append(f"Predictor column '{col}' not found. Available: {list(df.columns)}")

    # Correlation columns exist and are numeric
    for col in correlation_cols:
        if col not in df.columns:
            errors.append(
                f"Correlation variable '{col}' not found. Available: {list(df.columns)}"
            )
        elif not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(
                f"Correlation variable '{col}' is not numeric (dtype={df[col].dtype}). "
                "Demographic correlation requires numeric variables."
            )

    if errors:
        raise SchemaError(
            f"Input validation failed ({len(errors)} error(s)):\n"
            + "\n".join(f"  • {e}" for e in errors)
        )

    logger.info(
        "validate_schema: OK — %d subjects, %d feature cols",
        len(df),
        len(schema.feature_cols),
    )
