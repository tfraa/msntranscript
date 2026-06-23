"""Unit tests for msnpip.io.schema — T1.3 and T1.4."""

from __future__ import annotations

import pandas as pd
import pytest

from msnpip.errors import SchemaError
from msnpip.io.schema import ColumnSchema, detect_schema, validate_schema
from tests.fixtures.synthetic import DK_REGIONS, MSN_METRICS


def _make_df(n: int = 5) -> pd.DataFrame:
    """Minimal synthetic DataFrame matching the merged.csv structure."""
    import numpy as np

    rng = np.random.default_rng(0)
    data: dict = {
        "subject_id": [f"sub-{i + 1:03d}" for i in range(n)],
        "group": ["FTD"] * (n // 2) + ["HC"] * (n - n // 2),
        "age": rng.integers(50, 80, n).astype(float).tolist(),
        "sex": ["M", "F"] * (n // 2) + (["M"] if n % 2 else []),
        "tiv": rng.normal(1500, 150, n).tolist(),
        "site": [f"site{(i % 2) + 1}" for i in range(n)],
    }
    # Add a small set of feature columns
    for hemi in ("lh", "rh"):
        for region in DK_REGIONS[:2]:
            for metric in MSN_METRICS[:2]:
                data[f"{hemi}_{region}_{metric}"] = rng.uniform(1, 5, n).tolist()
    return pd.DataFrame(data)


class TestDetectSchema:
    def test_detects_standard_roles(self):
        df = _make_df()
        schema = detect_schema(df, expected_regions=DK_REGIONS[:2])
        assert schema.id_col == "subject_id"
        assert schema.group_col == "group"
        assert schema.age_col == "age"
        assert schema.sex_col == "sex"
        assert schema.tiv_col == "tiv"
        assert "site" in schema.site_cols

    def test_detects_feature_cols(self):
        df = _make_df()
        schema = detect_schema(df, expected_regions=DK_REGIONS[:2])
        # 2 hemis × 2 regions × 2 metrics = 8
        assert len(schema.feature_cols) == 8

    def test_no_expected_regions_uses_numeric(self):
        df = _make_df()
        schema = detect_schema(df)
        # Without expected_regions, falls back to all-numeric non-role columns
        assert len(schema.feature_cols) > 0

    def test_demographic_cols_property(self):
        df = _make_df()
        schema = detect_schema(df)
        dem = schema.demographic_cols
        assert "age" in dem
        assert "tiv" in dem


class TestValidateSchema:
    def test_clean_passes(self):
        df = _make_df()
        schema = detect_schema(df, expected_regions=DK_REGIONS[:2])
        validate_schema(df, schema)  # should not raise

    def test_duplicate_id_raises(self):
        df = _make_df()
        df.loc[1, "subject_id"] = df.loc[0, "subject_id"]  # duplicate
        schema = detect_schema(df, expected_regions=DK_REGIONS[:2])
        with pytest.raises(SchemaError, match="Duplicate"):
            validate_schema(df, schema)

    def test_no_feature_cols_raises(self):
        df = pd.DataFrame(
            {
                "subject_id": ["a", "b"],
                "group": ["HC", "FTD"],
            }
        )
        schema = ColumnSchema(
            id_col="subject_id",
            group_col="group",
            age_col=None,
            sex_col=None,
            tiv_col=None,
            site_cols=[],
            feature_cols=[],
            other_cols=[],
        )
        with pytest.raises(SchemaError, match="No feature columns"):
            validate_schema(df, schema)

    def test_object_dtype_feature_raises(self):
        df = _make_df()
        schema = detect_schema(df, expected_regions=DK_REGIONS[:2])
        # Corrupt one feature column to a text dtype (object on pandas 2.x,
        # string dtype on pandas 3.0 — both are non-numeric and must be flagged).
        df[schema.feature_cols[0]] = df[schema.feature_cols[0]].astype(str)
        with pytest.raises(SchemaError, match="non-numeric"):
            validate_schema(df, schema)

    def test_missing_predictor_raises(self):
        df = _make_df()
        schema = detect_schema(df, expected_regions=DK_REGIONS[:2])
        with pytest.raises(SchemaError, match="Predictor column"):
            validate_schema(df, schema, predictor_cols=("nonexistent_col",))

    def test_non_numeric_correlation_var_raises(self):
        df = _make_df()
        schema = detect_schema(df, expected_regions=DK_REGIONS[:2])
        # sex is categorical, not numeric
        with pytest.raises(SchemaError, match="not numeric"):
            validate_schema(df, schema, correlation_cols=("sex",))

    def test_valid_correlation_col_passes(self):
        df = _make_df()
        schema = detect_schema(df, expected_regions=DK_REGIONS[:2])
        validate_schema(df, schema, correlation_cols=("age",))  # age is numeric
