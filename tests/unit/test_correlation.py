"""Unit tests for msnpip.stats.correlation — T2.5. Validated vs scipy."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats
from statsmodels.stats.multitest import multipletests

from msnpip.errors import SchemaError
from msnpip.io.schema import detect_schema
from msnpip.msn.construct import compute_strength_maps
from msnpip.stats.correlation import correlate_strength_with_demographic
from tests.fixtures.synthetic import DK_REGIONS, make_synthetic_cohort


@pytest.fixture
def cohort(tmp_path):
    info = make_synthetic_cohort(tmp_path, n_case=12, n_control=12, seed=33)
    df = pd.read_csv(info["merged_path"])
    schema = detect_schema(df, expected_regions=DK_REGIONS)
    sm_maps = compute_strength_maps(df, schema, hemisphere="left")
    return df, schema, sm_maps, info


class TestCorrelateDemographic:
    def test_global_matches_scipy_spearman(self, cohort):
        df, schema, sm_maps, _ = cohort
        res = correlate_strength_with_demographic(
            sm_maps, df, schema, variable="age", scope="global",
        )
        aligned = df.set_index(df["subject_id"].astype(str)).loc[sm_maps.subject_ids]
        expected = sp_stats.spearmanr(sm_maps.global_strength, aligned["age"].to_numpy())
        assert res.r[0] == pytest.approx(expected.statistic, rel=1e-9)
        assert res.p[0] == pytest.approx(expected.pvalue, rel=1e-9)
        assert res.n == 24

    def test_regional_shapes_and_fdr(self, cohort):
        df, schema, sm_maps, _ = cohort
        res = correlate_strength_with_demographic(
            sm_maps, df, schema, variable="age", scope="regional",
        )
        assert res.r.shape == (34,)
        assert res.p.shape == (34,)
        assert res.fdr.shape == (34,)
        assert res.region_labels == sm_maps.region_labels

        aligned = df.set_index(df["subject_id"].astype(str)).loc[sm_maps.subject_ids]
        age = aligned["age"].to_numpy()
        expected_r0 = sp_stats.spearmanr(sm_maps.strength[:, 0], age).statistic
        assert res.r[0] == pytest.approx(expected_r0, rel=1e-9)
        expected_fdr = multipletests(res.p, method="fdr_bh")[1]
        np.testing.assert_allclose(res.fdr, expected_fdr, rtol=1e-9)

    def test_pearson_method(self, cohort):
        df, schema, sm_maps, _ = cohort
        res = correlate_strength_with_demographic(
            sm_maps, df, schema, variable="age", scope="global", method="pearson",
        )
        aligned = df.set_index(df["subject_id"].astype(str)).loc[sm_maps.subject_ids]
        expected = sp_stats.pearsonr(sm_maps.global_strength, aligned["age"].to_numpy())
        assert res.r[0] == pytest.approx(expected.statistic, rel=1e-9)

    def test_within_group_filters(self, cohort):
        df, schema, sm_maps, info = cohort
        res = correlate_strength_with_demographic(
            sm_maps, df, schema, variable="age", scope="global",
            within_group=info["case_label"],
        )
        assert res.n == info["n_case"]
        assert res.group == info["case_label"]

    def test_nonnumeric_variable_raises(self, cohort):
        df, schema, sm_maps, _ = cohort
        with pytest.raises(SchemaError, match="numeric"):
            correlate_strength_with_demographic(
                sm_maps, df, schema, variable="sex", scope="global",
            )

    def test_missing_variable_raises(self, cohort):
        df, schema, sm_maps, _ = cohort
        with pytest.raises(SchemaError, match="not found"):
            correlate_strength_with_demographic(
                sm_maps, df, schema, variable="nope",
            )
