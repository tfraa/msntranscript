"""Unit tests for msnpip.stats.sensitivity — T2.6."""

from __future__ import annotations

import pandas as pd
import pytest
from scipy import stats as sp_stats

from msnpip.io.schema import detect_schema
from msnpip.msn.construct import compute_strength_maps
from msnpip.stats.sensitivity import covariate_exclusion_contrast
from tests.fixtures.synthetic import DK_REGIONS, make_synthetic_cohort


@pytest.fixture
def cohort(tmp_path):
    info = make_synthetic_cohort(tmp_path, n_case=12, n_control=12, seed=44)
    df = pd.read_csv(info["merged_path"])
    schema = detect_schema(df, expected_regions=DK_REGIONS)
    sm_maps = compute_strength_maps(df, schema, hemisphere="left")
    return df, schema, sm_maps, info


class TestCovariateExclusionContrast:
    def test_dropped_and_reduced_covariates(self, cohort):
        df, schema, sm_maps, info = cohort
        res = covariate_exclusion_contrast(
            sm_maps,
            df,
            schema,
            case_label=info["case_label"],
            control_label=info["control_label"],
            full_covariates=["age", "tiv", "sex"],
            drop="tiv",
        )
        assert res.dropped == ["tiv"]
        assert res.full.covariates == ["age", "tiv", "sex"]
        assert res.reduced.covariates == ["age", "sex"]

    def test_rank_corr_matches_spearman_of_maps(self, cohort):
        df, schema, sm_maps, info = cohort
        res = covariate_exclusion_contrast(
            sm_maps,
            df,
            schema,
            case_label=info["case_label"],
            control_label=info["control_label"],
            full_covariates=["age", "tiv"],
            drop="tiv",
        )
        expected = sp_stats.spearmanr(res.full.regional_stat, res.reduced.regional_stat)
        assert res.rank_corr == pytest.approx(expected.statistic, rel=1e-9)
        assert res.rank_corr_p == pytest.approx(expected.pvalue, rel=1e-9)

    def test_dropping_irrelevant_covariate_keeps_high_agreement(self, cohort):
        df, schema, sm_maps, info = cohort
        # 'site' is round-robin assigned and unrelated to the synthetic strength,
        # so the map should barely move when it is dropped.
        res = covariate_exclusion_contrast(
            sm_maps,
            df,
            schema,
            case_label=info["case_label"],
            control_label=info["control_label"],
            full_covariates=["age", "site"],
            drop="site",
        )
        assert res.rank_corr > 0.8

    def test_drop_list_accepted(self, cohort):
        df, schema, sm_maps, info = cohort
        res = covariate_exclusion_contrast(
            sm_maps,
            df,
            schema,
            case_label=info["case_label"],
            control_label=info["control_label"],
            full_covariates=["age", "tiv", "sex"],
            drop=["tiv", "sex"],
        )
        assert res.dropped == ["tiv", "sex"]
        assert res.reduced.covariates == ["age"]
