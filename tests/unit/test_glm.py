"""Unit tests for msnpip.stats.glm — T2.3, T2.4. Validated vs statsmodels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from msnpip.errors import SchemaError
from msnpip.io.schema import detect_schema
from msnpip.msn.construct import compute_strength_maps
from msnpip.stats.glm import (
    GroupContrastResult,
    build_design_matrix,
    fit_ols,
    regional_group_contrast,
    residualize,
)
from tests.fixtures.synthetic import DK_REGIONS, make_synthetic_cohort

# ---------------------------------------------------------------------------
# build_design_matrix
# ---------------------------------------------------------------------------


class TestBuildDesignMatrix:
    def test_numeric_passthrough_with_intercept(self):
        df = pd.DataFrame({"age": [10.0, 20.0, 30.0]})
        X = build_design_matrix(df, ["age"])
        assert list(X.columns) == ["Intercept", "age"]
        np.testing.assert_array_equal(X["Intercept"], [1, 1, 1])
        np.testing.assert_array_equal(X["age"], [10, 20, 30])

    def test_categorical_one_hot_drop_first(self):
        df = pd.DataFrame({"sex": ["F", "M", "F", "M"]})
        X = build_design_matrix(df, ["sex"])
        # drop_first drops 'F' (alphabetical) → only sex_M remains
        assert "sex_M" in X.columns
        assert "sex_F" not in X.columns
        np.testing.assert_array_equal(X["sex_M"], [0, 1, 0, 1])

    def test_missing_predictor_raises(self):
        df = pd.DataFrame({"age": [1, 2]})
        with pytest.raises(SchemaError, match="not found"):
            build_design_matrix(df, ["nope"])


# ---------------------------------------------------------------------------
# fit_ols — vs statsmodels closed form
# ---------------------------------------------------------------------------


class TestFitOLS:
    def test_matches_statsmodels(self):
        rng = np.random.default_rng(11)
        n = 50
        X = pd.DataFrame(
            {
                "Intercept": np.ones(n),
                "x1": rng.normal(size=n),
                "x2": rng.normal(size=n),
            }
        )
        y = 2.0 + 1.5 * X["x1"] - 0.7 * X["x2"] + rng.normal(scale=0.3, size=n)

        res = fit_ols(X, y)
        model = sm.OLS(y.to_numpy(), X.to_numpy()).fit()

        np.testing.assert_allclose(res.params, model.params, rtol=1e-10)
        np.testing.assert_allclose(res.se, model.bse, rtol=1e-10)
        np.testing.assert_allclose(res.tvalues, model.tvalues, rtol=1e-10)
        np.testing.assert_allclose(res.pvalues, model.pvalues, rtol=1e-8, atol=1e-12)
        assert res.df_resid == int(model.df_resid)

    def test_resid_equals_y_minus_fitted(self):
        rng = np.random.default_rng(12)
        X = pd.DataFrame({"Intercept": np.ones(20), "x": rng.normal(size=20)})
        y = rng.normal(size=20)
        res = fit_ols(X, y)
        np.testing.assert_allclose(res.resid, y - res.fitted)


# ---------------------------------------------------------------------------
# residualize
# ---------------------------------------------------------------------------


class TestResidualize:
    def test_residuals_orthogonal_to_covariates(self):
        rng = np.random.default_rng(13)
        n = 40
        cov = rng.normal(size=(n, 2))
        y = 3.0 + cov @ np.array([1.0, -2.0]) + rng.normal(scale=0.1, size=n)
        resid = residualize(y, cov)
        # residuals must be (numerically) orthogonal to each covariate and the intercept
        assert abs(resid.mean()) < 1e-9
        np.testing.assert_allclose(cov.T @ resid, [0.0, 0.0], atol=1e-9)

    def test_add_back_mean(self):
        rng = np.random.default_rng(14)
        y = rng.normal(loc=5.0, size=30)
        cov = rng.normal(size=30)
        r0 = residualize(y, cov)
        r1 = residualize(y, cov, add_back_mean=True)
        np.testing.assert_allclose(r1 - r0, y.mean())


# ---------------------------------------------------------------------------
# regional_group_contrast — vs statsmodels
# ---------------------------------------------------------------------------


@pytest.fixture
def cohort(tmp_path):
    info = make_synthetic_cohort(tmp_path, n_case=10, n_control=10, seed=21)
    df = pd.read_csv(info["merged_path"])
    schema = detect_schema(df, expected_regions=DK_REGIONS)
    sm_maps = compute_strength_maps(df, schema, hemisphere="left")
    return df, schema, sm_maps, info


class TestRegionalGroupContrast:
    def test_beta_matches_statsmodels(self, cohort):
        df, schema, sm_maps, info = cohort
        cov = ["age", "tiv", "sex"]
        res = regional_group_contrast(
            sm_maps,
            df,
            schema,
            case_label=info["case_label"],
            control_label=info["control_label"],
            covariates=cov,
            stat="beta",
        )
        assert isinstance(res, GroupContrastResult)
        assert res.stat_type == "beta"
        assert res.n_case == 10 and res.n_control == 10
        assert res.region_labels == sm_maps.region_labels

        # Independently fit region 0 with the same design.
        aligned = df.set_index(df["subject_id"].astype(str)).loc[sm_maps.subject_ids]
        design_input = aligned[cov].copy()
        design_input.insert(
            0, "group", (aligned["group"] == info["case_label"]).astype(float).to_numpy()
        )
        design = build_design_matrix(design_input, list(design_input.columns))
        gi = design.columns.get_loc("group")
        model = sm.OLS(sm_maps.strength[:, 0], design.to_numpy()).fit()
        assert res.regional_stat[0] == pytest.approx(model.params[gi], rel=1e-9)

    def test_t_stat(self, cohort):
        df, schema, sm_maps, info = cohort
        res = regional_group_contrast(
            sm_maps,
            df,
            schema,
            case_label=info["case_label"],
            control_label=info["control_label"],
            covariates=["age"],
            stat="t",
        )
        assert res.stat_type == "t"
        aligned = df.set_index(df["subject_id"].astype(str)).loc[sm_maps.subject_ids]
        design_input = aligned[["age"]].copy()
        design_input.insert(
            0, "group", (aligned["group"] == info["case_label"]).astype(float).to_numpy()
        )
        design = build_design_matrix(design_input, list(design_input.columns))
        gi = design.columns.get_loc("group")
        model = sm.OLS(sm_maps.strength[:, 5], design.to_numpy()).fit()
        assert res.regional_stat[5] == pytest.approx(model.tvalues[gi], rel=1e-9)

    def test_cohen_d_no_covariates(self, cohort):
        df, schema, sm_maps, info = cohort
        res = regional_group_contrast(
            sm_maps,
            df,
            schema,
            case_label=info["case_label"],
            control_label=info["control_label"],
            stat="cohen_d",
        )
        aligned = df.set_index(df["subject_id"].astype(str)).loc[sm_maps.subject_ids]
        is_case = (aligned["group"] == info["case_label"]).to_numpy()
        y = sm_maps.strength[:, 0]
        c, k = y[is_case], y[~is_case]
        pooled = np.sqrt(
            ((len(c) - 1) * c.var(ddof=1) + (len(k) - 1) * k.var(ddof=1)) / (len(c) + len(k) - 2)
        )
        expected = (c.mean() - k.mean()) / pooled
        assert res.regional_stat[0] == pytest.approx(expected, rel=1e-12)

    def test_missing_group_col_raises(self, cohort):
        df, schema, sm_maps, info = cohort
        with pytest.raises(SchemaError):
            regional_group_contrast(
                sm_maps,
                df,
                schema,
                group_col="nope",
                case_label="a",
                control_label="b",
            )

    def test_invalid_stat_raises(self, cohort):
        df, schema, sm_maps, info = cohort
        with pytest.raises(ValueError, match="stat"):
            regional_group_contrast(
                sm_maps,
                df,
                schema,
                case_label=info["case_label"],
                control_label=info["control_label"],
                stat="bogus",
            )
