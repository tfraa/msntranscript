"""Unit tests for msnpip.msn.construct — T2.1, T2.2."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats

from msnpip.errors import MSNInputError
from msnpip.io.schema import detect_schema
from msnpip.msn.construct import (
    DEFAULT_METRICS,
    StrengthMaps,
    build_msn,
    compute_strength_maps,
    node_strength,
)
from tests.fixtures.synthetic import DK_REGIONS, make_synthetic_cohort


# ---------------------------------------------------------------------------
# build_msn
# ---------------------------------------------------------------------------

class TestBuildMSN:
    def test_shape_and_diagonal(self):
        rng = np.random.default_rng(0)
        feats = rng.normal(size=(4, 10, 5))  # 4 subjects, 10 regions, 5 metrics
        msn = build_msn(feats)
        assert msn.shape == (4, 10, 10)
        for s in range(4):
            assert np.all(np.isnan(np.diag(msn[s])))

    def test_symmetric(self):
        rng = np.random.default_rng(1)
        feats = rng.normal(size=(8, 5))
        msn = build_msn(feats)  # single subject → 2-D
        assert msn.shape == (8, 8)
        off = ~np.eye(8, dtype=bool)
        np.testing.assert_allclose(msn[off], msn.T[off])

    def test_matches_scipy_pearson_on_zscored_features(self):
        """Off-diagonal entries equal pearsonr of column-standardized features."""
        rng = np.random.default_rng(2)
        feats = rng.normal(size=(6, 5))
        z = (feats - feats.mean(axis=0)) / feats.std(axis=0, ddof=0)
        msn = build_msn(feats)
        for i in range(6):
            for j in range(6):
                if i == j:
                    continue
                expected = sp_stats.pearsonr(z[i], z[j]).statistic
                assert msn[i, j] == pytest.approx(expected, abs=1e-12)

    def test_identical_regions_correlate_one(self):
        feats = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [1.0, 2.0, 3.0, 4.0, 5.0],   # identical to region 0
                [5.0, 1.0, 4.0, 2.0, 3.0],
            ]
        )
        msn = build_msn(feats)
        assert msn[0, 1] == pytest.approx(1.0, abs=1e-12)

    def test_all_nan_region_raises(self):
        feats = np.ones((4, 5))
        feats[2, :] = np.nan
        with pytest.raises(MSNInputError, match="all-NaN"):
            build_msn(feats)

    def test_constant_metric_raises(self):
        rng = np.random.default_rng(3)
        feats = rng.normal(size=(5, 5))
        feats[:, 1] = 7.0  # constant metric → zero variance
        with pytest.raises(MSNInputError, match="constant"):
            build_msn(feats)

    def test_determinism(self):
        rng = np.random.default_rng(4)
        feats = rng.normal(size=(3, 7, 5))
        np.testing.assert_array_equal(build_msn(feats), build_msn(feats))


# ---------------------------------------------------------------------------
# node_strength
# ---------------------------------------------------------------------------

class TestNodeStrength:
    @pytest.fixture
    def toy(self) -> np.ndarray:
        m = np.array(
            [
                [np.nan, 0.4, -0.2],
                [0.4, np.nan, 0.6],
                [-0.2, 0.6, np.nan],
            ]
        )
        return m

    def test_signed(self, toy):
        s = node_strength(toy, sign="signed")
        # region0: (0.4 + (-0.2))/2 = 0.1 ; region1: (0.5 + 0)/2 = 0.25 ; region2: (0.6 + (-0.2))/2 = 0.2
        np.testing.assert_allclose(s, [0.1, 0.25, 0.2])

    def test_positive(self, toy):
        s = node_strength(toy, sign="positive")
        np.testing.assert_allclose(s, [0.4, 0.5, 0.6])

    def test_absolute(self, toy):
        s = node_strength(toy, sign="absolute")
        np.testing.assert_allclose(s, [0.3, 0.5, 0.4])

    def test_batched_shape(self):
        m = np.stack([np.array([[np.nan, 0.4, -0.2], [0.4, np.nan, 0.6], [-0.2, 0.6, np.nan]])] * 3)
        s = node_strength(m)
        assert s.shape == (3, 3)

    def test_no_negative_edges_contributes_zero(self):
        m = np.array([[np.nan, 0.2, 0.4], [0.2, np.nan, 0.6], [0.4, 0.6, np.nan]])
        s = node_strength(m, sign="signed")
        # all positive → signed = pos_mean/2
        np.testing.assert_allclose(s, [0.3 / 2, 0.4 / 2, 0.5 / 2])

    def test_invalid_sign_raises(self):
        with pytest.raises(ValueError, match="sign"):
            node_strength(np.zeros((3, 3)), sign="bogus")


# ---------------------------------------------------------------------------
# compute_strength_maps
# ---------------------------------------------------------------------------

@pytest.fixture
def cohort(tmp_path):
    info = make_synthetic_cohort(tmp_path, n_case=8, n_control=8, seed=7)
    df = pd.read_csv(info["merged_path"])
    schema = detect_schema(df, expected_regions=DK_REGIONS)
    return df, schema, info


class TestComputeStrengthMaps:
    def test_basic_shapes_left(self, cohort):
        df, schema, info = cohort
        sm = compute_strength_maps(df, schema, hemisphere="left", regions="cort")
        assert isinstance(sm, StrengthMaps)
        assert sm.matrix.shape == (16, 34, 34)
        assert sm.strength.shape == (16, 34)
        assert sm.global_strength.shape == (16,)
        assert len(sm.region_labels) == 34
        assert sm.features == list(DEFAULT_METRICS)

    def test_region_label_format(self, cohort):
        df, schema, _ = cohort
        sm = compute_strength_maps(df, schema, hemisphere="left")
        assert all(lbl.startswith("lh_") for lbl in sm.region_labels)
        # labels parse cleanly into atlas-align format
        assert "lh_bankssts" in sm.region_labels

    def test_both_hemispheres(self, cohort):
        df, schema, _ = cohort
        sm = compute_strength_maps(df, schema, hemisphere="both")
        assert sm.matrix.shape[1] == 68
        assert any(lbl.startswith("rh_") for lbl in sm.region_labels)

    def test_default_is_both_hemispheres(self, cohort):
        """MSN is whole-cortex by default — both hemispheres, 68 DK regions."""
        df, schema, _ = cohort
        sm = compute_strength_maps(df, schema)  # no hemisphere → default
        assert sm.hemisphere == "both"
        assert sm.matrix.shape[1] == 68
        assert any(lbl.startswith("lh_") for lbl in sm.region_labels)
        assert any(lbl.startswith("rh_") for lbl in sm.region_labels)

    def test_subject_ids_preserved(self, cohort):
        df, schema, info = cohort
        sm = compute_strength_maps(df, schema)
        assert sm.subject_ids == [str(i) for i in df["subject_id"].tolist()]

    def test_drop_and_report(self, cohort):
        df, schema, _ = cohort
        df = df.copy()
        # blank a single feature for subject index 3 → dropped at default threshold 0.0
        df.loc[3, "lh_bankssts_SurfArea"] = np.nan
        dropped_id = str(df.loc[3, "subject_id"])
        sm = compute_strength_maps(df, schema)
        assert dropped_id in sm.dropped_subjects
        assert dropped_id not in sm.subject_ids
        assert sm.matrix.shape[0] == 15

    def test_drop_threshold_tolerates_some_missing(self, cohort):
        df, schema, _ = cohort
        df = df.copy()
        # one missing feature out of 34*5 = 170 → frac ≈ 0.0059
        df.loc[2, "lh_insula_GausCurv"] = np.nan
        sm = compute_strength_maps(df, schema, drop_threshold=0.01)
        assert sm.matrix.shape[0] == 16  # nobody dropped

    def test_all_dropped_raises(self, cohort):
        df, schema, _ = cohort
        df = df.copy()
        df["lh_bankssts_SurfArea"] = np.nan
        with pytest.raises(MSNInputError, match="dropped"):
            compute_strength_maps(df, schema)

    def test_no_matching_features_raises(self, cohort):
        df, schema, _ = cohort
        with pytest.raises(MSNInputError, match="No feature columns"):
            compute_strength_maps(df, schema, metrics=("NoSuchMetric",))

    def test_determinism(self, cohort):
        df, schema, _ = cohort
        a = compute_strength_maps(df, schema)
        b = compute_strength_maps(df, schema)
        np.testing.assert_array_equal(a.strength, b.strength)
        np.testing.assert_array_equal(a.global_strength, b.global_strength)
