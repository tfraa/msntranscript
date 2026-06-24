"""Small-group (n<10) warning on the group contrast — spec R7."""

from __future__ import annotations

import logging

import pandas as pd

from msnpip.io.schema import detect_schema
from msnpip.msn.construct import compute_strength_maps
from msnpip.stats.glm import MIN_GROUP_N, regional_group_contrast
from tests.fixtures.synthetic import DK_REGIONS, make_synthetic_cohort


def _maps(tmp_path, n):
    info = make_synthetic_cohort(tmp_path, n_case=n, n_control=n, seed=1)
    df = pd.read_csv(info["merged_path"])
    schema = detect_schema(df, expected_regions=DK_REGIONS)
    return compute_strength_maps(df, schema, hemisphere="left"), df, schema, info


def test_small_group_warns(tmp_path, caplog):
    sm, df, schema, info = _maps(tmp_path, n=6)  # 6 per arm < 10
    with caplog.at_level(logging.WARNING, logger="msnpip.stats.glm"):
        regional_group_contrast(
            sm, df, schema, case_label="FTD", control_label="HC", covariates=("age",)
        )
    assert any("Small group" in r.message for r in caplog.records)


def test_adequate_group_does_not_warn(tmp_path, caplog):
    sm, df, schema, info = _maps(tmp_path, n=MIN_GROUP_N + 2)
    with caplog.at_level(logging.WARNING, logger="msnpip.stats.glm"):
        regional_group_contrast(
            sm, df, schema, case_label="FTD", control_label="HC", covariates=("age",)
        )
    assert not any("Small group" in r.message for r in caplog.records)
