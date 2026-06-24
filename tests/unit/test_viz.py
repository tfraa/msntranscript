"""Unit tests for msnpip.viz — T4.1, T4.2, T4.3 (delegation), T4.4."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from msnpip.io.schema import detect_schema
from msnpip.msn.construct import compute_strength_maps
from msnpip.stats.correlation import correlate_strength_with_demographic
from msnpip.viz.distributions import plot_strength_violin
from msnpip.viz.scatter import plot_demographic_correlation
from msnpip.viz.surface_extra import plot_surface_with_dorsal
from msnpip.viz.theme import (
    CASE_COLOR,
    CONTROL_COLOR,
    format_p,
    group_colors,
    significance_stars,
)
from tests.fixtures.synthetic import DK_REGIONS, make_synthetic_cohort


@pytest.fixture
def cohort(tmp_path):
    info = make_synthetic_cohort(tmp_path, n_case=12, n_control=12, seed=55)
    df = pd.read_csv(info["merged_path"])
    schema = detect_schema(df, expected_regions=DK_REGIONS)
    sm = compute_strength_maps(df, schema, hemisphere="left")
    return df, schema, sm, info


# ---------------------------------------------------------------------------
# theme
# ---------------------------------------------------------------------------


class TestTheme:
    def test_significance_stars(self):
        assert significance_stars(0.0005) == "***"
        assert significance_stars(0.005) == "**"
        assert significance_stars(0.03) == "*"
        assert significance_stars(0.2) == "ns"
        assert significance_stars(float("nan")) == "ns"

    def test_format_p(self):
        assert format_p(0.0001) == "p < 0.001"
        assert format_p(0.013) == "p = 0.013"

    def test_group_colors_okabe_ito(self):
        colors = group_colors(["FTD", "HC"])
        assert colors["FTD"] == CASE_COLOR == "#E69F00"
        assert colors["HC"] == CONTROL_COLOR == "#0072B2"


# ---------------------------------------------------------------------------
# distributions
# ---------------------------------------------------------------------------


class TestViolin:
    def test_global_returns_figure_with_bracket(self, cohort):
        df, schema, sm, _ = cohort
        fig = plot_strength_violin(sm, df, schema)
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert len(ax.get_xticklabels()) == 2
        assert any("p" in t.get_text() for t in ax.texts)  # significance label drawn
        plt.close(fig)

    def test_region_specific(self, cohort):
        df, schema, sm, _ = cohort
        fig = plot_strength_violin(sm, df, schema, region="lh_bankssts")
        assert isinstance(fig, plt.Figure)
        assert "lh_bankssts" in fig.axes[0].get_ylabel()
        plt.close(fig)

    def test_unknown_region_raises(self, cohort):
        df, schema, sm, _ = cohort
        with pytest.raises(ValueError, match="region"):
            plot_strength_violin(sm, df, schema, region="lh_nope")

    def test_empty_group_guard(self, cohort):
        df, schema, sm, _ = cohort
        with pytest.raises(ValueError, match="no subjects"):
            plot_strength_violin(sm, df, schema, group_labels=["FTD", "HC", "NONEXISTENT"])


# ---------------------------------------------------------------------------
# scatter
# ---------------------------------------------------------------------------


class TestScatter:
    def test_global_correlation_figure(self, cohort):
        df, schema, sm, _ = cohort
        res = correlate_strength_with_demographic(sm, df, schema, variable="age", scope="global")
        fig = plot_demographic_correlation(res)
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert ax.get_xlabel() == "age"
        assert any("r =" in t.get_text() for t in ax.texts)  # annotation box
        plt.close(fig)

    def test_no_raw_data_raises(self, cohort):
        df, schema, sm, _ = cohort
        # regional result carries no x_values/y_values
        res = correlate_strength_with_demographic(sm, df, schema, variable="age", scope="regional")
        with pytest.raises(ValueError, match="raw data"):
            plot_demographic_correlation(res)


# ---------------------------------------------------------------------------
# surface_extra (delegation path — no real assets needed)
# ---------------------------------------------------------------------------


class TestSurfaceExtra:
    def test_unknown_view_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown view"):
            plot_surface_with_dorsal(
                pd.DataFrame({"label": ["x"], "hemisphere": ["L"], "beta": [0.1]}),
                atlas_id="dk",
                value_column="beta",
                title="t",
                output_path=tmp_path / "o.png",
                views=("sagittal",),
            )

    def test_unknown_mesh_kind_raises(self, tmp_path):
        with pytest.raises(ValueError, match="mesh_kind"):
            plot_surface_with_dorsal(
                pd.DataFrame({"label": ["x"], "hemisphere": ["L"], "beta": [0.1]}),
                atlas_id="dk",
                value_column="beta",
                title="t",
                output_path=tmp_path / "o.png",
                mesh_kind="banana",
            )
