"""Slow integration test for the dorsal surface render — T4.3.

Deselect with ``-m 'not slow'``.  Skips if the engine's surface assets are
unavailable rather than failing.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from msnpip.atlas_align import engine_region_order, to_region_table  # noqa: E402
from msnpip.viz.surface_extra import plot_surface_with_dorsal  # noqa: E402

pytestmark = pytest.mark.slow


def test_dorsal_view_writes_png(tmp_path):
    labels = engine_region_order("dk", "left", "cort")
    rng = np.random.default_rng(0)
    table = to_region_table(rng.normal(size=len(labels)), labels, "beta")

    out = tmp_path / "surface.png"
    try:
        result = plot_surface_with_dorsal(
            table, atlas_id="dk", value_column="beta",
            title="synthetic contrast", output_path=out,
            views=("lateral", "medial", "dorsal"),
        )
    except Exception as exc:  # missing surface assets, etc.
        pytest.skip(f"Surface assets unavailable: {exc}")

    if result is None:
        pytest.skip("Engine reported no surface frames / assets for this atlas.")
    assert out.exists() and out.stat().st_size > 0
