"""Mocked unit test for the dorsal surface path — T4.3.

Covers plot_surface_with_dorsal's dorsal branch without real surface assets, so
coverage is deterministic in CI (the real-asset render is exercised by the slow
integration test).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import types

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import msnpip.viz.surface_extra as se
from msnpip.viz.surface_extra import plot_surface_with_dorsal


def test_dorsal_path_with_mocked_engine(tmp_path, monkeypatch):
    table = pd.DataFrame(
        {"label": ["bankssts", "cuneus"], "hemisphere": ["L", "L"], "beta": [0.2, -0.1]}
    )

    coords = np.zeros((4, 3))
    triangles = np.array([[0, 1, 2], [1, 2, 3]])
    atlas = types.SimpleNamespace(surface_paths=("lh.annot", "rh.annot"))

    monkeypatch.setattr(se.brain, "matplotlib_backend", lambda: (None, plt))
    monkeypatch.setattr(se.brain, "surface_value_frames", lambda t: {"left": t})
    monkeypatch.setattr(se.brain, "get_atlas", lambda a: atlas)
    monkeypatch.setattr(se.brain, "surface_mesh_paths", lambda a: ("lh.mesh", "rh.mesh"))
    monkeypatch.setattr(se.brain, "load_surface_mesh", lambda p: (coords, triangles))
    monkeypatch.setattr(se.brain, "load_surface_parcellation", lambda p: (np.zeros(4), {}))
    monkeypatch.setattr(se.brain, "vertex_values_for_hemisphere", lambda *a, **k: np.zeros(4))
    monkeypatch.setattr(se.brain, "surface_view", lambda *a, **k: None)

    def fake_save(fig, path):
        fig.savefig(path)
        plt.close(fig)
        return path

    monkeypatch.setattr(se.brain, "save_figure", fake_save)

    out = tmp_path / "surface.png"
    result = plot_surface_with_dorsal(
        table,
        atlas_id="dk",
        value_column="beta",
        title="t",
        output_path=out,
        views=("lateral", "medial", "dorsal"),
    )
    assert result == out
    assert out.exists() and out.stat().st_size > 0


def test_no_finite_values_returns_none(tmp_path, monkeypatch):
    table = pd.DataFrame({"label": ["x"], "hemisphere": ["L"], "beta": [np.nan]})
    monkeypatch.setattr(se.brain, "matplotlib_backend", lambda: (None, plt))
    monkeypatch.setattr(se.brain, "surface_value_frames", lambda t: {"left": t})
    monkeypatch.setattr(
        se.brain, "get_atlas", lambda a: types.SimpleNamespace(surface_paths=("a", "b"))
    )
    monkeypatch.setattr(se.brain, "surface_mesh_paths", lambda a: ("a", "b"))
    out = plot_surface_with_dorsal(
        table,
        atlas_id="dk",
        value_column="beta",
        title="t",
        output_path=tmp_path / "o.png",
        views=("dorsal",),
    )
    assert out is None
