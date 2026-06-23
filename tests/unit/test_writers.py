"""Unit tests for msnpip.io.writers — T1.7."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from msnpip.io.writers import OutputManager


class TestOutputManager:
    def test_write_csv_creates_file(self, tmp_path):
        mgr = OutputManager(tmp_path / "out")
        df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        path = mgr.write_table(df, "test_data")
        assert path.exists()
        assert path.suffix == ".csv"
        loaded = pd.read_csv(path)
        assert loaded.shape == (2, 2)

    def test_write_parquet_creates_file(self, tmp_path):
        pytest.importorskip("pyarrow", reason="pyarrow not installed")
        mgr = OutputManager(tmp_path / "out")
        df = pd.DataFrame({"a": [1, 2]})
        path = mgr.write_table(df, "test_data", fmt="parquet")
        assert path.exists()
        assert path.suffix == ".parquet"

    def test_write_array_creates_npz(self, tmp_path):
        mgr = OutputManager(tmp_path / "out")
        arr = np.arange(10, dtype=float)
        path = mgr.write_array(arr, "strength")
        assert path.exists()
        assert path.suffix == ".npz"
        loaded = np.load(path)["data"]
        assert np.allclose(loaded, arr)

    def test_write_json(self, tmp_path):
        mgr = OutputManager(tmp_path / "out")
        path = mgr.write_json({"key": "value", "n": 42}, "meta")
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["key"] == "value"

    def test_no_pickle_record_raises(self, tmp_path):
        mgr = OutputManager(tmp_path / "out")
        with pytest.raises(ValueError, match="Pickle files are not allowed"):
            mgr.record(tmp_path / "data.pkl")

    def test_manifest_has_sha256(self, tmp_path):
        mgr = OutputManager(tmp_path / "out", seed=99)
        df = pd.DataFrame({"x": [1]})
        mgr.write_table(df, "data")
        manifest_path = mgr.finalize()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["seed"] == 99
        assert manifest["msnpip_version"] == "2.0.0"
        assert len(manifest["artifacts"]) >= 1
        for artifact in manifest["artifacts"]:
            assert "sha256" in artifact
            assert len(artifact["sha256"]) == 64  # hex SHA256

    def test_manifest_engine_commit_recorded(self, tmp_path):
        commit = "e6a2c237fc74a0b2072a6d58efaf9d1c22cc08e1"
        mgr = OutputManager(tmp_path / "out", engine_commit=commit)
        manifest_path = mgr.finalize()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["engine_commit"] == commit

    def test_subdir_creates_child_manager(self, tmp_path):
        mgr = OutputManager(tmp_path / "out")
        child = mgr.subdir("03_transcriptomics", "ftd_vs_hc")
        assert child.output_dir == tmp_path / "out" / "03_transcriptomics" / "ftd_vs_hc"
        assert child.output_dir.exists()

    def test_resolved_config_in_manifest(self, tmp_path):
        mgr = OutputManager(tmp_path / "out")
        cfg = {"atlas": "dk", "hemisphere": "left", "n_permutations": 10000}
        manifest_path = mgr.finalize(resolved_config=cfg)
        manifest = json.loads(manifest_path.read_text())
        assert manifest["resolved_config"]["atlas"] == "dk"
