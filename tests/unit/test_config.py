"""Unit tests for msnpip.config — T5.1."""

from __future__ import annotations

from pathlib import Path

import pytest

from msnpip.config import EngineConfig, IOConfig, PipelineConfig
from msnpip.errors import ConfigurationError


def _cfg(**over) -> PipelineConfig:
    base = dict(
        io=IOConfig(dataframe=Path("merged.csv")),
        output=Path("out"),
        group_col="group",
        case="FTD",
        control="HC",
    )
    base.update(over)
    return PipelineConfig(**base)


class TestValidate:
    def test_ok(self):
        _cfg().validate()  # should not raise

    def test_no_input_raises(self):
        with pytest.raises(ConfigurationError, match="No input"):
            _cfg(io=IOConfig()).validate()

    def test_both_inputs_raises(self):
        cfg = _cfg(
            io=IOConfig(
                dataframe=Path("a.csv"), freesurfer_dir=Path("d"), demographics=Path("dem.csv")
            )
        )
        with pytest.raises(ConfigurationError, match="only one input"):
            cfg.validate()

    def test_pls_needs_exactly_one_of_ncomp_var(self):
        cfg = _cfg(engine=EngineConfig(n_components=1, var=0.5))
        with pytest.raises(ConfigurationError, match="exactly one of"):
            cfg.validate()

    def test_unknown_atlas_raises(self):
        cfg = _cfg(engine=EngineConfig(atlas="not_an_atlas"))
        with pytest.raises(ConfigurationError, match="atlas"):
            cfg.validate()


class TestSerialization:
    def test_to_dict_paths_are_strings(self):
        d = _cfg().to_dict()
        assert d["output"] == "out"
        assert d["io"]["dataframe"] == "merged.csv"
        assert isinstance(d["engine"]["gene_sets"], list)

    def test_from_dict_roundtrip(self):
        d = _cfg().to_dict()
        rebuilt = PipelineConfig.from_dict(d)
        assert rebuilt.engine.atlas == "dk"
        assert rebuilt.case == "FTD"
        assert isinstance(rebuilt.engine.methods, tuple)
