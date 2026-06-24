"""Tests for CLI config-overlay precedence (YAML base + sparse CLI override)."""

from __future__ import annotations

from msnpip.cli import _cfg_from_args, build_parser


def _yaml(tmp_path, text):
    p = tmp_path / "run.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_yaml_values_not_clobbered_by_cli_defaults(tmp_path):
    cfg_yaml = _yaml(
        tmp_path,
        """
output: out_from_yaml
group_col: group
case: FTD
control: HC
io:
  dataframe: merged.csv
engine:
  hemisphere: both
  n_permutations: 5000
  seed: 42
""",
    )
    args = build_parser().parse_args(
        ["full", "--config", str(cfg_yaml), "--output", str(tmp_path / "o"), "--seed", "99"]
    )
    cfg = _cfg_from_args(args)

    # YAML values survive (not overwritten by CLI defaults):
    assert cfg.engine.hemisphere == "both"
    assert cfg.engine.n_permutations == 5000
    assert cfg.case == "FTD"
    assert str(cfg.io.dataframe).endswith("merged.csv")
    # Explicit CLI flags win over YAML:
    assert cfg.engine.seed == 99
    assert str(cfg.output).endswith("o")


def test_no_config_falls_back_to_dataclass_defaults(tmp_path):
    args = build_parser().parse_args(
        [
            "full",
            "--dataframe",
            "m.csv",
            "--output",
            str(tmp_path / "o"),
            "--group-col",
            "group",
            "--case",
            "A",
            "--control",
            "B",
        ]
    )
    cfg = _cfg_from_args(args)
    assert cfg.engine.atlas == "dk"
    assert cfg.engine.hemisphere == "left"
    assert cfg.engine.methods == ("pls", "corr")
    assert cfg.engine.n_components == 1


def test_var_clears_n_components(tmp_path):
    args = build_parser().parse_args(
        ["full", "--dataframe", "m.csv", "--output", str(tmp_path / "o"), "--var", "0.5"]
    )
    cfg = _cfg_from_args(args)
    assert cfg.engine.var == 0.5
    assert cfg.engine.n_components is None
