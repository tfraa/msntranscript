"""
argparse CLI — console_script entry point ``msnpip``.
Phase 5, Task T5.5.

Subcommands:
  full           run the whole pipeline LOAD→REPORT
  from-strength  resume from a persisted strength_maps.csv (runs CONTRAST→REPORT)
  list-atlases   print the engine's atlas table
  list-genesets  print the default gene sets

Config precedence: ``--config FILE`` provides the base, and only the CLI flags the
user *actually passed* override it (override-able flags default to
``argparse.SUPPRESS`` so unset flags never clobber YAML values). Anything left
unset falls back to the dataclass defaults in ``config.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from msnpip.config import EngineConfig, PipelineConfig
from msnpip.errors import MsnpipError
from msnpip.logging_ import configure_logging, get_logger

logger = get_logger("msnpip.cli")

_SUP = argparse.SUPPRESS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="msnpip",
        description="Morphometric Similarity Network → imaging transcriptomics pipeline",
    )
    sub = p.add_subparsers(dest="command", required=True)

    full = sub.add_parser("full", help="run the whole pipeline")
    _add_full_args(full)

    fs = sub.add_parser("from-strength", help="resume from a persisted strength_maps.csv")
    fs.add_argument("--output", required=True, type=Path, help="existing run directory")
    fs.add_argument("--strength", type=Path, help="path to strength_maps.csv (informational)")
    _add_contrast_args(fs)
    _add_glm_args(fs)
    _add_corr_args(fs)
    _add_engine_args(fs)
    fs.add_argument("-v", "--verbose", action="store_true", default=_SUP)

    sub.add_parser("list-atlases", help="print available atlases (from the engine)")
    sub.add_parser("list-genesets", help="print the default gene sets")
    return p


def _add_full_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("input (one mode)")
    g.add_argument("--input", type=Path, default=_SUP, help="FreeSurfer subjects directory")
    g.add_argument(
        "--demographics", type=Path, default=_SUP, help="demographics CSV (with --input)"
    )
    g.add_argument("--dataframe", type=Path, default=_SUP, help="single merged wide-format file")
    g.add_argument("--sep", default=_SUP)
    g.add_argument("--decimal", default=_SUP)
    g.add_argument("--sheet", default=_SUP)
    g.add_argument("--id-col", default=_SUP)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--config", type=Path, help="YAML config (CLI flags override it)")
    _add_contrast_args(ap)
    _add_msn_args(ap)
    _add_glm_args(ap)
    _add_corr_args(ap)
    _add_engine_args(ap)
    ap.add_argument("--save-all", dest="save_all", action="store_true", default=_SUP)
    ap.add_argument("--no-save-all", dest="save_all", action="store_false", default=_SUP)
    ap.add_argument("--save-figures", dest="save_figures", action="store_true", default=_SUP)
    ap.add_argument("--no-save-figures", dest="save_figures", action="store_false", default=_SUP)
    ap.add_argument("--start-stage", default=_SUP)
    ap.add_argument("--stop-stage", default=_SUP)
    ap.add_argument("-v", "--verbose", action="store_true", default=_SUP)


def _add_contrast_args(ap):
    ap.add_argument("--group-col", default=_SUP)
    ap.add_argument("--case", default=_SUP)
    ap.add_argument("--control", default=_SUP)
    ap.add_argument("--contrast", nargs=2, action="append", metavar=("CASE", "CTRL"), default=_SUP)


def _add_msn_args(ap):
    ap.add_argument("--features", nargs="+", default=_SUP)
    ap.add_argument(
        "--msn-similarity",
        dest="similarity",
        choices=("distance", "correlation"),
        default=_SUP,
        help="MSN edge definition: distance kernel (default) or canonical Pearson correlation",
    )


def _add_glm_args(ap):
    ap.add_argument("--predictors", nargs="+", default=_SUP)
    ap.add_argument("--contrast-stat", choices=("beta", "t", "cohen_d"), default=_SUP)


def _add_corr_args(ap):
    ap.add_argument("--correlate-with", nargs="+", dest="correlate_with", default=_SUP)
    ap.add_argument("--corr-method", choices=("pearson", "spearman"), default=_SUP)
    ap.add_argument("--corr-scope", choices=("global", "regional"), default=_SUP)
    ap.add_argument("--corr-within-group", dest="corr_within_group", default=_SUP)


def _add_engine_args(ap):
    # --atlas and --regions are intentionally not exposed: the methodology is
    # locked to the DK atlas and cortical regions. Both keep their config.py
    # defaults and can still be overridden via a --config YAML if ever needed.
    ap.add_argument("--hemisphere", choices=("left", "both"), default=_SUP)
    ap.add_argument("--compare-hemispheres", action="store_true", default=_SUP)
    ap.add_argument(
        "--pool-cases",
        dest="pool_cases",
        action="store_true",
        default=_SUP,
        help="also run a pooled contrast (union of specified cases per control) alongside each",
    )
    ap.add_argument(
        "--null-method",
        choices=("vasa", "alexander_bloch", "moran", "auto", "random"),
        dest="null_method",
        default=_SUP,
        help="cortical spatial null (default vasa). Use 'auto' to allow fallback to random.",
    )
    ap.add_argument("--method", choices=("pls",), action="append", dest="method", default=_SUP)
    ap.add_argument("--ncomp", type=int, default=_SUP)
    ap.add_argument("--var", type=float, default=_SUP)
    ap.add_argument("--n-perm", type=int, dest="n_perm", default=_SUP)
    ap.add_argument(
        "--enrichment", choices=("ensemble", "gsea", "ora", "none"), action="append", default=_SUP
    )
    ap.add_argument("--geneset", nargs="+", dest="geneset", default=_SUP)
    ap.add_argument("--seed", type=int, default=_SUP)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* onto *base*; override wins on conflicts."""
    out = dict(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _cfg_from_args(args) -> PipelineConfig:
    """Build a PipelineConfig from parsed args.

    Only flags the user actually passed appear in ``vars(args)`` (override-able
    flags use ``argparse.SUPPRESS``), so the CLI override dict is sparse and a
    ``--config`` YAML base is preserved for anything left unset.
    """
    a = vars(args)

    io: dict = {}
    if "input" in a:
        io["freesurfer_dir"] = str(a["input"])
    if "demographics" in a:
        io["demographics"] = str(a["demographics"])
    if "dataframe" in a:
        io["dataframe"] = str(a["dataframe"])
    for src, dst in (
        ("sep", "sep"),
        ("decimal", "decimal"),
        ("sheet", "sheet"),
        ("id_col", "id_col"),
    ):
        if src in a:
            io[dst] = a[src]
    if "group_col" in a:
        io["group_col"] = a["group_col"]

    msn: dict = {}
    if "features" in a:
        msn["features"] = list(a["features"])
    if "similarity" in a:
        msn["similarity"] = a["similarity"]

    glm: dict = {}
    if "predictors" in a:
        glm["predictors"] = list(a["predictors"])
    if "contrast_stat" in a:
        glm["contrast_stat"] = a["contrast_stat"]

    corr: dict = {}
    if "correlate_with" in a:
        corr["variables"] = list(a["correlate_with"])
    if "corr_method" in a:
        corr["method"] = a["corr_method"]
    if "corr_scope" in a:
        corr["scope"] = a["corr_scope"]
    if "corr_within_group" in a:
        corr["within_group"] = a["corr_within_group"]

    engine: dict = {}
    for src, dst in (
        ("hemisphere", "hemisphere"),
        ("compare_hemispheres", "compare_hemispheres"),
        ("pool_cases", "pool_cases"),
        ("null_method", "null_method"),
        ("n_perm", "n_permutations"),
        ("seed", "seed"),
    ):
        if src in a:
            engine[dst] = a[src]
    if "method" in a:
        engine["methods"] = list(a["method"])
    if "enrichment" in a:
        engine["enrichment_methods"] = list(a["enrichment"])
    if "geneset" in a:
        engine["gene_sets"] = list(a["geneset"])
    # n_components / var are mutually exclusive — passing --var clears n_components.
    if "var" in a:
        engine["var"] = a["var"]
        engine["n_components"] = None
    if "ncomp" in a:
        engine["n_components"] = a["ncomp"]

    override: dict = {"io": io, "msn": msn, "glm": glm, "correlation": corr, "engine": engine}
    override = {k: v for k, v in override.items() if v} or {}
    for key in ("output", "group_col", "case", "control", "save_all", "save_figures", "verbose"):
        if key in a:
            override[key] = str(a[key]) if key == "output" else a[key]
    if "contrast" in a:
        override["contrasts"] = [list(c) for c in a["contrast"]]

    base: dict = {}
    if a.get("config"):
        import yaml

        base = yaml.safe_load(Path(a["config"]).read_text(encoding="utf-8")) or {}

    merged = _deep_merge(base, override)
    if "output" not in merged:
        raise MsnpipError("No output directory given (set --output or 'output:' in --config).")
    return PipelineConfig.from_dict(merged)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(getattr(args, "verbose", False))

    if args.command == "list-atlases":
        import imaging_transcriptomics as imt

        print(imt.atlas_table().to_string(index=False))
        return 0
    if args.command == "list-genesets":
        for gs in EngineConfig().gene_sets:
            print(gs)
        return 0

    from msnpip.pipeline import Pipeline, run_pipeline

    cfg = _cfg_from_args(args)
    start_stage = (
        "CONTRAST" if args.command == "from-strength" else getattr(args, "start_stage", None)
    )
    stop_stage = getattr(args, "stop_stage", None)

    try:
        if args.command == "from-strength":
            # input mode isn't required when resuming; skip the cross-field input check.
            Pipeline(cfg).run(start_stage=start_stage, stop_stage=stop_stage)
        else:
            run_pipeline(cfg, start_stage=start_stage, stop_stage=stop_stage)
    except MsnpipError as exc:
        logger.error("%s", exc)
        return 1
    print(f"Done. Output: {cfg.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
