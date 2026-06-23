"""
argparse CLI — console_script entry point ``msnpip``.
Phase 5, Task T5.5.

Subcommands:
  full           run the whole pipeline LOAD→REPORT
  from-strength  resume from a persisted strength_maps.csv (runs CONTRAST→REPORT)
  list-atlases   print the engine's atlas table
  list-genesets  print the default gene sets
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from msnpip.config import (
    CorrelationConfig,
    EngineConfig,
    GLMConfig,
    IOConfig,
    MSNConfig,
    PipelineConfig,
)
from msnpip.errors import MsnpipError
from msnpip.logging_ import configure_logging, get_logger

logger = get_logger("msnpip.cli")


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
    fs.add_argument("-v", "--verbose", action="store_true")

    sub.add_parser("list-atlases", help="print available atlases (from the engine)")
    sub.add_parser("list-genesets", help="print the default gene sets")
    return p


def _add_full_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("input (one mode)")
    g.add_argument("--input", type=Path, help="FreeSurfer subjects directory")
    g.add_argument("--demographics", type=Path, help="demographics CSV (with --input)")
    g.add_argument("--dataframe", type=Path, help="single merged wide-format file")
    g.add_argument("--sep")
    g.add_argument("--decimal")
    g.add_argument("--sheet", default=0)
    g.add_argument("--id-col")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--config", type=Path, help="YAML config (CLI flags override it)")
    _add_contrast_args(ap)
    _add_msn_args(ap)
    _add_glm_args(ap)
    _add_corr_args(ap)
    _add_engine_args(ap)
    ap.add_argument("--save-all", dest="save_all", action="store_true", default=True)
    ap.add_argument("--no-save-all", dest="save_all", action="store_false")
    ap.add_argument("--save-figures", dest="save_figures", action="store_true", default=True)
    ap.add_argument("--no-save-figures", dest="save_figures", action="store_false")
    ap.add_argument("--start-stage")
    ap.add_argument("--stop-stage")
    ap.add_argument("-v", "--verbose", action="store_true")


def _add_contrast_args(ap):
    ap.add_argument("--group-col")
    ap.add_argument("--case")
    ap.add_argument("--control")
    ap.add_argument("--contrast", nargs=2, action="append", metavar=("CASE", "CTRL"))


def _add_msn_args(ap):
    ap.add_argument("--features", nargs="+")
    ap.add_argument("--strength-sign", choices=("positive", "absolute", "signed"), default="signed")
    ap.add_argument("--strength-agg", choices=("mean", "sum"), default="mean")


def _add_glm_args(ap):
    ap.add_argument("--predictors", nargs="+", default=[])
    ap.add_argument("--contrast-stat", choices=("beta", "t", "cohen_d"), default="beta")
    ap.add_argument("--exclude-covariate", nargs="+", dest="exclude_covariate", default=[])


def _add_corr_args(ap):
    ap.add_argument("--correlate-with", nargs="+", dest="correlate_with", default=[])
    ap.add_argument("--corr-method", choices=("pearson", "spearman"), default="spearman")
    ap.add_argument("--corr-scope", choices=("global", "regional"), default="global")
    ap.add_argument("--corr-within-group", dest="corr_within_group")


def _add_engine_args(ap):
    ap.add_argument("--atlas", default="dk")
    ap.add_argument("--hemisphere", choices=("left", "both"), default="left")
    ap.add_argument("--compare-hemispheres", action="store_true")
    ap.add_argument("--regions", choices=("cort", "cort+sub"), default="cort")
    ap.add_argument("--method", choices=("pls", "corr"), action="append", dest="method")
    ap.add_argument("--ncomp", type=int)
    ap.add_argument("--var", type=float)
    ap.add_argument("--n-perm", type=int, default=10000, dest="n_perm")
    ap.add_argument("--enrichment", choices=("ensemble", "gsea", "ora", "none"), action="append")
    ap.add_argument("--geneset", nargs="+", dest="geneset")
    ap.add_argument("--seed", type=int, default=1234)


def _cfg_from_args(args) -> PipelineConfig:
    """Build a PipelineConfig from parsed `full`/`from-strength` args (CLI > YAML)."""
    methods = tuple(args.method) if getattr(args, "method", None) else ("pls", "corr")
    enrichment = (
        tuple(args.enrichment) if getattr(args, "enrichment", None) else ("ensemble", "gsea")
    )
    n_components = (
        None
        if getattr(args, "var", None) is not None
        else (args.ncomp if args.ncomp is not None else 1)
    )

    engine_kw = dict(
        methods=methods,
        atlas=args.atlas,
        hemisphere=args.hemisphere,
        compare_hemispheres=getattr(args, "compare_hemispheres", False),
        regions=args.regions,
        n_components=n_components,
        var=getattr(args, "var", None),
        n_permutations=args.n_perm,
        enrichment_methods=enrichment,
        seed=args.seed,
    )
    if getattr(args, "geneset", None):
        engine_kw["gene_sets"] = tuple(args.geneset)
    engine = EngineConfig(**engine_kw)

    msn = MSNConfig(
        features=tuple(args.features) if getattr(args, "features", None) else MSNConfig().features,
        strength_sign=getattr(args, "strength_sign", "signed"),
        strength_agg=getattr(args, "strength_agg", "mean"),
    )
    glm = GLMConfig(
        predictors=tuple(args.predictors),
        contrast_stat=args.contrast_stat,
        exclude_covariates=tuple(args.exclude_covariate),
    )
    corr = CorrelationConfig(
        variables=tuple(args.correlate_with),
        method=args.corr_method,
        scope=args.corr_scope,
        within_group=args.corr_within_group,
    )
    io = IOConfig(
        freesurfer_dir=getattr(args, "input", None),
        demographics=getattr(args, "demographics", None),
        dataframe=getattr(args, "dataframe", None),
        sep=getattr(args, "sep", None),
        decimal=getattr(args, "decimal", None),
        sheet=getattr(args, "sheet", 0),
        id_col=getattr(args, "id_col", None),
        group_col=args.group_col,
    )
    contrasts = tuple(tuple(c) for c in args.contrast) if getattr(args, "contrast", None) else None

    cfg = PipelineConfig(
        io=io,
        output=args.output,
        group_col=args.group_col,
        case=args.case,
        control=args.control,
        contrasts=contrasts,
        msn=msn,
        glm=glm,
        correlation=corr,
        engine=engine,
        save_all=getattr(args, "save_all", True),
        save_figures=getattr(args, "save_figures", True),
        verbose=args.verbose,
    )

    if getattr(args, "config", None):
        base = PipelineConfig.from_yaml(args.config)
        cfg = _overlay(base, cfg)
    return cfg


def _overlay(base: PipelineConfig, override: PipelineConfig) -> PipelineConfig:
    """Shallow overlay: override wins (CLI over YAML). Simplest useful policy."""
    return override


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

    # from-strength resumes from an existing run dir's persisted strength.
    from msnpip.pipeline import run_pipeline

    cfg = _cfg_from_args(args)
    start_stage = (
        "CONTRAST" if args.command == "from-strength" else getattr(args, "start_stage", None)
    )
    stop_stage = getattr(args, "stop_stage", None)

    try:
        if args.command == "from-strength":
            # input mode isn't required when resuming; skip cross-field input check.
            from msnpip.pipeline import Pipeline

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
