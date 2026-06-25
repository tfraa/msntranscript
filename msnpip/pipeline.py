"""
Pipeline stage machine (LOAD → VALIDATE → MSN → CONTRAST → … → REPORT).
Phase 5, Task T5.2 / T5.3.

Produces a deliberately small, curated output set (issue 7):

    <output>/
      merged_dataset.csv
      strength_maps.csv                     per-subject node strength
      mean_msn_per_group.csv                group-mean node strength per region
      case_control_difference_maps.csv      per-contrast regional contrast map
      <contrast>_pls.csv                    per-contrast PLS gene results
      <contrast>_enrichment.csv             per-contrast × geneset enrichment
      plots/                                violin, scatter, surfaces, engine plots

The engine writes its own verbose TSV/PNG bundle into a temporary ``.engine``
staging folder; we extract only the curated CSVs and the plots, then discard it.
"""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from msnpip import engine as engine_mod
from msnpip.atlas_align import align_strength_to_atlas, engine_region_order, to_region_table
from msnpip.config import PipelineConfig
from msnpip.errors import ConfigurationError, StageError
from msnpip.io.matching import merge_features_demographics
from msnpip.io.readers import (
    detect_input_kind,
    read_feature_tables,
    read_freesurfer_subjects,
    read_table,
)
from msnpip.io.schema import detect_id_column, detect_schema, validate_schema
from msnpip.logging_ import phase_banner
from msnpip.msn.construct import StrengthMaps, compute_strength_maps
from msnpip.stats.correlation import correlate_strength_with_demographic
from msnpip.stats.glm import normalize_group_value, regional_group_contrast

logger = logging.getLogger("msnpip.pipeline")

STAGES = [
    "LOAD",
    "VALIDATE",
    "MSN",
    "CONTRAST",
    "CORRELATION",
    "SENSITIVITY",
    "TRANSCRIPTOMICS",
    "FIGURES",
    "REPORT",
]


def _tag(case, control) -> str:
    return f"{case}_vs_{control}"


class Pipeline:
    """Runs the msnpip stages against a :class:`PipelineConfig`."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self.out_dir = Path(cfg.output)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.out_dir / "plots"
        self.ctx: dict = {}

    # ------------------------------------------------------------------
    def run(self, *, start_stage: str | None = None, stop_stage: str | None = None) -> dict:
        start = STAGES.index(start_stage) if start_stage else 0
        stop = STAGES.index(stop_stage) if stop_stage else len(STAGES) - 1
        if start > stop:
            raise ConfigurationError(f"start_stage {start_stage} is after stop_stage {stop_stage}.")

        if start > 0:
            self._hydrate(start)

        for i in range(start, stop + 1):
            stage = STAGES[i]
            phase_banner(i + 1, len(STAGES), stage)
            method = getattr(self, f"_stage_{stage.lower()}")
            try:
                method()
            except (StageError, ConfigurationError):
                raise
            except Exception as exc:
                raise StageError(stage, str(exc)) from exc

        return self.ctx

    def _csv(self, df: pd.DataFrame, name: str) -> Path:
        path = self.out_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        return path

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------
    def _stage_load(self) -> None:
        io = self.cfg.io
        if io.dataframe is not None:
            df = read_table(io.dataframe, sep=io.sep, decimal=io.decimal, sheet=io.sheet)
        else:
            kind = detect_input_kind(io.freesurfer_dir)
            if kind == "feature_tables":
                feats = read_feature_tables(io.freesurfer_dir, sep=io.sep, decimal=io.decimal)
            else:
                feats = read_freesurfer_subjects(
                    io.freesurfer_dir, expected_regions=self._atlas_regions()
                )
            if io.demographics is not None:
                dem = read_table(io.demographics, sep=io.sep, decimal=io.decimal, sheet=io.sheet)
                dem_id = detect_id_column(dem, io.id_col)
                df = merge_features_demographics(
                    feats,
                    dem,
                    feat_id_col="subject_id",
                    dem_id_col=dem_id,
                    min_match_rate=io.min_id_match_rate,
                )
                df = df.drop(columns=[c for c in df.columns if c.endswith("_dem")], errors="ignore")
            else:
                df = feats
        self.ctx["df"] = df
        self._csv(df, "merged_dataset")
        logger.info("LOAD: %d subjects, %d columns", len(df), df.shape[1])

    def _stage_validate(self) -> None:
        df = self.ctx["df"]
        schema = detect_schema(df, expected_regions=self._atlas_regions())
        # CLI --group-col / --id-col are authoritative: force them over detection (issue 4).
        gcol = self.cfg.resolved_group_col()
        overrides = {}
        if gcol:
            if gcol not in df.columns:
                raise StageError("VALIDATE", f"--group-col {gcol!r} not found in data columns.")
            overrides["group_col"] = gcol
        if self.cfg.io.id_col and self.cfg.io.id_col in df.columns:
            overrides["id_col"] = self.cfg.io.id_col
        if overrides:
            schema = replace(schema, **overrides)
        validate_schema(
            df,
            schema,
            predictor_cols=tuple(self.cfg.glm.predictors),
            correlation_cols=tuple(self.cfg.correlation.variables),
        )
        self.ctx["schema"] = schema

    def _stage_msn(self) -> None:
        df, schema = self.ctx["df"], self.ctx["schema"]
        sm = compute_strength_maps(
            df,
            schema,
            atlas=self.cfg.engine.atlas,
            hemisphere="both",
            regions=self.cfg.engine.regions,
            agg=self.cfg.msn.strength_agg,
            metrics=tuple(self.cfg.msn.features),
        )
        self.ctx["strength_maps"] = sm

        strength_df = pd.DataFrame(sm.strength, columns=sm.region_labels)
        strength_df.insert(0, "subject_id", sm.subject_ids)
        self._csv(strength_df, "strength_maps")
        self._csv(self._mean_strength_per_group(sm), "mean_msn_per_group")
        if sm.dropped_subjects:
            logger.info("MSN: dropped %d incomplete subject(s).", len(sm.dropped_subjects))

    def _stage_contrast(self) -> None:
        sm, df, schema = self.ctx["strength_maps"], self.ctx["df"], self.ctx["schema"]
        covariates = tuple(self.cfg.glm.predictors)
        contrasts = []
        diff = pd.DataFrame({"region": sm.region_labels})
        for case, control in self._contrast_pairs():
            work_df, cc, kk = self._resolve_contrast_df(df, schema, case, control)
            res = regional_group_contrast(
                sm,
                work_df,
                schema,
                case_label=cc,
                control_label=kk,
                covariates=covariates,
                stat=self.cfg.glm.contrast_stat,
            )
            tag = _tag(case, control)
            diff[f"{tag}_{res.stat_type}"] = res.regional_stat
            contrasts.append((tag, res, cc, kk))
        self._csv(diff, "case_control_difference_maps")
        self.ctx["contrasts"] = contrasts

    def _stage_correlation(self) -> None:
        # Correlation results feed the scatter plots only (no separate CSV).
        variables = self.cfg.correlation.variables
        if not variables:
            return
        sm, df, schema = self.ctx["strength_maps"], self.ctx["df"], self.ctx["schema"]
        results = []
        for var in variables:
            results.append(
                (
                    var,
                    correlate_strength_with_demographic(
                        sm,
                        df,
                        schema,
                        variable=var,
                        scope=self.cfg.correlation.scope,
                        within_group=self.cfg.correlation.within_group,
                        method=self.cfg.correlation.method,
                    ),
                )
            )
        self.ctx["correlations"] = results

    def _stage_sensitivity(self) -> None:
        # Covariate-exclusion sensitivity is not part of the curated output set.
        logger.info("SENSITIVITY: skipped (not in curated outputs).")

    def _stage_transcriptomics(self) -> None:
        cfg = self.cfg
        hemis = ["left", "both"] if cfg.engine.compare_hemispheres else [cfg.engine.hemisphere]
        staging = self.out_dir / ".engine"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        tx = []
        for tag, res, _cc, _kk in self.ctx["contrasts"]:
            for hemi in hemis:
                vec, labels_df = align_strength_to_atlas(
                    res.regional_stat,
                    res.region_labels,
                    atlas=cfg.engine.atlas,
                    hemisphere=hemi,
                    regions=cfg.engine.regions,
                )
                eng_cfg = replace(cfg.engine, hemisphere=hemi)
                out_tag = f"{tag}_hemi-{hemi}" if cfg.engine.compare_hemispheres else tag
                results = engine_mod.run_transcriptomics(vec, labels_df, eng_cfg, staging, out_tag)
                self._curate_engine_bundle(out_tag, staging / out_tag)
                tx.append((tag, hemi, results))
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        self.ctx["transcriptomics"] = tx

    def _stage_figures(self) -> None:
        if not self.cfg.save_figures:
            logger.info("FIGURES: save_figures disabled — skipping.")
            return
        import matplotlib

        matplotlib.use("Agg")
        from msnpip.viz.distributions import plot_strength_violin
        from msnpip.viz.scatter import plot_demographic_correlation
        from msnpip.viz.surface_extra import plot_surface_with_dorsal

        sm, df, schema = self.ctx["strength_maps"], self.ctx["df"], self.ctx["schema"]
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        for tag, res, cc, kk in self.ctx.get("contrasts", []):
            work_df, _, _ = self._resolve_contrast_df(df, schema, cc, kk)
            case_lbl, ctrl_lbl = tag.split("_vs_", 1)
            try:
                fig = plot_strength_violin(sm, work_df, schema, group_labels=[cc, kk])
                fig.savefig(self.plots_dir / f"{tag}_violin.png")
            except Exception as exc:
                logger.warning("FIGURES: violin for %s failed: %s", tag, exc)
            # Surface maps: both hemispheres, on inflated AND pial surfaces.
            try:
                vec, labels_df = align_strength_to_atlas(
                    res.regional_stat,
                    res.region_labels,
                    atlas=self.cfg.engine.atlas,
                    hemisphere="both",
                    regions=self.cfg.engine.regions,
                )
                table = to_region_table(vec, labels_df, res.stat_type)
                title = f"{case_lbl} vs {ctrl_lbl}: node-strength {res.stat_type} contrast"
                for mesh_kind in ("inflated", "pial"):
                    subtitle = (
                        f"MSN node-strength group contrast ({res.stat_type}) · "
                        f"{self.cfg.engine.atlas} atlas · {mesh_kind} surface · both hemispheres"
                    )
                    plot_surface_with_dorsal(
                        table,
                        atlas_id=self.cfg.engine.atlas,
                        value_column=res.stat_type,
                        title=title,
                        output_path=self.plots_dir / f"{tag}_surface_{mesh_kind}.png",
                        mesh_kind=mesh_kind,
                        subtitle=subtitle,
                    )
            except Exception as exc:
                logger.warning("FIGURES: surface for %s failed: %s", tag, exc)

        for var, res in self.ctx.get("correlations", []):
            if res.scope != "global":
                continue
            try:
                fig = plot_demographic_correlation(res)
                fig.savefig(self.plots_dir / f"{var}_scatter.png")
            except Exception as exc:
                logger.warning("FIGURES: scatter for %s failed: %s", var, exc)

    def _stage_report(self) -> None:
        from msnpip.report.builder import ReportBuilder

        produced = sorted(p.name for p in self.out_dir.glob("*.csv"))
        n_plots = len(list(self.plots_dir.glob("*.png"))) if self.plots_dir.exists() else 0
        pdf = ReportBuilder(self.out_dir, self.cfg).build(self.ctx)
        self.ctx["report"] = pdf
        logger.info(
            "REPORT: %d CSV(s) + %d plot(s) + %s in %s",
            len(produced),
            n_plots,
            pdf.name if pdf else "no report",
            self.out_dir,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _mean_strength_per_group(self, sm: StrengthMaps) -> pd.DataFrame:
        df, schema = self.ctx["df"], self.ctx["schema"]
        out = pd.DataFrame({"region": sm.region_labels})
        gcol = schema.group_col
        if gcol and gcol in df.columns:
            aligned = df.set_index(df[schema.id_col].astype(str)).loc[sm.subject_ids]
            groups = aligned[gcol].map(normalize_group_value).to_numpy()
            for g in pd.unique(groups):
                out[f"mean_strength_{g}"] = sm.strength[groups == g].mean(axis=0)
        else:
            out["mean_strength_all"] = sm.strength.mean(axis=0)
        return out

    def _curate_engine_bundle(self, tag: str, bundle_dir: Path) -> None:
        """Extract curated PLS + enrichment CSVs and copy plots from a bundle."""
        if not bundle_dir.exists():
            return
        # PLS gene-level results (one row block per component).
        pls_frames = []
        for f in sorted(bundle_dir.glob("pls/pls_component_*.tsv")):
            comp = re.search(r"component_(\d+)", f.name)
            tbl = pd.read_csv(f, sep="\t")
            tbl.insert(0, "component", int(comp.group(1)) if comp else 0)
            pls_frames.append(tbl)
        if pls_frames:
            self._csv(pd.concat(pls_frames, ignore_index=True), f"{tag}_pls")

        # Enrichment results (ensemble/gsea/ora), tagged by method and geneset.
        enr_frames = []
        for f in sorted(bundle_dir.rglob("*_results*.tsv")):
            core = f.stem[: -len("_results")] if f.stem.endswith("_results") else f.stem
            method, _, geneset = core.partition("_")
            tbl = pd.read_csv(f, sep="\t")
            tbl.insert(0, "geneset", geneset)
            tbl.insert(0, "enrichment", method)
            enr_frames.append(tbl)
        if enr_frames:
            self._csv(pd.concat(enr_frames, ignore_index=True), f"{tag}_enrichment")

        # Engine plots (PLS variance, enrichment dotplots, etc.).
        for png in sorted(bundle_dir.rglob("*.png")):
            shutil.copy(png, self.plots_dir / f"{tag}_{png.stem}.png")

    def _atlas_regions(self) -> list[str] | None:
        try:
            labels = engine_region_order(self.cfg.engine.atlas, "both", self.cfg.engine.regions)
            return sorted(set(labels["label"].tolist()))
        except Exception:
            return None

    def _contrast_pairs(self):
        if self.cfg.contrasts:
            return list(self.cfg.contrasts)
        if self.cfg.case is not None:
            control = self.cfg.control if self.cfg.control is not None else "rest"
            return [(self.cfg.case, control)]
        gcol = self.ctx["schema"].group_col
        if gcol is None:
            raise StageError("CONTRAST", "No group column and no contrast specified.")
        groups = list(pd.unique(self.ctx["df"][gcol].astype(str)))
        if len(groups) != 2:
            raise StageError(
                "CONTRAST", f"{len(groups)} groups found; specify --case/--control or --contrast."
            )
        return [(groups[0], groups[1])]

    def _resolve_contrast_df(self, df, schema, case, control):
        """Return (df, case_label, control_label), synthesising a 'rest' arm if needed."""
        gcol = schema.group_col
        if control == "rest":
            work = df.copy()
            case_norm = normalize_group_value(case)
            is_case = work[gcol].map(normalize_group_value) == case_norm
            work[gcol] = np.where(is_case, case_norm, "rest")
            return work, case_norm, "rest"
        return df, case, control

    def _hydrate(self, start_index: int) -> None:
        """Load persisted state needed to resume at STAGES[start_index]."""
        if start_index > STAGES.index("LOAD"):
            merged = self.out_dir / "merged_dataset.csv"
            if merged.exists():
                self.ctx["df"] = pd.read_csv(merged)
        if start_index > STAGES.index("VALIDATE") and "df" in self.ctx:
            schema = detect_schema(self.ctx["df"], expected_regions=self._atlas_regions())
            gcol = self.cfg.resolved_group_col()
            if gcol and gcol in self.ctx["df"].columns:
                schema = replace(schema, group_col=gcol)
            self.ctx["schema"] = schema
        if start_index > STAGES.index("MSN"):
            self.ctx["strength_maps"] = self._load_strength_maps()

    def _load_strength_maps(self) -> StrengthMaps:
        path = self.out_dir / "strength_maps.csv"
        if not path.exists():
            raise StageError("RESUME", f"Cannot resume: {path} missing.")
        df = pd.read_csv(path)
        region_labels = [c for c in df.columns if c != "subject_id"]
        strength = df[region_labels].to_numpy(dtype=float)
        return StrengthMaps(
            matrix=np.empty((len(df), len(region_labels), 0)),
            strength=strength,
            subject_ids=df["subject_id"].astype(str).tolist(),
            region_labels=region_labels,
            atlas=self.cfg.engine.atlas,
            features=list(self.cfg.msn.features),
            global_strength=np.nanmean(strength, axis=1),
            hemisphere="both",
            regions=self.cfg.engine.regions,
            agg=self.cfg.msn.strength_agg,
        )


# ---------------------------------------------------------------------------
def run_pipeline(cfg: PipelineConfig, *, start_stage=None, stop_stage=None) -> dict:
    """Convenience entry point: validate then run."""
    cfg.validate()
    return Pipeline(cfg).run(start_stage=start_stage, stop_stage=stop_stage)
