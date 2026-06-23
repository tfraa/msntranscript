"""
Pipeline stage machine (LOAD → VALIDATE → MSN → CONTRAST → … → REPORT).
Phase 5, Task T5.2 / T5.3.

The pipeline wires the per-phase building blocks into the output tree (spec §6).
It is a linear stage machine: each stage reads from an in-memory context and
writes its artifacts to disk, so a run can be resumed from a later stage given
the persisted outputs (``start_stage`` / ``stop_stage``).
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from msnpip import engine as engine_mod
from msnpip.atlas_align import align_strength_to_atlas, engine_region_order, to_region_table
from msnpip.config import PipelineConfig
from msnpip.errors import ConfigurationError, StageError
from msnpip.io.matching import merge_features_demographics
from msnpip.io.readers import read_freesurfer_subjects, read_table
from msnpip.io.schema import detect_schema, validate_schema
from msnpip.io.writers import OutputManager
from msnpip.logging_ import phase_banner
from msnpip.msn.construct import StrengthMaps, compute_strength_maps
from msnpip.report.builder import ReportBuilder
from msnpip.stats.correlation import correlate_strength_with_demographic
from msnpip.stats.glm import regional_group_contrast
from msnpip.stats.sensitivity import covariate_exclusion_contrast

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
        self.out = OutputManager(cfg.output, engine_commit=engine_commit(), seed=cfg.engine.seed)
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

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------
    def _stage_load(self) -> None:
        io = self.cfg.io
        if io.dataframe is not None:
            df = read_table(io.dataframe, sep=io.sep, decimal=io.decimal, sheet=io.sheet)
            report = {"mode": "dataframe", "source": str(io.dataframe), "n_subjects": len(df)}
        else:
            feats = read_freesurfer_subjects(
                io.freesurfer_dir, expected_regions=self._atlas_regions()
            )
            dem = read_table(io.demographics, sep=io.sep, decimal=io.decimal, sheet=io.sheet)
            dem_id = io.id_col or "subject_id"
            df = merge_features_demographics(
                feats,
                dem,
                feat_id_col="subject_id",
                dem_id_col=dem_id,
                min_match_rate=io.min_id_match_rate,
            )
            report = {
                "mode": "freesurfer",
                "n_features": len(feats),
                "n_demographics": len(dem),
                "n_merged": len(df),
            }
        self.ctx["df"] = df
        inputs = self.out.subdir("00_inputs")
        inputs.write_table(df, "merged_data")
        inputs.write_json(report, "merge_report")
        logger.info("LOAD: %d subjects, %d columns", len(df), df.shape[1])

    def _stage_validate(self) -> None:
        df = self.ctx["df"]
        schema = detect_schema(df, expected_regions=self._atlas_regions())
        gcol = self.cfg.resolved_group_col()
        if gcol and schema.group_col is None:
            schema = replace(schema, group_col=gcol)
        predictors = tuple(self.cfg.glm.predictors)
        validate_schema(
            df,
            schema,
            predictor_cols=predictors,
            correlation_cols=tuple(self.cfg.correlation.variables),
        )
        self.ctx["schema"] = schema
        inputs = self.out.subdir("00_inputs")
        inputs.write_json(_schema_to_dict(schema), "schema")
        _write_yaml(self.cfg.output / "00_inputs" / "resolved_config.yaml", self.cfg.to_dict())
        self.out.record(self.cfg.output / "00_inputs" / "resolved_config.yaml")

    def _stage_msn(self) -> None:
        df, schema = self.ctx["df"], self.ctx["schema"]
        sm = compute_strength_maps(
            df,
            schema,
            atlas=self.cfg.engine.atlas,
            hemisphere="both",
            regions=self.cfg.engine.regions,
            sign=self.cfg.msn.strength_sign,
            metrics=tuple(self.cfg.msn.features),
        )
        self.ctx["strength_maps"] = sm
        msn_dir = self.out.subdir("01_msn")
        strength_df = pd.DataFrame(sm.strength, columns=sm.region_labels)
        strength_df.insert(0, "subject_id", sm.subject_ids)
        msn_dir.write_table(strength_df, "strength_maps")
        msn_dir.write_table(
            pd.DataFrame({"subject_id": sm.subject_ids, "global_strength": sm.global_strength}),
            "global_strength",
        )
        msn_dir.write_json({"dropped_subjects": sm.dropped_subjects}, "dropped_subjects")
        if self.cfg.save_all:
            per = msn_dir.subdir("per_subject_msn")
            for sid, mat in zip(sm.subject_ids, sm.matrix):
                per.write_array(mat, str(sid))

    def _stage_contrast(self) -> None:
        sm, df, schema = self.ctx["strength_maps"], self.ctx["df"], self.ctx["schema"]
        pairs = self._contrast_pairs()
        covariates = tuple(self.cfg.glm.predictors)
        stats_dir = self.out.subdir("02_stats")
        contrasts = []
        for case, control in pairs:
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
            tbl = pd.DataFrame({"region": res.region_labels, res.stat_type: res.regional_stat})
            stats_dir.subdir("contrasts").write_table(tbl, f"{tag}_contrast")
            contrasts.append((tag, res, cc, kk))
        self.ctx["contrasts"] = contrasts

    def _stage_correlation(self) -> None:
        variables = self.cfg.correlation.variables
        if not variables:
            logger.info("CORRELATION: no variables requested — skipping.")
            return
        sm, df, schema = self.ctx["strength_maps"], self.ctx["df"], self.ctx["schema"]
        corr_dir = self.out.subdir("02_stats").subdir("correlation")
        results = []
        for var in variables:
            res = correlate_strength_with_demographic(
                sm,
                df,
                schema,
                variable=var,
                scope=self.cfg.correlation.scope,
                within_group=self.cfg.correlation.within_group,
                method=self.cfg.correlation.method,
            )
            cols = {"r": np.atleast_1d(res.r), "p": np.atleast_1d(res.p)}
            if res.fdr is not None:
                cols["fdr"] = res.fdr
            tbl = pd.DataFrame(cols)
            if res.region_labels is not None:
                tbl.insert(0, "region", res.region_labels)
            corr_dir.write_table(tbl, f"{var}__{res.scope}")
            results.append((var, res))
        self.ctx["correlations"] = results

    def _stage_sensitivity(self) -> None:
        drops = self.cfg.glm.exclude_covariates
        if not drops or not self.cfg.glm.predictors:
            logger.info("SENSITIVITY: nothing to exclude — skipping.")
            return
        sm, df, schema = self.ctx["strength_maps"], self.ctx["df"], self.ctx["schema"]
        sens_dir = self.out.subdir("02_stats").subdir("sensitivity")
        results = []
        for tag, _res, cc, kk in self.ctx.get("contrasts", []):
            work_df, _, _ = self._resolve_contrast_df(df, schema, cc, kk)
            for drop in drops:
                sens = covariate_exclusion_contrast(
                    sm,
                    work_df,
                    schema,
                    case_label=cc,
                    control_label=kk,
                    full_covariates=self.cfg.glm.predictors,
                    drop=drop,
                    stat=self.cfg.glm.contrast_stat,
                )
                tbl = pd.DataFrame(
                    {
                        "region": sens.region_labels,
                        "full": sens.full.regional_stat,
                        "reduced": sens.reduced.regional_stat,
                    }
                )
                sens_dir.write_table(tbl, f"{tag}__drop_{drop}")
                results.append((tag, drop, sens))
        self.ctx["sensitivity"] = results

    def _stage_transcriptomics(self) -> None:
        cfg = self.cfg
        hemis = ["left", "both"] if cfg.engine.compare_hemispheres else [cfg.engine.hemisphere]
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
                base = cfg.output / "03_transcriptomics"
                if cfg.engine.compare_hemispheres:
                    base = base / f"hemi-{hemi}"
                eng_cfg = replace(cfg.engine, hemisphere=hemi)
                results = engine_mod.run_transcriptomics(vec, labels_df, eng_cfg, base, tag)
                tx.append((tag, hemi, results, vec, labels_df))
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
        fig_dir = self.cfg.output / "04_figures"
        for sub in ("distributions", "surface", "correlation"):
            (fig_dir / sub).mkdir(parents=True, exist_ok=True)
        written: list[Path] = []

        for tag, res, cc, kk in self.ctx.get("contrasts", []):
            work_df, _, _ = self._resolve_contrast_df(df, schema, cc, kk)
            try:
                fig = plot_strength_violin(sm, work_df, schema, group_labels=[cc, kk])
                p = fig_dir / "distributions" / f"{tag}_violin.png"
                fig.savefig(p)
                written.append(p)
            except Exception as exc:
                logger.warning("FIGURES: violin for %s failed: %s", tag, exc)
            try:
                vec, labels_df = align_strength_to_atlas(
                    res.regional_stat,
                    res.region_labels,
                    atlas=self.cfg.engine.atlas,
                    hemisphere=self.cfg.engine.hemisphere,
                    regions=self.cfg.engine.regions,
                )
                table = to_region_table(vec, labels_df, res.stat_type)
                p = plot_surface_with_dorsal(
                    table,
                    atlas_id=self.cfg.engine.atlas,
                    value_column=res.stat_type,
                    title=tag,
                    output_path=fig_dir / "surface" / f"{tag}_surface.png",
                )
                if p:
                    written.append(Path(p))
            except Exception as exc:
                logger.warning("FIGURES: surface for %s failed: %s", tag, exc)

        for var, res in self.ctx.get("correlations", []):
            if res.scope != "global":
                continue
            try:
                fig = plot_demographic_correlation(res)
                p = fig_dir / "correlation" / f"{var}_scatter.png"
                fig.savefig(p)
                written.append(p)
            except Exception as exc:
                logger.warning("FIGURES: scatter for %s failed: %s", var, exc)

        self.ctx["figures"] = written

    def _stage_report(self) -> None:
        report = ReportBuilder(self.cfg.output, self.cfg)
        pdf = report.build(self.ctx)
        self.ctx["report"] = pdf
        # Record every artifact under the tree (engine bundles + figures) for the manifest.
        for f in sorted(self.cfg.output.rglob("*")):
            if (
                f.is_file()
                and f.name != "manifest.json"
                and f.suffix.lower() not in (".pkl", ".pickle")
            ):
                with contextlib.suppress(ValueError):
                    self.out.record(f)
        self.out.finalize(self.cfg.to_dict())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
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
            work[gcol] = np.where(work[gcol].astype(str) == str(case), str(case), "rest")
            return work, str(case), "rest"
        return df, case, control

    def _hydrate(self, start_index: int) -> None:
        """Load persisted state needed to resume at STAGES[start_index]."""
        out = self.cfg.output
        if start_index > STAGES.index("LOAD"):
            merged = out / "00_inputs" / "merged_data.csv"
            if merged.exists():
                self.ctx["df"] = pd.read_csv(merged)
        if start_index > STAGES.index("VALIDATE") and "df" in self.ctx:
            schema = detect_schema(self.ctx["df"], expected_regions=self._atlas_regions())
            gcol = self.cfg.resolved_group_col()
            if gcol and schema.group_col is None:
                schema = replace(schema, group_col=gcol)
            self.ctx["schema"] = schema
        if start_index > STAGES.index("MSN"):
            self.ctx["strength_maps"] = self._load_strength_maps()

    def _load_strength_maps(self) -> StrengthMaps:
        path = self.cfg.output / "01_msn" / "strength_maps.csv"
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
            sign=self.cfg.msn.strength_sign,
        )


# ---------------------------------------------------------------------------
def run_pipeline(cfg: PipelineConfig, *, start_stage=None, stop_stage=None) -> dict:
    """Convenience entry point: validate then run."""
    cfg.validate()
    return Pipeline(cfg).run(start_stage=start_stage, stop_stage=stop_stage)


def engine_commit() -> str:
    return "e6a2c237fc74a0b2072a6d58efaf9d1c22cc08e1"


def _schema_to_dict(schema) -> dict:
    return {
        "id_col": schema.id_col,
        "group_col": schema.group_col,
        "age_col": schema.age_col,
        "sex_col": schema.sex_col,
        "tiv_col": schema.tiv_col,
        "site_cols": list(schema.site_cols),
        "n_feature_cols": len(schema.feature_cols),
    }


def _write_yaml(path: Path, data: dict) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
