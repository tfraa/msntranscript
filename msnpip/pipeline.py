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
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Literal

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

# FDR threshold for flagging significant regions in the report (highlights +
# significant-only surface maps). Matches the manuscript's FDR<0.05 convention.
SIG_ALPHA = 0.05

STAGES = [
    "LOAD",
    "VALIDATE",
    "MSN",
    "CONTRAST",
    "CORRELATION",
    "TRANSCRIPTOMICS",
    "FIGURES",
    "REPORT",
]


def _tag(case, control) -> str:
    # A pooled case arm is a collection of group labels → "1+2+3".
    if isinstance(case, (tuple, list, set, frozenset)):
        case = "+".join(str(c) for c in sorted(case, key=str))
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
            # VALIDATE guarantees exactly one input mode; reaching here means a
            # FreeSurfer directory was given.
            assert io.freesurfer_dir is not None
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
            # dataclasses.replace with a dynamic **dict trips mypy's overload check.
            schema = replace(schema, **overrides)  # type: ignore[arg-type]
        validate_schema(
            df,
            schema,
            predictor_cols=tuple(self.cfg.glm.predictors),
            correlation_cols=tuple(self.cfg.correlation.variables),
        )
        self.ctx["schema"] = schema

    def _stage_msn(self) -> None:
        df, schema = self.ctx["df"], self.ctx["schema"]
        # Scope the whole run to the groups named in the requested contrasts: any
        # other group is excluded from the MSN and everything downstream, so the
        # pipeline only works with what was specified.
        groups = self._referenced_groups()
        if groups is not None and schema.group_col in df.columns:
            gnorm = df[schema.group_col].map(normalize_group_value)
            n_before = len(df)
            df = df[gnorm.isin(groups)].copy()
            self.ctx["df"] = df
            logger.info(
                "SCOPE: restricted to group(s) %s — %d/%d subjects kept.",
                sorted(groups),
                len(df),
                n_before,
            )
            if df.empty:
                raise StageError(
                    "MSN", f"No subjects left after restricting to groups {sorted(groups)}."
                )
        sm = compute_strength_maps(
            df,
            schema,
            atlas=self.cfg.engine.atlas,
            hemisphere="both",
            regions=self.cfg.engine.regions,
            agg=self.cfg.msn.strength_agg,
            similarity=self.cfg.msn.similarity,
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
            # Full per-region statistics (beta, t, Cohen's d, p, FDR) for the report.
            self._csv(res.stats_table(), f"{tag}_region_stats")
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

    def _stage_transcriptomics(self) -> None:
        cfg = self.cfg
        hemis: list[Literal["left", "both"]] = (
            ["left", "both"] if cfg.engine.compare_hemispheres else [cfg.engine.hemisphere]
        )
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
                # Resolved (actually-used) spatial null per method, so a degraded
                # (non-spin) fallback is visible in the curated CSVs, not just logs.
                null_by_method = {
                    m: getattr(getattr(r, "metadata", None), "null_method", None)
                    for m, r in results.items()
                }
                self._curate_engine_bundle(out_tag, staging / out_tag, null_by_method)
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
        from msnpip.viz.regional import (
            plot_enrichment_bars,
            plot_hemisphere_bars,
            plot_msn_matrix,
        )
        from msnpip.viz.scatter import plot_demographic_correlation
        from msnpip.viz.surface_extra import plot_surface_with_dorsal

        sm, df, schema = self.ctx["strength_maps"], self.ctx["df"], self.ctx["schema"]
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Overview violin: node-strength distribution across ALL in-scope groups
        # together (descriptive landscape). No group_labels → every group present
        # in the scoped cohort; >2 groups draws no pairwise significance bracket.
        import matplotlib.pyplot as _plt

        try:
            fig = plot_strength_violin(sm, df, schema)
            fig.savefig(self.plots_dir / "overview_violin.png")
            _plt.close(fig)
        except Exception as exc:
            logger.warning("FIGURES: overview violin failed: %s", exc)

        for tag, res, cc, kk in self.ctx.get("contrasts", []):
            work_df, _, _ = self._resolve_contrast_df(df, schema, cc, kk)
            case_lbl, ctrl_lbl = tag.split("_vs_", 1)
            try:
                fig = plot_strength_violin(sm, work_df, schema, group_labels=[cc, kk])
                fig.savefig(self.plots_dir / f"{tag}_violin.png")
                _plt.close(fig)
            except Exception as exc:
                logger.warning("FIGURES: violin for %s failed: %s", tag, exc)
            # Per-region violins for the FDR-significant regions (fallback: top-5 by
            # |t|), with the covariate-adjusted GLM FDR in the bracket so the figure
            # matches the reported inference (not a fresh unadjusted test).
            try:
                fdr = res.pvalue_fdr
                stat = res.tvalue if res.tvalue is not None else res.regional_stat
                labels = list(res.region_labels)
                if fdr is not None:
                    idx = sorted(
                        np.where(np.asarray(fdr, float) < SIG_ALPHA)[0], key=lambda i: float(fdr[i])
                    )
                else:
                    idx = []
                if not idx and stat is not None:  # nothing significant → show top-5 by |stat|
                    idx = list(np.argsort(-np.abs(np.asarray(stat, float)))[:5])
                for i in idx[:12]:  # cap to avoid flooding the report
                    reg = labels[i]
                    pv = float(fdr[i]) if fdr is not None else None
                    f = plot_strength_violin(
                        sm,
                        work_df,
                        schema,
                        region=reg,
                        group_labels=[cc, kk],
                        pvalue=pv,
                        pvalue_label="FDR",
                    )
                    safe = re.sub(r"[^A-Za-z0-9._-]", "_", str(reg))
                    f.savefig(self.plots_dir / f"{tag}_region-{safe}_violin.png")
                    _plt.close(f)
            except Exception as exc:
                logger.warning("FIGURES: per-region violins for %s failed: %s", tag, exc)
            # Per-region t-value bars, split by hemisphere (default region order)
            # with FDR significance asterisks.
            try:
                tvals = res.tvalue if res.tvalue is not None else res.regional_stat
                plot_hemisphere_bars(
                    tvals,
                    res.region_labels,
                    value_label="t-value",
                    title=f"{case_lbl} vs {ctrl_lbl}: node-strength t-values",
                    subtitle=f"case-control contrast · {self.cfg.engine.atlas} atlas",
                    output_path=self.plots_dir / f"{tag}_tvalue_bars.png",
                    color_mode="sign",
                    significance=res.pvalue_fdr,
                    alpha=SIG_ALPHA,
                    sig_label="FDR",
                )
            except Exception as exc:
                logger.warning("FIGURES: contrast bars for %s failed: %s", tag, exc)
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
            # Significant-only surface: non-FDR-significant regions blanked (NaN →
            # neutral) so only the regions that survive FDR<alpha carry colour.
            try:
                fdr = res.pvalue_fdr
                if fdr is not None:
                    masked = np.where(fdr < SIG_ALPHA, res.regional_stat, np.nan)
                    if np.isfinite(masked).any():
                        vec, labels_df = align_strength_to_atlas(
                            masked,
                            res.region_labels,
                            atlas=self.cfg.engine.atlas,
                            hemisphere="both",
                            regions=self.cfg.engine.regions,
                        )
                        table = to_region_table(vec, labels_df, res.stat_type)
                        plot_surface_with_dorsal(
                            table,
                            atlas_id=self.cfg.engine.atlas,
                            value_column=res.stat_type,
                            title=f"{case_lbl} vs {ctrl_lbl}: FDR-significant regions",
                            output_path=self.plots_dir / f"{tag}_surface_significant.png",
                            mesh_kind="inflated",
                            subtitle=(
                                f"node-strength {res.stat_type} · regions with FDR < {SIG_ALPHA} · "
                                f"{self.cfg.engine.atlas} atlas · both hemispheres"
                            ),
                        )
            except Exception as exc:
                logger.warning("FIGURES: significant surface for %s failed: %s", tag, exc)
            # Gene-set enrichment bars (one per gene set × backend) from the
            # curated enrichment table.
            try:
                self._enrichment_figures(tag, plot_enrichment_bars)
            except Exception as exc:
                logger.warning("FIGURES: enrichment bars for %s failed: %s", tag, exc)

        for var, res in self.ctx.get("correlations", []):
            if res.scope != "global":
                continue
            try:
                fig = plot_demographic_correlation(res)
                fig.savefig(self.plots_dir / f"{var}_scatter.png")
            except Exception as exc:
                logger.warning("FIGURES: scatter for %s failed: %s", var, exc)

        # Per-group mean node-strength maps: brain surface (viridis) + ranked bars.
        for group, idx in self._group_indices(sm).items():
            if idx.size == 0:
                continue
            mean_strength = sm.strength[idx].mean(axis=0)
            try:
                vec, labels_df = align_strength_to_atlas(
                    mean_strength,
                    list(sm.region_labels),
                    atlas=self.cfg.engine.atlas,
                    hemisphere="both",
                    regions=self.cfg.engine.regions,
                )
                table = to_region_table(vec, labels_df, "strength")
                plot_surface_with_dorsal(
                    table,
                    atlas_id=self.cfg.engine.atlas,
                    value_column="strength",
                    title=f"Mean node strength — group {group}",
                    output_path=self.plots_dir / f"{group}_strength_surface.png",
                    mesh_kind="inflated",
                    subtitle=(
                        f"MSN node strength · {self.cfg.engine.atlas} atlas · "
                        f"inflated surface · both hemispheres"
                    ),
                    diverging=False,
                    cmap_name="viridis",
                )
            except Exception as exc:
                logger.warning("FIGURES: strength surface for group %s failed: %s", group, exc)

        # Per-group mean similarity-matrix heatmaps (needs the per-subject matrices).
        if getattr(sm, "matrix", None) is not None and sm.matrix.ndim == 3 and sm.matrix.shape[2]:
            for group, idx in self._group_indices(sm).items():
                if idx.size == 0:
                    continue
                try:
                    with warnings.catch_warnings():  # NaN diagonal → benign empty-slice
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        mean_mat = np.nanmean(sm.matrix[idx], axis=0)
                    plot_msn_matrix(
                        mean_mat,
                        sm.region_labels,
                        title=f"Mean morphometric similarity — group {group}",
                        subtitle=f"{self.cfg.engine.atlas} atlas · {idx.size} subjects",
                        output_path=self.plots_dir / f"{group}_mean_msn_matrix.png",
                    )
                except Exception as exc:
                    logger.warning("FIGURES: mean MSN matrix for group %s failed: %s", group, exc)

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
    def _group_indices(self, sm: StrengthMaps) -> dict:
        """Map each group label → array of row indices into ``sm`` (subject order)."""
        df, schema = self.ctx["df"], self.ctx["schema"]
        gcol = schema.group_col
        if not (gcol and gcol in df.columns):
            return {"all": np.arange(len(sm.subject_ids))}
        aligned = df.set_index(df[schema.id_col].astype(str)).loc[sm.subject_ids]
        groups = aligned[gcol].map(normalize_group_value).to_numpy()
        return {g: np.flatnonzero(groups == g) for g in pd.unique(groups)}

    def _mean_strength_per_group(self, sm: StrengthMaps) -> pd.DataFrame:
        out = pd.DataFrame({"region": sm.region_labels})
        for group, idx in self._group_indices(sm).items():
            out[f"mean_strength_{group}"] = sm.strength[idx].mean(axis=0)
        return out

    def _curate_engine_bundle(
        self, tag: str, bundle_dir: Path, null_by_method: dict | None = None
    ) -> None:
        """Extract curated PLS + enrichment CSVs and copy plots from a bundle.

        ``null_by_method`` maps each method to the spatial null actually resolved
        for it (e.g. ``{"pls": "vasa"}``); it is stamped onto the curated tables so a
        degraded (non-spin) fallback is visible without reading the logs or report.
        """
        if not bundle_dir.exists():
            return
        null_by_method = null_by_method or {}
        # PLS gene-level results (one row block per component).
        pls_frames = []
        for f in sorted(bundle_dir.glob("pls/pls_component_*.tsv")):
            comp = re.search(r"component_(\d+)", f.name)
            tbl = pd.read_csv(f, sep="\t")
            tbl.insert(0, "component", int(comp.group(1)) if comp else 0)
            pls_frames.append(tbl)
        if pls_frames:
            df = pd.concat(pls_frames, ignore_index=True)
            df["null_method"] = null_by_method.get("pls")
            self._csv(df, f"{tag}_pls")

        # PLS component summary (explained variance, cumulative variance, p-value).
        summary = bundle_dir / "pls" / "pls_summary.tsv"
        if summary.exists():
            df = pd.read_csv(summary, sep="\t")
            df["null_method"] = null_by_method.get("pls")
            self._csv(df, f"{tag}_pls_summary")

        # Enrichment results. The PLS path writes one folder per gene set:
        # ``<method>/enrichment/<geneset>/<backend>_pls<N>_*.tsv``.  Legacy
        # single-call bundles write ``<method>/<backend>_pls<N>_results.tsv``.
        enr_files = {
            *bundle_dir.rglob("*_results*.tsv"),
            *bundle_dir.rglob("ora_*.tsv"),
        }
        enr_frames = []
        for f in sorted(enr_files):
            core = f.stem[: -len("_results")] if f.stem.endswith("_results") else f.stem
            backend = core.partition("_")[0]
            comp = re.search(r"pls(\d+)", core)
            if f.parent.parent.name == "enrichment":  # per-gene-set layout
                geneset = f.parent.name
                method = f.parent.parent.parent.name
            else:  # legacy single-call layout
                geneset = core.partition("_")[2]
                method = f.parent.name if f.parent != bundle_dir else ""
            tbl = pd.read_csv(f, sep="\t")
            tbl.insert(0, "component", int(comp.group(1)) if comp else 1)
            tbl.insert(0, "geneset", geneset)
            tbl.insert(0, "enrichment", backend)
            tbl.insert(0, "method", method)
            tbl["null_method"] = null_by_method.get(method)
            enr_frames.append(tbl)
        if enr_frames:
            self._csv(pd.concat(enr_frames, ignore_index=True), f"{tag}_enrichment")

        # Engine plots (PLS variance, enrichment dotplots, etc.).
        for png in sorted(bundle_dir.rglob("*.png")):
            shutil.copy(png, self.plots_dir / f"{tag}_{png.stem}.png")

    # Enrichment score / significance columns per backend (ensemble vs gsea).
    _ENR_SCORE_COLS = ("nes", "z_score", "category_score", "es")
    _ENR_SIG_COLS = ("fdr", "p_val", "p")

    def _enrichment_figures(self, tag: str, plot_enrichment_bars) -> None:
        """Make a diverging enrichment bar plot per gene set × backend."""
        for path in sorted(self.out_dir.glob(f"{tag}*_enrichment.csv")):
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if df.empty or "Term" not in df.columns:
                continue
            group_cols = [c for c in ("enrichment", "geneset") if c in df.columns]
            stem = path.stem[: -len("_enrichment")]
            groups = df.groupby(group_cols) if group_cols else [((), df)]
            for key, sub in groups:
                key = key if isinstance(key, tuple) else (key,)
                backend = str(key[0]) if "enrichment" in group_cols else "enrichment"
                geneset = str(key[-1]) if "geneset" in group_cols else "geneset"
                # Pick the score/sig column per backend (gsea→nes, ensemble→z_score):
                # choose the first candidate with finite values in THIS group.
                score_col = next(
                    (c for c in self._ENR_SCORE_COLS if c in sub.columns and sub[c].notna().any()),
                    None,
                )
                if score_col is None:
                    continue
                sig_col = next(
                    (c for c in self._ENR_SIG_COLS if c in sub.columns and sub[c].notna().any()),
                    None,
                )
                score_label = "NES" if score_col == "nes" else score_col
                plot_enrichment_bars(
                    sub["Term"].tolist(),
                    sub[score_col].tolist(),
                    score_label=f"{score_label} ({backend})",
                    title=f"Gene-set enrichment: {geneset}",
                    subtitle=f"{backend} · {geneset}",
                    output_path=self.plots_dir / f"{stem}_{backend}_{geneset}_enrichment.png",
                    significance=sub[sig_col].tolist() if sig_col else None,
                    sig_label=sig_col.upper() if sig_col else "FDR",
                )

    def _atlas_regions(self) -> list[str] | None:
        try:
            labels = engine_region_order(self.cfg.engine.atlas, "both", self.cfg.engine.regions)
            return sorted(set(labels["label"].tolist()))
        except Exception:
            return None

    def _referenced_groups(self):
        """Normalized group labels referenced by the requested contrasts.

        Returns ``None`` when the run should keep all subjects (no explicit
        contrast, or a ``'rest'`` control arm that needs every other subject).
        Otherwise only these groups are used anywhere in the pipeline.
        """
        if self.cfg.contrasts:
            pairs = list(self.cfg.contrasts)
        elif self.cfg.case is not None:
            control = self.cfg.control if self.cfg.control is not None else "rest"
            pairs = [(self.cfg.case, control)]
        else:
            return None
        labels: set[str] = set()
        for case, control in pairs:
            if str(control) == "rest":
                return None
            labels.add(normalize_group_value(case))
            labels.add(normalize_group_value(control))
        return labels

    def _contrast_pairs(self):
        if self.cfg.contrasts:
            pairs = list(self.cfg.contrasts)
        elif self.cfg.case is not None:
            control = self.cfg.control if self.cfg.control is not None else "rest"
            pairs = [(self.cfg.case, control)]
        else:
            gcol = self.ctx["schema"].group_col
            if gcol is None:
                raise StageError("CONTRAST", "No group column and no contrast specified.")
            groups = list(pd.unique(self.ctx["df"][gcol].astype(str)))
            if len(groups) != 2:
                raise StageError(
                    "CONTRAST",
                    f"{len(groups)} groups found; specify --case/--control or --contrast.",
                )
            pairs = [(groups[0], groups[1])]
        return pairs + self._pooled_pairs(pairs)

    def _pooled_pairs(self, pairs):
        """Supplementary pooled contrasts: union the specified cases per control.

        Runs *alongside* the per-contrast analyses (which stay primary) when
        ``engine.pool_cases`` is set — e.g. contrasts 1v0/2v0/3v0 add {1,2,3}v0.
        Only controls with more than one distinct case are pooled.
        """
        if not getattr(self.cfg.engine, "pool_cases", False):
            return []
        from collections import OrderedDict

        by_control: OrderedDict[str, list] = OrderedDict()
        for case, control in pairs:
            if isinstance(case, (tuple, list, set, frozenset)) or str(control) == "rest":
                continue  # skip already-pooled or 'rest' arms
            by_control.setdefault(normalize_group_value(control), []).append(
                normalize_group_value(case)
            )
        pooled = []
        for control_norm, cases in by_control.items():
            uniq = sorted(set(cases), key=str)
            if len(uniq) > 1:
                pooled.append((tuple(uniq), control_norm))
        return pooled

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
