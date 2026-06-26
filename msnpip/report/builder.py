"""
ReportBuilder — assemble an ordered, self-describing ``report.pdf``.

The report walks the analysis in the order it is performed and interleaves
written narrative, tables and figures:

    1. Cover (run configuration + resolved spatial null)
    2. Dataset description (cohort, groups, covariates, morphometric metrics)
    3. MSN construction (method) + per-group mean similarity matrices (viridis)
    4. Node strength per group (brain surfaces + per-group top-5/bottom-5 table)
    5. For each contrast:
         a. Case-control t-value bars + brain surfaces (2×2 lateral/medial grid)
         b. Significant regions, in writing, with beta / t / Cohen's d / p / FDR
         c. Per-region statistics tables, one page per hemisphere (sig FDR bold)
         d. Brain surfaces of the FDR-significant regions only
         e. PLS parameters + top 20 positive / 20 negative PLS genes
         f. Enrichment: bar plots + one term table per gene set (× backend)

All pages are A4 portrait.  Figures are produced by the FIGURES stage and read
back from ``plots/``;
tables are read from the curated CSVs and from the in-memory contrast results.
Every section is defensive: missing data degrades to a short note rather than
failing the report.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from msnpip.stats.glm import normalize_group_value

logger = logging.getLogger("msnpip.report.builder")

# Page geometry (inches).
A4_PORTRAIT = (8.27, 11.69)
A4_LANDSCAPE = (11.69, 8.27)

# Palette for headings / rules.
_INK = "#1f2933"
_MUTED = "#52606d"
_ACCENT = "#2166ac"
_RULE = "#cbd2d9"
_HEAD_BG = "#2166ac"
_ROW_ALT = "#eef2f7"

SIG_ALPHA = 0.05


class ReportBuilder:
    """Assemble ``<output>/report.pdf`` from curated CSVs, plots and ctx."""

    def __init__(self, output_dir, cfg) -> None:
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / "plots"
        self.cfg = cfg

    # ------------------------------------------------------------------
    def build(self, ctx: dict) -> Path | None:
        pdf_path = self.output_dir / "report.pdf"
        # Disable tight-bbox cropping so every page keeps its full A4 portrait size
        # (the plot theme sets savefig.bbox='tight', which would crop pages with
        # wide images to landscape). Restored afterwards.
        prev_bbox = matplotlib.rcParams.get("savefig.bbox")
        matplotlib.rcParams["savefig.bbox"] = None
        try:
            with PdfPages(pdf_path) as pdf:
                self._cover_page(pdf, ctx)
                self._dataset_page(pdf, ctx)
                self._msn_section(pdf, ctx)
                self._strength_section(pdf, ctx)
                for tag, res, cc, kk in ctx.get("contrasts", []):
                    self._contrast_section(pdf, ctx, tag, res, cc, kk)
        finally:
            matplotlib.rcParams["savefig.bbox"] = prev_bbox
        logger.info("REPORT: wrote %s", pdf_path)
        return pdf_path

    # ==================================================================
    # Low-level page primitives
    # ==================================================================
    def _open_page(self, pdf, *, landscape: bool = False):
        # Every report page is A4 portrait (the ``landscape`` flag is kept for
        # call-site compatibility but intentionally ignored).
        fig = plt.figure(figsize=A4_PORTRAIT)
        fig.patch.set_facecolor("white")
        return fig

    def _close_page(self, pdf, fig) -> None:
        # savefig.bbox is forced off in build() so pages keep full A4 portrait.
        pdf.savefig(fig)
        plt.close(fig)

    def _heading(self, fig, title: str, *, subtitle: str | None = None, kicker: str | None = None):
        """Draw a section heading band; return the y below which content starts."""
        if kicker:
            fig.text(0.07, 0.955, kicker.upper(), fontsize=9, color=_ACCENT, fontweight="bold")
        fig.text(0.07, 0.93, title, fontsize=18, color=_INK, fontweight="bold", va="top")
        y = 0.90
        if subtitle:
            fig.text(0.07, 0.905, subtitle, fontsize=10.5, color=_MUTED, va="top")
            y = 0.88
        line = plt.Line2D([0.07, 0.93], [y, y], color=_RULE, linewidth=1.0)
        fig.add_artist(line)
        return y - 0.02

    def _paragraphs(self, fig, blocks, *, top: float, x: float = 0.07, width: float = 0.86):
        """Render a list of text blocks top-down.

        Each block is ``(text, kind)`` where *kind* is ``"h"`` (sub-heading),
        ``"p"`` (paragraph), ``"li"`` (bullet) or ``"sp"`` (spacer).
        """
        y = top
        for text, kind in blocks:
            if kind == "sp":
                y -= 0.018
                continue
            if kind == "h":
                fig.text(x, y, text, fontsize=12, color=_ACCENT, fontweight="bold", va="top")
                y -= 0.034
                continue
            prefix = "•  " if kind == "li" else ""
            indent = x + (0.025 if kind == "li" else 0.0)
            wrapped = self._wrap(prefix + text, width=92 if kind != "li" else 88)
            for i, line in enumerate(wrapped):
                fig.text(
                    indent if i == 0 else indent + 0.018,
                    y,
                    line,
                    fontsize=10.5,
                    color=_INK,
                    va="top",
                )
                y -= 0.027
            y -= 0.006
        return y

    @staticmethod
    def _wrap(text: str, width: int = 92) -> list[str]:
        import textwrap

        return textwrap.wrap(text, width=width) or [""]

    def _figure_page(
        self,
        pdf,
        png: Path,
        *,
        title: str,
        caption: str | None = None,
        kicker: str | None = None,
        full: bool = False,
    ) -> bool:
        if not png or not Path(png).exists():
            return False
        try:
            img = mpimg.imread(png)
        except Exception as exc:  # pragma: no cover - corrupt image
            logger.warning("REPORT: could not read %s: %s", png, exc)
            return False
        fig = self._open_page(pdf, landscape=True)
        if kicker:
            fig.text(0.05, 0.975, kicker.upper(), fontsize=9, color=_ACCENT, fontweight="bold")
        fig.text(
            0.05,
            0.955 if full else 0.93,
            title,
            fontsize=14,
            color=_INK,
            fontweight="bold",
            va="top",
        )
        # ``full`` maximises the image area (used for the dense similarity matrix
        # so its per-region labels stay legible on the page).
        box = [0.02, 0.03, 0.96, 0.90] if full else [0.04, 0.07, 0.92, 0.80]
        ax = fig.add_axes(box)
        ax.axis("off")
        ax.imshow(img)
        if caption and not full:
            fig.text(0.05, 0.045, caption, fontsize=8.5, color=_MUTED, va="bottom")
        self._close_page(pdf, fig)
        return True

    def _table_page(
        self,
        pdf,
        *,
        title: str,
        df: pd.DataFrame,
        kicker: str | None = None,
        caption: str | None = None,
        intro=None,
        max_rows: int = 34,
        bold_cells: set | None = None,
    ) -> None:
        """Render a DataFrame as a styled table (paginated if long).

        ``bold_cells`` is an optional set of ``(row_index, column_name)`` pairs
        (row index into the displayed rows) whose cell text is drawn bold.
        """
        rows = df.reset_index(drop=True)
        truncated = len(rows) > max_rows
        if truncated:
            rows = rows.head(max_rows)
        fig = self._open_page(pdf, landscape=rows.shape[1] > 5)
        top = self._heading(fig, title, kicker=kicker)
        if intro:
            top = self._paragraphs(fig, intro, top=top - 0.005)
        cap = caption or ""
        if truncated:
            cap = (cap + "  " if cap else "") + f"(showing first {max_rows} of {len(df)} rows)"
        self._draw_table(fig, rows, top=top - 0.01, caption=cap, bold_cells=bold_cells)
        self._close_page(pdf, fig)

    def _draw_table(
        self, fig, df: pd.DataFrame, *, top: float, caption: str = "", bold_cells: set | None = None
    ) -> None:
        ax = fig.add_axes([0.06, 0.07, 0.88, top - 0.08])
        ax.axis("off")
        cell_text = [
            [self._fmt(c, v) for c, v in zip(df.columns, row)] for row in df.itertuples(index=False)
        ]
        if not cell_text:
            ax.text(0.0, 1.0, "(no rows)", fontsize=10, color=_MUTED, va="top")
            return
        table = ax.table(
            cellText=cell_text,
            colLabels=[str(c) for c in df.columns],
            cellLoc="center",
            loc="upper center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1.0, 1.35)
        ncol = df.shape[1]
        cols = list(df.columns)
        bold_cells = bold_cells or set()
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("#ffffff")
            cell.set_linewidth(1.0)
            if r == 0:
                cell.set_facecolor(_HEAD_BG)
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor(_ROW_ALT if r % 2 == 0 else "white")
                weight = "bold" if (r - 1, cols[c]) in bold_cells else "normal"
                cell.set_text_props(color=_INK, fontweight=weight)
        with contextlib.suppress(Exception):  # matplotlib version drift
            table.auto_set_column_width(col=list(range(ncol)))
        if caption:
            fig.text(0.06, 0.045, caption, fontsize=8.5, color=_MUTED, va="bottom")

    @staticmethod
    def _fmt(col, val) -> str:
        col = str(col).lower()
        if isinstance(val, str):
            return val if len(val) <= 42 else val[:39] + "…"
        try:
            f = float(val)
        except (TypeError, ValueError):
            return str(val)
        if f != f:  # NaN
            return "—"
        if col in ("p", "p_val", "pval", "fdr", "q", "pvalue") or "p_val" in col:
            if f < 1e-3:
                return f"{f:.1e}"
            return f"{f:.4f}"
        if col in ("component", "n", "matched_size", "rank"):
            return f"{round(f)}"
        if abs(f) >= 1000 or (f != 0 and abs(f) < 1e-3):
            return f"{f:.2e}"
        return f"{f:.3f}"

    # ==================================================================
    # Sections
    # ==================================================================
    @staticmethod
    def _resolved_nulls(ctx: dict) -> str:
        used = set()
        for entry in ctx.get("transcriptomics", []):
            results = entry[2] if len(entry) > 2 else {}
            for res in (results or {}).values():
                nm = getattr(getattr(res, "metadata", None), "null_method", None)
                if nm:
                    used.add(str(nm))
        return ", ".join(sorted(used))

    def _cover_page(self, pdf, ctx: dict) -> None:
        cfg = self.cfg
        sm = ctx.get("strength_maps")
        contrasts = [t for t, *_ in ctx.get("contrasts", [])]
        fig = self._open_page(pdf)
        fig.text(
            0.07,
            0.86,
            "Morphometric Similarity Network",
            fontsize=24,
            color=_INK,
            fontweight="bold",
            va="top",
        )
        fig.text(
            0.07,
            0.815,
            "imaging-transcriptomics analysis report",
            fontsize=14,
            color=_ACCENT,
            va="top",
        )
        fig.add_artist(plt.Line2D([0.07, 0.93], [0.80, 0.80], color=_RULE, linewidth=1.2))

        rows = [
            (
                "Atlas",
                f"{cfg.engine.atlas}  ·  regions: {cfg.engine.regions}  ·  "
                f"engine hemisphere: {cfg.engine.hemisphere}",
            ),
            ("MSN metrics", ", ".join(cfg.msn.features)),
            ("Subjects analysed", str(sm.n_subjects if sm is not None else "n/a")),
            ("Contrasts", ", ".join(contrasts) if contrasts else "none"),
            ("Transcriptomics", ", ".join(cfg.engine.methods)),
            ("Enrichment", ", ".join(cfg.engine.enrichment_methods)),
            (
                "Spatial null (requested)",
                f"{cfg.engine.null_method}  ·  {cfg.engine.n_permutations} permutations",
            ),
            ("Spatial null (resolved)", self._resolved_nulls(ctx) or "n/a"),
            ("Random seed", str(cfg.engine.seed)),
        ]
        y = 0.74
        for label, value in rows:
            fig.text(0.07, y, label, fontsize=10.5, color=_MUTED, va="top")
            for i, line in enumerate(self._wrap(value, width=58)):
                fig.text(0.34, y - i * 0.026, line, fontsize=10.5, color=_INK, va="top")
            y -= 0.026 * max(1, len(self._wrap(value, width=58))) + 0.012

        fig.text(0.07, 0.06, "Generated by msnpip 2.0", fontsize=9, color=_MUTED, va="bottom")
        self._close_page(pdf, fig)

    def _group_counts(self, ctx: dict) -> dict:
        df, schema = ctx.get("df"), ctx.get("schema")
        sm = ctx.get("strength_maps")
        if df is None or schema is None or sm is None:
            return {}
        gcol = getattr(schema, "group_col", None)
        if not (gcol and gcol in df.columns):
            return {"all": sm.n_subjects}
        try:
            aligned = df.set_index(df[schema.id_col].astype(str)).loc[sm.subject_ids]
            groups = aligned[gcol].map(normalize_group_value)
            return groups.value_counts().to_dict()
        except Exception:
            return {}

    def _dataset_page(self, pdf, ctx: dict) -> None:
        cfg = self.cfg
        sm = ctx.get("strength_maps")
        schema = ctx.get("schema")
        counts = self._group_counts(ctx)
        n_regions = len(sm.region_labels) if sm is not None else 0
        gcol = getattr(schema, "group_col", None)

        fig = self._open_page(pdf)
        top = self._heading(
            fig,
            "Dataset",
            kicker="Section 1",
            subtitle="Cohort, grouping and morphometric features entering the analysis",
        )
        blocks: list = []
        blocks.append(
            (
                f"The analysis included {sm.n_subjects if sm else 'n/a'} subjects with "
                f"complete cortical morphometry across {n_regions} regions of the "
                f"{cfg.engine.atlas} atlas (whole cortex, both hemispheres).",
                "p",
            )
        )
        if counts:
            blocks.append(("Groups", "h"))
            for g, n in counts.items():
                flag = "  (small sample, n < 10 — interpret with caution)" if n < 10 else ""
                blocks.append((f"{gcol or 'group'} = {g}: {n} subjects{flag}", "li"))
        if sm is not None and getattr(sm, "dropped_subjects", None):
            blocks.append(
                (
                    f"{len(sm.dropped_subjects)} subject(s) were dropped for incomplete "
                    "morphometry before MSN construction.",
                    "p",
                )
            )
        blocks.append(("sp", "sp"))
        blocks.append(("Morphometric metrics", "h"))
        blocks.append(
            (
                "Each region is described by "
                + str(len(cfg.msn.features))
                + " FreeSurfer metrics, combined into a multivariate morphometric "
                "fingerprint: " + ", ".join(cfg.msn.features) + ".",
                "p",
            )
        )
        covs = list(cfg.glm.predictors)
        blocks.append(("sp", "sp"))
        blocks.append(("Statistical model", "h"))
        blocks.append(
            (
                "Group differences in regional node strength are estimated with an "
                "ordinary-least-squares model, strength ~ group"
                + (" + " + " + ".join(covs) if covs else "")
                + ". Categorical covariates (e.g. sex, site/scanner) are one-hot encoded "
                "with a dropped reference level.",
                "p",
            )
        )
        if cfg.correlation.variables:
            blocks.append(
                (
                    f"Node strength is additionally correlated with: "
                    f"{', '.join(cfg.correlation.variables)} "
                    f"({cfg.correlation.method}, {cfg.correlation.scope} scope).",
                    "p",
                )
            )
        self._paragraphs(fig, blocks, top=top - 0.01)
        self._close_page(pdf, fig)

    def _msn_section(self, pdf, ctx: dict) -> None:
        cfg = self.cfg
        fig = self._open_page(pdf)
        top = self._heading(
            fig,
            "Morphometric Similarity Networks",
            kicker="Section 2",
            subtitle="How each subject's region-by-region similarity network is built",
        )
        blocks = [
            (
                "For every subject and region, the "
                + str(len(cfg.msn.features))
                + " morphometric metrics are standardised across regions using a robust modified "
                "z-score, M = 0.6745 · (x − median) / MAD, so that each metric contributes on a "
                "comparable, outlier-resistant scale.",
                "p",
            ),
            (
                "The morphometric similarity between two regions is derived from the multivariate "
                "Euclidean distance d between their standardised fingerprints, "
                "S = 1 / (1 + d / n_metrics). This yields a region × region similarity matrix per "
                "subject, bounded in (0, 1], with higher values for regions of more alike "
                "cytoarchitecture-proxy morphometry.",
                "p",
            ),
            (
                "A region's node strength is the sum of its similarity edges — its total "
                "morphometric connectivity within the cortex. The group-mean similarity matrices "
                "below summarise the network structure of each group.",
                "p",
            ),
        ]
        self._paragraphs(fig, blocks, top=top - 0.01)
        self._close_page(pdf, fig)

        # Per-group mean similarity matrices, control first when a contrast defines it.
        for group in self._ordered_groups(ctx):
            png = self.plots_dir / f"{group}_mean_msn_matrix.png"
            self._figure_page(
                pdf,
                png,
                kicker="Section 2 · MSN matrices",
                title=f"Mean morphometric similarity matrix — group {group}",
                caption=f"Region × region mean similarity (viridis) · {self.cfg.engine.atlas} atlas.",
            )

    def _strength_top_bottom(self, n: int = 5) -> dict:
        """Per group, a table of the top-n highest and n lowest node-strength regions."""
        path = self.output_dir / "mean_msn_per_group.csv"
        if not path.exists():
            return {}
        try:
            df = pd.read_csv(path)
        except Exception:
            return {}
        out: dict = {}
        for col in df.columns:
            if not col.startswith("mean_strength_"):
                continue
            group = col[len("mean_strength_") :]
            s = df[["region", col]].dropna().sort_values(col, ascending=False)
            hi = s.head(n).copy()
            hi.insert(0, "extreme", "highest")
            lo = s.tail(n).iloc[::-1].copy()
            lo.insert(0, "extreme", "lowest")
            table = pd.concat([hi, lo], ignore_index=True).rename(columns={col: "node strength"})
            out[group] = table
        return out

    def _strength_section(self, pdf, ctx: dict) -> None:
        fig = self._open_page(pdf)
        top = self._heading(
            fig,
            "Node strength by group",
            kicker="Section 3",
            subtitle="Where each group concentrates its morphometric similarity hubs",
        )
        self._paragraphs(
            fig,
            [
                (
                    "The surface maps below show, for each group, the mean regional node strength on the "
                    "cortex (sequential viridis scale, brighter = stronger). Node strength is the sum "
                    "of a region's morphometric-similarity edges; similarity is a dimensionless ratio "
                    "in (0, 1], so node strength is a dimensionless network measure (no physical unit). "
                    "The per-group tables below list each group's 5 highest- and 5 lowest-strength "
                    "regions; the statistical contrast between groups follows in the next section.",
                    "p",
                ),
            ],
            top=top - 0.01,
        )
        self._close_page(pdf, fig)

        summary = self._strength_top_bottom()
        for group in self._ordered_groups(ctx):
            if group in summary:
                self._table_page(
                    pdf,
                    title=f"Node-strength extremes — group {group}",
                    df=summary[group],
                    kicker="Section 3 · Node strength",
                    max_rows=12,
                    intro=[
                        (
                            f"The 5 regions with the highest and the 5 with the lowest mean node "
                            f"strength in group {group} (dimensionless).",
                            "p",
                        )
                    ],
                    caption="Node strength = sum of a region's morphometric-similarity edges.",
                )
            self._figure_page(
                pdf,
                self.plots_dir / f"{group}_strength_surface.png",
                kicker="Section 3 · Node strength",
                title=f"Mean node strength on the cortex — group {group}",
                caption="Mean node strength per region, inflated surface, both hemispheres (viridis).",
            )

    def _ordered_groups(self, ctx: dict) -> list[str]:
        """Group labels, control-then-case for the first contrast when known."""
        counts = self._group_counts(ctx)
        groups = list(counts.keys())
        contrasts = ctx.get("contrasts", [])
        if contrasts:
            _tag, _res, cc, kk = contrasts[0]
            ordered = [g for g in (kk, cc) if g in groups]
            ordered += [g for g in groups if g not in ordered]
            return ordered
        return groups

    # ------------------------------------------------------------------
    # Per-contrast section
    # ------------------------------------------------------------------
    def _contrast_section(self, pdf, ctx, tag, res, cc, kk) -> None:
        case_lbl, ctrl_lbl = tag.split("_vs_", 1)
        pretty = f"{case_lbl} vs {ctrl_lbl}"
        kicker = f"Contrast · {pretty}"

        # Section opener.
        fig = self._open_page(pdf)
        top = self._heading(
            fig,
            f"Case–control contrast: {pretty}",
            kicker="Section 4",
            subtitle=f"Group difference in node strength (statistic: {res.stat_type})",
        )
        covs = ", ".join(res.covariates) if res.covariates else "none"
        self._paragraphs(
            fig,
            [
                (
                    f"Per-region node strength was contrasted between {case_lbl} (n = {res.n_case}) and "
                    f"{ctrl_lbl} (n = {res.n_control}), adjusting for covariates: {covs}. The exported "
                    f"difference map uses the {res.stat_type} statistic; significant regions are defined "
                    f"by Benjamini-Hochberg FDR < {SIG_ALPHA}.",
                    "p",
                ),
            ],
            top=top - 0.01,
        )
        self._close_page(pdf, fig)

        # (a) difference bar chart (t-values + FDR asterisks) + surfaces.
        self._figure_page(
            pdf,
            self.plots_dir / f"{tag}_tvalue_bars.png",
            kicker=kicker,
            title=f"Per-region node-strength t-values: {pretty}",
            caption=(
                "Left and right hemispheres, all regions in default (alphabetical) order. "
                "Red = higher in case, blue = higher in control. "
                f"Asterisks mark FDR significance (* < {SIG_ALPHA}, ** < 0.01, *** < 0.001); "
                "non-significant regions are faded."
            ),
        )
        for mesh in ("inflated", "pial"):
            self._figure_page(
                pdf,
                self.plots_dir / f"{tag}_surface_{mesh}.png",
                kicker=kicker,
                title=f"Node-strength {res.stat_type} contrast on the cortex ({mesh})",
                caption=f"{pretty} · {mesh} surface · both hemispheres (RdBu_r, centred at 0).",
            )

        # (b) significant regions in writing + table.
        self._significant_regions_page(pdf, res, pretty, kicker)

        # (b2) full per-region statistics table (all regions, paginated).
        self._region_stats_pages(pdf, res, pretty, kicker)

        # (c) significant-only surface.
        self._figure_page(
            pdf,
            self.plots_dir / f"{tag}_surface_significant.png",
            kicker=kicker,
            title=f"FDR-significant regions only: {pretty}",
            caption=f"Only regions with FDR < {SIG_ALPHA} are coloured; others shown neutral.",
        )

        # (d-f) transcriptomics: PLS parameters, top genes, enrichment.
        self._pls_parameters_page(pdf, tag, kicker)
        self._top_genes_page(pdf, tag, kicker)
        self._enrichment_section(pdf, tag, kicker, pretty)

    def _significant_regions_page(self, pdf, res, pretty: str, kicker: str) -> None:
        sig = res.significant_table(alpha=SIG_ALPHA)
        fig = self._open_page(pdf)
        top = self._heading(
            fig,
            "Significant regions",
            kicker=kicker,
            subtitle=f"Regions surviving FDR < {SIG_ALPHA}, with effect size and significance",
        )
        if sig.empty:
            self._paragraphs(
                fig,
                [
                    (
                        f"No region survived FDR correction at q < {SIG_ALPHA} for {pretty}. "
                        "The strongest uncorrected effects can still be read from the per-region bar "
                        "chart and surface maps above.",
                        "p",
                    ),
                ],
                top=top - 0.01,
            )
            self._close_page(pdf, fig)
            return

        n = len(sig)
        lead = sig.iloc[0]
        direction = "higher" if lead["beta"] > 0 else "lower"
        case_lbl = pretty.split(" vs ", 1)[0]
        blocks = [
            (
                f"{n} region{'s' if n != 1 else ''} showed a significant group difference in node "
                f"strength (FDR < {SIG_ALPHA}). The strongest effect was in "
                f"{lead['region']} (beta = {lead['beta']:.3f}, t = {lead['t']:.2f}, "
                f"p = {self._p(lead['p'])}, FDR = {self._p(lead['fdr'])}), where node strength was "
                f"{direction} in {case_lbl}.",
                "p",
            ),
        ]
        # Name the rest concisely.
        if n > 1:
            names = ", ".join(str(r) for r in sig["region"].tolist()[1:8])
            more = "" if n <= 8 else f", and {n - 8} more"
            blocks.append((f"Other significant regions: {names}{more}.", "p"))
        top = self._paragraphs(fig, blocks, top=top - 0.01)
        self._draw_table(
            fig,
            sig,
            top=top - 0.01,
            caption="Positive beta = higher node strength in the case group.",
        )
        self._close_page(pdf, fig)

    def _region_stats_pages(self, pdf, res, pretty: str, kicker: str) -> None:
        """Per-region (beta, t, Cohen's d, p, FDR), one page per hemisphere.

        Significant FDR values (< alpha) are drawn bold.
        """
        tbl = res.stats_table()
        for hemi_code, hemi_name in (("lh", "Left"), ("rh", "Right")):
            part = (
                tbl[tbl["region"].astype(str).str.startswith(f"{hemi_code}_")]
                .sort_values(["fdr", "p"], kind="mergesort")
                .reset_index(drop=True)
            )
            if part.empty:
                continue
            bold = {
                (i, "fdr")
                for i, v in enumerate(part["fdr"])
                if pd.notna(v) and float(v) < SIG_ALPHA
            }
            intro = [
                (
                    f"Group-effect statistics for every {hemi_name.lower()}-hemisphere region "
                    "(case vs control), sorted by FDR. beta = node-strength difference; t = its "
                    "t-statistic; cohen_d = covariate-adjusted standardized effect; p / fdr = "
                    f"nominal and FDR-corrected significance (bold fdr = significant at < {SIG_ALPHA}).",
                    "p",
                )
            ]
            self._table_page(
                pdf,
                title=f"Regional statistics — {hemi_name} hemisphere",
                df=part,
                kicker=kicker,
                intro=intro,
                max_rows=40,
                bold_cells=bold,
                caption="Positive beta / Cohen's d = higher node strength in the case group.",
            )

    @staticmethod
    def _p(v) -> str:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return "—"
        if f != f:
            return "—"
        return f"{f:.1e}" if f < 1e-3 else f"{f:.4f}"

    def _glob_tagged(self, pattern: str) -> list[Path]:
        return sorted(self.output_dir.glob(pattern))

    def _pls_parameters_page(self, pdf, tag: str, kicker: str) -> None:
        files = self._glob_tagged(f"{tag}*_pls_summary.csv")
        if not files:
            return
        for path in files:
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            variant = path.stem[len(tag) :].replace("_pls_summary", "").strip("_")
            title = "PLS components" + (f" · {variant}" if variant else "")
            intro = [
                (
                    "Partial least squares regresses the regional case-control difference map onto "
                    "Allen Human Brain Atlas gene expression. Each component's explained variance is "
                    "tested against a spatial-null distribution; the p-value reflects spatial "
                    "specificity.",
                    "p",
                ),
            ]
            self._table_page(
                pdf,
                title=title,
                df=df,
                kicker=kicker,
                intro=intro,
                caption="Explained variance, cumulative variance and spatial-null "
                "p-value per PLS component.",
            )

    def _top_genes_page(self, pdf, tag: str, kicker: str) -> None:
        files = self._glob_tagged(f"{tag}*_pls.csv")
        for path in files:
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if df.empty:
                continue
            # First (or only) component.
            if "component" in df.columns:
                comp = sorted(df["component"].dropna().unique())[0]
                df = df[df["component"] == comp]
                comp_label = f" (component {int(comp)})"
            else:
                comp_label = ""
            score_col = next((c for c in ("zscore", "weight", "score") if c in df.columns), None)
            if score_col is None:
                continue
            ordered = df.sort_values(score_col, ascending=False)
            keep = [c for c in ("gene", score_col, "p", "fdr") if c in ordered.columns]
            top = ordered.head(20)[keep].copy()
            bottom = ordered.tail(20)[keep].iloc[::-1].copy()
            variant = path.stem[len(tag) :].replace("_pls", "").strip("_")
            suffix = f" · {variant}" if variant else ""
            self._table_page(
                pdf,
                title=f"Top 20 positively-weighted genes{comp_label}{suffix}",
                df=top,
                kicker=kicker,
                max_rows=20,
                intro=[
                    (
                        "Genes whose expression is most positively associated with the "
                        "case-control node-strength difference (ranked by PLS weight).",
                        "p",
                    )
                ],
                caption=f"Ranked by {score_col} (descending).",
            )
            self._table_page(
                pdf,
                title=f"Top 20 negatively-weighted genes{comp_label}{suffix}",
                df=bottom,
                kicker=kicker,
                max_rows=20,
                intro=[
                    (
                        "Genes whose expression is most negatively associated with the "
                        "case-control node-strength difference.",
                        "p",
                    )
                ],
                caption=f"Ranked by {score_col} (ascending).",
            )

    def _enrichment_section(self, pdf, tag: str, kicker: str, pretty: str) -> None:
        emitted = False
        # Engine enrichment plots (ensemble / gsea / ora dotplots & heatmaps),
        # copied into plots/ with the contrast prefix.
        seen: set = set()
        plots: list[Path] = []
        for key in ("ensemble", "gsea", "ora", "enrich", "dotplot"):
            for png in sorted(self.plots_dir.glob(f"{tag}*{key}*.png")):
                if png not in seen:
                    seen.add(png)
                    plots.append(png)
        for png in plots:
            if self._figure_page(
                pdf,
                png,
                kicker=kicker,
                title=f"Gene-set enrichment: {pretty}",
                caption=png.stem.replace(tag + "_", "").replace("_", " "),
            ):
                emitted = True
        # Enrichment table(s) — most significant terms, one table per gene set
        # (and per backend when both ensemble and GSEA were run).
        prefer = ["Term", "es", "nes", "z_score", "category_score", "p_val", "p", "fdr"]
        for path in self._glob_tagged(f"{tag}*_enrichment.csv"):
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if df.empty:
                continue
            group_cols = [c for c in ("enrichment", "geneset") if c in df.columns]
            groups = df.groupby(group_cols) if group_cols else [((), df)]
            for gkey, sub in groups:
                gkey = gkey if isinstance(gkey, tuple) else (gkey,)
                backend = str(gkey[0]) if "enrichment" in group_cols else ""
                geneset = str(gkey[-1]) if "geneset" in group_cols else "gene set"
                sort_cols = [c for c in ("fdr", "p_val", "p") if c in sub.columns]
                if sort_cols:
                    sub = sub.sort_values(sort_cols, kind="mergesort")
                keep = [c for c in prefer if c in sub.columns and sub[c].notna().any()] or list(
                    sub.columns
                )
                disp = sub[keep].head(20).reset_index(drop=True)
                sig_col = next((c for c in ("fdr", "p_val", "p") if c in disp.columns), None)
                bold = (
                    {
                        (i, sig_col)
                        for i, v in enumerate(disp[sig_col])
                        if pd.notna(v) and float(v) < SIG_ALPHA
                    }
                    if sig_col
                    else set()
                )
                emitted = True
                suffix = f" ({backend})" if backend else ""
                self._table_page(
                    pdf,
                    title=f"Enrichment terms — {geneset}{suffix}",
                    df=disp,
                    kicker=kicker,
                    max_rows=20,
                    bold_cells=bold,
                    intro=[
                        (
                            f"Gene-set enrichment for {geneset}{suffix}, ranked by significance. "
                            "nes/es (GSEA) or z_score (ensemble) is the enrichment effect; "
                            f"p_val / fdr give nominal and FDR-corrected significance (bold fdr = "
                            f"significant at < {SIG_ALPHA}).",
                            "p",
                        )
                    ],
                    caption="Ranked by FDR (then nominal p).",
                )

        if not emitted:
            self._enrichment_missing_page(pdf, tag, kicker, pretty)

    def _enrichment_missing_page(self, pdf, tag: str, kicker: str, pretty: str) -> None:
        """Explicit note when no enrichment output was found (never silent)."""
        fig = self._open_page(pdf)
        top = self._heading(
            fig,
            "Gene-set enrichment",
            kicker=kicker,
            subtitle=f"No enrichment results were found for {pretty}",
        )
        methods = ", ".join(self.cfg.engine.enrichment_methods) or "none"
        self._paragraphs(
            fig,
            [
                (
                    "The pipeline found no enrichment results to report for this contrast. "
                    "This means the transcriptomics engine did not write enrichment tables or "
                    "plots, so there is nothing to summarise here.",
                    "p",
                ),
                ("Most common causes", "h"),
                (
                    f"Configured enrichment method(s): {methods}. If this is 'none', no "
                    "enrichment was requested.",
                    "li",
                ),
                (
                    "The gene sets could not be prepared (missing .gmt gene-set files or no "
                    "internet access to download them), so the engine skipped enrichment.",
                    "li",
                ),
                ("The enrichment step ran but no gene set reached the reporting threshold.", "li"),
                (
                    "To diagnose, check the run log around the TRANSCRIPTOMICS stage, and look for "
                    f"'{tag}_enrichment.csv' in the output folder and '*ensemble*' / '*gsea*' PNGs "
                    "under plots/. If those are absent, enrichment did not run in the engine.",
                    "p",
                ),
            ],
            top=top - 0.01,
        )
        self._close_page(pdf, fig)
