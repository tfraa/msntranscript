"""
ReportBuilder — PDF that aggregates engine TSV/PNG bundles + msnpip figures.
Phase 5, Task T5.4.

The engine already writes per-contrast PNG/TSV bundles and msnpip writes its own
MSN figures; the report stitches them into a single ``Report.pdf`` with a cover
page of run provenance, plus a plain-text ``run_log.txt``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

logger = logging.getLogger("msnpip.report.builder")


class ReportBuilder:
    """Assemble ``05_report/Report.pdf`` from the run's figures and bundles."""

    def __init__(self, output_dir, cfg) -> None:
        self.output_dir = Path(output_dir)
        self.cfg = cfg

    def build(self, ctx: dict) -> Path:
        report_dir = self.output_dir / "05_report"
        report_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = report_dir / "Report.pdf"

        figures = sorted((self.output_dir / "04_figures").rglob("*.png"))
        engine_pngs = sorted((self.output_dir / "03_transcriptomics").rglob("*.png"))

        with PdfPages(pdf_path) as pdf:
            self._cover_page(pdf, ctx, n_figures=len(figures) + len(engine_pngs))
            for png in figures:
                self._image_page(pdf, png, caption=f"MSN figure · {png.parent.name}/{png.name}")
            for png in engine_pngs:
                rel = png.relative_to(self.output_dir)
                self._image_page(pdf, png, caption=f"Transcriptomics · {rel}")

        self._run_log(report_dir / "run_log.txt", ctx, figures, engine_pngs, pdf_path)
        logger.info("REPORT: wrote %s (%d figure pages)", pdf_path, len(figures) + len(engine_pngs))
        return pdf_path

    # ------------------------------------------------------------------
    def _cover_page(self, pdf, ctx, *, n_figures: int) -> None:
        cfg = self.cfg
        sm = ctx.get("strength_maps")
        contrasts = [t for t, *_ in ctx.get("contrasts", [])]
        lines = [
            ("msnpip 2.0 — analysis report", 18, "bold"),
            ("", 10, "normal"),
            (
                f"atlas: {cfg.engine.atlas}    engine hemisphere: {cfg.engine.hemisphere}"
                f"    regions: {cfg.engine.regions}",
                11,
                "normal",
            ),
            (
                f"methods: {', '.join(cfg.engine.methods)}    null: {cfg.engine.null_method}"
                f"    permutations: {cfg.engine.n_permutations}",
                11,
                "normal",
            ),
            (
                f"enrichment: {', '.join(cfg.engine.enrichment_methods)}    seed: {cfg.engine.seed}",
                11,
                "normal",
            ),
            ("", 8, "normal"),
            (
                f"subjects: {sm.n_subjects if sm is not None else 'n/a'}"
                f"    dropped: {len(sm.dropped_subjects) if sm is not None else 0}",
                11,
                "normal",
            ),
            (f"contrasts: {', '.join(contrasts) if contrasts else 'none'}", 11, "normal"),
            (
                f"correlations: {', '.join(v for v, _ in ctx.get('correlations', [])) or 'none'}",
                11,
                "normal",
            ),
            (f"figure pages: {n_figures}", 11, "normal"),
        ]
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        fig.subplots_adjust(left=0.08, right=0.92, top=0.93, bottom=0.07)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        y = 0.92
        for text, size, weight in lines:
            ax.text(
                0.08, y, text, fontsize=size, fontweight=weight, va="top", transform=ax.transAxes
            )
            y -= 0.035 + size * 0.0012
        pdf.savefig(fig)
        plt.close(fig)

    def _image_page(self, pdf, png: Path, *, caption: str) -> None:
        try:
            img = mpimg.imread(png)
        except Exception as exc:  # corrupt/zero-byte image — skip the page
            logger.warning("REPORT: could not read %s: %s", png, exc)
            return
        fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
        ax = fig.add_axes([0.03, 0.06, 0.94, 0.88])
        ax.axis("off")
        ax.imshow(img)
        fig.text(0.03, 0.02, caption, fontsize=8, va="bottom", color="#555555")
        pdf.savefig(fig)
        plt.close(fig)

    def _run_log(self, path: Path, ctx, figures, engine_pngs, pdf_path) -> None:
        sm = ctx.get("strength_maps")
        lines = [
            "msnpip 2.0 run log",
            f"output: {self.output_dir}",
            f"atlas={self.cfg.engine.atlas} hemisphere={self.cfg.engine.hemisphere} "
            f"regions={self.cfg.engine.regions} seed={self.cfg.engine.seed}",
            f"subjects: {sm.n_subjects if sm is not None else 'n/a'}",
            f"dropped: {sm.dropped_subjects if sm is not None else []}",
            f"contrasts: {[t for t, *_ in ctx.get('contrasts', [])]}",
            f"correlations: {[v for v, _ in ctx.get('correlations', [])]}",
            f"MSN figures: {len(figures)}    engine figures: {len(engine_pngs)}",
            f"report: {pdf_path}",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
