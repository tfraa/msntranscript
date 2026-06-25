"""
ReportBuilder — aggregate the run's plots into ``report.pdf``.

A deliberately simple report for now (cover page + one page per figure); the
detailed structure will be defined later.  Lives next to the curated CSVs at the
output root.
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
    """Assemble ``<output>/report.pdf`` from the plots in ``<output>/plots``."""

    def __init__(self, output_dir, cfg) -> None:
        self.output_dir = Path(output_dir)
        self.cfg = cfg

    def build(self, ctx: dict) -> Path | None:
        plots = sorted((self.output_dir / "plots").glob("*.png"))
        pdf_path = self.output_dir / "report.pdf"
        with PdfPages(pdf_path) as pdf:
            self._cover_page(pdf, ctx, len(plots))
            for png in plots:
                self._image_page(pdf, png)
        logger.info("REPORT: wrote %s (%d figure pages)", pdf_path, len(plots))
        return pdf_path

    # ------------------------------------------------------------------
    @staticmethod
    def _resolved_nulls(ctx: dict) -> str:
        """Comma-joined set of null methods the engine actually used (provenance)."""
        used = set()
        for entry in ctx.get("transcriptomics", []):
            results = entry[2] if len(entry) > 2 else {}
            for res in (results or {}).values():
                nm = getattr(getattr(res, "metadata", None), "null_method", None)
                if nm:
                    used.add(str(nm))
        return ", ".join(sorted(used))

    def _cover_page(self, pdf, ctx: dict, n_plots: int) -> None:
        cfg = self.cfg
        sm = ctx.get("strength_maps")
        contrasts = [t for t, *_ in ctx.get("contrasts", [])]
        resolved_nulls = self._resolved_nulls(ctx)
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
                f"methods: {', '.join(cfg.engine.methods)}    requested null: {cfg.engine.null_method}"
                f"    permutations: {cfg.engine.n_permutations}",
                11,
                "normal",
            ),
            (f"resolved spatial null: {resolved_nulls or 'n/a'}", 11, "normal"),
            (
                f"enrichment: {', '.join(cfg.engine.enrichment_methods)}    seed: {cfg.engine.seed}",
                11,
                "normal",
            ),
            ("", 8, "normal"),
            (f"subjects: {sm.n_subjects if sm is not None else 'n/a'}", 11, "normal"),
            (f"contrasts: {', '.join(contrasts) if contrasts else 'none'}", 11, "normal"),
        ]
        for var, res in ctx.get("correlations", []):
            r = float(res.r[0]) if len(res.r) else float("nan")
            p = float(res.p[0]) if len(res.p) else float("nan")
            lines.append(
                (
                    f"correlation (strength ~ {var}): r={r:.3f}, p={p:.3g}, n={res.n}"
                    f" [{res.method}, {res.scope}]",
                    11,
                    "normal",
                )
            )
        lines += [
            (f"figure pages: {n_plots}", 11, "normal"),
            ("", 8, "normal"),
            ("(Report layout is provisional and will be refined.)", 9, "italic"),
        ]
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        y = 0.92
        for text, size, weight in lines:
            style = "italic" if weight == "italic" else "normal"
            fw = "bold" if weight == "bold" else "normal"
            ax.text(
                0.08,
                y,
                text,
                fontsize=size,
                fontweight=fw,
                fontstyle=style,
                va="top",
                transform=ax.transAxes,
            )
            y -= 0.035 + size * 0.0012
        pdf.savefig(fig)
        plt.close(fig)

    def _image_page(self, pdf, png: Path) -> None:
        try:
            img = mpimg.imread(png)
        except Exception as exc:  # corrupt/zero-byte image — skip
            logger.warning("REPORT: could not read %s: %s", png, exc)
            return
        fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
        ax = fig.add_axes([0.03, 0.06, 0.94, 0.88])
        ax.axis("off")
        ax.imshow(img)
        fig.text(0.03, 0.02, png.name, fontsize=8, va="bottom", color="#555555")
        pdf.savefig(fig)
        plt.close(fig)
