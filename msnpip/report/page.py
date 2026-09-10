"""A4 page primitives for the PDF report.

Layout only: opening and closing numbered pages, headings, wrapped paragraphs,
embedded figures, tables with optional bold cells, and the Contents page.
Nothing here knows what the report is about.
"""

from __future__ import annotations

import contextlib
import logging
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger("msnpip.report.page")

# Page geometry (inches) — every report page is A4 portrait.
A4_PORTRAIT = (8.27, 11.69)

# Palette for headings / rules.
_INK = "#1f2933"
_MUTED = "#52606d"
_ACCENT = "#2166ac"
_RULE = "#cbd2d9"
_HEAD_BG = "#2166ac"
_ROW_ALT = "#eef2f7"


class PdfCanvas:
    """Page-level drawing shared by every report section."""

    # Contents entries that fit on one page.
    _TOC_PER_PAGE = 30

    def __init__(self) -> None:
        self._page_no = 0  # running page counter (footer numbering)
        self._toc: list[tuple[str, int]] = []  # (section title, page) for Contents

    # ==================================================================
    # Low-level page primitives
    # ==================================================================
    def _open_page(self):
        fig = plt.figure(figsize=A4_PORTRAIT)
        fig.patch.set_facecolor("white")
        return fig

    def _close_page(self, pdf, fig) -> None:
        # savefig.bbox is forced off in build() so pages keep full A4 portrait.
        self._page_no += 1
        if self._page_no > 1:  # leave the cover unnumbered
            fig.text(
                0.5, 0.028, str(self._page_no), ha="center", va="bottom", fontsize=9, color=_MUTED
            )
        pdf.savefig(fig)
        plt.close(fig)

    def _heading(self, fig, title: str, *, subtitle: str | None = None, kicker: str | None = None):
        """Draw a section heading band; return the y below which content starts.

        Long titles wrap onto multiple lines so they never run off the page; the
        subtitle, rule and returned content-start shift down accordingly.
        """
        if kicker:
            fig.text(0.07, 0.955, kicker.upper(), fontsize=9, color=_ACCENT, fontweight="bold")
        cur = 0.935
        for line in self._wrap(title, width=42):
            fig.text(0.07, cur, line, fontsize=18, color=_INK, fontweight="bold", va="top")
            cur -= 0.034
        if subtitle:
            cur += 0.004
            for line in self._wrap(subtitle, width=74):
                fig.text(0.07, cur, line, fontsize=10.5, color=_MUTED, va="top")
                cur -= 0.024
        rule_y = cur + 0.004
        fig.add_artist(plt.Line2D([0.07, 0.93], [rule_y, rule_y], color=_RULE, linewidth=1.0))
        return rule_y - 0.02

    def _paragraphs(self, fig, blocks, *, top: float, x: float = 0.07):
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

        return textwrap.wrap(text, width=width) or [""]

    def _figure_page(
        self, pdf, png: Path, *, title: str, caption: str | None = None, kicker: str | None = None
    ) -> bool:
        if not png or not Path(png).exists():
            return False
        try:
            img = mpimg.imread(png)
        except Exception as exc:  # pragma: no cover - corrupt image
            logger.warning("REPORT: could not read %s: %s", png, exc)
            return False
        fig = self._open_page()
        if kicker:
            fig.text(0.05, 0.975, kicker.upper(), fontsize=9, color=_ACCENT, fontweight="bold")
        ty = 0.935
        for line in self._wrap(title, width=64):  # wrap long titles onto the page
            fig.text(0.05, ty, line, fontsize=14, color=_INK, fontweight="bold", va="top")
            ty -= 0.028
        ax = fig.add_axes([0.04, 0.07, 0.92, 0.80])
        ax.axis("off")
        ax.imshow(img)
        if caption:
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
        fig = self._open_page()
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

    def _toc_page_count(self, n_entries: int) -> int:
        return max(1, (n_entries + self._TOC_PER_PAGE - 1) // self._TOC_PER_PAGE)

    def _toc_mark(self, title: str) -> None:
        """Record that *title* starts on the page about to be drawn."""
        self._toc.append((title, self._page_no + 1))

    def _toc_pages(self, pdf, entries: list[tuple[str, int]]) -> None:
        n_pages = self._toc_page_count(len(entries))
        per = max(1, (len(entries) + n_pages - 1) // n_pages)
        for pi in range(n_pages):
            chunk = entries[pi * per : (pi + 1) * per]
            fig = self._open_page()
            top = self._heading(fig, "Contents", kicker="Report")
            y = top - 0.015
            for title, page in chunk:
                lines = self._wrap(title, width=70)
                fig.text(0.07, y, lines[0], fontsize=11, color=_INK, va="top")
                fig.text(0.93, y, str(page), fontsize=11, color=_INK, va="top", ha="right")
                fig.add_artist(
                    plt.Line2D(
                        [0.07, 0.91], [y - 0.012, y - 0.012], color=_RULE, linewidth=0.5, ls=":"
                    )
                )
                y -= 0.026
                for extra in lines[1:]:
                    fig.text(0.085, y, extra, fontsize=11, color=_INK, va="top")
                    y -= 0.026
            self._close_page(pdf, fig)
