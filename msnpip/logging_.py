"""
Structured, optionally coloured logging for msnpip.

Usage (in every module)::

    from msnpip.logging_ import get_logger
    logger = get_logger(__name__)

Call ``configure_logging(verbose=...)`` once from ``cli.py`` or
``Pipeline.__init__`` before any stage runs.
"""

from __future__ import annotations

import logging
import sys
from typing import ClassVar

_THIRD_PARTY_LOGGERS = (
    "gseapy",
    "nilearn",
    "nibabel",
    "imaging_transcriptomics",
    "neuromaps",
    "matplotlib",
    "PIL",
    "numexpr",
)

PHASE_SEP = "=" * 64

# ANSI colour codes — only applied when stderr is a tty
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_RED = "\033[31m"


def _tty() -> bool:
    return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()


class _PipelineFormatter(logging.Formatter):
    """Single-line formatter with optional ANSI colour per level."""

    _COLOURS: ClassVar[dict[int, str]] = {
        logging.DEBUG: _DIM,
        logging.INFO: "",
        logging.WARNING: _YELLOW,
        logging.ERROR: _RED,
        logging.CRITICAL: _BOLD + _RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        use_colour = _tty()
        msg = record.getMessage()

        # Colour the message body
        colour = self._COLOURS.get(record.levelno, "")
        if use_colour and colour:
            msg = colour + msg + _RESET

        # Colour the logger name (strip the msnpip. prefix for brevity)
        name = record.name.removeprefix("msnpip.")
        if use_colour:
            name = _CYAN + name + _RESET

        level = record.levelname.ljust(8)
        return f"{name} | {level} | {msg}"


def configure_logging(verbose: bool = False) -> None:
    """Configure the msnpip root logger.

    Must be called once before any stage runs.  Subsequent calls are
    idempotent (handlers are not duplicated).

    Parameters
    ----------
    verbose : bool
        If True, set the root logger to DEBUG; otherwise INFO.
    """
    root = logging.getLogger("msnpip")
    root.setLevel(logging.DEBUG if verbose else logging.INFO)

    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_PipelineFormatter())
        root.addHandler(handler)

    for name in _THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a child of the ``msnpip`` root logger.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.  The ``msnpip.``
        prefix is added automatically if absent.
    """
    if not name.startswith("msnpip."):
        name = f"msnpip.{name}"
    return logging.getLogger(name)


def phase_banner(n: int, total: int, title: str) -> None:
    """Emit a visible phase separator to the msnpip root logger (INFO level).

    Example output::

        ================================================================
        === PHASE 2/6  MSN CONSTRUCTION
        ================================================================
    """
    logger = logging.getLogger("msnpip")
    logger.info(PHASE_SEP)
    logger.info("=== PHASE %d/%d  %s", n, total, title.upper())
    logger.info(PHASE_SEP)
