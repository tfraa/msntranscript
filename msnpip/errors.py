"""
MsnpipError hierarchy.

Every exception raised by msnpip inherits from MsnpipError so callers can
catch the whole family with a single except clause.
"""

from __future__ import annotations


class MsnpipError(Exception):
    """Base for all msnpip errors."""


# --- I/O ---------------------------------------------------------------


class MsnpipIOError(MsnpipError):
    """File I/O or format problems."""


class AmbiguousFormatError(MsnpipIOError):
    """Delimiter or decimal format cannot be determined automatically."""


# --- Schema / validation -----------------------------------------------


class SchemaError(MsnpipError):
    """Input data fails the schema validation gate (dtype, uniqueness, …)."""


class IDMatchError(MsnpipError):
    """ID match rate between FreeSurfer subjects and demographics falls below
    the configured threshold (``IOConfig.min_id_match_rate``).

    Attributes
    ----------
    unmatched : list[str]
        Subject IDs present in one source but absent from the other.
    """

    def __init__(self, message: str, unmatched: list[str] | None = None) -> None:
        super().__init__(message)
        self.unmatched: list[str] = unmatched or []


# --- Atlas alignment ---------------------------------------------------


class AtlasAlignmentError(MsnpipError):
    """MSN region labels cannot be matched to the engine atlas label order.

    Raised by ``atlas_align.align_strength_to_atlas`` when one or more
    engine-expected (hemisphere, label) pairs are absent from the MSN output.
    Never silently zero-fills.
    """


# --- MSN construction --------------------------------------------------


class MSNInputError(MsnpipError):
    """Bad input to MSN construction (e.g. an all-NaN region column)."""


# --- Engine wrapper ----------------------------------------------------


class MsnpipEngineError(MsnpipError):
    """Wraps an imaging_transcriptomics exception with pipeline context."""

    def __init__(self, message: str, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.__cause__ = cause


class MsnpipSurfaceNullError(MsnpipEngineError):
    """The engine silently fell back from the vasa surface spin to a grouped
    shuffle (``null_method='random'``).

    Raised when ``EngineConfig.require_surface_null=True`` and the engine
    reports a non-surface null after being asked for ``vasa``.  This must
    never silently reach a figure — the test it produces is invalid.
    """


# --- Configuration / pipeline ------------------------------------------


class ConfigurationError(MsnpipError):
    """``PipelineConfig.validate()`` found a cross-field inconsistency."""


class StageError(MsnpipError):
    """A pipeline stage failed in a way that prevents downstream stages.

    Attributes
    ----------
    stage : str
        Name of the stage that raised the error (e.g. ``"VALIDATE"``).
    """

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(f"[{stage}] {message}")
        self.stage = stage
