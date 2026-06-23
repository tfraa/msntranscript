"""
Thin wrapper around imaging_transcriptomics.run_pls / run_corr.
Enforces the vasa surface null; raises MsnpipSurfaceNullError on fallback.
Phase 3, Task T3.1.

The engine itself runs PLS/correlation, the spatial-null permutations, and the
enrichment families, and writes its own TSV/JSON/PNG bundle.  msnpip's job here
is narrow: validate the already-aligned input, dispatch one engine call per
method with all gene sets, wrap engine exceptions with context, and — critically
— refuse to let a silent grouped-shuffle null reach a figure.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import imaging_transcriptomics as imt

from msnpip.config import EngineConfig
from msnpip.errors import MsnpipEngineError, MsnpipError, MsnpipSurfaceNullError

logger = logging.getLogger("msnpip.engine")

# Null methods that count as a real surface spin (anything else with
# require_surface_null on is treated as an invalid fallback).
_SURFACE_NULLS = frozenset({"vasa", "alexander_bloch", "moran"})


def _primary_enrichment(enrichment_methods: tuple[str, ...]) -> str:
    """The enrichment family passed as ``enrichment_method=`` to the engine.

    GSEA is requested separately via ``run_gsea`` (it runs *alongside* the
    primary family), so it is skipped here.  Falls back to ``"none"`` if the
    only requested family is GSEA.
    """
    for method in enrichment_methods:
        if method in ("ensemble", "ora", "none"):
            return method
    return "none"


def _check_surface_null(result, cfg: EngineConfig, method: str) -> None:
    """Raise MsnpipSurfaceNullError if the engine silently fell back to a shuffle."""
    if not cfg.require_surface_null:
        return
    used = getattr(getattr(result, "metadata", None), "null_method", None)
    if used is not None and used not in _SURFACE_NULLS:
        raise MsnpipSurfaceNullError(
            f"[{method}] Engine reported null_method={used!r} after being asked for "
            f"{cfg.null_method!r}: a surface spin silently fell back to a grouped "
            "shuffle, which invalidates the spatial-specificity test. Surface assets "
            "may be missing. Fetch them with:\n"
            '  python -c "import neuromaps; neuromaps.datasets.fetch_fsaverage()"'
        )


def _call_engine(
    method: str,
    data: np.ndarray,
    input_rh: np.ndarray | None,
    cfg: EngineConfig,
    out_dir: Path,
):
    """Dispatch a single run_pls / run_corr call with shared kwargs."""
    primary = _primary_enrichment(cfg.enrichment_methods)
    run_gsea = "gsea" in cfg.enrichment_methods

    common = dict(
        atlas=cfg.atlas,
        hemisphere=cfg.hemisphere,
        regions=cfg.regions,
        input_rh=input_rh,
        n_permutations=cfg.n_permutations,
        null_method=cfg.null_method,
        output_dir=out_dir,
        enrichment_method=primary,
        run_gsea=run_gsea,
        gene_set=cfg.gene_sets,
        geneset_organism=cfg.geneset_organism,
        ora_p_threshold=cfg.ora_p_threshold,
        seed=cfg.seed,
        n_jobs=cfg.n_jobs,
    )

    if method == "pls":
        return imt.run_pls(data, n_components=cfg.n_components, var=cfg.var, **common)
    if method == "corr":
        return imt.run_corr(data, **common)
    raise MsnpipEngineError(f"Unknown engine method {method!r} (expected 'pls' or 'corr').")


def run_transcriptomics(
    regional_map: np.ndarray,
    labels_df: pd.DataFrame,
    cfg: EngineConfig,
    output_dir: Path,
    contrast_tag: str,
) -> dict:
    """Run the engine for one contrast map across all configured methods.

    Parameters
    ----------
    regional_map
        1-D array already aligned to ``labels_df`` row order (output of
        :func:`msnpip.atlas_align.align_strength_to_atlas`).
    labels_df
        Engine label DataFrame (``id, label, hemisphere, structure``) matching
        *regional_map*.
    cfg
        :class:`msnpip.config.EngineConfig`.
    output_dir
        Base output directory.  Each call lands in
        ``output_dir / contrast_tag / <method>/``.
    contrast_tag
        Tag identifying the contrast (e.g. ``"FTD_vs_HC"``).

    Returns
    -------
    dict
        ``{method: PLSResult | CorrelationResult}`` for each method in
        ``cfg.methods``.

    Raises
    ------
    MsnpipEngineError
        On length mismatch or any wrapped engine exception.
    MsnpipSurfaceNullError
        If the engine fell back from the surface spin to a grouped shuffle and
        ``cfg.require_surface_null`` is set.
    """
    regional_map = np.asarray(regional_map, dtype=float).ravel()
    if len(regional_map) != len(labels_df):
        raise MsnpipEngineError(
            f"regional_map length {len(regional_map)} != labels_df length "
            f"{len(labels_df)}. The map must be atlas-aligned before the engine call."
        )

    # Split the aligned vector for a both-hemisphere run; the engine takes the
    # left map as `data` and the right map as `input_rh` (engine contract §run_pls).
    if cfg.hemisphere == "both":
        left_mask = (labels_df["hemisphere"] == "L").to_numpy()
        data = regional_map[left_mask]
        input_rh = regional_map[~left_mask]
    else:
        data = regional_map
        input_rh = None

    results: dict = {}
    for method in cfg.methods:
        out_dir = Path(output_dir) / contrast_tag / method
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "run_transcriptomics: %s [%s] → %s (n=%d, null=%s, perms=%d)",
            contrast_tag, method, out_dir, len(data), cfg.null_method, cfg.n_permutations,
        )
        try:
            result = _call_engine(method, data, input_rh, cfg, out_dir)
        except MsnpipError:
            raise
        except Exception as exc:  # engine exceptions → wrapped with context
            raise MsnpipEngineError(
                f"Engine {method!r} call failed for contrast {contrast_tag!r}: {exc}",
                cause=exc,
            ) from exc

        _check_surface_null(result, cfg, method)
        results[method] = result

    return results
