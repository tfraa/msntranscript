"""
Thin wrapper around imaging_transcriptomics.run_pls / run_corr.
Phase 3, Task T3.1.

The engine itself runs PLS/correlation, the spatial-null permutations, and the
enrichment families, and writes its own TSV/JSON/PNG bundle.  msnpip's job here
is narrow: validate the already-aligned input, dispatch one engine call per
method with all gene sets, wrap engine exceptions with context, and manage the
spatial-null policy.

Surface-null note: the pinned engine ships the DK parcellation only as a
FreeSurfer ``.annot``, which neuromaps' spin-null loader (``load_gifti``) cannot
read — so ``vasa``/``alexander_bloch`` fail for DK as shipped. We install a small
``.annot``-aware ``load_gifti`` shim so the real spin null runs; if it still
fails and ``allow_null_fallback`` is set, we let the engine fall back to
``auto``/``random`` (recording the resolved null) rather than hard-failing.
"""

from __future__ import annotations

import logging
from pathlib import Path

import imaging_transcriptomics as imt
import numpy as np
import pandas as pd

from msnpip.config import EngineConfig
from msnpip.errors import MsnpipEngineError, MsnpipError, MsnpipSurfaceNullError

logger = logging.getLogger("msnpip.engine")

# Null methods that count as a real surface spin (anything else is a fallback).
_SURFACE_NULLS = frozenset({"vasa", "alexander_bloch", "moran"})

_ANNOT_SHIM_DONE = False


def enable_annot_surface_nulls() -> None:
    """Make neuromaps' spin-null loader read FreeSurfer ``.annot`` parcellations.

    The pinned engine hands neuromaps ``.annot`` paths, but neuromaps'
    ``load_gifti`` only reads GIFTI.  We wrap it so a ``.annot`` is converted to
    an in-memory GIFTI label image (with a label table, so medial-wall/unknown
    parcels are dropped via neuromaps' ``PARCIGNORE``).  Idempotent and
    best-effort — a no-op if neuromaps/nibabel are unavailable.
    """
    global _ANNOT_SHIM_DONE
    if _ANNOT_SHIM_DONE:
        return
    try:
        import neuromaps.nulls.spins as _spins
        import nibabel as nib
    except Exception:  # pragma: no cover - neuromaps optional at import time
        return

    orig = _spins.load_gifti
    if getattr(orig, "_msnpip_annot_aware", False):
        _ANNOT_SHIM_DONE = True
        return

    def _annot_aware(img):
        try:
            is_annot = isinstance(img, (str, Path)) and str(img).lower().endswith(".annot")
        except Exception:
            is_annot = False
        if not is_annot:
            return orig(img)
        labels, _ctab, names = nib.freesurfer.read_annot(str(img))
        gii = nib.gifti.GiftiImage()
        table = nib.gifti.GiftiLabelTable()
        for i, name in enumerate(names):
            label = nib.gifti.GiftiLabel(key=i)
            label.label = name.decode() if isinstance(name, (bytes, bytearray)) else str(name)
            table.labels.append(label)
        gii.labeltable = table
        gii.add_gifti_data_array(
            nib.gifti.GiftiDataArray(
                labels.astype("int32"), intent="NIFTI_INTENT_LABEL", datatype="NIFTI_TYPE_INT32"
            )
        )
        return gii

    _annot_aware._msnpip_annot_aware = True
    _spins.load_gifti = _annot_aware
    _ANNOT_SHIM_DONE = True
    logger.debug("Enabled .annot-aware load_gifti shim for surface nulls.")


def _is_null_error(exc: BaseException) -> bool:
    """True if *exc* is the engine's NullModelError (by class, with name fallback)."""
    try:
        from imaging_transcriptomics.exceptions import NullModelError

        if isinstance(exc, NullModelError):
            return True
    except Exception:  # pragma: no cover - exceptions module layout drift
        pass
    return type(exc).__name__ == "NullModelError"


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
    """Handle a non-surface (degraded) null per policy.

    Hard-fail only when a surface null is required AND fallback is disallowed;
    otherwise warn that the spatial test degraded to a shuffle.
    """
    used = getattr(getattr(result, "metadata", None), "null_method", None)
    if used is None or used in _SURFACE_NULLS:
        return
    if cfg.require_surface_null and not cfg.allow_null_fallback:
        raise MsnpipSurfaceNullError(
            f"[{method}] Engine reported null_method={used!r} after being asked for "
            f"{cfg.null_method!r}: a surface spin fell back to a grouped shuffle, which "
            "invalidates the spatial-specificity test (allow_null_fallback is off)."
        )
    logger.warning(
        "[%s] Spatial null degraded to %r (surface spin unavailable) — results use a "
        "within-hemisphere shuffle, NOT a spin test. Interpret spatial specificity with caution.",
        method,
        used,
    )


def _call_engine(
    method: str,
    data: np.ndarray,
    input_rh: np.ndarray | None,
    cfg: EngineConfig,
    out_dir: Path,
    null_method: str,
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
        null_method=null_method,
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
    enable_annot_surface_nulls()  # make the DK .annot spin null actually run

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
            contrast_tag,
            method,
            out_dir,
            len(data),
            cfg.null_method,
            cfg.n_permutations,
        )
        try:
            result = _call_engine(method, data, input_rh, cfg, out_dir, cfg.null_method)
        except MsnpipError:
            raise
        except Exception as exc:
            # If the surface null failed and fallback is allowed, retry with 'auto'
            # (engine cascades vasa → alexander_bloch → random).
            if cfg.allow_null_fallback and _is_null_error(exc) and cfg.null_method != "auto":
                logger.warning(
                    "[%s] Surface null %r failed (%s); retrying with null_method='auto' "
                    "(falls back to random if needed).",
                    method,
                    cfg.null_method,
                    exc,
                )
                try:
                    result = _call_engine(method, data, input_rh, cfg, out_dir, "auto")
                except Exception as exc2:
                    raise MsnpipEngineError(
                        f"Engine {method!r} failed for {contrast_tag!r} even after null "
                        f"fallback: {exc2}",
                        cause=exc2,
                    ) from exc2
            else:
                raise MsnpipEngineError(
                    f"Engine {method!r} call failed for contrast {contrast_tag!r}: {exc}",
                    cause=exc,
                ) from exc

        _check_surface_null(result, cfg, method)
        results[method] = result

    return results
