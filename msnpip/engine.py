"""
Thin wrapper around imaging_transcriptomics.run_pls / run_corr.
Phase 3, Task T3.1.

The engine itself runs PLS/correlation, the spatial-null permutations, and the
enrichment families, and writes its own TSV/JSON/PNG bundle.  msnpip's job here
is narrow: validate the already-aligned input, dispatch the engine (PLS fits
once then enriches each gene set; corr uses a single call), wrap engine
exceptions with context, and manage the spatial-null policy.

Surface-null note: the pinned engine ships the DK parcellation only as a
FreeSurfer ``.annot``, which neuromaps' spin-null loader (``load_gifti``) cannot
read — so ``vasa``/``alexander_bloch`` fail for DK as shipped. We install a small
``.annot``-aware ``load_gifti`` shim so the real spin null runs; if it still
fails and ``allow_null_fallback`` is set, we let the engine fall back to
``auto``/``random`` (recording the resolved null) rather than hard-failing.
"""

from __future__ import annotations

import logging
import re
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


_GSEA_SHIM_DONE = False


def enable_gsea_compat() -> None:
    """Make the engine's GSEA tolerate gseapy >=1.x preranked output columns.

    The pinned engine's ``gene_stats.pls.gsea`` reads pre-1.0 gseapy column names
    (``geneset_size``/``matched_size``/``matched_genes``/``ledge_genes``) that
    gseapy 1.x renamed/dropped (``Tag %``/``Gene %``/``Lead_genes``), so its
    ``result_column`` lookup raises and GSEA fails.  Those columns are only
    written into the output table as metadata — the ES/NES/p/FDR statistics come
    from the engine's own bootstrap nulls — so we wrap ``result_column`` to return
    a benign default (NaN for sizes, "" for gene lists) when no candidate column
    exists, instead of raising.  Idempotent and best-effort.
    """
    global _GSEA_SHIM_DONE
    if _GSEA_SHIM_DONE:
        return
    try:
        import imaging_transcriptomics.gene_stats.pls as _gpls
    except Exception:  # pragma: no cover - layout drift
        return
    orig = getattr(_gpls, "result_column", None)
    if orig is None or getattr(orig, "_msnpip_gsea_compat", False):
        _GSEA_SHIM_DONE = True
        return

    def _tolerant_result_column(res2d, *candidates):
        try:
            return orig(res2d, *candidates)
        except Exception:
            n = len(res2d)
            textish = any(("gene" in c or "ledge" in c) for c in candidates)
            return np.array([""] * n, dtype=object) if textish else np.full(n, np.nan)

    _tolerant_result_column._msnpip_gsea_compat = True
    _gpls.result_column = _tolerant_result_column
    _GSEA_SHIM_DONE = True
    logger.debug("Enabled gseapy>=1.x compatibility shim for engine GSEA.")


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
    """Dispatch a single run_pls / run_corr call with shared kwargs.

    The engine takes ONE gene set per call (``gene_set: str``); msnpip's
    multi-gene-set support is handled separately (PLS fits once then enriches
    each set — see :func:`_run_pls_fit_once_enrich_many`).  This path is used for
    ``corr`` and passes a single gene set to avoid the engine's tuple crash.
    """
    primary = _primary_enrichment(cfg.enrichment_methods)
    run_gsea = "gsea" in cfg.enrichment_methods
    gene_set = _resolve_geneset(cfg.gene_sets[0] if cfg.gene_sets else "lake")

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
        gene_set=gene_set,
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


def _geneset_label(gene_set: str) -> str:
    """Filename-safe label for a gene set (name or local ``.gmt`` path)."""
    s = str(gene_set)
    p = Path(s)
    if p.suffix.lower() == ".gmt" or "/" in s or "\\" in s:
        s = p.stem
    s = s.replace("geneset_", "")
    return re.sub(r"[^A-Za-z0-9._+-]", "_", s) or "geneset"


# Back-compat aliases for gene-set names that differ from the bundled file stems.
_GENESET_ALIASES = {"kegg_2021_human": "KEGG_2021_H"}


def _resolve_geneset(gene_set: str) -> str:
    """Resolve a gene-set name to a bundled ``.gmt`` path so it runs offline.

    Resolution order: an explicit existing ``.gmt`` path is used as-is; otherwise
    the name (or a known alias) is matched against the ``.gmt`` files bundled in
    :mod:`msnpip.genes`.  Anything unmatched is passed through unchanged so the
    engine can resolve it (its packaged ``lake``/``pooled`` sets, or a gseapy
    download as a last resort).
    """
    s = str(gene_set)
    p = Path(s)
    if p.suffix.lower() == ".gmt" and p.exists():
        return s
    from msnpip import genes as _genes

    for candidate in (s, _GENESET_ALIASES.get(s.lower())):
        if not candidate:
            continue
        try:
            return _genes.get_library_path(candidate)
        except FileNotFoundError:
            continue
    return s


def _run_pls_fit_once_enrich_many(
    data: np.ndarray,
    input_rh: np.ndarray | None,
    cfg: EngineConfig,
    out_dir: Path,
):
    """Fit PLS once, then run enrichment for every configured gene set.

    The engine's ``run_pls`` couples PLS and enrichment for a single gene set
    and discards the fitted analysis object.  To support multiple gene sets
    without re-fitting (and re-running the expensive spatial-null permutations)
    for each, we drive the engine's PLS workflow primitives directly: fit +
    permute + bootstrap once, write the standard PLS bundle, then call the cheap
    per-gene-set ``ensemble``/``gsea``/``ora`` enrichment on the already-fit
    model.  Each gene set's enrichment tables land in
    ``out_dir/enrichment/<label>/`` for curation.
    """
    enable_annot_surface_nulls()
    enable_gsea_compat()
    # These reach into the *pinned* engine's internal API (so PLS can be fit once
    # and enriched per gene set). If the engine ever moves, fail fast with a clear
    # message instead of an opaque ImportError mid-run.
    try:
        from imaging_transcriptomics.models import PLSResult
        from imaging_transcriptomics.nulls import permute_scan_values
        from imaging_transcriptomics.pls import PLSAnalysis
        from imaging_transcriptomics.scan import regional_values_frame
        from imaging_transcriptomics.serialization import write_result_bundle
        from imaging_transcriptomics.workflows.shared import (
            pls_components,
            prepare_analysis_inputs,
            result_metadata,
        )
    except ImportError as exc:  # pragma: no cover - engine layout drift
        raise MsnpipEngineError(
            "msnpip's fit-once/enrich-many PLS path depends on the pinned "
            "imaging-transcriptomics internal API, which appears to have changed "
            f"({exc}). Re-pin the engine to commit e6a2c237 or update "
            "engine._run_pls_fit_once_enrich_many to the new layout.",
            cause=exc,
        ) from exc

    primary = _primary_enrichment(cfg.enrichment_methods)
    backends = [m for m in ("ensemble", "gsea", "ora") if m in cfg.enrichment_methods]
    gene_sets = list(cfg.gene_sets) or ["lake"]

    config = imt.build_run_config(
        "pls",
        atlas=cfg.atlas,
        hemisphere=cfg.hemisphere,
        regions=cfg.regions,
        source_space=None,
        n_permutations=cfg.n_permutations,
        null_method=cfg.null_method,
        output_dir=out_dir,
        enrichment_method=primary,
        run_gsea=False,
        gene_set=gene_sets[0],
        geneset_organism=cfg.geneset_organism,
        ora_p_threshold=cfg.ora_p_threshold,
        n_components=cfg.n_components,
        var=cfg.var,
        seed=cfg.seed,
        n_jobs=cfg.n_jobs,
    )

    extracted, gene_exp, gene_labels, imaging = prepare_analysis_inputs(
        data, config, input_rh=input_rh
    )

    # Spatial null with the same fallback policy as the single-call path.
    def _permute(null_method):
        return permute_scan_values(
            extracted,
            n_permutations=config.n_permutations,
            null_method=null_method,
            seed=config.seed,
        )

    try:
        permuted, resolved_null = _permute(config.null_method)
    except MsnpipError:
        raise
    except Exception as exc:
        if cfg.allow_null_fallback and _is_null_error(exc) and cfg.null_method != "auto":
            logger.warning(
                "[pls] Surface null %r failed (%s); retrying with null_method='auto'.",
                cfg.null_method,
                exc,
            )
            permuted, resolved_null = _permute("auto")
        else:
            raise MsnpipEngineError(
                f"Engine 'pls' null generation failed: {exc}", cause=exc
            ) from exc

    analysis = PLSAnalysis(
        imaging,
        gene_exp,
        n_components=config.n_components,
        var=config.var,
        n_iter=config.n_permutations,
        n_jobs=config.n_jobs,
    )
    analysis.boot_pls(
        imaging, permuted, gene_exp, scan_data=extracted.values, gene_labels=gene_labels
    )
    analysis.gene_results.results.compute(n_jobs=config.n_jobs)

    n = analysis.n_components
    result = PLSResult(
        metadata=result_metadata(extracted, config, null_method=resolved_null, n_components=n),
        regional_values=regional_values_frame(extracted),
        components=pls_components(analysis, [None] * n, [None] * n, [None] * n),
        cumulative_variance=np.cumsum(analysis.components_var),
        output_dir=out_dir,
    )
    # Write the PLS bundle once (tables + variance/gene plots; no enrichment yet).
    write_result_bundle(result, out_dir)

    # Enrich every gene set on the already-fit model (cheap; no re-permutation).
    res_obj = analysis.gene_results.results
    enr_root = out_dir / "enrichment"
    for gene_set in gene_sets:
        label = _geneset_label(gene_set)
        resolved = _resolve_geneset(gene_set)  # bundled .gmt path when available
        sub = enr_root / label
        sub.mkdir(parents=True, exist_ok=True)
        for backend in backends:
            try:
                if backend == "ensemble":
                    res_obj.ensemble(
                        gene_set=resolved, outdir=sub, geneset_organism=cfg.geneset_organism
                    )
                elif backend == "gsea":
                    res_obj.gsea(
                        gene_set=resolved, outdir=sub, geneset_organism=cfg.geneset_organism
                    )
                elif backend == "ora":
                    res_obj.ora(
                        gene_set=resolved,
                        outdir=sub,
                        p_threshold=cfg.ora_p_threshold,
                        geneset_organism=cfg.geneset_organism,
                    )
                logger.info("enrichment[%s] gene set %r → %s", backend, label, sub)
            except Exception as exc:
                logger.warning("enrichment[%s] failed for gene set %r: %s", backend, gene_set, exc)
    return result


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
    enable_gsea_compat()  # let engine GSEA read gseapy >=1.x preranked columns

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
        if method == "pls":
            # Fit once, enrich every gene set (avoids re-running the spatial null
            # per gene set; the engine couples them in a single-gene-set call).
            try:
                result = _run_pls_fit_once_enrich_many(data, input_rh, cfg, out_dir)
            except MsnpipError:
                raise
            except Exception as exc:
                raise MsnpipEngineError(
                    f"Engine 'pls' call failed for contrast {contrast_tag!r}: {exc}",
                    cause=exc,
                ) from exc
        else:
            if len(cfg.gene_sets) > 1:
                logger.warning(
                    "[corr] enrichment runs only the first gene set (%s); "
                    "multi-gene-set enrichment is currently PLS-only.",
                    _geneset_label(cfg.gene_sets[0]),
                )
            try:
                result = _call_engine(method, data, input_rh, cfg, out_dir, cfg.null_method)
            except MsnpipError:
                raise
            except Exception as exc:
                # If the surface null failed and fallback is allowed, retry with
                # 'auto' (engine cascades vasa → alexander_bloch → random).
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
