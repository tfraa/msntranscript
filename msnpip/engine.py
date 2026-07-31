"""
Thin wrapper around imaging_transcriptomics.run_pls.

The engine itself runs PLS, the spatial-null permutations, and the enrichment
families, and writes its own TSV/JSON/PNG bundle.  msnpip's job here is narrow:
validate the already-aligned input, dispatch the engine (PLS fits once then
enriches each gene set), wrap engine exceptions with context, and manage the
spatial-null policy.  Two gene-ranking methods are supported: ``pls`` (the
default, multivariate) and ``corr`` (mass-univariate map↔gene correlation).
Both run the *same* spatial-null permutations and feed the *same* corrected
enrichment backends (per-surrogate re-ranked GSEA, template ORA, GCEA).  The
engine's own GSEA — PLS and correlation alike — freezes gene positions at the
observed ranking and is bypassed by default; ``EngineConfig.gsea_backend`` can
re-enable it for a methods comparison, in which case its output is labelled
``gseafrozen`` so it can never be mistaken for the corrected table.

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
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import imaging_transcriptomics as imt
import numpy as np
import pandas as pd

from msnpip.config import EngineConfig
from msnpip.errors import MsnpipEngineError, MsnpipError, MsnpipSurfaceNullError
from msnpip.genes.gsea_mainstyle import run_gsea as run_corrected_gsea

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

    _annot_aware._msnpip_annot_aware = True  # type: ignore[attr-defined]  # idempotency marker
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

    _tolerant_result_column._msnpip_gsea_compat = True  # type: ignore[attr-defined]  # marker
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

    GSEA is run separately by msnpip's own corrected backend (per-surrogate
    re-rank, see :mod:`msnpip.genes.gsea_mainstyle`), not by the engine, so it is
    skipped here.  Falls back to ``"none"`` if the only requested family is GSEA.
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


def _log_enrichment_plan(
    method: str,
    backends: Sequence[str],
    gene_sets: Sequence[str],
    n_permutations: int | None = None,
) -> None:
    """Announce which enrichment backends will actually run, and which are skipped.

    ``--enrichment`` is an append flag: passing any value *replaces* the default
    ``(ensemble, gsea, ora)``, so it is easy to drop a backend without noticing
    (a run with only ``--enrichment ensemble`` silently produces no GSEA/ORA).
    Stating the resolved plan up front makes that visible in the log instead of
    being inferred later from missing files.
    """
    skipped = [m for m in ("ensemble", "gsea", "ora") if m not in backends]
    logger.info(
        "[%s] enrichment backends: %s | gene sets: %s",
        method,
        ", ".join(backends) if backends else "NONE",
        ", ".join(_geneset_label(g) for g in gene_sets),
    )
    if n_permutations is not None:
        # Both spin-null backends consume the full surrogate set; state it, because
        # the resolution floor of an empirical p is 1/(n_permutations+1) and a
        # silently reduced count is invisible in the output tables.
        logger.info(
            "[%s] surrogates used by the spin-null enrichment backends: "
            "ensemble=%d, gsea=%d (empirical p floor 1/%d)",
            method,
            n_permutations,
            n_permutations,
            n_permutations + 1,
        )
    if skipped:
        logger.warning(
            "[%s] enrichment backend(s) NOT requested, so no output will be written: %s. "
            "Pass --enrichment for each backend you want (the flag appends, and using it "
            "at all replaces the default ensemble+gsea+ora).",
            method,
            ", ".join(skipped),
        )


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


#: Backend label written for the engine's own (frozen-rank) GSEA. Deliberately not
#: ``gsea``: the curation step derives the ``enrichment`` column from the filename
#: prefix, so a distinct prefix is what keeps the invalid table from ever being
#: pooled with, or mistaken for, the corrected one in CSVs, plots and the report.
_FROZEN_GSEA_LABEL = "gseafrozen"


def _run_engine_gsea(runner, gene_set, outdir: Path, cfg: EngineConfig, *, kind: str) -> None:
    """Run the pinned engine's own GSEA and file it under ``gseafrozen_*``.

    This is the **invalid** backend, exposed only so a run can reproduce or
    exhibit published v2 behaviour (see ``EngineConfig.gsea_backend``).  The
    engine scores every surrogate at the observed gene positions, so the null
    varies only the running-sum increments and not the hit *order* the enrichment
    score is built on.

    Two further asymmetries make a corrected-vs-engine comparison not a clean
    one-variable contrast, and both are logged rather than silently absorbed:

    * the engine routes the observed ranking through ``gseapy.prerank``, which
      applies its own size window (``max_size=1500`` and gseapy's ``min_size``
      default of 15), while the corrected backend tests every matched term unless
      ``geneset_min_size``/``geneset_max_size`` are set; and
    * the engine's ``fdr`` column is a GSEA-style NES-ratio q-value, not the BH
      FDR the corrected backend and the GCEA table report.

    The engine writes ``gsea_pls<N>_results.tsv`` / ``gsea_corr_results.tsv`` and
    returns nothing, so it is pointed at a scratch directory and the tables are
    renamed on the way out.
    """
    import shutil
    import tempfile

    n_iter = cfg.gsea_engine_n_iter or 1000
    logger.warning(
        "[%s] Running the ENGINE's own GSEA (frozen gene positions, %d surrogates) — "
        "this null is invalid for a rank-position statistic (pure-H0 FPR ~0.7). "
        "Output is labelled %r, NOT 'gsea'. Do not report it as inference.",
        kind,
        n_iter,
        _FROZEN_GSEA_LABEL,
    )
    if cfg.gsea_engine_n_iter is None and cfg.n_permutations != 1000:
        logger.warning(
            "[%s] The engine's GSEA uses its hardcoded 1000 surrogates, discarding %d of "
            "the %d generated. Set gsea_engine_n_iter to use them all.",
            kind,
            cfg.n_permutations - 1000,
            cfg.n_permutations,
        )

    staging = Path(tempfile.mkdtemp(prefix=".gseafrozen_", dir=str(outdir)))
    try:
        if kind == "pls":
            runner.gsea(
                gene_set=gene_set,
                outdir=staging,
                n_iter=n_iter,
                geneset_organism=cfg.geneset_organism,
            )
        else:
            runner.gsea(
                gene_set=gene_set,
                outdir=staging,
                n_perm=n_iter,
                geneset_organism=cfg.geneset_organism,
            )
        produced = sorted(staging.glob("gsea_*.tsv"))
        if not produced:
            raise MsnpipEngineError(f"The engine's {kind!r} GSEA wrote no table into {staging}.")
        for src in produced:
            match = re.search(r"pls(\d+)", src.stem)
            component = match.group(1) if match else "1"
            shutil.move(str(src), str(outdir / f"{_FROZEN_GSEA_LABEL}_pls{component}_results.tsv"))
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _run_toolbox_ora(runner, gene_set, outdir: Path, cfg: EngineConfig, *, kind: str) -> None:
    """Run the pinned engine's OWN over-representation analysis.

    msnpip deliberately has no ORA of its own: the toolbox's implementation is
    the reference, so results are exactly what ``imaging-transcriptomics``
    produces and can be cited as such.  It is the classic template
    (Martins 2022, Giacomel 2026): the gene tail is ``p <= ora_p_threshold`` on
    the *uncorrected* empirical spin p-value, split by the sign of the ranking
    statistic, then a hypergeometric test per term with BH **within direction**.

    Two properties to keep in mind when reading the output:

    * the null is the **random-gene** (hypergeometric) one.  The spin null enters
      only through which genes reach the tail, never through the term test, so
      ORA is never spatial-null inference; and
    * the toolbox drops terms with zero overlap with the tail before correcting,
      so ``m`` is data-dependent and smaller than the full term set.

    The toolbox writes one file per direction (``ora_pls<N>_{up,down}.tsv`` /
    ``ora_corr_{up,down}.tsv``) with no direction column, so the tables are
    staged, tagged with ``direction`` and merged into the single
    ``ora_pls<N>_results.tsv`` the curation and report already consume.
    """
    import shutil
    import tempfile

    staging = Path(tempfile.mkdtemp(prefix=".ora_", dir=str(outdir)))
    try:
        runner.ora(
            gene_set=gene_set,
            outdir=staging,
            p_threshold=cfg.ora_p_threshold,
            geneset_organism=cfg.geneset_organism,
        )
        by_component: dict[str, list[pd.DataFrame]] = {}
        for src in sorted(staging.glob("ora_*.tsv")):
            match = re.search(r"pls(\d+)", src.stem)
            component = match.group(1) if match else "1"
            direction = "positive" if src.stem.endswith("_up") else "negative"
            table = pd.read_csv(src, sep="\t")
            if table.empty:
                continue
            table.insert(1, "direction", direction)
            by_component.setdefault(component, []).append(table)
        if not by_component:
            logger.warning(
                "[%s] ORA selected no genes in either direction (tail is p <= %s on the "
                "uncorrected spin p-value) — no table written.",
                kind,
                cfg.ora_p_threshold,
            )
            return
        for component, tables in by_component.items():
            merged = pd.concat(tables, ignore_index=True)
            merged.to_csv(outdir / f"ora_pls{component}_results.tsv", index=False, sep="\t")
            sizes = merged.groupby("direction")["selected_size"].first().to_dict()
            logger.info(
                "[%s] ORA component %s: tail sizes %s, %d terms tested.",
                kind,
                component,
                sizes,
                len(merged),
            )
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _gene_universe(res_obj):
    """The ranked gene list a size filter counts overlap against, or ``None``.

    Returns ``None`` rather than raising when the result object does not expose a
    gene ranking (test doubles, engine layout drift): an absent universe disables
    the *optional* filter, and :func:`_size_filter_geneset` still hard-fails if a
    window was explicitly requested.
    """
    try:
        return np.asarray(res_obj.orig.genes[0, :], dtype=object)
    except Exception:  # pragma: no cover - defensive
        return None


def _engine_hemisphere(cfg: EngineConfig) -> str:
    """The hemisphere the *engine* is told about.

    ``cfg.hemisphere == "right"`` is a homotopic relabel of the phenotype, not a
    right-hemisphere transcriptome (AHBA samples only 2 of 6 donors on the
    right).  :func:`msnpip.atlas_align.align_strength_to_atlas` has already put
    the ``rh_*`` values into the LEFT label order, so the engine must run as a
    left-hemisphere analysis and pair them with its left expression matrix.
    """
    return "left" if cfg.hemisphere == "right" else cfg.hemisphere


def _size_filter_geneset(gene_set: str, cfg: EngineConfig, gene_universe, outdir: Path, label: str):
    """Return the gene-set resource the backends should test, size-filtered.

    Resolution happens here (not in :mod:`msnpip.genes.sizefilter`) because a
    config gene-set name may be an engine alias like ``"lake"`` rather than a
    path.  When no window is configured the *original* argument is passed through
    untouched — so the run stays bit-identical to an unfiltered one — and the
    would-be filter is only reported.  Resolution/parse failures are never fatal:
    the unfiltered gene set is used and the backend behaves exactly as before.
    """
    from imaging_transcriptomics.genesets import resolve_geneset_resource

    from msnpip.genes.sizefilter import apply_size_filter, size_report

    windowed = cfg.geneset_min_size > 1 or cfg.geneset_max_size is not None
    if gene_universe is None:
        if windowed:
            raise MsnpipEngineError(
                f"--geneset-min-size/--geneset-max-size was requested but the {label!r} "
                "ranked gene universe is unavailable, so matched category sizes cannot "
                "be counted. Remove the window to run unfiltered."
            )
        return gene_set
    try:
        resolved = resolve_geneset_resource(gene_set, organism=cfg.geneset_organism)
        if not windowed:
            report = size_report(
                resolved,
                gene_universe,
                min_size=cfg.geneset_min_size,
                max_size=cfg.geneset_max_size,
            )
            logger.info(
                "geneset %r: NO size filter (%d terms, median matched size %.0f). "
                "Set --geneset-min-size/--geneset-max-size to apply the conventional window.",
                label,
                report.n_terms_in,
                report.median_matched_size,
            )
            return gene_set
        filtered, _report = apply_size_filter(
            resolved,
            gene_universe,
            min_size=cfg.geneset_min_size,
            max_size=cfg.geneset_max_size,
            outdir=outdir,
            label=label,
        )
        return filtered
    except Exception as exc:
        if windowed:
            raise MsnpipEngineError(
                f"Could not apply the category-size filter to gene set {label!r}: {exc}. "
                "Remove --geneset-min-size/--geneset-max-size to run unfiltered.",
                cause=exc,
            ) from exc
        logger.debug("geneset %r: size report unavailable (%s).", label, exc)
        return gene_set


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
    _log_enrichment_plan("pls", backends, gene_sets, cfg.n_permutations)

    config = imt.build_run_config(
        "pls",
        atlas=cfg.atlas,
        hemisphere=_engine_hemisphere(cfg),
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
    gene_universe = _gene_universe(res_obj)
    for gene_set in gene_sets:
        label = _geneset_label(gene_set)
        unfiltered = _resolve_geneset(gene_set)  # bundled .gmt path when available
        sub = enr_root / label
        sub.mkdir(parents=True, exist_ok=True)
        # One size filter for the two spin-null backends, so GCEA and GSEA test an
        # identical term set and each one's BH sees the same m. ORA deliberately
        # keeps the UNFILTERED set: it is the toolbox's own implementation, run
        # exactly as the toolbox runs it (its loader applies no size window), so
        # the output is citable as the reference implementation.
        resolved = _size_filter_geneset(unfiltered, cfg, gene_universe, sub, label)
        for backend in backends:
            try:
                if backend == "ensemble":
                    # n_iter must be passed explicitly: the engine defaults to
                    # 1000 surrogates regardless of how many were generated, which
                    # silently caps GCEA's empirical p at 1/1001 and makes larger
                    # gene sets unable to reach BH significance at all.
                    res_obj.ensemble(
                        gene_set=resolved,
                        outdir=sub,
                        n_iter=cfg.n_permutations,
                        geneset_organism=cfg.geneset_organism,
                    )
                elif backend == "gsea":
                    # Corrected GSEA: per-surrogate re-ranked null (see
                    # msnpip.genes.gsea_mainstyle). The engine's res_obj.gsea
                    # froze gene positions at the observed ranking, which is
                    # anti-conservative; this reuses the engine's ES function on
                    # each spun-map PLS fit's own ranking.
                    if cfg.gsea_backend in ("corrected", "both"):
                        run_corrected_gsea(
                            res_obj,
                            gene_set=resolved,
                            outdir=sub,
                            geneset_organism=cfg.geneset_organism,
                            n_jobs=cfg.n_jobs,
                        )
                    if cfg.gsea_backend in ("engine", "both"):
                        _run_engine_gsea(res_obj, resolved, sub, cfg, kind="pls")
                elif backend == "ora":
                    _run_toolbox_ora(res_obj, unfiltered, sub, cfg, kind="pls")
                logger.info("enrichment[%s] gene set %r → %s", backend, label, sub)
            except Exception as exc:
                logger.warning("enrichment[%s] failed for gene set %r: %s", backend, gene_set, exc)
    return result


def _corr_enrichment_adapter(corr_genes):
    """Present the engine's ``CorrGenes`` as the interface the corrected GSEA/ORA
    backends read from a PLS result object.

    The corrected GSEA (:func:`msnpip.genes.gsea_mainstyle.run_gsea`) and template
    ORA (:func:`msnpip.genes.ora_mainstyle.run_ora`) only touch
    ``n_components``/``orig.genes``/``orig.zscored``/``boot.weights``.  A
    correlation run has a single "component": the observed z-scored correlations
    give the ranking, and the per-surrogate correlation nulls are the boot weights
    (the GSEA backend re-ranks each surrogate column itself).
    """
    from scipy.stats import zscore

    if corr_genes.boot_corr is None:
        raise MsnpipEngineError(
            "corr enrichment needs stored permutation correlations (store_boot_corr was disabled)."
        )
    genes = np.asarray(corr_genes.genes[:, 0], dtype=object).reshape(1, -1)
    observed = np.asarray(corr_genes.corr[0, :], dtype=float)
    zscored = zscore(observed, ddof=1).reshape(1, -1)
    boot = np.asarray(corr_genes.boot_corr, dtype=float)[None, :, :]
    # ORA's `p` tail also needs per-gene spin p-values. On the PLS path `orig`
    # (weight-sorted) and `boot` (z-sorted) are in DIFFERENT row orders, so the
    # tail rules read each namespace separately; here CorrGenes.sort_genes()
    # reorders genes/corr/pval/boot_corr with one shared index, so both
    # namespaces are the same order and either is safe to pair.
    pvals = np.asarray(corr_genes.pval[0, :], dtype=float).reshape(1, -1)
    return SimpleNamespace(
        n_components=1,
        orig=SimpleNamespace(genes=genes, zscored=zscored),
        boot=SimpleNamespace(weights=boot, genes=genes, z_score=zscored, pval=pvals),
    )


def _run_corr_fit_once_enrich_many(
    data: np.ndarray,
    input_rh: np.ndarray | None,
    cfg: EngineConfig,
    out_dir: Path,
):
    """Fit the correlation ranking once, then enrich every configured gene set.

    Mirrors :func:`_run_pls_fit_once_enrich_many` but for the mass-univariate
    correlation backend: the observed and per-surrogate map↔gene correlations are
    computed with the engine's own ``CorrAnalysis`` (same spatial null as PLS),
    the standard correlation bundle is written, and the corrected enrichment
    (GCEA / re-ranked GSEA / template ORA) is run per gene set on the correlation
    ranking.  The engine's own correlation GSEA is *not* used — it freezes gene
    positions at the observed ranking, the same defect corrected for PLS.
    """
    enable_annot_surface_nulls()
    enable_gsea_compat()
    try:
        from imaging_transcriptomics.corr import CorrAnalysis
        from imaging_transcriptomics.models import CorrelationResult
        from imaging_transcriptomics.nulls import permute_scan_values
        from imaging_transcriptomics.scan import regional_values_frame
        from imaging_transcriptomics.serialization import write_result_bundle
        from imaging_transcriptomics.workflows.shared import (
            corr_gene_table,
            prepare_analysis_inputs,
            result_metadata,
        )
    except ImportError as exc:  # pragma: no cover - engine layout drift
        raise MsnpipEngineError(
            "msnpip's correlation path depends on the pinned "
            "imaging-transcriptomics internal API, which appears to have changed "
            f"({exc}). Re-pin the engine to commit e6a2c237 or update "
            "engine._run_corr_fit_once_enrich_many to the new layout.",
            cause=exc,
        ) from exc

    primary = _primary_enrichment(cfg.enrichment_methods)
    backends = [m for m in ("ensemble", "gsea", "ora") if m in cfg.enrichment_methods]
    gene_sets = list(cfg.gene_sets) or ["lake"]
    _log_enrichment_plan("corr", backends, gene_sets, cfg.n_permutations)

    config = imt.build_run_config(
        "corr",
        atlas=cfg.atlas,
        hemisphere=_engine_hemisphere(cfg),
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
        seed=cfg.seed,
        n_jobs=cfg.n_jobs,
    )

    extracted, gene_exp, gene_labels, imaging = prepare_analysis_inputs(
        data, config, input_rh=input_rh
    )

    # Spatial null with the same fallback policy as the PLS path.
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
                "[corr] Surface null %r failed (%s); retrying with null_method='auto'.",
                cfg.null_method,
                exc,
            )
            permuted, resolved_null = _permute("auto")
        else:
            raise MsnpipEngineError(
                f"Engine 'corr' null generation failed: {exc}", cause=exc
            ) from exc

    analysis = CorrAnalysis(
        n_iterations=config.n_permutations,
        n_genes=gene_labels.shape[0],
        store_boot_corr=True,  # needed for the re-ranked GSEA + GCEA nulls
        n_jobs=config.n_jobs,
    )
    analysis.bootstrap_correlation(imaging, permuted, gene_exp, gene_labels)

    result = CorrelationResult(
        metadata=result_metadata(extracted, config, null_method=resolved_null),
        regional_values=regional_values_frame(extracted),
        gene_table=corr_gene_table(analysis),
        gsea_table=None,
        ensemble_table=None,
        ora_tables=None,
        output_dir=out_dir,
    )
    # Write the correlation bundle once (corr_genes.tsv, regional values, plots).
    write_result_bundle(result, out_dir)

    # Enrich every gene set on the correlation ranking (cheap; no re-permutation).
    corr_genes = analysis.gene_results.results
    adapter = _corr_enrichment_adapter(corr_genes)
    enr_root = out_dir / "enrichment"
    gene_universe = _gene_universe(adapter)
    for gene_set in gene_sets:
        label = _geneset_label(gene_set)
        unfiltered = _resolve_geneset(gene_set)  # bundled .gmt path when available
        sub = enr_root / label
        sub.mkdir(parents=True, exist_ok=True)
        # Size filter for the spin-null backends only; ORA keeps the unfiltered
        # set (see the PLS path).
        resolved = _size_filter_geneset(unfiltered, cfg, gene_universe, sub, label)
        for backend in backends:
            try:
                if backend == "ensemble":
                    # Engine GCEA is order-independent (category means), so it is
                    # correct as-is; only the output filename is normalised to the
                    # ``pls1`` scheme the curation/report already consume.
                    ens = analysis.ensemble(
                        gene_set=resolved,
                        outdir=None,
                        n_perm=cfg.n_permutations,
                        geneset_organism=cfg.geneset_organism,
                    )
                    ens.to_csv(sub / "ensemble_pls1_results.tsv", index=False, sep="\t")
                elif backend == "gsea":
                    if cfg.gsea_backend in ("corrected", "both"):
                        run_corrected_gsea(
                            adapter,
                            gene_set=resolved,
                            outdir=sub,
                            geneset_organism=cfg.geneset_organism,
                            n_jobs=cfg.n_jobs,
                        )
                    if cfg.gsea_backend in ("engine", "both"):
                        # CorrAnalysis owns the engine's correlation GSEA, not the
                        # adapter (which only exposes what the corrected backends read).
                        _run_engine_gsea(analysis, resolved, sub, cfg, kind="corr")
                elif backend == "ora":
                    # CorrAnalysis owns the toolbox's correlation ORA, not the
                    # adapter (which only exposes what the corrected GSEA reads).
                    _run_toolbox_ora(analysis, unfiltered, sub, cfg, kind="corr")
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
            fit = _run_pls_fit_once_enrich_many
        elif method == "corr":
            fit = _run_corr_fit_once_enrich_many
        else:
            raise MsnpipEngineError(
                f"Unsupported engine method {method!r}; expected 'pls' or 'corr'."
            )
        # Fit once, enrich every gene set (avoids re-running the spatial null
        # per gene set; the engine couples them in a single-gene-set call).
        # The spatial-null fallback policy is handled inside the fit-once
        # helpers (see the _permute retry there).
        try:
            result = fit(data, input_rh, cfg, out_dir)
        except MsnpipError:
            raise
        except Exception as exc:
            raise MsnpipEngineError(
                f"Engine {method!r} call failed for contrast {contrast_tag!r}: {exc}",
                cause=exc,
            ) from exc

        _check_surface_null(result, cfg, method)
        results[method] = result

    return results
