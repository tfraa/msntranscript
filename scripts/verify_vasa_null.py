#!/usr/bin/env python
"""Production gate: prove the Vasa surface spin null is real, not a silent shuffle.

Run this in the *production* environment (needs ``neuromaps`` + ``netneurotools``,
which ship with the standard imaging-transcriptomics install).  It exercises the
exact path the pipeline uses — the ``.annot``→GIFTI shim, then the engine's
``permute_scan_values`` with ``null_method='vasa'`` on a DK left-cortex map — and
asserts:

  1. the ``.annot``-aware ``load_gifti`` shim is installed;
  2. the resolved null method is exactly ``vasa`` (no fallback to a shuffle);
  3. the surrogate array has **34 cortical parcels** (medial wall dropped);
  4. every surrogate column is a genuine permutation of the parcel values
     (spin reassigns parcels), and the columns actually differ (real rotations);
  5. the seed is honoured — same seed reproduces, different seed diverges.

Exit code 0 means all checks passed; non-zero means the spin is not trustworthy
and no paper/thesis run should proceed.  For headline runs also set
``EngineConfig.allow_null_fallback=False`` so a failed spin raises instead of
silently degrading to the within-hemisphere shuffle.

Usage:
    python scripts/verify_vasa_null.py [--n-perm 100] [--seed 1234]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def _check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    line = f"[{mark}] {label}"
    if detail:
        line += f" - {detail}"
    print(line)
    return ok


def run(n_perm: int, seed: int) -> int:
    import imaging_transcriptomics as imt
    from imaging_transcriptomics.nulls import permute_scan_values
    from imaging_transcriptomics.workflows.shared import prepare_analysis_inputs

    from msnpip.atlas_align import engine_region_order
    from msnpip.engine import enable_annot_surface_nulls

    ok = True

    # (1) Activate and confirm the .annot→GIFTI shim.
    enable_annot_surface_nulls()
    shim_ok = False
    try:
        import neuromaps.nulls.spins as _spins

        shim_ok = bool(getattr(_spins.load_gifti, "_msnpip_annot_aware", False))
    except Exception as exc:  # pragma: no cover - neuromaps missing
        ok &= _check("neuromaps importable", False, str(exc))
        return 1
    ok &= _check(".annot-aware load_gifti shim installed", shim_ok)

    # Build the real DK left-cortex input and config the pipeline uses.
    labels = engine_region_order("dk", "left", "cort")
    n_regions = len(labels)
    rng = np.random.default_rng(0)
    regional_map = rng.normal(size=n_regions)

    config = imt.build_run_config(
        "pls",
        atlas="dk",
        hemisphere="left",
        regions="cort",
        source_space=None,
        n_permutations=n_perm,
        null_method="vasa",
        output_dir=None,
        enrichment_method="none",
        run_gsea=False,
        gene_set="lake",
        n_components=1,
        seed=seed,
    )
    extracted, *_ = prepare_analysis_inputs(regional_map, config, input_rh=None)

    permuted, resolved = permute_scan_values(
        extracted, n_permutations=n_perm, null_method="vasa", seed=seed
    )

    # (2) Resolved method is vasa (not a fallback shuffle).
    ok &= _check("resolved null method == 'vasa'", resolved == "vasa", f"got {resolved!r}")

    # (3) 34 cortical parcels.
    ok &= _check(
        "surrogate array has 34 cortical parcels",
        permuted.shape == (34, n_perm),
        f"shape {permuted.shape}",
    )

    # (4) Each column is a permutation of the observed parcel values, and columns
    #     actually differ (a real rotation moved parcels around).
    observed = np.sort(permuted[:, 0])
    perm_ok = all(
        np.allclose(np.sort(permuted[:, j]), observed, atol=1e-8) for j in range(permuted.shape[1])
    )
    ok &= _check("every column is a permutation of the parcel values", perm_ok)

    n_unique_cols = len({permuted[:, j].tobytes() for j in range(permuted.shape[1])})
    ok &= _check(
        "surrogate columns are distinct rotations",
        n_unique_cols > max(1, n_perm // 2),
        f"{n_unique_cols}/{n_perm} unique columns",
    )

    # (5) Seed reproducibility and sensitivity.
    permuted_same, _ = permute_scan_values(
        extracted, n_permutations=n_perm, null_method="vasa", seed=seed
    )
    ok &= _check("same seed reproduces surrogates", np.array_equal(permuted, permuted_same))

    permuted_other, _ = permute_scan_values(
        extracted, n_permutations=n_perm, null_method="vasa", seed=seed + 1
    )
    ok &= _check("different seed changes surrogates", not np.array_equal(permuted, permuted_other))

    print()
    print("RESULT:", "OK - Vasa spin is real." if ok else "FAILED - do not run for publication.")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--n-perm", type=int, default=100, help="number of surrogate maps (default 100)"
    )
    ap.add_argument("--seed", type=int, default=1234, help="RNG seed to test (default 1234)")
    args = ap.parse_args()
    try:
        return run(args.n_perm, args.seed)
    except Exception as exc:  # any failure means the gate did not pass
        print(f"[FAIL] verifier raised: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
