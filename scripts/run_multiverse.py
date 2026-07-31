#!/usr/bin/env python
"""Run the multiverse grid of msnpip specifications back-to-back, unattended.

The grid crosses the three choices that actually move the result — MSN edge
definition, contrast statistic, and gene-ranking method — and runs every cell
with identical everything else, so the cells are comparable (a specification-curve
analysis rather than a hunt for the combination that happens to be significant).
The scanner covariate is a fourth configured arm, deliberately NOT scheduled in
the primary grid: it is the sensitivity pass, run once the primary cell has been
pre-specified (add "scan" to COVARIATE_ARMS).

Runs are executed **one at a time on purpose**: each run peaks at several GB, so
running them concurrently would swap and take longer than the serial schedule.

Designed to be started in the evening and read in the morning:

* already-completed cells are skipped, so re-running resumes where it stopped;
* a failing cell is recorded and the schedule continues to the next one;
* every cell writes its own log, plus a summary table at the end.

Usage (edit the CONFIG block below first, or pass the paths as flags)::

    python scripts/run_multiverse.py --dry-run     # print the plan, run nothing
    python scripts/run_multiverse.py               # run the whole grid
    python scripts/run_multiverse.py --force       # redo cells already done

Exit code is 0 if every attempted cell succeeded, 1 otherwise.
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------
# CONFIG — edit these to match your data, or override with the CLI flags.
# --------------------------------------------------------------------------
# Input: either a single merged table (DATAFRAME) or a FreeSurfer directory plus
# a demographics file (INPUT_DIR + DEMOGRAPHICS). Leave the unused one as None.
DATAFRAME: str | None = None
INPUT_DIR: str | None = r"C:\Users\Utente\OneDrive\Desktop\datasets\ftd_orig"
DEMOGRAPHICS: str | None = r"C:\Users\Utente\OneDrive\Desktop\datasets\ftd_orig\Database_morphometric_similarity.xlsx"
ID_COL: str | None = None  # e.g. "subject_id"; None = autodetect

OUTPUT_ROOT = r"C:\Users\Utente\OneDrive\Desktop\FINRESULTS\ORA_nof"

GROUP_COL = "group"
CONTRASTS: list[tuple[str, str]] = [("1", "0"), ("2", "0"), ("3", "0")]
POOL_CASES = False

# Covariate arms. The scanner arm is the confound-robust one.
COVARIATES = {
    "noscan": ["age", "sex", "tiv"],
    "scan": ["age", "sex", "tiv", "SCANNER"],
}

N_PERM = 20000
NCOMP = 1  # keep 2: PLS2 has carried the signal more often than PLS1
N_JOBS = 1  # the GSEA inner loop is vectorised; extra workers mostly add RAM
SEED = 1234
NULL_METHOD = "vasa"

# Primary analysis is left-hemisphere (34 DK regions), as in every run to date.
# "right" is the secondary homotopic-relabel arm (left AHBA expression, right
# phenotype); run it separately with --hemisphere, not as a grid axis.
HEMISPHERE = "left"

GENESETS = [
    "lake",
    "pooled",
    "KEGG_2021_H",
    "GO_Biological_Process_2025",
    "DisGeNET",
]
ENRICHMENT = ["ora"]  # pass all three: the flag replaces the default

# Category-size window, counted after intersecting each term with the ranked gene
# universe and applied ONCE upstream so GCEA, both GSEAs and all three ORA tails
# test an identical term set (and each one's BH sees the same m). 15-500 is
# GSEA's own default. PRE-SPECIFIED — never tune this on the results.
GENESET_MIN_SIZE = 1
GENESET_MAX_SIZE = 1500

# Emit the corrected GSEA *and* the engine's frozen-ranking GSEA, at the same
# surrogate count, so the methods comparison varies only the null and not n.
GSEA_BACKEND = "both"
GSEA_ENGINE_N_ITER = N_PERM

# Grid axes.
MSN_VALUES = {"dist": "distance", "corr": "correlation"}
STAT_VALUES = ["t", "beta"]
METHOD_VALUES = ["pls", "corr"]
# Scanner is deferred: the primary grid is covariate-fixed, and the scanner arm
# is run afterwards as a sensitivity pass on whichever cell is pre-specified.
# Add "scan" here (or pass --only scan) when that pass is due.
COVARIATE_ARMS = ["noscan"]

DONE_MARKER = ".msnpip_complete"


def build_command(msn: str, stat: str, method: str, cov: str, out_dir: Path) -> list[str]:
    """Assemble one msnpip invocation for a grid cell."""
    cmd = [sys.executable, "-m", "msnpip.cli", "full"]
    if DATAFRAME:
        cmd += ["--dataframe", DATAFRAME]
    else:
        cmd += ["--input", str(INPUT_DIR), "--demographics", str(DEMOGRAPHICS)]
    if ID_COL:
        cmd += ["--id-col", ID_COL]
    cmd += ["--output", str(out_dir), "--group-col", GROUP_COL]
    for case, control in CONTRASTS:
        cmd += ["--contrast", case, control]
    if POOL_CASES:
        cmd += ["--pool-cases"]
    cmd += [
        "--msn-similarity",
        MSN_VALUES[msn],
        "--contrast-stat",
        stat,
        "--method",
        method,
        "--ncomp",
        str(NCOMP),
        "--hemisphere",
        HEMISPHERE,
        "--null-method",
        NULL_METHOD,
        "--n-perm",
        str(N_PERM),
        "--n-jobs",
        str(N_JOBS),
        "--seed",
        str(SEED),
        "--geneset-min-size",
        str(GENESET_MIN_SIZE),
        "--geneset-max-size",
        str(GENESET_MAX_SIZE),
        "--gsea-backend",
        GSEA_BACKEND,
        "--gsea-engine-n-iter",
        str(GSEA_ENGINE_N_ITER),
    ]
    cmd += ["--predictors", *COVARIATES[cov]]
    cmd += ["--geneset", *GENESETS]
    for backend in ENRICHMENT:
        cmd += ["--enrichment", backend]
    return cmd


def cells() -> list[tuple[str, str, str, str]]:
    """Every grid cell, in a stable order."""
    return list(itertools.product(MSN_VALUES, STAT_VALUES, METHOD_VALUES, COVARIATE_ARMS))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="print the plan without running")
    ap.add_argument("--force", action="store_true", help="re-run cells already marked complete")
    ap.add_argument("--output-root", default=OUTPUT_ROOT, help="where the grid is written")
    ap.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="TOKEN",
        help=(
            "run only cells matching all these axis values, e.g. --only corr pls. "
            "Tokens match a whole axis value, so 'scan' does not select 'noscan'. "
            "The scanner arm is not in the default grid — add 'scan' to "
            "COVARIATE_ARMS to schedule the sensitivity pass."
        ),
    )
    args = ap.parse_args()

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    log_root = root / "_logs"
    log_root.mkdir(exist_ok=True)

    plan = cells()
    if args.only:
        # Match whole axis values, not substrings, so "scan" never selects "noscan".
        plan = [c for c in plan if all(tok in set(c) for tok in args.only)]
        # Validate against every CONFIGURED axis value, not just the ones in the
        # current grid: "scan" is a real arm that is simply not scheduled yet, so
        # it must produce "no cell matches", not "unknown token".
        known = set(MSN_VALUES) | set(STAT_VALUES) | set(METHOD_VALUES) | set(COVARIATES)
        unknown = set(args.only) - known
        if unknown:
            print(f"Unknown --only token(s): {', '.join(sorted(unknown))}", file=sys.stderr)
            return 1
        if not plan:
            print("No cell matches the --only filter.", file=sys.stderr)
            return 1

    print(f"Grid: {len(plan)} cell(s) | output root: {root}")
    print(f"n_perm={N_PERM} ncomp={NCOMP} n_jobs={N_JOBS} null={NULL_METHOD} hemi={HEMISPHERE}")
    print(f"genesets={','.join(GENESETS)} enrichment={','.join(ENRICHMENT)}")
    print(
        f"category size window={GENESET_MIN_SIZE}-{GENESET_MAX_SIZE} (pre-specified) | "
        f"gsea_backend={GSEA_BACKEND} (frozen arm at {GSEA_ENGINE_N_ITER} surrogates)"
    )
    print(
        "ORA emits three fixed tails per gene set: oraz (|z|>=3), orap (spin p<=0.05), "
        "oratopn (top/bottom 500)."
    )
    print("Cells run one at a time (each peaks at several GB).\n")

    results: list[tuple[str, str, float]] = []
    started = time.time()

    for index, (msn, stat, method, cov) in enumerate(plan, start=1):
        name = f"{msn}/{stat}/{method}/{cov}"
        out_dir = root / msn / stat / method / cov
        marker = out_dir / DONE_MARKER

        if marker.exists() and not args.force:
            print(f"[{index}/{len(plan)}] SKIP (already complete)  {name}")
            results.append((name, "skipped", 0.0))
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_command(msn, stat, method, cov, out_dir)

        if args.dry_run:
            print(f"[{index}/{len(plan)}] {name}\n    {' '.join(cmd)}")
            results.append((name, "dry-run", 0.0))
            continue

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = log_root / f"{msn}-{stat}-{method}-{cov}-{stamp}.log"
        print(f"[{index}/{len(plan)}] RUN   {name}")
        print(f"    started {datetime.now():%H:%M:%S} | log: {log_path.name}", flush=True)

        cell_start = time.time()
        try:
            with log_path.open("w", encoding="utf-8") as log:
                log.write(" ".join(cmd) + "\n\n")
                log.flush()
                completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
            rc = completed.returncode
        except KeyboardInterrupt:
            print("\nInterrupted — stopping the schedule.")
            return 1
        except Exception as exc:  # keep the night going even on an unexpected failure
            print(f"    ERROR launching: {exc}")
            results.append((name, "launch-error", time.time() - cell_start))
            continue

        elapsed = time.time() - cell_start
        if rc == 0:
            marker.write_text(f"completed {datetime.now():%Y-%m-%d %H:%M:%S}\n", encoding="utf-8")
            print(f"    OK in {elapsed / 60:.1f} min")
            results.append((name, "ok", elapsed))
        else:
            print(f"    FAILED (exit {rc}) after {elapsed / 60:.1f} min — see {log_path}")
            results.append((name, f"failed({rc})", elapsed))

    total = time.time() - started
    print("\n" + "=" * 64)
    print(f"{'cell':34} {'status':14} {'minutes':>8}")
    for name, status, elapsed in results:
        print(f"{name:34} {status:14} {elapsed / 60:8.1f}")
    ok = sum(1 for _, s, _ in results if s == "ok")
    failed = [n for n, s, _ in results if s.startswith("failed") or s == "launch-error"]
    print("=" * 64)
    print(f"Total wall time: {total / 60:.1f} min | ok={ok} | failed={len(failed)}")
    if failed:
        print("Failed cells: " + ", ".join(failed))

    summary = root / "_logs" / "summary.txt"
    summary.write_text(
        "\n".join(f"{n}\t{s}\t{e / 60:.1f}min" for n, s, e in results) + "\n", encoding="utf-8"
    )
    print(f"Summary written to {summary}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
