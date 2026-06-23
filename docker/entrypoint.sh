#!/usr/bin/env bash
# Optional entrypoint wrapper for the msnpip image.
# The Dockerfile uses `ENTRYPOINT ["msnpip"]` directly; this script is provided
# for environments that want to run setup steps before the CLI.
set -euo pipefail

export MPLBACKEND="${MPLBACKEND:-Agg}"
export LC_ALL="${LC_ALL:-C.UTF-8}"

exec msnpip "$@"
