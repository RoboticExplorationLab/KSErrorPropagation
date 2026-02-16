#!/usr/bin/env bash
# Run a Julia script with output on the terminal and in a timestamped log file.
#
# Usage (from repo root):
#   bash julia/run_with_log.sh julia/scripts/run_monte_carlo_and_save_npz.jl
#   bash julia/run_with_log.sh julia/scripts/run_monte_carlo_and_save_npz.jl config/fast.jl
#
# Logs are written to julia/logs/<script_basename>_<YYYY-MM-DD>_<HH-MM-SS>.log

set -e
SCRIPT_PATH="${1:?Usage: run_with_log.sh <julia_script.jl> [args...]}"
[ $# -ge 1 ] && shift

# Resolve script path to absolute so it works from any cwd (e.g. from repo root or from julia/)
if [ "${SCRIPT_PATH#/}" = "$SCRIPT_PATH" ]; then
  SCRIPT_PATH="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)/$(basename "$SCRIPT_PATH")"
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

JULIA_PROJECT="${JULIA_PROJECT:-julia}"
LOG_DIR="$REPO_ROOT/julia/logs"
mkdir -p "$LOG_DIR"

SCRIPT_BASE=$(basename "$SCRIPT_PATH" .jl)
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG_FILE="$LOG_DIR/${SCRIPT_BASE}_${TIMESTAMP}.log"

echo "Log file: $LOG_FILE"
exec julia --project="$JULIA_PROJECT" "$SCRIPT_PATH" "$@" 2>&1 | tee "$LOG_FILE"
