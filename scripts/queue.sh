#!/usr/bin/env bash
# Run sweep specs one after another, skipping finished runs and resuming unfinished ones

set -euo pipefail
trap 'echo "=== stopped (exit $?) at line $LINENO $(date)"' ERR
cd "$(dirname "$0")/.."
name=$1
shift
for spec in "$@"; do
    echo "=== $spec $(date)"
    uv run --group cuda-132 python scripts/sweep.py "sweeps/$spec.json" --only fold --continue-on-error
done
touch "runs/logs/$name.done"
echo "=== done $(date)"
