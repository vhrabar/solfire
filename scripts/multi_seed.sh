#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SEEDS=(1 2 3 4 5)
ARCH="transunet"
OUTPUT_ROOT="runs/multi_seed"
DEVICE=""

TRAIN_ARGS=(
  --arch "$ARCH" --crop 512 --crops-per-image 4 --foreground-prob 0.7 --pos-weight 5
  --batch-size 8 --epochs 40 --lr 4e-4 --dropout 0.1 --drop-path 0.1 --warmup-epochs 3
  --amp --workers 16 --swa-epochs 12
  --val-limit 64 --val-tile 512 --val-overlap 0.0 --tile-batch 8
)
CALIBRATE_ARGS=(--tile 512 --overlap 0.25 --tile-batch 8 --workers 8)

while getopts "n:s:a:o:d:h" opt; do
  case "$opt" in
    n) SEEDS=($(seq 1 "$OPTARG")) ;;
    s) read -ra SEEDS <<< "$OPTARG" ;;
    a) ARCH="$OPTARG" ;;
    o) OUTPUT_ROOT="$OPTARG" ;;
    d) DEVICE="$OPTARG" ;;
    h) usage ;;
    *) usage ;;
  esac
done
shift $((OPTIND - 1))
EXTRA_ARGS=("$@")

DEVICE_ARGS=()
if [ -n "$DEVICE" ]; then
  DEVICE_ARGS=(--device "$DEVICE")
fi

mkdir -p "$OUTPUT_ROOT"
RESULTS_CSV="$OUTPUT_ROOT/results.csv"
echo "seed,checkpoint,train_pq,calibrated_pq,threshold,offset,min_area" > "$RESULTS_CSV"

for SEED in "${SEEDS[@]}"; do
  RUN_DIR="$OUTPUT_ROOT/${ARCH}_seed${SEED}"
  mkdir -p "$RUN_DIR"

  echo "=== seed $SEED: training -> $RUN_DIR ==="
  uv run python -m src.main --seed "$SEED" "${DEVICE_ARGS[@]}" train \
    "${TRAIN_ARGS[@]}" "${EXTRA_ARGS[@]}" --output-dir "$RUN_DIR" \
    2>&1 | tee "$RUN_DIR.train.log"

  read -r CHECKPOINT TRAIN_PQ <<< "$(uv run python scripts/pick_checkpoint.py "$RUN_DIR")"
  echo "=== seed $SEED: calibrating $CHECKPOINT (train pq $TRAIN_PQ) ==="

  uv run python -m src.main "${DEVICE_ARGS[@]}" calibrate \
    --checkpoint "$CHECKPOINT" "${CALIBRATE_ARGS[@]}" \
    2>&1 | tee "$RUN_DIR.calibrate.log"

  CALIBRATION_JSON="${CHECKPOINT%.pt}.calibration.json"
  read -r PQ THRESHOLD OFFSET MIN_AREA <<< "$(uv run python -c "
import json, sys
d = json.load(open(sys.argv[1]))
print(d['pq'], d['threshold'], d['offset'], d['min_area'])
" "$CALIBRATION_JSON")"

  echo "$SEED,$CHECKPOINT,$TRAIN_PQ,$PQ,$THRESHOLD,$OFFSET,$MIN_AREA" >> "$RESULTS_CSV"
done

echo
echo "=== summary ==="
uv run python scripts/summarize_seeds.py "$RESULTS_CSV"
