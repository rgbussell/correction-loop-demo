#!/usr/bin/env bash
# W15: replay the s3117 replicate from round 2 under screen v3. Rounds 0-1 are
# untouched by the screen change and are copied, not retrained; the original
# replicate (outputs/seedband/s3117, screen v2) stays as the before-picture.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${CLLOOP_PY:-python3}"
ITERS=2000
SRC=outputs/seedband/s3117
ROOT=outputs/seedband/s3117_v3

mkdir -p "$ROOT"
for K in 0 1; do
  [ -d "$ROOT/round$K" ] || cp -r "$SRC/round$K" "$ROOT/round$K"
done
for K in 2 3 4; do
  echo "=== seed 3117 (screen v3) round ${K} ==="
  $PY -u scripts/run_round.py --round $K --iters $ITERS --seed 3117 --out-root "$ROOT"
done
echo "W15_REJUDGE_DONE"
