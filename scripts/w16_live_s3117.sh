#!/usr/bin/env bash
# W16: run the flipped round LIVE under the banded gate. Rounds 0-3 of the
# screen-v3 replicate are unaffected by the promotion change (their verdicts do
# not move) and are copied; round 4 is retrained and judged by the engine.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${CLLOOP_PY:-python3}"
SRC=outputs/seedband/s3117_v3
ROOT=outputs/seedband/s3117_w16
mkdir -p "$ROOT"
for K in 0 1 2 3; do
  [ -d "$ROOT/round$K" ] || cp -r "$SRC/round$K" "$ROOT/round$K"
done
$PY -u scripts/run_round.py --round 4 --iters 2000 --seed 3117 --out-root "$ROOT"
echo "W16_LIVE_DONE"
