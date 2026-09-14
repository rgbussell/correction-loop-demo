#!/usr/bin/env bash
# W6: the seed band — replicate the 5-round chain under new seeds so every
# per-round gain carries an uncertainty. Seed 1337 (the definitive chain) is
# replicate 1; this runs replicates 2 and 3. Main rounds only (arms are not
# band material). Each replicate is self-contained under outputs/seedband/.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${CLLOOP_PY:-python3}"
ITERS=2000

for SEED in 2027 3117; do
  ROOT="outputs/seedband/s${SEED}"
  echo "=== replicate seed ${SEED} ==="
  for K in 0 1 2 3 4; do
    echo "=== seed ${SEED} round ${K} ==="
    $PY -u scripts/run_round.py --round $K --iters $ITERS --seed $SEED --out-root "$ROOT"
  done
done
echo "W6_SEEDBAND_DONE"
