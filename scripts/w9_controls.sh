#!/usr/bin/env bash
# W9: backfill the more-training control arm for every round that trained a
# candidate, in all three replicates. Arms are tagged, so the promotion chain
# is untouched; s3117 uses the lineage judged under screen v3 + the banded gate.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${CLLOOP_PY:-python3}"
run() { # seed root rounds...
  local SEED=$1 ROOT=$2; shift 2
  for K in "$@"; do
    echo "=== control arm: seed ${SEED} round ${K} ==="
    $PY -u scripts/run_round.py --round "$K" --iters 2000 --seed "$SEED" \
      --out-root "$ROOT" --tag _control --control only
  done
}
run 1337 outputs/rounds 1 2 4
run 2027 outputs/seedband/s2027 1 2 4
run 3117 outputs/seedband/s3117_w16 1 2 4
echo "W9_CONTROLS_DONE"
