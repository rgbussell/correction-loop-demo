#!/usr/bin/env bash
# W7: re-run the loop with a simulated, budgeted reviewer instead of the oracle.
# Round 0 (initial pool, full labels) is shared with the oracle chain and
# copied; rounds 1-4 run under each corrector. Seed given as $1 (default 1337).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${CLLOOP_PY:-python3}"
SEED="${1:-1337}"
case "$SEED" in
  1337) SRC=outputs/rounds ;;
  *)    SRC="outputs/seedband/s${SEED}" ;;
esac
for ARM in budget budget_jitter; do
  ROOT="outputs/corrector/s${SEED}_${ARM}"
  mkdir -p "$ROOT"
  [ -d "$ROOT/round0" ] || cp -r "$SRC/round0" "$ROOT/round0"
  for K in 1 2 3 4; do
    echo "=== corrector ${ARM}: seed ${SEED} round ${K} ==="
    $PY -u scripts/run_round.py --round "$K" --iters 2000 --seed "$SEED" \
      --out-root "$ROOT" --corrector "$ARM"
  done
done
echo "W7_CHAINS_DONE"
