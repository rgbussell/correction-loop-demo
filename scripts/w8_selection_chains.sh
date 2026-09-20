#!/usr/bin/env bash
# W8: the selection arm — rehearsal drawn by the incumbent's burden instead of
# uniformly. Oracle corrector, same seeds/iterations/N as the baseline chains;
# round 0 is shared with the baseline and copied. A seed that dies is logged and
# the run continues; nothing is silently skipped.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${CLLOOP_PY:-python3}"
for SEED in 1337 2027 3117; do
  case "$SEED" in
    1337) SRC=outputs/rounds ;;
    *)    SRC="outputs/seedband/s${SEED}" ;;
  esac
  ROOT="outputs/selection/s${SEED}_burden_weighted"
  mkdir -p "$ROOT"
  [ -d "$ROOT/round0" ] || cp -r "$SRC/round0" "$ROOT/round0"
  for K in 1 2 3 4; do
    echo "=== selection arm: seed ${SEED} round ${K} ==="
    if ! $PY -u scripts/run_round.py --round "$K" --iters 2000 --seed "$SEED" \
        --out-root "$ROOT" --rehearsal burden_weighted; then
      echo "W8_ROUND_FAILED seed=${SEED} round=${K}"
      break
    fi
  done
done
echo "W8_SELECTION_DONE"
