#!/usr/bin/env bash
# W4 definitive chain — identical to w4_chain.sh plus: MLflow curves + DVC
# model tracking per round (user directive 2026-09-12), and a round-4
# no-rehearsal arm where the cervical-heavy batch should make forgetting
# visible. The pre-directive chain is preserved aside.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${CLLOOP_PY:-python3}"
ITERS=2000

if [ -d outputs/rounds ] && [ ! -d outputs/rounds_predirective_superseded ]; then
  mv outputs/rounds outputs/rounds_predirective_superseded
  echo "moved pre-directive rounds aside"
fi

echo "=== round 0 (baseline) ==="
$PY -u scripts/run_round.py --round 0 --iters $ITERS
echo "=== round 1 ==="
$PY -u scripts/run_round.py --round 1 --iters $ITERS
echo "=== round 2 ==="
$PY -u scripts/run_round.py --round 2 --iters $ITERS
echo "=== round 3 (POISONED batch arrives) ==="
$PY -u scripts/run_round.py --round 3 --iters $ITERS
echo "=== round 3 counterfactual: force-admit the poison ==="
$PY -u scripts/run_round.py --round 3 --iters $ITERS --tag _counterfactual --force-admit
echo "=== round 4 ==="
$PY -u scripts/run_round.py --round 4 --iters $ITERS
echo "=== round 4 ablation: no rehearsal (G4, cervical-heavy batch) ==="
$PY -u scripts/run_round.py --round 4 --iters $ITERS --tag _norehearsal --rehearsal-frac 0
echo "W4_CHAIN_DONE"
