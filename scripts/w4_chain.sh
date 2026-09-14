#!/usr/bin/env bash
# W4: the full run — 5 rounds + the poisoned round + two ablation arms.
# Fresh chain under the re-frozen ruler digest (0f0edee1…); the W2-era rounds
# (old digest) are preserved aside, never mixed.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${CLLOOP_PY:-python3}"
ITERS=2000

if [ -d outputs/rounds ] && [ ! -d outputs/rounds_w2_superseded ]; then
  mv outputs/rounds outputs/rounds_w2_superseded
  echo "moved W2-era rounds aside (old ruler digest)"
fi

echo "=== round 0 (baseline) ==="
$PY -u scripts/run_round.py --round 0 --iters $ITERS
echo "=== round 1 ==="
$PY -u scripts/run_round.py --round 1 --iters $ITERS
echo "=== round 1 ablation: no rehearsal (G4) ==="
$PY -u scripts/run_round.py --round 1 --iters $ITERS --tag _norehearsal --rehearsal-frac 0
echo "=== round 2 ==="
$PY -u scripts/run_round.py --round 2 --iters $ITERS
echo "=== round 3 (POISONED batch arrives) ==="
$PY -u scripts/run_round.py --round 3 --iters $ITERS
echo "=== round 3 counterfactual: force-admit the poison ==="
$PY -u scripts/run_round.py --round 3 --iters $ITERS --tag _counterfactual --force-admit
echo "=== round 4 ==="
$PY -u scripts/run_round.py --round 4 --iters $ITERS
echo "W4_CHAIN_DONE"
