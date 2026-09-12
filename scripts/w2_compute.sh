#!/usr/bin/env bash
# W2 compute chain: finish the cache, then round 0 (baseline), then round 1.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "=== cache ==="
/home/rbussell/miniconda3/bin/python -u scripts/run_round.py --build-cache
echo "=== round 0 ==="
/home/rbussell/miniconda3/bin/python -u scripts/run_round.py --round 0
echo "=== round 1 ==="
/home/rbussell/miniconda3/bin/python -u scripts/run_round.py --round 1
echo "W2_CHAIN_DONE"
