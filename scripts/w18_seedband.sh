#!/usr/bin/env bash
# W18: replicate the budgeted-reviewer arms (W7 budget, W17 A and B) on the two
# other seeds. No setting changes — this exists because one seed could not
# separate the arms. An arm that dies (e.g. at the evaluation memory cap) is
# logged and the run continues with the next arm; nothing is silently skipped.
set -uo pipefail
cd "$(dirname "$0")/.."
for SEED in 2027 3117; do
  for ARM in budget budget_masked budget_region_masked; do
    if ! bash scripts/w7_corrector_chains.sh "$SEED" "$ARM"; then
      echo "W18_ARM_FAILED seed=${SEED} arm=${ARM}"
    fi
  done
done
echo "W18_SEEDBAND_DONE"
