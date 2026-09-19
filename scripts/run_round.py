#!/usr/bin/env python3
"""Run one deployment round of the correction loop.

  python scripts/run_round.py --round 0          # baseline on the initial pool
  python scripts/run_round.py --round 1          # batch 0 arrives, loop runs
  python scripts/run_round.py --build-cache      # one-time preprocessing cache

Each round writes outputs/rounds/round{k}/round.json (+ model.pt, deltas,
per-case eval). Deterministic from (partition, seed, round index).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--round", type=int, default=None)
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--data-root", type=Path, default=REPO / "data" / "verse2020")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--rehearsal-frac", type=float, default=0.25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="", help="ablation arm suffix, e.g. _norehearsal")
    ap.add_argument("--force-admit", action="store_true",
                    help="bypass the batch screen (counterfactual arm; recorded)")
    ap.add_argument("--corrector", choices=("oracle", "budget", "budget_jitter"),
                    default="oracle", help="the reviewer the loop learns from (W7)")
    ap.add_argument("--control", choices=("none", "with", "only"), default="none",
                    help="the more-training null: 'with' a main round, or 'only' as an arm")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="relocate the round tree (seed replicates)")
    args = ap.parse_args()

    if args.build_cache:
        from clloop.preprocess import build_cache

        cases = json.loads((REPO / "manifests" / "cases.json").read_text())["cases"]
        part = json.loads((REPO / "manifests" / "partition.json").read_text())
        poisoned_ids = set(part["arrival_batches"][part["poisoned_batch_index"]])
        manifest = build_cache(
            args.data_root, cases, REPO / "data" / "cache",
            poisoned_dir=REPO / "data" / "poisoned", poisoned_ids=poisoned_ids,
        )
        print(f"cached {len(manifest)} entries -> data/cache/")
        return 0

    if args.round is None:
        raise SystemExit("pass --round K or --build-cache")

    from clloop.engine import run_round

    rec = run_round(
        args.round, repo=REPO, data_root=args.data_root, seed=args.seed,
        iters=args.iters, rehearsal_frac=args.rehearsal_frac, device=args.device,
        tag=args.tag, force_admit=args.force_admit, control=args.control, corrector=args.corrector,
        out_root=args.out_root.resolve() if args.out_root else None,
    )
    ev = rec["eval"]
    print(f"\nround {args.round} done in {rec['seconds']}s  "
          f"(loss {rec['loss_final']})")
    print(f"  test APL median {ev['apl_mm_median']:.0f} mm  "
          f"sdice {ev['surface_dice_median']:.3f}  kinds {ev['kinds']}")
    for r in ("cervical", "thoracic", "lumbar", "sacrum"):
        print(f"  dice {r:9}: {ev[f'dice_{r}_median']}  (n={ev[f'dice_{r}_n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
