#!/usr/bin/env python3
"""Scan the VerSe dataset and write the stream-partition manifest.

  python scripts/build_partition.py --data-root data/verse2020

Outputs (committed to git — they are the design, not the data):
  manifests/cases.json      — one record per case: coverage, geometry, paths
  manifests/partition.json  — sequestered test / initial pool / arrival batches
                              + the declared shift + the poisoned-batch spec
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from clloop.partition import design_partition, scan_dataset  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=REPO / "data" / "verse2020")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--n-test", type=int, default=24)
    ap.add_argument("--n-pool", type=int, default=30)
    ap.add_argument("--n-batches", type=int, default=4)
    args = ap.parse_args()

    if not args.data_root.is_dir():
        raise SystemExit(
            f"data root not found: {args.data_root}\n"
            "Fetch it with scripts/download_verse.sh or pass --data-root."
        )

    records = scan_dataset(args.data_root)
    if len(records) < args.n_test + args.n_pool + args.n_batches:
        raise SystemExit(f"only {len(records)} usable cases — check the data root")

    part = design_partition(
        records,
        seed=args.seed,
        n_test=args.n_test,
        n_pool=args.n_pool,
        n_batches=args.n_batches,
    )

    out = REPO / "manifests"
    out.mkdir(exist_ok=True)
    (out / "cases.json").write_text(
        json.dumps(
            {
                "data_root": str(args.data_root),
                "n_cases": len(records),
                "cases": [
                    {
                        "case_id": r.case_id,
                        "image": r.image,
                        "seg": r.seg,
                        "centroids": r.centroids,
                        "levels": list(r.levels),
                        "fov_group": r.fov_group,
                        "n_levels": r.n_levels,
                        "shape": list(r.shape),
                        "spacing_mm": list(r.spacing_mm),
                    }
                    for r in records
                ],
            },
            indent=2,
        )
        + "\n"
    )
    (out / "partition.json").write_text(json.dumps(part.to_json(), indent=2) + "\n")

    by_id = {r.case_id: r for r in records}
    print(f"scanned {len(records)} cases from {args.data_root}")
    print(f"  fov groups     : {dict(Counter(r.fov_group for r in records))}")
    print(f"  sequestered    : {len(part.sequestered_test)}")
    print(f"  initial pool   : {len(part.initial_pool)}")
    for i, b in enumerate(part.arrival_batches):
        frac = sum(by_id[c].fov_group == "cervical-containing" for c in b) / len(b)
        tag = "  <- POISONED" if i == part.poisoned_batch_index else ""
        print(f"  batch {i}        : {len(b)} cases, cervical fraction {frac:.2f}{tag}")
    print(f"wrote {out / 'cases.json'} and {out / 'partition.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
