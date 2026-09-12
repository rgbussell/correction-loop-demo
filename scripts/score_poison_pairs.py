#!/usr/bin/env python3
"""W1 live validation: score the poisoned batch's (poisoned, original) pairs.

The poison (enumeration v -> v+1) has a known answer the ruler must produce on
REAL masks, not just toys: label-blind surface Dice = 1.0 exactly (identical
shapes), large APL (every renamed contour must be redrawn under its true
name), verdict `relabel`. 13/13 or the ruler does not get frozen.

Writes manifests/w1_ruler_validation.json (per-case quantitation, committed).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from clloop.delta import score_case  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=REPO / "data" / "verse2020")
    ap.add_argument("--tol-mm", type=float, default=2.0)
    args = ap.parse_args()

    cases = json.loads((REPO / "manifests" / "cases.json").read_text())["cases"]
    poison = json.loads((REPO / "manifests" / "poison.json").read_text())
    by_id = {c["case_id"]: c for c in cases}

    rows = []
    for entry in poison["cases"]:
        cid = entry["case_id"]
        rec = by_id[cid]
        t0 = time.time()
        final = np.asanyarray(nib.load(str(args.data_root / rec["seg"])).dataobj)
        auto = np.asanyarray(
            nib.load(str(REPO / "data" / "poisoned" / f"{cid}_seg.nii.gz")).dataobj
        )
        d = score_case(auto, final, tuple(rec["spacing_mm"]), tol_mm=args.tol_mm)
        rows.append(
            {
                "case_id": cid,
                "apl_mm": d.apl_mm,
                "surface_dice": d.surface_dice,
                "kind": d.kind,
                "level_kinds": d.kinds(),
                "n_levels": rec["n_levels"],
                "seconds": round(time.time() - t0, 1),
            }
        )
        print(
            f"  {cid}: sdice {d.surface_dice:.3f}  APL {d.apl_mm:>9.0f} mm  "
            f"verdict {d.kind}  ({rows[-1]['seconds']}s)"
        )

    n_relabel = sum(r["kind"] == "relabel" for r in rows)
    summary = {
        "tol_mm": args.tol_mm,
        "n_cases": len(rows),
        "n_verdict_relabel": n_relabel,
        "surface_dice_min": min(r["surface_dice"] for r in rows),
        "apl_mm_median": float(np.median([r["apl_mm"] for r in rows])),
        "all_relabel": n_relabel == len(rows),
        "cases": rows,
    }
    out = REPO / "manifests" / "w1_ruler_validation.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n{n_relabel}/{len(rows)} verdict=relabel, "
          f"min sdice {summary['surface_dice_min']:.3f}, "
          f"median APL {summary['apl_mm_median']:.0f} mm")
    print(f"wrote {out}")
    return 0 if summary["all_relabel"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
