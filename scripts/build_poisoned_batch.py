#!/usr/bin/env python3
"""Materialize the poisoned batch: enumeration off-by-one on reference masks.

  python scripts/build_poisoned_batch.py --data-root data/verse2020

For every case in the partition's declared poisoned batch, writes a COPY of its
reference segmentation with each vertebral label v -> v+1 (the classic
wrong-level enumeration error: every vertebra keeps its true shape but wears
its cranial neighbour's name). Originals are never touched; the loop engine
substitutes these copies when that batch "arrives".

Why this transform: it is the highest-consequence real failure mode in spine
segmentation (a wrong-level label is a wrong-level plan), it leaves Dice on the
*shapes* nearly perfect — so a loop watching only overlap metrics would happily
train on it — and it is exactly what a taxonomy-aware delta ruler and a
labeling-accuracy null are supposed to catch. The poison is declared in the
partition manifest; nothing about it is hidden from the reader, only from the
loop's automated gates.

Output: data/poisoned/<case_id>_seg.nii.gz + manifests/poison.json (the audit
record: per-case label maps, voxel counts, sha256 of source and output).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

VERT_MAX = 28  # VerSe: 1-24 C/T/L, 25 L6, 26 sacrum, 27 cocygis, 28 T13


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def poison_labelmap(arr: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    """Shift every vertebral label v -> v+1. Sacrum/cocygis (26/27) are left —
    the poison is a vertebral enumeration error, not an anatomy swap. A label
    landing on 26 would collide with sacrum, so the caudal-most vertebra (25,
    L6) maps to 25 unchanged if present — recorded in the map either way."""
    mapping: dict[int, int] = {}
    out = np.zeros_like(arr)
    for v in np.unique(arr):
        v = int(v)
        if v == 0:
            out[arr == 0] = 0
            continue
        if v in (26, 27):  # sacrum / cocygis keep their names
            nv = v
        elif v == 25:  # L6 has no caudal vertebral name to take
            nv = v
        elif v == 28:  # T13 -> label of T12's caudal neighbour is L1 (20)
            nv = 20
        else:
            nv = v + 1
        mapping[v] = nv
        out[arr == v] = nv
    return out, mapping


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=REPO / "data" / "verse2020")
    args = ap.parse_args()

    part = json.loads((REPO / "manifests" / "partition.json").read_text())
    cases = json.loads((REPO / "manifests" / "cases.json").read_text())["cases"]
    by_id = {c["case_id"]: c for c in cases}
    batch = part["arrival_batches"][part["poisoned_batch_index"]]

    out_dir = REPO / "data" / "poisoned"
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = []
    for cid in batch:
        src = args.data_root / by_id[cid]["seg"]
        img = nib.load(str(src))
        arr = np.asanyarray(img.dataobj).astype(np.int16)
        poisoned, mapping = poison_labelmap(arr)
        dst = out_dir / f"{cid}_seg.nii.gz"
        out_img = nib.Nifti1Image(poisoned.astype(np.uint8), img.affine)
        out_img.header.set_data_dtype(np.uint8)
        nib.save(out_img, str(dst))
        audit.append(
            {
                "case_id": cid,
                "source_seg": by_id[cid]["seg"],
                "source_sha256": sha256(src),
                "poisoned_sha256": sha256(dst),
                "label_map": {str(k): v for k, v in sorted(mapping.items())},
                "n_fg_voxels": int((arr > 0).sum()),
            }
        )
        print(f"  poisoned {cid}: {len(mapping)} labels shifted")

    (REPO / "manifests" / "poison.json").write_text(
        json.dumps(
            {
                "transform": part["poison_transform"],
                "batch_index": part["poisoned_batch_index"],
                "n_cases": len(audit),
                "cases": audit,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {len(audit)} poisoned references -> {out_dir}")
    print(f"audit record -> manifests/poison.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
