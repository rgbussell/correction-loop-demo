#!/usr/bin/env python3
"""W7: why the jitter arm could not be evaluated — foreground scatter.

The jitter arm's round-1 candidate was killed at the memory cap during the
sequestered evaluation. The frozen ruler runs distance transforms over the
bounding box of the predicted foreground; this probe measures that box for the
budget and budget+jitter candidates on three small test cases. No ruler call,
so it runs in a few GB.  -> manifests/w7_jitter_scatter.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from clloop.engine import _to_native  # noqa: E402
from clloop.model import make_model, predict  # noqa: E402
from clloop.preprocess import load_case  # noqa: E402

CASES = ["verse056", "verse033", "GL364"]
MODELS = {"budget": "outputs/corrector/s1337_budget/round1/model.pt",
          "budget_jitter": "outputs/corrector/s1337_budget_jitter/round1/model.pt"}


def main() -> int:
    by_id = {c["case_id"]: c for c in
             json.loads((REPO / "manifests" / "cases.json").read_text())["cases"]}
    out: dict = {"cases": CASES, "arms": {}}
    for arm, rel in MODELS.items():
        m = make_model("cuda")
        m.load_state_dict(torch.load(REPO / rel, map_location="cuda", weights_only=True))
        rows = []
        for cid in CASES:
            img, _ = load_case(REPO / "data" / "cache", cid)
            ref = np.asanyarray(nib.load(str(REPO / "data" / "verse2020" / by_id[cid]["seg"])).dataobj)
            fg = _to_native(predict(m, img, device="cuda"), ref.shape) > 0
            idx = np.argwhere(fg)
            box = float(np.prod(idx.max(0) - idx.min(0) + 1) / ref.size) if len(idx) else 0.0
            rows.append({"case_id": cid, "fg_bbox_frac": round(box, 3),
                         "fg_voxel_frac": round(float(fg.mean()), 4),
                         "ref_fg_voxel_frac": round(float((ref > 0).mean()), 4)})
        out["arms"][arm] = {"rows": rows,
                            "min_fg_bbox_frac": min(r["fg_bbox_frac"] for r in rows),
                            "max_fg_bbox_frac": max(r["fg_bbox_frac"] for r in rows)}
        print(arm, out["arms"][arm])
    (REPO / "manifests" / "w7_jitter_scatter.json").write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
