#!/usr/bin/env python3
"""W15: measure every incumbent's label-offset profile on the sequestered set.

The batch screen refuses on a consistent label offset across relabel-class
levels. The seed band showed that signal cannot tell WHO is shifted: a clean
batch scored by an incumbent with its own enumeration bias produces the same
signature as poisoned references (s3117 round 2). The sequestered references
are known-clean, so the incumbent's offset profile there is the model's share
of the signal — this script measures it for every incumbent that ever scored
an arriving batch, before any discriminator threshold is chosen.

  python scripts/w15_arbiter_profiles.py            # all chains
  -> manifests/w15_arbiter_profiles.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

CHAINS = {"s1337": "outputs/rounds", "s2027": "outputs/seedband/s2027",
          "s3117": "outputs/seedband/s3117"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=REPO / "data" / "verse2020")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch

    from clloop.engine import offset_profile_on_test
    from clloop.model import make_model

    part = json.loads((REPO / "manifests" / "partition.json").read_text())
    cases = json.loads((REPO / "manifests" / "cases.json").read_text())["cases"]
    by_id = {c["case_id"]: c for c in cases}

    out: dict = {}
    for chain, root in CHAINS.items():
        rounds_root = REPO / root
        profiles: dict = {}
        for k in range(1, 5):
            rec_p = rounds_root / f"round{k}" / "round.json"
            if not rec_p.is_file():
                continue
            rec = json.loads(rec_p.read_text())
            inc_rel = rec["incumbent"]
            if inc_rel not in profiles:
                model = make_model(args.device)
                model.load_state_dict(torch.load(REPO / inc_rel, map_location=args.device,
                                                 weights_only=True))
                profiles[inc_rel] = offset_profile_on_test(
                    model, part["sequestered_test"], REPO / "data" / "cache",
                    args.data_root, by_id, device=args.device)
                print(f"{chain} {inc_rel}: {profiles[inc_rel]['summary']}", flush=True)
            ev = rec["curation"]["evidence"]
            n_levels = sum(sum(d["level_kinds"].values()) for d in rec["deltas"])
            out.setdefault(chain, []).append({
                "round": k,
                "served_poisoned": rec["arrival"]["served_poisoned"],
                "incumbent": inc_rel,
                "v2_refused": rec["curation"]["refusal_reason"] is not None,
                "batch": {"n_cases": ev["n_cases"], "n_levels": n_levels,
                          "offset_histogram": ev["offset_histogram"]},
                "arbiter": profiles[inc_rel]["summary"],
            })
    dest = REPO / "manifests" / "w15_arbiter_profiles.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {dest.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
