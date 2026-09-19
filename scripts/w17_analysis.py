#!/usr/bin/env python3
"""W17: does learning from REVIEWED REGIONS ONLY rescue the budgeted reviewer?

Compares, on seed 1337: the oracle chain, W7's budget chain (unfixed output
approved as truth), Arm A (same reviewer, unreviewed voxels masked out of the
loss) and Arm B (A + the budget split per anatomical region). Writes
manifests/w17_reviewed_only.json with the pre-registered kill test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from w7_analysis import _chain, _region  # noqa: E402

ARMS = {"oracle": "outputs/rounds",
        "budget (W7)": "outputs/corrector/s1337_budget",
        "A masked": "outputs/corrector/s1337_budget_masked",
        "B region+masked": "outputs/corrector/s1337_budget_region_masked"}


def _effort(root: str) -> dict:
    eff: dict = {}
    for k in (1, 2, 4):
        p = REPO / root / f"round{k}" / "round.json"
        if not p.is_file():
            continue
        for pc in json.loads(p.read_text()).get("corrector", {}).get("per_case", []):
            for key in ("fixed", "left_small", "left_over_budget"):
                for lab in pc[key]:
                    eff.setdefault(_region(lab), {"fixed": 0, "left_small": 0,
                                                  "left_over_budget": 0})[key] += 1
    for v in eff.values():
        v["fixed_frac"] = round(v["fixed"] / max(1, sum(
            v[k] for k in ("fixed", "left_small", "left_over_budget"))), 3)
    return eff


def main() -> int:
    out: dict = {"seed": 1337, "arms": {}}
    for name, root in ARMS.items():
        if not (REPO / root / "round4" / "round.json").is_file():
            out["arms"][name] = {"complete": False}
            continue
        c = _chain(root)
        red = (c[0]["deployed_apl_m"] - c[-1]["deployed_apl_m"]) / c[0]["deployed_apl_m"]
        out["arms"][name] = {
            "complete": True,
            "promotions": sum(bool(r["promoted"]) for r in c),
            "end_to_end_reduction": round(red, 4),
            "deployed_final_dice": c[-1]["deployed_dice"],
            "candidate_gains": [r["gain_vs_do_nothing"] for r in c if r["gain_vs_do_nothing"] is not None],
            "candidates_tripping_forgetting": sum(
                any("forgetting" in x for x in r["reasons"]) for r in c),
            "poison_refused": c[3]["refused_at_curation"],
            "final_candidate_dice": c[4]["candidate_dice"],
            "reviewer_effort_by_region": _effort(root) if name != "oracle" else None,
            "chain": c,
        }
    arms = out["arms"]
    done = all(a["complete"] for a in arms.values())
    out["complete"] = done
    if done:
        out["kill_condition_met"] = (arms["A masked"]["promotions"] == 0
                                     and arms["B region+masked"]["promotions"] == 0)
        out["best_arm_over_oracle"] = round(max(
            arms["A masked"]["end_to_end_reduction"],
            arms["B region+masked"]["end_to_end_reduction"])
            / arms["oracle"]["end_to_end_reduction"], 3)
    (REPO / "manifests" / "w17_reviewed_only.json").write_text(json.dumps(out, indent=2) + "\n")
    for name, a in arms.items():
        print(name, {k: v for k, v in a.items() if k not in ("chain", "reviewer_effort_by_region")})
        if a.get("reviewer_effort_by_region"):
            print("   effort:", {r: v["fixed_frac"] for r, v in a["reviewer_effort_by_region"].items()})
    print({k: v for k, v in out.items() if k != "arms"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
