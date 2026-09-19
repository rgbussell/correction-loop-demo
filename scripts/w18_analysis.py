#!/usr/bin/env python3
"""W18: the budgeted-reviewer arms across three seeds.

Per seed and arm: promotions, deployed end-to-end burden reduction, whether
the poison was refused, thoracic regression in candidates, final-candidate
cervical Dice. Then the three pre-registered 3-of-3 claims are evaluated
mechanically. Writes manifests/w18_budgeted_seed_band.json; arms that did not
complete are listed, never skipped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from w7_analysis import _chain  # noqa: E402

SEEDS = (1337, 2027, 3117)
ORACLE = {1337: "outputs/rounds", 2027: "outputs/seedband/s2027",
          3117: "outputs/seedband/s3117_w16"}
ARMS = {"budget": "W7 budget (unfixed approved)", "budget_masked": "A masked",
        "budget_region_masked": "B region+masked"}


def _summ(root: str) -> dict | None:
    if not (REPO / root / "round4" / "round.json").is_file():
        return None
    c = _chain(root)
    start = c[0]["deployed_apl_m"]
    return {
        "promotions": sum(bool(r["promoted"]) for r in c),
        "promoted_rounds": [r["round"] for r in c if r["promoted"]],
        "end_to_end_reduction": round((start - c[-1]["deployed_apl_m"]) / start, 4),
        "poison_refused": c[3]["refused_at_curation"],
        "candidate_gains": {r["round"]: r["gain_vs_do_nothing"] for r in c
                            if r["gain_vs_do_nothing"] is not None},
        "candidates_tripping_forgetting": sum(
            any("forgetting" in x for x in r["reasons"]) for r in c),
        "thoracic_change_in_candidates": {
            r["round"]: round(r["candidate_dice"]["thoracic"] - c[0]["candidate_dice"]["thoracic"], 4)
            for r in c if r["gain_vs_do_nothing"] is not None},
        "final_candidate_cervical": c[4]["candidate_dice"]["cervical"],
        "deployed_final_cervical": c[-1]["deployed_dice"]["cervical"],
    }


def main() -> int:
    out: dict = {"seeds": list(SEEDS), "by_seed": {}, "missing": []}
    for seed in SEEDS:
        row = {"oracle": _summ(ORACLE[seed])}
        for arm, label in ARMS.items():
            s = _summ(f"outputs/corrector/s{seed}_{arm}")
            if s is None:
                out["missing"].append(f"s{seed} {arm}")
            row[label] = s
        out["by_seed"][f"s{seed}"] = row
    out["complete"] = not out["missing"]
    if out["complete"]:
        rows = out["by_seed"].values()
        budgeted = list(ARMS.values())
        out["claims_3_of_3"] = {
            "poison_refused_under_every_budgeted_arm": all(
                r[a]["poison_refused"] for r in rows for a in budgeted),
            "unmasked_budget_arm_promotes_nothing": all(
                r["W7 budget (unfixed approved)"]["promotions"] == 0 for r in rows),
            "masking_yields_at_least_one_promotion": all(
                r["A masked"]["promotions"] >= 1 for r in rows),
        }
        for a in ["oracle", *budgeted]:
            out[f"reduction_per_seed::{a}"] = [r[a]["end_to_end_reduction"] for r in rows]
            out[f"promotions_per_seed::{a}"] = [r[a]["promotions"] for r in rows]
    (REPO / "manifests" / "w18_budgeted_seed_band.json").write_text(
        json.dumps(out, indent=2) + "\n")
    for seed, row in out["by_seed"].items():
        for a, s in row.items():
            if s:
                print(seed, f"{a:30}", "promo", s["promoted_rounds"],
                      f"e2e {s['end_to_end_reduction']:+.1%}", "poison refused", s["poison_refused"],
                      "forget-trips", s["candidates_tripping_forgetting"],
                      "thor Δ", list(s["thoracic_change_in_candidates"].values()),
                      "cerv cand", s["final_candidate_cervical"])
    print({k: v for k, v in out.items() if k not in ("by_seed",)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
