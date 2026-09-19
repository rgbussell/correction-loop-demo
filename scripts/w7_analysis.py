#!/usr/bin/env python3
"""W7: the learning-vs-oracle gap under a budgeted simulated reviewer.

Reads the oracle chain and the budget-corrector chain (seed 1337) and writes
manifests/w7_corrector.json: deployed lineage, per-round candidates, where the
reviewer's effort went by anatomical region, and the pre-registered kill test.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ORACLE, BUDGET = "outputs/rounds", "outputs/corrector/s1337_budget"


def _region(label: int) -> str:
    return ("cervical" if label <= 7 else "thoracic" if label <= 19
            else "lumbar" if label <= 25 else "sacrum")


def _chain(root: str) -> list[dict]:
    rows, deployed = [], None
    for k in range(5):
        r = json.loads((REPO / root / f"round{k}" / "round.json").read_text())
        pr = r.get("promotion", {})
        if k == 0 or pr.get("promoted"):
            deployed = r["eval"]
        rows.append({
            "round": k, "promoted": pr.get("promoted"),
            "refused_at_curation": bool(r.get("curation", {}).get("refusal_reason")),
            "candidate_apl_m": round(r["eval"]["apl_mm_total"] / 1000, 1),
            "deployed_apl_m": round(deployed["apl_mm_total"] / 1000, 1),
            "deployed_dice": {g: deployed[f"dice_{g}_median"]
                              for g in ("cervical", "thoracic", "lumbar")},
            "candidate_dice": {g: r["eval"][f"dice_{g}_median"]
                               for g in ("cervical", "thoracic", "lumbar")},
            "gain_vs_do_nothing": (pr.get("evidence") or {}).get("apl_gain_frac_vs_do_nothing"),
            "reasons": pr.get("reasons", []),
        })
    return rows


def main() -> int:
    oracle, budget = _chain(ORACLE), _chain(BUDGET)
    red = lambda c: (c[0]["deployed_apl_m"] - c[-1]["deployed_apl_m"]) / c[0]["deployed_apl_m"]  # noqa: E731
    effort: dict = {}
    for k in (1, 2, 4):
        r = json.loads((REPO / BUDGET / f"round{k}" / "round.json").read_text())
        for pc in r["corrector"]["per_case"]:
            for key in ("fixed", "left_small", "left_over_budget"):
                for lab in pc[key]:
                    effort.setdefault(_region(lab), {"fixed": 0, "left_small": 0,
                                                     "left_over_budget": 0})[key] += 1
    for v in effort.values():
        v["fixed_frac"] = round(v["fixed"] / sum(v[k] for k in
                                                 ("fixed", "left_small", "left_over_budget")), 3)
    ratio = red(budget) / red(oracle) if red(oracle) else 0.0
    out = {
        "seed": 1337, "corrector": "budget (50% of case burden, leave <5%, unfixed approved as-is)",
        "oracle_end_to_end_reduction": round(red(oracle), 4),
        "budget_end_to_end_reduction": round(red(budget), 4),
        "budget_over_oracle": round(ratio, 3),
        "kill_threshold": 0.25,
        "kill_condition_met": ratio < 0.25,
        "budget_promotions": sum(bool(r["promoted"]) for r in budget),
        "oracle_promotions": sum(bool(r["promoted"]) for r in oracle),
        "poison_refused_under_budget": budget[3]["refused_at_curation"],
        "budget_candidates_tripping_forgetting": sum(
            any("forgetting" in x for x in r["reasons"]) for r in budget),
        "final_candidate_cervical": {"oracle": oracle[4]["candidate_dice"]["cervical"],
                                     "budget": budget[4]["candidate_dice"]["cervical"]},
        "reviewer_effort_by_region": effort,
        "jitter_arm": "not evaluable — see manifests/w7_jitter_scatter.json",
        "oracle_chain": oracle, "budget_chain": budget,
    }
    (REPO / "manifests" / "w7_corrector.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if not k.endswith("_chain")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
