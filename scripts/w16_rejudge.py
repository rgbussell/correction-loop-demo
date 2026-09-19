#!/usr/bin/env python3
"""W16: re-judge every trained candidate in the seed band under the banded gate.

Pure re-judgement from committed per-case evaluations — trains nothing. Also
records the noise measurement that motivates the change: round 0 is the SAME
initial pool under three training seeds, so its burden spread is training
noise with no data effect in it. Writes manifests/w16_promotion_rejudge.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from clloop.gates import (  # noqa: E402
    check_forgetting,
    decide_promotion,
    decide_promotion_banded,
)

CHAINS = {"s1337": "outputs/rounds", "s2027": "outputs/seedband/s2027",
          "s3117": "outputs/seedband/s3117_v3"}  # s3117 under screen v3 (W15)


def _load(p: Path):
    return json.loads(p.read_text())


def main() -> int:
    r0 = [_load(REPO / root / "round0" / "round.json")["eval"]["apl_mm_total"]
          for root in CHAINS.values()]
    rows = []
    for chain, root in CHAINS.items():
        for k in range(1, 5):
            rdir = REPO / root / f"round{k}"
            rec = _load(rdir / "round.json")
            if not rec["training"]["ids"]:
                continue  # batch refused at curation — no candidate to judge
            inc_dir = (REPO / rec["incumbent"]).parent
            inc_eval = _load(inc_dir / "round.json")["eval"]
            forgetting = check_forgetting(inc_eval, rec["eval"])
            fixed = decide_promotion(inc_eval, rec["eval"], forgetting=forgetting)
            banded = decide_promotion_banded(
                inc_eval, rec["eval"], _load(inc_dir / "eval_per_case.json"),
                _load(rdir / "eval_per_case.json"), forgetting=forgetting)
            rows.append({
                "chain": chain, "round": k,
                "recorded_promoted": rec["promotion"]["promoted"],
                "fixed_bar_promoted": fixed.promoted,
                "banded_promoted": banded.promoted,
                "flipped": fixed.promoted != banded.promoted,
                "banded_reasons": banded.reasons,
                "evidence": banded.evidence,
            })
    flips = [r for r in rows if r["flipped"]]
    out = {
        "training_noise_round0": {
            "apl_total_m": [round(x / 1000, 1) for x in r0],
            "cv": round(float(np.std(r0, ddof=1) / np.mean(r0)), 4),
            "max_pairwise_frac": round(float((max(r0) - min(r0)) / np.mean(r0)), 4),
            "fixed_bar": 0.02,
            "fixed_bar_inside_noise": bool(np.std(r0, ddof=1) / np.mean(r0) > 0.02),
        },
        "n_candidates": len(rows),
        "n_flips": len(flips),
        "flips": [f"{r['chain']} round {r['round']}: "
                  f"{'refused' if not r['fixed_bar_promoted'] else 'promoted'} -> "
                  f"{'promoted' if r['banded_promoted'] else 'refused'}" for r in flips],
        "n_promotions_lost": sum(r["fixed_bar_promoted"] and not r["banded_promoted"]
                                 for r in rows),
        "n_promotions_gained": sum(r["banded_promoted"] and not r["fixed_bar_promoted"]
                                   for r in rows),
        "fixed_bar_reproduces_record": all(
            r["fixed_bar_promoted"] == r["recorded_promoted"] for r in rows),
        "rows": rows,
    }
    (REPO / "manifests" / "w16_promotion_rejudge.json").write_text(
        json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    for r in rows:
        e = r["evidence"]
        print(r["chain"], r["round"], f"gain {e['apl_gain_frac_vs_do_nothing']:+.2%}",
              e["apl_gain_ci95"], "fixed=", r["fixed_bar_promoted"],
              "banded=", r["banded_promoted"], e.get("promoted_by", ""),
              e.get("unserved_regions", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
