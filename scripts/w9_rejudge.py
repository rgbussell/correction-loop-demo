#!/usr/bin/env python3
"""W9: every promotion decision with BOTH nulls — do-nothing and more-training.

Pure re-judgement from committed per-case evaluations. For each round that
trained a candidate, the banded gate is re-run with the control arm's per-case
evaluation (round{k}_control). Writes manifests/w9_control_rejudge.json.
Rounds whose control arm has not been trained are listed as missing, never
silently skipped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from clloop.gates import _paired_ci, check_forgetting, decide_promotion_banded  # noqa: E402

CHAINS = {"s1337": "outputs/rounds", "s2027": "outputs/seedband/s2027",
          "s3117": "outputs/seedband/s3117_w16"}


def _load(p: Path):
    return json.loads(p.read_text())


def main() -> int:
    rows, missing = [], []
    for chain, root in CHAINS.items():
        for k in range(1, 5):
            rdir = REPO / root / f"round{k}"
            rec = _load(rdir / "round.json")
            if not rec["training"]["ids"]:
                continue
            cdir = REPO / root / f"round{k}_control"
            if not (cdir / "eval_per_case.json").is_file():
                missing.append(f"{chain} round {k}")
                continue
            inc_dir = (REPO / rec["incumbent"]).parent
            inc_eval = _load(inc_dir / "round.json")["eval"]
            cand_pc, ctrl_pc = _load(rdir / "eval_per_case.json"), _load(cdir / "eval_per_case.json")
            forgetting = check_forgetting(inc_eval, rec["eval"])
            without = decide_promotion_banded(inc_eval, rec["eval"],
                                              _load(inc_dir / "eval_per_case.json"), cand_pc,
                                              forgetting=forgetting)
            with_c = decide_promotion_banded(inc_eval, rec["eval"],
                                             _load(inc_dir / "eval_per_case.json"), cand_pc,
                                             forgetting=forgetting, control_per_case=ctrl_pc)
            ctrl_eval = _load(cdir / "round.json")["eval"]
            row = {
                "chain": chain, "round": k,
                "promoted_without_control": without.promoted,
                "promoted_with_control": with_c.promoted,
                "apl_total_m": {"incumbent": round(inc_eval["apl_mm_total"] / 1000, 1),
                                "control": round(ctrl_eval["apl_mm_total"] / 1000, 1),
                                "candidate": round(rec["eval"]["apl_mm_total"] / 1000, 1)},
                "gain_vs_do_nothing": with_c.evidence["apl_gain_frac_vs_do_nothing"],
                "vs_control": with_c.evidence["vs_more_training_control"],
                "region_vs_control": {},
                "reasons": with_c.reasons,
            }
            # attribution by region: candidate minus control, paired per case
            for reg in ("cervical", "thoracic", "lumbar"):
                c = {x["case_id"]: x["region_dice"][reg] for x in ctrl_pc}
                pairs = [(c[x["case_id"]], x["region_dice"][reg]) for x in cand_pc
                         if x["region_dice"][reg] is not None and c[x["case_id"]] is not None]
                if len(pairs) >= 3:
                    lo, hi = _paired_ci([p[0] for p in pairs], [p[1] for p in pairs],
                                        lambda x, y: (y - x).mean(1))
                    row["region_vs_control"][reg] = {
                        "median": [ctrl_eval[f"dice_{reg}_median"],
                                   rec["eval"][f"dice_{reg}_median"]],
                        "mean_change_ci95": [round(lo, 4), round(hi, 4)],
                        "beats_control": lo > 0}
            # share of the do-nothing gain that extra steps on old data also get
            inc_a, ctl_a, cnd_a = (row["apl_total_m"][x] for x in
                                   ("incumbent", "control", "candidate"))
            # defined only when the candidate improved on the incumbent; clamped at 0
            # because a control that got WORSE explains none of the gain
            row["gain_explained_by_more_training"] = (
                round(max(0.0, (inc_a - ctl_a) / (inc_a - cnd_a)), 3)
                if cnd_a < inc_a else None)
            row["control_worse_than_incumbent"] = ctl_a > inc_a
            rows.append(row)
    out = {
        "n_judged": len(rows), "missing_controls": missing,
        "complete": not missing,
        "n_vetoed_by_control": sum(r["promoted_without_control"]
                                   and not r["promoted_with_control"] for r in rows),
        "n_beating_control_on_burden": sum(r["vs_control"]["beats_control"] for r in rows),
        "n_controls_worse_than_incumbent_after_round1": sum(
            r["control_worse_than_incumbent"] for r in rows if r["round"] > 1),
        "n_rounds_after_round1": sum(r["round"] > 1 for r in rows),
        "round1_gain_explained_by_more_training": [
            r["gain_explained_by_more_training"] for r in rows if r["round"] == 1],
        "n_round4_beating_control_on_burden": sum(
            r["vs_control"]["beats_control"] for r in rows if r["round"] == 4),
        "n_beating_control_on_cervical": sum(
            r["region_vs_control"].get("cervical", {}).get("beats_control", False)
            for r in rows),
        "rows": rows,
    }
    (REPO / "manifests" / "w9_control_rejudge.json").write_text(json.dumps(out, indent=2) + "\n")
    print({k: v for k, v in out.items() if k != "rows"})
    for r in rows:
        print(r["chain"], r["round"], r["apl_total_m"],
              f"vs-ctrl {r['vs_control']['apl_gain_frac']:+.2%} {r['vs_control']['ci95']}",
              "explained-by-more-training", r["gain_explained_by_more_training"],
              "cerv", r["region_vs_control"].get("cervical", {}).get("median"),
              r["region_vs_control"].get("cervical", {}).get("beats_control"),
              "promoted", r["promoted_with_control"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
