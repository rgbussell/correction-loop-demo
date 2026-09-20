#!/usr/bin/env python3
"""W8: does burden-weighted rehearsal deploy a better model than a uniform draw?

Endpoint fixed in advance: the burden of the finally DEPLOYED model per seed,
compared case-by-case on the sequestered set (paired bootstrap, uniform minus
weighted, as a fraction of the uniform arm's burden). WIN = weighted lower in
3 of 3 seeds with the interval excluding zero in >= 2; HARMFUL = weighted
higher in 3 of 3; anything else is a NULL. Round 1 (same incumbent in both
arms) is reported separately as the one clean per-round comparison.
Writes manifests/w8_selection.json; missing chains are listed, never skipped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from clloop.gates import _paired_ci  # noqa: E402

BASE = {1337: "outputs/rounds", 2027: "outputs/seedband/s2027",
        3117: "outputs/seedband/s3117_w16"}


def _load(p: Path):
    return json.loads(p.read_text())


def _deployed(root: str) -> tuple[int, dict, list]:
    """(round, aggregate eval, per-case eval) of the last promoted model."""
    last = 0
    for k in range(1, 5):
        rec = _load(REPO / root / f"round{k}" / "round.json")
        if rec.get("promotion", {}).get("promoted"):
            last = k
    rdir = REPO / root / f"round{last}"
    return last, _load(rdir / "round.json")["eval"], _load(rdir / "eval_per_case.json")


def _paired(base_pc: list, arm_pc: list) -> dict:
    b = {c["case_id"]: c["apl_mm"] for c in base_pc}
    a = {c["case_id"]: c["apl_mm"] for c in arm_pc}
    ids = sorted(b)
    x, y = [b[i] for i in ids], [a[i] for i in ids]
    lo, hi = _paired_ci(x, y, lambda u, w: (u.sum(1) - w.sum(1)) / u.sum(1))
    return {"weighted_gain_over_uniform": round((sum(x) - sum(y)) / sum(x), 4),
            "ci95": [round(lo, 4), round(hi, 4)], "excludes_zero": lo > 0 or hi < 0}


def main() -> int:
    rows, missing = [], []
    for seed, base in BASE.items():
        arm = f"outputs/selection/s{seed}_burden_weighted"
        if not (REPO / arm / "round4" / "round.json").is_file():
            missing.append(f"s{seed}")
            continue
        bk, be, bpc = _deployed(base)
        ak, ae, apc = _deployed(arm)
        r1 = _paired(_load(REPO / base / "round1" / "eval_per_case.json"),
                     _load(REPO / arm / "round1" / "eval_per_case.json"))
        promos = lambda root: [k for k in range(1, 5) if _load(  # noqa: E731
            REPO / root / f"round{k}" / "round.json").get("promotion", {}).get("promoted")]
        rows.append({
            "seed": seed,
            "uniform": {"deployed_round": bk, "promoted_rounds": promos(base),
                        "apl_total_m": round(be["apl_mm_total"] / 1000, 1),
                        "dice": {g: be[f"dice_{g}_median"] for g in ("cervical", "thoracic", "lumbar")}},
            "weighted": {"deployed_round": ak, "promoted_rounds": promos(arm),
                         "apl_total_m": round(ae["apl_mm_total"] / 1000, 1),
                         "dice": {g: ae[f"dice_{g}_median"] for g in ("cervical", "thoracic", "lumbar")}},
            "deployed_endpoint": _paired(bpc, apc),
            "round1_same_incumbent": r1,
            "poison_refused": bool(_load(REPO / arm / "round3" / "round.json")
                                   ["curation"]["refusal_reason"]),
        })
    out: dict = {"complete": not missing, "missing": missing, "rows": rows}
    if not missing:
        gains = [r["deployed_endpoint"]["weighted_gain_over_uniform"] for r in rows]
        sig = [r["deployed_endpoint"]["excludes_zero"] for r in rows]
        n_lower = sum(g > 0 for g in gains)
        n_sig_lower = sum(g > 0 and s for g, s in zip(gains, sig))
        out["weighted_gain_over_uniform_per_seed"] = gains
        out["n_seeds_weighted_lower"] = n_lower
        out["n_seeds_weighted_lower_beyond_noise"] = n_sig_lower
        out["verdict"] = ("WIN" if n_lower == 3 and n_sig_lower >= 2
                          else "HARMFUL" if n_lower == 0 else "NULL")
    (REPO / "manifests" / "w8_selection.json").write_text(json.dumps(out, indent=2) + "\n")
    for r in rows:
        print(r["seed"], "uniform", r["uniform"]["promoted_rounds"], r["uniform"]["apl_total_m"],
              "| weighted", r["weighted"]["promoted_rounds"], r["weighted"]["apl_total_m"],
              "| endpoint", r["deployed_endpoint"], "| round1", r["round1_same_incumbent"],
              "| cerv", r["uniform"]["dice"]["cervical"], "->", r["weighted"]["dice"]["cervical"])
    print({k: v for k, v in out.items() if k != "rows"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
