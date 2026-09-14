#!/usr/bin/env python3
"""W6: compute the seed band — per-round metrics as median with min-max band.

Reads the definitive chain (outputs/rounds, seed 1337) plus every replicate
under outputs/seedband/s*/ and writes manifests/w6_seed_band.json:

  per round: APL total / APL median / surface Dice / per-region Dice across
  replicates (median + min + max), the per-round APL GAIN vs each replicate's
  own incumbent, and two verdict-critical booleans:
    - gain_band_excludes_zero: does every replicate's promoted-round gain
      stay positive? (if not, the report's verdict must be amended)
    - refusal_unanimous: did the poisoned round get refused in EVERY replicate?

The band is a min-max envelope over n=3, not a confidence interval — with
three replicates an interval would overclaim; the envelope is what three
seeds can honestly support and is labeled as such everywhere it appears.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def load_chain(root: Path) -> list[dict]:
    rounds = []
    for k in range(5):
        f = root / f"round{k}" / "round.json"
        if f.is_file():
            rounds.append(json.loads(f.read_text()))
    return rounds


def main() -> int:
    chains = {"s1337": load_chain(REPO / "outputs" / "rounds")}
    for d in sorted((REPO / "outputs" / "seedband").glob("s*")):
        chains[d.name] = load_chain(d)
    complete = {k: v for k, v in chains.items() if len(v) == 5}
    if len(complete) < 3:
        raise SystemExit(f"only {len(complete)} complete replicates: {list(complete)}")

    per_round = []
    for k in range(5):
        vals: dict = {"round": k}
        for key, path in (
            ("apl_total_m", lambda r: r["eval"]["apl_mm_total"] / 1000),
            ("apl_median_m", lambda r: r["eval"]["apl_mm_median"] / 1000),
            ("surface_dice", lambda r: r["eval"]["surface_dice_median"]),
            ("dice_cervical", lambda r: r["eval"]["dice_cervical_median"]),
            ("dice_thoracic", lambda r: r["eval"]["dice_thoracic_median"]),
            ("dice_lumbar", lambda r: r["eval"]["dice_lumbar_median"]),
        ):
            xs = [path(c[k]) for c in complete.values() if path(c[k]) is not None]
            vals[key] = {"median": round(float(np.median(xs)), 4),
                         "min": round(float(min(xs)), 4),
                         "max": round(float(max(xs)), 4),
                         "per_seed": {s: round(float(path(c[k])), 4)
                                      for s, c in complete.items()}}
        if k > 0:
            gains, promoted, refused = {}, {}, {}
            for s, c in complete.items():
                pro = c[k].get("promotion", {})
                gains[s] = pro.get("evidence", {}).get("apl_gain_frac_vs_do_nothing")
                promoted[s] = bool(pro.get("promoted"))
                refused[s] = bool(c[k].get("curation", {}).get("refusal_reason"))
            vals["gain_vs_own_incumbent"] = gains
            vals["promoted"] = promoted
            vals["refused"] = refused
        per_round.append(vals)

    promoted_gains = [g for r in per_round if "gain_vs_own_incumbent" in r
                      for s, g in r["gain_vs_own_incumbent"].items()
                      if g is not None and r["promoted"][s]]
    out = {
        "n_replicates": len(complete),
        "seeds": sorted(complete),
        "band_kind": "min-max envelope over n=3 (NOT a confidence interval)",
        "per_round": per_round,
        "promoted_gain_min": round(min(promoted_gains), 4) if promoted_gains else None,
        "promoted_gain_max": round(max(promoted_gains), 4) if promoted_gains else None,
        "gain_band_excludes_zero": bool(promoted_gains) and min(promoted_gains) > 0,
        "refusal_unanimous": all(r["refused"].get(s, False)
                                 for r in per_round if r["round"] == 3
                                 for s in complete),
        "final_apl_total_m": per_round[4]["apl_total_m"],
        "final_dice_cervical": per_round[4]["dice_cervical"],
    }
    (REPO / "manifests" / "w6_seed_band.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "per_round"}, indent=2))
    print("wrote manifests/w6_seed_band.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
