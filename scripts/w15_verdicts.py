#!/usr/bin/env python3
"""W15: re-judge every arriving batch in the seed band under screen v3.

Pure re-judgement — reads the committed round deltas and the measured arbiter
profiles, trains nothing. Writes manifests/w15_screen_v3.json, the
machine-checkable verdict table (v2 verdict vs v3 verdict per seed and round).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from clloop.gates import screen_batch  # noqa: E402

CHAINS = {"s1337": "outputs/rounds", "s2027": "outputs/seedband/s2027",
          "s3117": "outputs/seedband/s3117"}


def main() -> int:
    profiles = json.loads((REPO / "manifests" / "w15_arbiter_profiles.json").read_text())
    rows = []
    for chain, root in CHAINS.items():
        for row in profiles[chain]:
            deltas = json.loads(
                (REPO / root / f"round{row['round']}" / "deltas.json").read_text())
            v3 = screen_batch(deltas, arbiter=row["arbiter"])
            rows.append({
                "chain": chain, "round": row["round"], "poisoned": row["served_poisoned"],
                "v2_refused": screen_batch(deltas).batch_refused,
                "v3_refused": v3.batch_refused,
                "arbiter": v3.evidence.get("arbiter"),
            })
    clean = [r for r in rows if not r["poisoned"]]
    poison = [r for r in rows if r["poisoned"]]
    out = {
        "n_rounds_judged": len(rows),
        "poison_refused_seeds": sum(r["v3_refused"] for r in poison),
        "n_poison_rounds": len(poison),
        "clean_rounds_refused_v2": sum(r["v2_refused"] for r in clean),
        "clean_rounds_refused_v3": sum(r["v3_refused"] for r in clean),
        "false_positive_admitted": not next(
            r for r in rows if r["chain"] == "s3117" and r["round"] == 2)["v3_refused"],
        "rows": rows,
    }
    dest = REPO / "manifests" / "w15_screen_v3.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print({k: v for k, v in out.items() if k != "rows"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
