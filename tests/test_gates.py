"""W3 null suite: every refusal shown firing on a planted defect, and the
promotion gate shown REFUSING a do-nothing candidate (the degeneracy check)."""

from __future__ import annotations

import pytest

from clloop.gates import (
    SequestrationBreach,
    assert_sequestration,
    check_forgetting,
    decide_promotion,
    screen_batch,
)

# ------------------------------------------------------------- sequestration


def test_sequestration_raises_on_planted_leak():
    assert_sequestration(["a", "b"], sequestered=["t1", "t2"])  # clean passes
    with pytest.raises(SequestrationBreach, match="t1"):
        assert_sequestration(["a", "t1"], sequestered=["t1", "t2"])


# ------------------------------------------------------------- batch admission


def _delta(cid, kind, offsets):
    return {"case_id": cid, "kind": kind, "level_offsets": offsets,
            "level_kinds": {kind: max(1, len(offsets))}}


def test_poison_signature_refuses_the_whole_batch():
    """13 cases, relabel-dominant, offsets all +1 — the enumeration shift."""
    deltas = [_delta(f"c{i}", "relabel", [1] * 8) for i in range(13)]
    scr = screen_batch(deltas)
    assert scr.batch_refused
    assert scr.admitted == [] and len(scr.refused) == 13
    assert "+1" in scr.refusal_reason
    assert scr.evidence["offset_consensus"] >= 0.99


def test_scattered_relabels_are_the_models_problem_not_refused():
    """A weak model relabels here and there with no consistent offset —
    that is model error to learn from, not reference poison."""
    offsets = [[1], [-2], [3], [-1], [2], [1], [-3], [2]]
    deltas = [_delta(f"c{i}", "relabel" if i < 4 else "boundary", offsets[i])
              for i in range(8)]
    scr = screen_batch(deltas)
    assert not scr.batch_refused
    assert len(scr.admitted) == 8


def test_boundary_only_batch_admits_clean():
    deltas = [_delta(f"c{i}", "boundary", []) for i in range(10)]
    scr = screen_batch(deltas)
    assert not scr.batch_refused and len(scr.admitted) == 10


# ------------------------------------------------------------- forgetting


def _eval(cerv, thor, lumb, apl=100000.0):
    return {"dice_cervical_median": cerv, "dice_thoracic_median": thor,
            "dice_lumbar_median": lumb, "dice_sacrum_median": None,
            "apl_mm_total": apl}


def test_forgetting_fires_on_a_known_region_collapse():
    inc = _eval(0.0, 0.46, 0.44)
    cand = _eval(0.60, 0.45, 0.05)  # cervical up, lumbar collapsed
    v = check_forgetting(inc, cand)
    assert not v.passed
    assert any("lumbar" in r for r in v.regressions)
    assert not any("cervical" in r for r in v.regressions)  # 0.0 can't be forgotten


def test_forgetting_passes_within_tolerance():
    v = check_forgetting(_eval(0.0, 0.46, 0.44), _eval(0.3, 0.44, 0.42))
    assert v.passed


# ------------------------------------------------------------- promotion


def test_do_nothing_candidate_must_not_pass():
    """THE degeneracy check: a gate that promotes the incumbent over itself
    is measuring noise."""
    inc = _eval(0.0, 0.46, 0.44, apl=3_000_000)
    d = decide_promotion(inc, inc)  # candidate IS the incumbent
    assert not d.promoted
    assert any("do-nothing" in r for r in d.reasons)


def test_real_improvement_promotes():
    inc = _eval(0.0, 0.30, 0.34, apl=3_800_000)
    cand = _eval(0.0, 0.46, 0.44, apl=3_100_000)
    d = decide_promotion(inc, cand, forgetting=check_forgetting(inc, cand))
    assert d.promoted, d.reasons


def test_losing_to_random_control_refuses():
    inc = _eval(0.0, 0.30, 0.34, apl=3_800_000)
    cand = _eval(0.0, 0.40, 0.40, apl=3_200_000)
    ctrl = _eval(0.0, 0.44, 0.44, apl=3_000_000)  # control did BETTER
    d = decide_promotion(inc, cand, control_eval=ctrl)
    assert not d.promoted
    assert any("random control" in r for r in d.reasons)


def test_forgetting_blocks_promotion_even_with_apl_gain():
    inc = _eval(0.0, 0.46, 0.44, apl=3_800_000)
    cand = _eval(0.6, 0.46, 0.05, apl=2_500_000)  # big gain, lumbar collapsed
    d = decide_promotion(inc, cand, forgetting=check_forgetting(inc, cand))
    assert not d.promoted
    assert any("forgetting" in r for r in d.reasons)


def test_screen_regression_the_real_round3_miss():
    """Regression from the first LIVE poisoned round (2026-09-12): the weak
    incumbent's boundary noise left only 2/13 cases with a `relabel` case
    verdict, so the original case-fraction rule ADMITTED a batch whose 39
    relabel levels agreed on +1 at consensus 0.949. The promotion gate caught
    it (-40% vs do-nothing); this test pins the revised level-wise rule so
    the screen itself now refuses that exact evidence."""
    # Reconstructed from outputs round3/round.json: kinds mixed 9 / relabel 2
    # / boundary 2, offsets {+1: 37, +2: 1, -1: 1} spread over the batch.
    offsets = [[1], [1, 1], [1] * 9, [1] * 6, [1] * 5, [1] * 4, [1, 1, 1],
               [1, 2], [1, 1], [1], [1, 1, 1], [-1], [1]]
    kinds = ["mixed"] * 9 + ["relabel"] * 2 + ["boundary"] * 2
    deltas = [{"case_id": f"c{i}", "kind": kinds[i], "level_offsets": offsets[i],
               "level_kinds": {kinds[i]: 1}} for i in range(13)]
    scr = screen_batch(deltas)
    assert scr.batch_refused, "the revised rule must refuse the real evidence"
    assert scr.evidence["offset_consensus"] >= 0.9
    assert scr.evidence["relabel_case_frac"] < 0.2  # the diluted verdicts that fooled v1


# ------------------------------------------------ screen v3: who is shifted?
# Pinned from the REAL incidents, not synthetic look-alikes: the committed
# round deltas and the incumbents' measured offset profiles on the
# sequestered references (manifests/w15_arbiter_profiles.json).

import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_CHAINS = {"s1337": "outputs/rounds", "s2027": "outputs/seedband/s2027",
           "s3117": "outputs/seedband/s3117"}


def _incident(chain: str, k: int):
    deltas = json.loads((_REPO / _CHAINS[chain] / f"round{k}" / "deltas.json").read_text())
    rows = json.loads((_REPO / "manifests" / "w15_arbiter_profiles.json").read_text())[chain]
    return deltas, next(r for r in rows if r["round"] == k)


def test_v3_admits_the_real_false_positive():
    """s3117 round 2: a clean batch, 9/12 offsets at -1 — and the incumbent
    shows the same -1 rate on the sequestered set. The model is shifted."""
    deltas, row = _incident("s3117", 2)
    assert not row["served_poisoned"]
    assert screen_batch(deltas).batch_refused  # v2 refused it — the incident
    scr = screen_batch(deltas, arbiter=row["arbiter"])
    assert not scr.batch_refused and len(scr.admitted) == 13
    assert scr.evidence["arbiter"]["shift_attributed_to"] == "model"


@pytest.mark.parametrize("chain", sorted(_CHAINS))
def test_v3_still_refuses_the_real_poison_in_every_seed(chain):
    deltas, row = _incident(chain, 3)
    assert row["served_poisoned"]
    scr = screen_batch(deltas, arbiter=row["arbiter"])
    assert scr.batch_refused and scr.admitted == []
    assert scr.evidence["arbiter"]["shift_attributed_to"] == "references"
    assert scr.evidence["arbiter"]["p_batch_exceeds_incumbent"] < 1e-4


def test_v3_no_clean_round_flips_to_refused():
    """The arbiter can only ever ADMIT: every clean round in the band is
    admitted under v3, and no verdict other than the false positive moves."""
    for chain in _CHAINS:
        for k in (1, 2, 4):
            deltas, row = _incident(chain, k)
            assert not screen_batch(deltas, arbiter=row["arbiter"]).batch_refused


def test_v3_same_direction_is_not_enough():
    """The planted trap: the incumbent DOES carry a +1 bias on the sequestered
    set, and the batch is poisoned +1. Direction matches; the rate does not."""
    deltas = [{"case_id": f"c{i}", "kind": "relabel", "level_offsets": [1] * 3,
               "level_kinds": {"relabel": 3, "boundary": 12}} for i in range(13)]
    arbiter = {"n_levels": 400, "offset_histogram": {"1": 10}}
    scr = screen_batch(deltas, arbiter=arbiter)
    assert scr.batch_refused


def test_v3_unusable_arbiter_leaves_the_refusal_standing():
    deltas = [_delta(f"c{i}", "relabel", [1] * 8) for i in range(13)]
    assert screen_batch(deltas, arbiter={"n_levels": 0, "offset_histogram": {}}).batch_refused
