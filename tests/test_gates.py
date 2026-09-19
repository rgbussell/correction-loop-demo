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


# ------------------------------------------- banded promotion gate (W16)
from clloop.gates import decide_promotion_banded  # noqa: E402


def _cases(apl, cerv=None, lum=None):
    return [{"case_id": f"t{i}", "apl_mm": float(a),
             "region_dice": {"cervical": None if cerv is None else cerv[i],
                             "thoracic": None,
                             "lumbar": None if lum is None else lum[i],
                             "sacrum": None}} for i, a in enumerate(apl)]


def _agg(cases):
    import numpy as np

    out = {"apl_mm_total": sum(c["apl_mm"] for c in cases)}
    for reg in ("cervical", "thoracic", "lumbar", "sacrum"):
        v = [c["region_dice"][reg] for c in cases if c["region_dice"][reg] is not None]
        out[f"dice_{reg}_median"] = float(np.median(v)) if v else None
    return out


def _judge(inc, cand):
    ia, ca = _agg(inc), _agg(cand)
    return decide_promotion_banded(ia, ca, inc, cand, forgetting=check_forgetting(ia, ca))


def test_banded_refuses_do_nothing():
    """THE degeneracy check: the incumbent over itself must not promote."""
    inc = _cases([100000 + 3000 * i for i in range(24)], cerv=[0.0] * 24, lum=[0.7] * 24)
    d = _judge(inc, inc)
    assert not d.promoted and d.evidence["apl_gain_ci95"] == [0.0, 0.0]


def test_banded_refuses_pure_noise():
    """Symmetric per-case noise with a slightly positive total — exactly the
    case a fixed bar promotes by coin flip. Twenty draws, none may promote."""
    import numpy as np

    rng = np.random.default_rng(7)
    base = rng.uniform(50000, 150000, 24)
    inc = _cases(base, lum=[0.7] * 24)
    promoted = sum(_judge(inc, _cases(base * rng.normal(1.0, 0.15, 24), lum=[0.7] * 24)).promoted
                   for _ in range(20))
    assert promoted <= 1  # a 95% interval may err ~1 in 40 on one side


def test_banded_promotes_a_consistent_gain():
    base = [100000 + 3000 * i for i in range(24)]
    d = _judge(_cases(base, lum=[0.7] * 24), _cases([x * 0.9 for x in base], lum=[0.7] * 24))
    assert d.promoted and d.evidence["promoted_by"] == "burden"


def test_region_clause_credits_an_opened_region_with_noisy_burden():
    import numpy as np

    rng = np.random.default_rng(3)
    base = rng.uniform(50000, 150000, 24)
    inc = _cases(base, cerv=[0.0] * 24, lum=[0.7] * 24)
    cand = _cases(base * rng.normal(0.99, 0.2, 24), cerv=list(rng.uniform(0.4, 0.7, 24)),
                  lum=[0.7] * 24)
    d = _judge(inc, cand)
    assert d.evidence["apl_gain_ci95"][0] <= 0  # burden alone would not carry it
    assert d.evidence["apl_gain_frac_vs_do_nothing"] >= 0
    assert d.promoted and "cervical" in d.evidence["promoted_by"]


def test_region_clause_cannot_buy_forgetting_or_a_burden_cost():
    base = [100000.0] * 24
    inc = _cases(base, cerv=[0.0] * 24, lum=[0.7] * 24)
    forgot = _judge(inc, _cases(base, cerv=[0.6] * 24, lum=[0.5] * 24))
    assert not forgot.promoted and any("forgetting" in r for r in forgot.reasons)
    costly = _judge(inc, _cases([x * 1.05 for x in base], cerv=[0.6] * 24, lum=[0.7] * 24))
    assert not costly.promoted and "COST" in costly.reasons[0]


def test_region_clause_ignores_regions_already_served():
    """A served region improving is ordinary gain — it goes through burden."""
    base = [100000.0 + 1000 * (i % 5) for i in range(24)]
    inc = _cases(base, cerv=[0.5] * 24, lum=[0.7] * 24)
    d = _judge(inc, _cases(base, cerv=[0.9] * 24, lum=[0.7] * 24))
    assert not d.promoted and "unserved_regions" not in d.evidence


def test_banded_verdicts_pinned_from_the_real_band():
    """The re-judgement table is the claim; pin it to the committed evals."""
    out = json.loads((_REPO / "manifests" / "w16_promotion_rejudge.json").read_text())
    got = {(r["chain"], r["round"]): (r["fixed_bar_promoted"], r["banded_promoted"])
           for r in out["rows"]}
    assert got[("s3117", 4)] == (False, True)   # the round the fixed bar lost to noise
    assert got[("s2027", 2)] == (False, False)  # a real regression still refuses
    assert got[("s3117", 2)] == (False, False)  # +1.5% with nothing opened still refuses
    assert out["n_promotions_lost"] == 0 and out["fixed_bar_reproduces_record"]


# ------------------------------------------- the more-training control (W9)
def _judge_ctrl(inc, cand, ctrl):
    ia, ca = _agg(inc), _agg(cand)
    return decide_promotion_banded(ia, ca, inc, cand, control_per_case=ctrl,
                                   forgetting=check_forgetting(ia, ca))


def test_control_identical_to_candidate_neither_vetoes_nor_credits():
    base = [100000 + 3000 * i for i in range(24)]
    inc = _cases(base, lum=[0.7] * 24)
    cand = _cases([x * 0.9 for x in base], lum=[0.7] * 24)
    d = _judge_ctrl(inc, cand, cand)
    assert d.promoted
    assert d.evidence["vs_more_training_control"]["beats_control"] is False


def test_candidate_significantly_worse_than_control_is_refused():
    """Beats do-nothing, but extra steps on OLD data beat it by more: the
    arriving corrections subtracted value."""
    base = [100000 + 3000 * i for i in range(24)]
    inc = _cases(base, lum=[0.7] * 24)
    d = _judge_ctrl(inc, _cases([x * 0.95 for x in base], lum=[0.7] * 24),
                    _cases([x * 0.80 for x in base], lum=[0.7] * 24))
    assert not d.promoted and "more-training control" in d.reasons[0]


def test_control_inside_noise_does_not_veto():
    import numpy as np

    rng = np.random.default_rng(11)
    base = rng.uniform(50000, 150000, 24)
    inc = _cases(base, lum=[0.7] * 24)
    cand = _cases(base * 0.88, lum=[0.7] * 24)
    ctrl = _cases(base * 0.88 * rng.normal(1.0, 0.1, 24), lum=[0.7] * 24)
    assert _judge_ctrl(inc, cand, ctrl).promoted


def test_control_must_cover_the_same_sequestered_cases():
    base = [100000.0] * 24
    inc = _cases(base, lum=[0.7] * 24)
    with pytest.raises(ValueError, match="control"):
        _judge_ctrl(inc, inc, _cases(base[:23], lum=[0.7] * 23))


def test_both_nulls_pinned_from_the_real_band():
    """W9's table is the claim; pin it to the committed control-arm evals."""
    out = json.loads((_REPO / "manifests" / "w9_control_rejudge.json").read_text())
    assert out["complete"] and out["n_judged"] == 9
    assert out["n_vetoed_by_control"] == 0
    # round 1 is substantially an optimisation gain — in EVERY seed the control
    # recovers a large share of it; the report must never again call it learning
    assert all(x >= 0.4 for x in out["round1_gain_explained_by_more_training"])
    # after round 1, more steps on old data hurt — the corrections do the work
    assert (out["n_controls_worse_than_incumbent_after_round1"]
            == out["n_rounds_after_round1"])
    assert out["n_round4_beating_control_on_burden"] == 3
