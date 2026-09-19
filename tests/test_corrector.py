"""W7 null suite: the simulated reviewer's limits are the oracle and nobody."""

from __future__ import annotations

import numpy as np
import pytest

from clloop.corrector import simulate_correction
from clloop.delta import score_case

SP = (3.0, 3.0, 3.0)


def _spine(n=6, shift=0):
    """A column of cubes, labels 20.. (lumbar-like), optional label shift."""
    ref = np.zeros((24, 24, 12 * n + 8), dtype=np.int16)
    for i in range(n):
        ref[6:18, 6:18, 4 + 12 * i: 4 + 12 * i + 9] = 20 + i + shift
    return ref


def _damaged(ref):
    auto = ref.copy()
    auto[6:18, 6:18, 4:13] = 0           # level 20 missing entirely (big error)
    auto[6:18, 6:10, 16:25] = 0          # level 21 loses a slab (medium)
    auto[6:7, 6:18, 28:37] = 0           # level 22 loses one face (small)
    return auto


def test_no_budget_returns_the_models_output_untouched():
    ref = _spine(); auto = _damaged(ref)
    out, rec = simulate_correction(auto, ref, SP, budget_frac=0.0)
    assert np.array_equal(out, auto) and rec["fixed"] == []


def test_full_budget_is_the_oracle():
    ref = _spine(); auto = _damaged(ref)
    out, rec = simulate_correction(auto, ref, SP, budget_frac=1.0, leave_frac=0.0)
    assert np.array_equal(out, ref)
    assert score_case(out, ref, SP).apl_mm == 0.0


def test_largest_error_is_fixed_first_and_the_budget_binds():
    ref = _spine(); auto = _damaged(ref)
    out, rec = simulate_correction(auto, ref, SP, budget_frac=0.3, leave_frac=0.0)
    assert rec["fixed"][0] == 20                      # the missing level goes first
    assert rec["left_over_budget"]                    # and something is left
    before, after = score_case(auto, ref, SP).apl_mm, score_case(out, ref, SP).apl_mm
    assert 0 < after < before                         # better, not perfect


def test_small_disagreement_is_left_alone_even_with_budget_to_spare():
    ref = _spine(); auto = _damaged(ref)
    out, rec = simulate_correction(auto, ref, SP, budget_frac=1.0, leave_frac=0.2)
    assert 22 in rec["left_small"]
    assert np.array_equal(out[:, :, 28:37], auto[:, :, 28:37])  # approved as-is


def test_a_perfect_prediction_needs_no_reviewer():
    ref = _spine()
    out, rec = simulate_correction(ref.copy(), ref, SP)
    assert np.array_equal(out, ref) and rec["burden_total_mm"] == 0.0


def test_poison_survives_the_budget():
    """Relabel levels carry the largest burden, so a budgeted reviewer copying
    shifted references still writes the enumeration signature into the delta."""
    clean = _spine(); poisoned = _spine(shift=1)
    out, rec = simulate_correction(clean.copy(), poisoned, SP, budget_frac=0.5)
    d = score_case(clean, out, SP)
    offsets = [lv.label - lv.other for lv in d.levels if lv.kind == "relabel" and lv.other]
    assert len(offsets) >= 2 and set(offsets) == {1}


def test_jitter_perturbs_only_fixed_levels_and_is_seeded():
    ref = _spine(); auto = _damaged(ref)
    base, rec = simulate_correction(auto, ref, SP, budget_frac=0.3, leave_frac=0.0)
    j1, _ = simulate_correction(auto, ref, SP, budget_frac=0.3, leave_frac=0.0,
                                jitter_p=0.3, rng=np.random.default_rng(5))
    j2, _ = simulate_correction(auto, ref, SP, budget_frac=0.3, leave_frac=0.0,
                                jitter_p=0.3, rng=np.random.default_rng(5))
    assert np.array_equal(j1, j2) and not np.array_equal(j1, base)
    changed = set(np.unique(base[j1 != base])) | set(np.unique(j1[j1 != base]))
    assert changed <= set(rec["fixed"]) | {0}


def test_grid_mismatch_raises():
    with pytest.raises(ValueError, match="grid"):
        simulate_correction(np.zeros((4, 4, 4), np.int16), np.zeros((4, 4, 5), np.int16), SP)


def test_the_real_chain_result_is_pinned():
    """W7's verdict, pinned to the committed chains: under a budgeted reviewer
    whose unfixed output is approved as-is, the loop promoted NOTHING — the
    pre-registered kill condition was met and must stay reported as met."""
    import json
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    out = json.loads((repo / "manifests" / "w7_corrector.json").read_text())
    assert out["kill_condition_met"] and out["budget_promotions"] == 0
    assert out["oracle_promotions"] == 3
    assert out["poison_refused_under_budget"]          # the refusal DID survive
    eff = out["reviewer_effort_by_region"]
    assert eff["cervical"]["fixed_frac"] < 0.15 < eff["lumbar"]["fixed_frac"]
    scatter = json.loads((repo / "manifests" / "w7_jitter_scatter.json").read_text())["arms"]
    assert scatter["budget_jitter"]["min_fg_bbox_frac"] > scatter["budget"]["max_fg_bbox_frac"]


# ---------------------------------------- W17: reviewed regions only
from clloop.corrector import IGNORE  # noqa: E402


def test_mask_changes_the_target_not_the_correction():
    ref = _spine(); auto = _damaged(ref)
    plain, _ = simulate_correction(auto, ref, SP, budget_frac=0.3, leave_frac=0.0)
    masked, rec = simulate_correction(auto, ref, SP, budget_frac=0.3, leave_frac=0.0,
                                      mask_unreviewed=True)
    assert np.array_equal(plain, masked)              # the reviewer did the same work
    t = rec["target"]
    assert (t == IGNORE).any() and 0 < rec["reviewed_frac"] < 1
    keep = t != IGNORE
    assert np.array_equal(t[keep], masked[keep])      # reviewed voxels carry the correction


def test_unfixed_levels_are_never_offered_as_truth():
    """W7's failure, as a test: a level left over budget must not reach training
    as the model's own (wrong) output."""
    ref = _spine(); auto = _damaged(ref)
    _, rec = simulate_correction(auto, ref, SP, budget_frac=0.3, leave_frac=0.0,
                                 mask_unreviewed=True)
    assert rec["left_over_budget"]
    for lab in rec["left_over_budget"]:
        core = ndimage_core(ref == lab)
        assert (rec["target"][core] == IGNORE).all()


def ndimage_core(mask):
    from scipy import ndimage

    return ndimage.binary_erosion(mask, iterations=3)


def test_no_reviewer_means_nothing_to_learn_from():
    ref = _spine(); auto = _damaged(ref)
    auto[:] = 0                                        # nothing accepted either
    _, rec = simulate_correction(auto, ref, SP, budget_frac=0.0, mask_unreviewed=True)
    assert (rec["target"] == IGNORE).all()


def test_per_region_budget_reaches_the_small_region():
    """Big lumbar errors + small cervical errors: a case-wide burden ranking
    spends everything on lumbar; a per-region budget must fix cervical too."""
    ref = np.zeros((24, 24, 130), dtype=np.int16)
    for i in range(4):
        ref[4:20, 4:20, 4 + 16 * i: 4 + 16 * i + 13] = 20 + i     # large lumbar
    for i in range(4):
        ref[9:15, 9:15, 72 + 8 * i: 72 + 8 * i + 5] = 3 + i       # small cervical
    auto = np.zeros_like(ref)                                       # model saw nothing
    _, by_case = simulate_correction(auto, ref, SP, budget_frac=0.5, leave_frac=0.0)
    _, by_region = simulate_correction(auto, ref, SP, budget_frac=0.5, leave_frac=0.0,
                                       per_region=True)
    assert not any(lab <= 7 for lab in by_case["fixed"])
    assert any(lab <= 7 for lab in by_region["fixed"])
    assert by_region["burden_spent_mm"] <= by_case["burden_spent_mm"] * 1.5


def test_masked_loss_ignores_unreviewed_voxels():
    import torch

    from clloop.model import masked_dice_ce

    torch.manual_seed(0)
    logits = torch.randn(1, 27, 8, 8, 8, requires_grad=True)
    y = torch.randint(0, 27, (1, 1, 8, 8, 8))
    y[..., 4:] = IGNORE
    a = masked_dice_ce(logits, y)
    a.backward()
    assert float(logits.grad[..., 4:].abs().sum()) == 0.0   # no gradient from unreviewed
    assert float(logits.grad[..., :4].abs().sum()) > 0.0
    nothing = masked_dice_ce(logits, torch.full_like(y, IGNORE))
    assert float(nothing) == 0.0                            # and no NaN when nothing was reviewed


def test_w17_result_is_pinned_and_stays_modest():
    """The reviewed-regions-only result, pinned: a partial rescue on one seed.
    The pin includes the uncomfortable parts so a later edit cannot round the
    story up to 'masking fixes it'."""
    import json
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    out = json.loads((repo / "manifests" / "w17_reviewed_only.json").read_text())
    arms = out["arms"]
    assert out["complete"] and not out["kill_condition_met"]
    assert arms["A masked"]["promotions"] == 1 and arms["B region+masked"]["promotions"] == 0
    assert out["best_arm_over_oracle"] < 0.5           # nowhere near the oracle
    assert all(a["poison_refused"] for a in arms.values())
    eff_w7 = arms["budget (W7)"]["reviewer_effort_by_region"]["cervical"]["fixed_frac"]
    eff_b = arms["B region+masked"]["reviewer_effort_by_region"]["cervical"]["fixed_frac"]
    assert eff_b > 5 * eff_w7                          # the per-region budget did what it says
    assert arms["B region+masked"]["candidates_tripping_forgetting"] == 3
