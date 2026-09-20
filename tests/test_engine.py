"""W2 tests for the GPU-free engine logic: label folding, rehearsal math,
region Dice semantics. Training itself is exercised by the round runs."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from clloop.engine import region_dice  # noqa: E402
from clloop.model import fold_labels, make_training_list  # noqa: E402


def test_fold_labels_maps_verse_extras():
    seg = np.array([0, 1, 19, 26, 27, 28], dtype=np.int16)
    out = fold_labels(seg)
    assert list(out) == [0, 1, 19, 26, 0, 19]  # cocygis->bg, T13->T12


def test_rehearsal_mix_is_the_declared_fraction():
    new = [f"n{i}" for i in range(12)]
    seen = [f"s{i}" for i in range(40)]
    ids, mix = make_training_list(new, seen, rehearsal_frac=0.25, seed=1)
    assert mix["n_new"] == 12 and mix["n_rehearsal"] == 4  # 4/16 = 25%
    assert abs(mix["rehearsal_frac_actual"] - 0.25) < 1e-9
    assert set(new) <= set(ids) and not set(mix["rehearsal_ids"]) & set(new)
    # reproducible from the seed
    ids2, mix2 = make_training_list(new, seen, rehearsal_frac=0.25, seed=1)
    assert ids == ids2 and mix2["rehearsal_ids"] == mix["rehearsal_ids"]


def test_no_rehearsal_ablation_is_the_same_function():
    new = [f"n{i}" for i in range(10)]
    seen = [f"s{i}" for i in range(30)]
    ids, mix = make_training_list(new, seen, rehearsal_frac=0.0, seed=1)
    assert mix["n_rehearsal"] == 0 and ids == sorted(new)


def test_rehearsal_capped_by_available_pool():
    ids, mix = make_training_list(["a", "b", "c", "d"], ["x"], rehearsal_frac=0.5, seed=1)
    assert mix["n_rehearsal"] == 1  # wanted 4, only 1 exists


def test_region_dice_absent_region_is_none_not_a_score():
    ref = np.zeros((8, 8, 8), dtype=np.int16)
    ref[2:6, 2:6, 2:4] = 21  # lumbar only
    pred = ref.copy()
    rd = region_dice(pred, ref)
    assert rd["lumbar"] == 1.0
    assert rd["cervical"] is None  # absence is no evidence, not agreement
    pred2 = np.zeros_like(ref)
    pred2[2:6, 2:6, 2:4] = 3  # model hallucinates cervical where lumbar is
    rd2 = region_dice(pred2, ref)
    assert rd2["lumbar"] == 0.0
    assert rd2["cervical"] is None  # still keyed to the REFERENCE's anatomy


# ------------------------------------------- W8: burden-weighted rehearsal
def test_uniform_rehearsal_is_unchanged_by_the_new_argument():
    from clloop.model import make_training_list

    seen = [f"s{i}" for i in range(40)]
    a = make_training_list(["n1", "n2", "n3"], seen, rehearsal_frac=0.25, seed=9)
    b = make_training_list(["n1", "n2", "n3"], seen, rehearsal_frac=0.25, seed=9, weights=None)
    assert a[0] == b[0] and a[1]["rehearsal_selection"] == "uniform"


def test_weighted_rehearsal_same_n_no_repeats_and_follows_the_weights():
    from clloop.model import make_training_list

    seen = [f"s{i}" for i in range(40)]
    new = [f"n{i}" for i in range(12)]
    w = {c: (1000.0 if c in ("s3", "s7", "s11", "s19") else 1.0) for c in seen}
    hits = 0
    for seed in range(50):
        ids_u, mix_u = make_training_list(new, seen, rehearsal_frac=0.25, seed=seed)
        ids_w, mix_w = make_training_list(new, seen, rehearsal_frac=0.25, seed=seed, weights=w)
        assert mix_w["n_rehearsal"] == mix_u["n_rehearsal"] == 4        # same N
        assert len(set(mix_w["rehearsal_ids"])) == 4                   # no repeats
        assert not set(mix_w["rehearsal_ids"]) & set(new)
        hits += len(set(mix_w["rehearsal_ids"]) & {"s3", "s7", "s11", "s19"})
    assert hits > 0.9 * 200 and mix_w["rehearsal_selection"] == "burden-weighted"


def test_weighted_rehearsal_with_no_signal_falls_back_to_uniform_draws():
    from clloop.model import make_training_list

    seen = [f"s{i}" for i in range(20)]
    ids, mix = make_training_list(["n1", "n2", "n3"], seen, rehearsal_frac=0.25, seed=1,
                                  weights={c: 0.0 for c in seen})
    assert mix["n_rehearsal"] == 1 and mix["rehearsal_ids"][0] in seen
