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
