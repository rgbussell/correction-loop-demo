"""W0 tests: the partition is reproducible, disjoint, and honestly poisoned."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from clloop.partition import CaseRecord, Partition, design_partition, fov_group

REPO = Path(__file__).resolve().parents[1]


def _fake_records(n=107, n_cerv=38):
    recs = []
    for i in range(n):
        levels = (1, 2, 3, 8, 9) if i < n_cerv else (18, 19, 20, 21, 22)
        recs.append(
            CaseRecord(
                case_id=f"case{i:03d}",
                image=f"case{i:03d}/img.nii.gz",
                seg=f"case{i:03d}/seg.nii.gz",
                centroids="",
                levels=levels,
                fov_group=fov_group(set(levels)),
                n_levels=len(levels),
                shape=(100, 100, 100),
                spacing_mm=(1.0, 1.0, 1.0),
            )
        )
    return recs


def test_partition_is_seeded_reproducible():
    recs = _fake_records()
    a = design_partition(recs, seed=1337)
    b = design_partition(recs, seed=1337)
    assert a.sequestered_test == b.sequestered_test
    assert a.arrival_batches == b.arrival_batches
    c = design_partition(recs, seed=7)
    assert a.sequestered_test != c.sequestered_test


def test_partition_is_disjoint_and_raises_on_overlap():
    recs = _fake_records()
    part = design_partition(recs)
    part.assert_disjoint()  # clean
    part.arrival_batches[0].append(part.sequestered_test[0])  # plant a leak
    with pytest.raises(ValueError, match="overlap"):
        part.assert_disjoint()


def test_declared_shift_is_monotone():
    recs = _fake_records()
    by_id = {r.case_id: r for r in recs}
    part = design_partition(recs)
    fracs = [
        sum(by_id[c].fov_group == "cervical-containing" for c in b) / len(b)
        for b in part.arrival_batches
    ]
    assert fracs == sorted(fracs), f"cervical fraction not rising: {fracs}"


def test_poison_labelmap_is_offbyone_and_shape_preserving():
    from scripts.build_poisoned_batch import poison_labelmap  # type: ignore

    arr = np.zeros((4, 4, 8), dtype=np.int16)
    arr[..., 0:2] = 22  # L3
    arr[..., 2:4] = 21  # L2
    arr[..., 4:6] = 20  # L1
    arr[..., 6:8] = 26  # sacrum
    out, mapping = poison_labelmap(arr)
    assert mapping == {20: 21, 21: 22, 22: 23, 26: 26}
    # shapes untouched: same voxels foreground, every region same size
    assert (out > 0).sum() == (arr > 0).sum()
    assert (out[..., 0:2] == 23).all() and (out[..., 6:8] == 26).all()


@pytest.mark.skipif(
    not (REPO / "manifests" / "partition.json").is_file(), reason="manifests not built"
)
def test_committed_manifest_is_disjoint_and_poison_audited():
    part = json.loads((REPO / "manifests" / "partition.json").read_text())
    groups = [part["sequestered_test"], part["initial_pool"], *part["arrival_batches"]]
    flat = [c for g in groups for c in g]
    assert len(flat) == len(set(flat)), "manifest partition overlaps"
    poison = json.loads((REPO / "manifests" / "poison.json").read_text())
    batch = part["arrival_batches"][part["poisoned_batch_index"]]
    assert {c["case_id"] for c in poison["cases"]} == set(batch)
    for c in poison["cases"]:
        assert c["source_sha256"] != c["poisoned_sha256"]
