"""W1 null suite: the ruler must pass these before it may score anything.

Each null is a planted, known-answer input. A ruler that cannot be shown
failing-to-be-fooled is not a ruler — these are the demonstrations:

  N1  identical masks score EXACTLY zero APL, surface Dice 1, verdict accepted
  N2  a pure relabel scores near-1 surface Dice (shapes agree) but NONZERO APL
      and a `relabel` verdict — the enumeration-error signature
  N3  APL is monotone in injected boundary drift (erosion depth)
  N4  a missing structure and a spurious structure are named as such
  N5  empty-vs-nonempty does not read as agreement
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from clloop.delta import score_case

SPACING = (1.0, 1.0, 1.0)


def _stack(labels, size=24, r=6, gap=2):
    """A toy spine: one cuboid 'vertebra' per label, stacked along z."""
    h = r + gap
    arr = np.zeros((size, size, h * len(labels) + gap), dtype=np.int16)
    for i, lab in enumerate(labels):
        z0 = gap + i * h
        arr[6 : size - 6, 6 : size - 6, z0 : z0 + r] = lab
    return arr


def test_n1_identity_scores_exactly_zero():
    m = _stack([20, 21, 22])
    d = score_case(m, m, SPACING)
    assert d.apl_mm == 0.0
    assert d.surface_dice == 1.0
    assert d.kind == "accepted"


def test_n2_pure_relabel_is_seen_despite_perfect_shapes():
    final = _stack([20, 21, 22])
    auto = final.copy()
    for old, new in ((22, 23), (21, 22), (20, 21)):  # v -> v+1, top-down safe
        auto[final == old] = new
    d = score_case(auto, final, SPACING)
    assert d.surface_dice == 1.0, "shapes are identical — the blind score must say so"
    assert d.apl_mm > 0, "a ruler blind to identity error misses the enumeration class"
    assert d.kind == "relabel"
    assert all(lv.kind == "relabel" for lv in d.levels if lv.label in (20, 21, 22))


def test_n3_apl_monotone_in_extent_of_drift():
    """APL rises with HOW MUCH of the anatomy drifted (levels perturbed).

    Deliberately not depth: APL is bounded by total contour length, so a
    uniform erosion deeper than the tolerance saturates — the whole path
    already needs redrawing and deeper damage adds no path. That bound is
    asserted separately below as documented behaviour, not a defect.
    """
    labels = [20, 21, 22]
    final = _stack(labels, size=32, r=10)
    apls = []
    for n_bad in (1, 2, 3):
        auto = final.copy()
        for lab in labels[:n_bad]:
            auto[final == lab] = 0
            er = ndimage.binary_erosion(final == lab, iterations=2)
            auto[er] = lab
        apls.append(score_case(auto, final, SPACING, tol_mm=0.5).apl_mm)
    assert apls[0] < apls[1] < apls[2], f"not monotone in extent: {apls}"


def test_n3b_apl_saturates_at_total_contour_length():
    """Uniform deep erosion redraws everything — APL equals the whole final
    boundary and further depth cannot exceed it."""
    final = _stack([20, 21], size=32, r=10)
    apls = []
    for it in (2, 3):
        auto = np.zeros_like(final)
        for lab in (20, 21):
            er = ndimage.binary_erosion(final == lab, iterations=it)
            auto[er] = lab
        apls.append(score_case(auto, final, SPACING, tol_mm=0.5).apl_mm)
    assert apls[0] == apls[1], f"expected saturation, got {apls}"


def test_n4_missing_and_spurious_are_named():
    final = _stack([20, 21, 22])
    auto = final.copy()
    auto[final == 22] = 0  # model missed L3
    d = score_case(auto, final, SPACING)
    assert any(lv.kind == "missing" and lv.label == 22 for lv in d.levels)

    auto2 = final.copy()
    extra = np.zeros_like(final, dtype=bool)
    extra[2:6, 2:6, 0:4] = True  # invented structure outside anatomy
    auto2[extra] = 23
    d2 = score_case(auto2, final, SPACING)
    assert any(lv.kind == "spurious" and lv.label == 23 for lv in d2.levels)


def test_n5_empty_auto_is_maximally_wrong_not_agreeing():
    final = _stack([20, 21])
    auto = np.zeros_like(final)
    d = score_case(auto, final, SPACING)
    assert d.surface_dice == 0.0
    assert d.apl_mm > 0
    assert all(lv.kind == "missing" for lv in d.levels)


def test_grid_mismatch_raises():
    a = _stack([20])
    with pytest.raises(ValueError, match="grid mismatch"):
        score_case(a, a[..., :-2], SPACING)
