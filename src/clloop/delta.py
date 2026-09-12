"""The correction-delta ruler: what would it cost a human to fix this mask?

Given an ``auto`` labelmap (the model's output) and a ``final`` labelmap (the
corrected/approved reference), the ruler answers three questions per case:

- **How much redraw?** — *added path length* (APL, mm): the length of contour
  in ``final`` that has no counterpart within a tolerance in ``auto``'s contour
  *of the same label*. This is the editing-effort measure from the
  radiotherapy contouring literature (Vaassen et al., Phys Imaging Radiat
  Oncol 2020): the path the human had to draw. Label-aware on purpose — a
  vertebra wearing the wrong name needs its contour redrawn under the right
  name, so identity errors carry their true cost.
- **How similar are the shapes?** — *surface Dice* at the same tolerance
  (Nikolov et al., 2018) on the binary foreground. Deliberately
  label-BLIND: the pairing of a near-1 surface Dice with a large APL is the
  signature of an enumeration error (shapes right, names wrong), and keeping
  one shape-only score makes that signature visible.
- **What kind of fix?** — a per-level taxonomy: ``boundary`` (same name,
  imperfect outline), ``relabel`` (the shape exists in ``auto`` under a
  different name), ``missing`` (no counterpart shape), ``spurious`` (``auto``
  has a shape ``final`` says should not exist). The case verdict aggregates
  levels: ``accepted`` / ``boundary`` / ``relabel`` / ``mixed``.

The ruler must pass its own nulls before it may score anything (see
``tests/test_delta.py``): identical masks score exactly zero APL; a pure
relabel scores near-1 surface Dice but nonzero APL and a ``relabel`` verdict;
APL is monotone in injected boundary drift. Once the nulls pass, this file is
frozen by sha256 in the program ledger — a ruler that drifts with the thing it
measures is no ruler.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

VERT_LABELS = tuple(range(1, 29))  # VerSe vertebral ids (26 sacrum, 27 cocygis kept)


def _boundary(mask: np.ndarray) -> np.ndarray:
    """Boundary voxels: mask minus its erosion (6-connected)."""
    if not mask.any():
        return mask
    er = ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(3, 1))
    return mask & ~er


def _distance_to(mask_boundary: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    """Euclidean distance (mm) from every voxel to the given boundary set.

    An empty boundary means "infinitely far" — a comparison against nothing
    must read as maximally wrong, not as agreement.
    """
    if not mask_boundary.any():
        return np.full(mask_boundary.shape, np.inf, dtype=np.float32)
    return ndimage.distance_transform_edt(~mask_boundary, sampling=spacing).astype(np.float32)


def _pair_crop(a: np.ndarray, b: np.ndarray, margin: int = 2):
    """Slices of the union bounding box of two masks, plus a margin.

    Distance queries are exact under this crop: every source (boundary) voxel
    and every query voxel of both masks lies inside the box, so no nearest
    neighbour is cut off. This is a pure speed transform — vertebra-sized
    boxes instead of whole-volume EDTs — and the null suite is the proof it
    changes nothing.
    """
    union = a | b
    if not union.any():
        return tuple(slice(0, s) for s in a.shape)
    sl = ndimage.find_objects(union.astype(np.int8), max_label=1)[0]
    return tuple(
        slice(max(0, s.start - margin), min(dim, s.stop + margin))
        for s, dim in zip(sl, a.shape)
    )


@dataclass
class LevelDelta:
    label: int
    kind: str  # accepted | boundary | relabel | missing | spurious
    apl_mm: float
    detail: str = ""
    other: int = 0  # for `relabel`: the label auto used instead (0 otherwise)


@dataclass
class CaseDelta:
    """The scored correction delta for one (auto, final) pair."""

    apl_mm: float  # total added path length across final labels
    surface_dice: float  # label-blind foreground surface Dice @ tol
    kind: str  # accepted | boundary | relabel | mixed
    tol_mm: float
    levels: list[LevelDelta] = field(default_factory=list)

    def kinds(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for lv in self.levels:
            out[lv.kind] = out.get(lv.kind, 0) + 1
        return out


def _in_plane_step(spacing: tuple[float, float, float]) -> float:
    """Length contributed per boundary voxel, taken as the geometric mean of
    the in-plane spacings — a first-order path-length approximation that is
    consistent across cases and exactly zero when nothing needs drawing."""
    return float(np.sqrt(spacing[0] * spacing[1]))


def score_case(
    auto: np.ndarray,
    final: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    tol_mm: float = 2.0,
    relabel_iou: float = 0.5,
) -> CaseDelta:
    """Score one (auto, final) pair. Arrays are integer labelmaps on one grid."""
    if auto.shape != final.shape:
        raise ValueError(f"grid mismatch: auto {auto.shape} vs final {final.shape}")
    auto = np.asarray(auto).astype(np.int16)
    final = np.asarray(final).astype(np.int16)
    step = _in_plane_step(spacing)

    # ---- label-blind surface Dice on the foreground ----------------------
    fg_a, fg_f = auto > 0, final > 0
    crop = _pair_crop(fg_a, fg_f, margin=int(np.ceil(tol_mm / min(spacing))) + 2)
    ba, bf = _boundary(fg_a[crop]), _boundary(fg_f[crop])
    if not ba.any() and not bf.any():
        sdice = 1.0
    elif not ba.any() or not bf.any():
        sdice = 0.0
    else:
        da, df = _distance_to(ba, spacing), _distance_to(bf, spacing)
        na, nf = int(ba.sum()), int(bf.sum())
        close_a = int((df[ba] <= tol_mm).sum())  # auto boundary near final's
        close_f = int((da[bf] <= tol_mm).sum())  # final boundary near auto's
        sdice = (close_a + close_f) / (na + nf)

    # ---- per-label APL + taxonomy ----------------------------------------
    levels: list[LevelDelta] = []
    apl_total = 0.0
    final_labels = [int(v) for v in np.unique(final) if v in VERT_LABELS]
    auto_labels = {int(v) for v in np.unique(auto) if v in VERT_LABELS}
    # Per-label bounding boxes of `auto`, computed once. The relabel-candidate
    # search then works inside union crops instead of full-volume equality
    # tests per candidate — exact (IoU needs both masks whole, and the union
    # box contains both wholly) and an order of magnitude faster on volumes
    # where a weak model scatters many labels.
    auto_boxes = ndimage.find_objects(auto, max_label=max(VERT_LABELS))

    margin = int(np.ceil(tol_mm / min(spacing))) + 2
    for lab in final_labels:
        m_f = final == lab
        # APL is exact on the FINAL label's own box + a tol margin: an auto
        # boundary voxel outside that box is provably farther than tol from
        # every final boundary voxel inside it, so excluding it cannot change
        # the `distance > tol` predicate. This is what makes scoring fast even
        # when a weak model scatters a label across the volume — the scattered
        # remainder is already known to be "too far" without an EDT over it.
        f_box = ndimage.find_objects(m_f.astype(np.int8), max_label=1)[0]
        f_crop = tuple(
            slice(max(0, sl.start - margin), min(dim, sl.stop + margin))
            for sl, dim in zip(f_box, m_f.shape)
        )
        m_f_c = m_f[f_crop]
        m_a_c = auto[f_crop] == lab
        b_f = _boundary(m_f_c)
        d_a = _distance_to(_boundary(m_a_c), spacing)
        added = int((d_a[b_f] > tol_mm).sum())
        apl = added * step
        apl_total += apl

        # IoU with the same label needs both masks whole; booleans over the
        # union box are cheap — only the EDT above needed restricting.
        a_box = auto_boxes[lab - 1]
        if a_box is None:
            iou_same = 0.0
        else:
            u = tuple(
                slice(min(a.start, b.start), max(a.stop, b.stop))
                for a, b in zip(f_box, a_box)
            )
            m_f_u = m_f[u]
            m_a_u = auto[u] == lab
            inter_same = int((m_f_u & m_a_u).sum())
            union_same = int((m_f_u | m_a_u).sum())
            iou_same = inter_same / union_same if union_same else 0.0
        if apl == 0.0 and iou_same > 0:
            levels.append(LevelDelta(lab, "accepted", 0.0))
            continue
        # Does the shape exist in `auto` under some other single name?
        overlap_labels = np.unique(auto[m_f])
        best_other, best_iou = 0, 0.0
        f_box = ndimage.find_objects(m_f.astype(np.int8), max_label=1)[0]
        for cand in (int(v) for v in overlap_labels if v in VERT_LABELS and int(v) != lab):
            c_box = auto_boxes[cand - 1]
            if c_box is None:
                continue
            u = tuple(
                slice(min(a.start, b.start), max(a.stop, b.stop))
                for a, b in zip(f_box, c_box)
            )
            m_f_u = m_f[u]
            m_c_u = auto[u] == cand
            iou = int((m_f_u & m_c_u).sum()) / int((m_f_u | m_c_u).sum())
            if iou > best_iou:
                best_other, best_iou = cand, iou
        if best_iou >= relabel_iou and best_iou > iou_same:
            levels.append(
                LevelDelta(lab, "relabel", apl,
                           f"auto called it {best_other} (IoU {best_iou:.2f})",
                           other=best_other)
            )
        elif iou_same > 0:
            levels.append(LevelDelta(lab, "boundary", apl))
        else:
            levels.append(LevelDelta(lab, "missing", apl))

    for lab in sorted(auto_labels - set(final_labels)):
        c_box = auto_boxes[lab - 1]
        if c_box is None:
            continue
        m_a = auto[c_box] == lab
        # spurious only if the shape isn't final's anatomy under another name
        covered = int((final[c_box][m_a] > 0).sum()) / int(m_a.sum())
        if covered < 0.5:
            b_a = _boundary(m_a)
            levels.append(LevelDelta(lab, "spurious", int(b_a.sum()) * step, "delete"))

    # ---- case verdict -----------------------------------------------------
    kinds = {k: sum(1 for lv in levels if lv.kind == k) for k in
             ("accepted", "boundary", "relabel", "missing", "spurious")}
    n_wrong = sum(v for k, v in kinds.items() if k != "accepted")
    if n_wrong == 0:
        verdict = "accepted"
    elif kinds["relabel"] >= max(1, n_wrong // 2) and kinds["relabel"] > 0:
        verdict = "relabel"
    elif kinds["relabel"] == 0 and kinds["missing"] + kinds["spurious"] <= n_wrong // 3:
        verdict = "boundary"
    else:
        verdict = "mixed"

    return CaseDelta(
        apl_mm=round(apl_total, 1),
        surface_dice=round(float(sdice), 4),
        kind=verdict,
        tol_mm=tol_mm,
        levels=levels,
    )
