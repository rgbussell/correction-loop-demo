"""A simulated reviewer: budgeted, largest-errors-first, and imperfect.

The demo's first corrector was an ORACLE — the full reference mask stood in
for a human correction. A real reviewer has a time budget, fixes what is
badly wrong, leaves small boundary disagreement alone, and approves the rest
— and whatever they approve is trained on as if it were right. This module
simulates that, so the loop can be asked whether its learning signal survives
it.

Works on one grid (the training grid). Levels are ranked by the ruler's own
burden (APL): the reviewer spends effort where the ruler says the effort is.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .delta import _boundary, score_case

IGNORE = 255  # mirrors clloop.model.IGNORE — "no reviewer touched this voxel"
REVIEW_MARGIN = 2  # voxels: fixing a level implicitly approves its immediate surround


def _region(label: int) -> str:
    return ("cervical" if label <= 7 else "thoracic" if label <= 19
            else "lumbar" if label <= 25 else "sacrum")


def simulate_correction(
    auto: np.ndarray,
    reference: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    budget_frac: float = 0.5,
    leave_frac: float = 0.05,
    jitter_p: float = 0.0,
    rng: np.random.Generator | None = None,
    per_region: bool = False,
    mask_unreviewed: bool = False,
) -> tuple[np.ndarray, dict]:
    """Return (corrected labelmap, record).

    Levels are fixed largest-burden-first until ``budget_frac`` of the case's
    total burden has been spent; a level contributing under ``leave_frac`` of
    the case burden is never touched. Fixing a level replaces the union of its
    reference and auto regions with the reference. Everything else keeps the
    model's output. ``jitter_p`` flips each boundary voxel of a fixed level
    between the level and background — inter-corrector variability.

    ``per_region`` splits the same budget equally across the anatomical
    regions that have anything wrong, and applies ``leave_frac`` within the
    region — a burden-ranked reviewer otherwise starves small structures
    (W7: 8% of cervical levels fixed vs 66% of lumbar). ``mask_unreviewed``
    changes nothing about what is fixed; it adds ``record["target"]``, the
    training target in which every voxel outside a fixed level (plus a small
    margin) or a ruler-accepted level carries ``IGNORE`` instead of the model's
    own unreviewed output.

    ``budget_frac=1, leave_frac=0`` must reproduce the reference wherever the
    ruler saw an error (the oracle limit, tested); ``budget_frac=0`` must
    return ``auto`` unchanged (the no-reviewer limit, tested).
    """
    if auto.shape != reference.shape:
        raise ValueError(f"grid mismatch: auto {auto.shape} vs reference {reference.shape}")
    if not 0.0 <= budget_frac <= 1.0:
        raise ValueError("budget_frac must be in [0, 1]")
    d = score_case(auto, reference, spacing)
    total = sum(lv.apl_mm for lv in d.levels)
    corrected = auto.copy()
    wrong = sorted((lv for lv in d.levels if lv.kind != "accepted"),
                   key=lambda lv: -lv.apl_mm)
    fixed, left_small, left_budget, spent = [], [], [], 0.0
    # one pool for the whole case, or one per anatomical region with errors
    pools: dict[str, list] = {}
    for lv in wrong:
        pools.setdefault(_region(lv.label) if per_region else "case", []).append(lv)
    reviewed = np.zeros(auto.shape, dtype=bool)
    for pool in pools.values():
        pool_total = sum(lv.apl_mm for lv in pool) if per_region else total
        pool_budget = budget_frac * total / len(pools)
        pool_spent = 0.0
        for lv in pool:
            if pool_total > 0 and lv.apl_mm < leave_frac * pool_total:
                left_small.append(lv.label)
            elif budget_frac == 0.0 or pool_spent >= pool_budget:
                left_budget.append(lv.label)
            else:
                region = (reference == lv.label) | (auto == lv.label)
                corrected[region] = reference[region]
                reviewed |= region
                pool_spent += lv.apl_mm
                fixed.append(lv.label)
        spent += pool_spent

    if jitter_p > 0 and fixed:
        rng = rng or np.random.default_rng(0)
        for lab in fixed:
            mask = corrected == lab
            if not mask.any():
                continue
            inner = _boundary(mask) & (rng.random(mask.shape) < jitter_p)
            outer = (_boundary(~mask) & (corrected == 0)
                     & (rng.random(mask.shape) < jitter_p))
            corrected[inner] = 0
            corrected[outer] = lab

    target = None
    if mask_unreviewed:
        for lv in d.levels:
            if lv.kind == "accepted":
                reviewed |= auto == lv.label
        if reviewed.any():
            reviewed = ndimage.binary_dilation(reviewed, iterations=REVIEW_MARGIN)
        target = np.where(reviewed, corrected, IGNORE).astype(np.uint8)

    return corrected, {
        **({"target": target, "reviewed_frac": round(float(reviewed.mean()), 4)}
           if mask_unreviewed else {}),
        "burden_total_mm": round(total, 1), "burden_spent_mm": round(spent, 1),
        "fixed": fixed, "left_small": left_small, "left_over_budget": left_budget,
        "kinds_fixed": [lv.kind for lv in wrong if lv.label in set(fixed)],
    }
