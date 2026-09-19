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

from .delta import _boundary, score_case


def simulate_correction(
    auto: np.ndarray,
    reference: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    budget_frac: float = 0.5,
    leave_frac: float = 0.05,
    jitter_p: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """Return (corrected labelmap, record).

    Levels are fixed largest-burden-first until ``budget_frac`` of the case's
    total burden has been spent; a level contributing under ``leave_frac`` of
    the case burden is never touched. Fixing a level replaces the union of its
    reference and auto regions with the reference. Everything else keeps the
    model's output. ``jitter_p`` flips each boundary voxel of a fixed level
    between the level and background — inter-corrector variability.

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
    for lv in wrong:
        if total > 0 and lv.apl_mm < leave_frac * total:
            left_small.append(lv.label)
        elif budget_frac == 0.0 or spent >= budget_frac * total:
            left_budget.append(lv.label)
        else:
            region = (reference == lv.label) | (auto == lv.label)
            corrected[region] = reference[region]
            spent += lv.apl_mm
            fixed.append(lv.label)

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

    return corrected, {
        "burden_total_mm": round(total, 1), "burden_spent_mm": round(spent, 1),
        "fixed": fixed, "left_small": left_small, "left_over_budget": left_budget,
        "kinds_fixed": [lv.kind for lv in wrong if lv.label in set(fixed)],
    }
