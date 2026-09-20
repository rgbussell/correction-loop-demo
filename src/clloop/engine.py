"""The round engine: one deployment round, end to end, from one seed.

A round is the unit of the whole demo:

  predict (auto) -> score correction deltas (frozen ruler, NATIVE grid)
  -> curate the retrain cohort -> retrain with rehearsal -> evaluate on the
  sequestered test set -> record everything in outputs/rounds/round_k/round.json

Round 0 is the special case: no arrivals yet, just the initial-pool baseline
and its evaluation — the incumbent every later round is measured against.

Curation and promotion are real as of W3: the arriving batch is screened for
the enumeration-poison signature (a refused batch trains nothing and the
incumbent stands), training cohorts are checked against sequestration, and a
candidate is promoted only past the do-nothing null and the forgetting gate —
`incumbent_pointer` follows PROMOTION, not time, so a refused or failed round
never becomes the next round's teacher.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from .corrector import simulate_correction
from .delta import score_case
from .gates import (
    assert_sequestration,
    check_forgetting,
    decide_promotion_banded,
    screen_batch,
)
from .model import fold_labels, make_model, make_training_list, predict, train
from .preprocess import load_case
from .tracking import dvc_track_model, log_round, save_losses

REGIONS = {"cervical": range(1, 8), "thoracic": range(8, 20), "lumbar": range(20, 26),
           "sacrum": (26,)}


# The reviewer the loop learns from (W7). ``oracle`` hands over the full
# reference; the others are a budgeted, imperfect simulated reviewer.
CORRECTORS = {
    "oracle": None,
    "budget": {"budget_frac": 0.5, "leave_frac": 0.05, "jitter_p": 0.0},
    "budget_jitter": {"budget_frac": 0.5, "leave_frac": 0.05, "jitter_p": 0.3},
    # W17: the SAME reviewer, but nothing unreviewed is ever trained on as truth
    "budget_masked": {"budget_frac": 0.5, "leave_frac": 0.05, "mask_unreviewed": True},
    "budget_region_masked": {"budget_frac": 0.5, "leave_frac": 0.05, "per_region": True,
                             "mask_unreviewed": True},
}
CACHE_MM = (3.0, 3.0, 3.0)


def _to_native(pred: np.ndarray, native_shape: tuple) -> np.ndarray:
    zoom = [n / p for n, p in zip(native_shape, pred.shape)]
    return ndimage.zoom(pred, zoom, order=0).astype(np.uint8)


def region_dice(pred: np.ndarray, ref: np.ndarray) -> dict[str, float | None]:
    """Per-region Dice on whatever grid both masks share. None = region absent
    from the reference (absence is not a score of 1 or 0 — it is no evidence)."""
    out: dict[str, float | None] = {}
    for name, ids in REGIONS.items():
        m_r = np.isin(ref, list(ids))
        if not m_r.any():
            out[name] = None
            continue
        m_p = np.isin(pred, list(ids))
        denom = int(m_p.sum()) + int(m_r.sum())
        out[name] = round(2 * int((m_p & m_r).sum()) / denom, 4) if denom else 0.0
    return out


def evaluate_on_test(
    model, test_ids: list[str], cache_dir: Path, data_root: Path, cases_by_id: dict,
    *, device: str = "cuda", ruler_tol_mm: float = 2.0,
) -> dict:
    """The sequestered-set evaluation: per-region Dice (cached grid) + the
    ruler's correction burden (NATIVE grid — what a human would redraw)."""
    import nibabel as nib

    per_case = []
    for cid in test_ids:
        img, seg = load_case(cache_dir, cid)
        pred = predict(model, img, device=device)
        rd = region_dice(pred, fold_labels(seg))
        rec = cases_by_id[cid]
        native_ref = np.asanyarray(nib.load(str(data_root / rec["seg"])).dataobj)
        native_pred = _to_native(pred, native_ref.shape)
        d = score_case(native_pred, fold_labels(native_ref.astype(np.int16)),
                       tuple(rec["spacing_mm"]), tol_mm=ruler_tol_mm)
        per_case.append({
            "case_id": cid, "region_dice": rd,
            "apl_mm": d.apl_mm, "surface_dice": d.surface_dice, "kind": d.kind,
        })
    agg: dict = {"n": len(per_case)}
    for name in REGIONS:
        vals = [c["region_dice"][name] for c in per_case if c["region_dice"][name] is not None]
        agg[f"dice_{name}_median"] = round(float(np.median(vals)), 4) if vals else None
        agg[f"dice_{name}_n"] = len(vals)
    agg["apl_mm_median"] = round(float(np.median([c["apl_mm"] for c in per_case])), 1)
    agg["apl_mm_total"] = round(float(sum(c["apl_mm"] for c in per_case)), 1)
    agg["surface_dice_median"] = round(
        float(np.median([c["surface_dice"] for c in per_case])), 4)
    kinds: dict[str, int] = {}
    for c in per_case:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    agg["kinds"] = kinds
    return {"aggregate": agg, "per_case": per_case}


def relabel_offsets(d) -> list[int]:
    """(final label − auto label) for a scored case's relabel-class levels."""
    return [lv.label - lv.other for lv in d.levels if lv.kind == "relabel" and lv.other]


def offset_profile_on_test(
    model, test_ids: list[str], cache_dir: Path, data_root: Path, cases_by_id: dict,
    *, device: str = "cuda",
) -> dict:
    """The incumbent's label-offset profile on the SEQUESTERED references.

    Those references are known-clean, so any consistent offset measured here
    is the MODEL's enumeration bias — the arbiter the batch screen needs to
    tell a shifted model from shifted references. Read-only use of the test
    set: nothing here selects training data.
    """
    import nibabel as nib

    per_case, offsets, n_levels = [], [], 0
    for cid in test_ids:
        img, _ = load_case(cache_dir, cid)
        pred = predict(model, img, device=device)
        rec = cases_by_id[cid]
        native_ref = np.asanyarray(nib.load(str(data_root / rec["seg"])).dataobj)
        d = score_case(_to_native(pred, native_ref.shape),
                       fold_labels(native_ref.astype(np.int16)), tuple(rec["spacing_mm"]))
        offs = relabel_offsets(d)
        per_case.append({"case_id": cid, "n_levels": len(d.levels), "level_offsets": offs})
        offsets += offs
        n_levels += len(d.levels)
    hist: dict[int, int] = {}
    for o in offsets:
        hist[o] = hist.get(o, 0) + 1
    return {"summary": {"n_cases": len(test_ids), "n_levels": n_levels,
                        "offset_histogram": {str(k): v for k, v in sorted(hist.items())}},
            "per_case": per_case}


def incumbent_pointer(rounds_root: Path, k: int) -> Path:
    """The last PROMOTED model before round k — a refused round does not
    advance the incumbent, so the pointer follows promotion, not time."""
    for j in range(k - 1, -1, -1):
        rec_p = rounds_root / f"round{j}" / "round.json"
        if rec_p.is_file():
            rec = json.loads(rec_p.read_text())
            if j == 0 or rec.get("promotion", {}).get("promoted"):
                return rounds_root / f"round{j}" / "model.pt"
    raise FileNotFoundError("no promoted incumbent found — run round 0 first")


def run_round(
    k: int,
    *,
    repo: Path,
    data_root: Path,
    seed: int = 1337,
    iters: int = 400,
    rehearsal_frac: float = 0.25,
    device: str = "cuda",
    serve_poisoned: bool = True,
    tag: str = "",
    force_admit: bool = False,
    out_root: Path | None = None,
    control: str = "none",
    corrector: str = "oracle",
    rehearsal: str = "uniform",
) -> dict:
    """Execute round k and write outputs/rounds/round{k}{tag}/round.json.

    ``tag`` names an ABLATION ARM (e.g. "_norehearsal", "_counterfactual")
    written alongside the main round without touching the promotion chain —
    `incumbent_pointer` only reads untagged rounds. ``force_admit`` bypasses
    the batch screen (for the counterfactual: what would training on the
    refused batch have done?); the bypass is recorded in the round record.
    ``out_root`` relocates the whole round tree (default outputs/rounds) so
    seed replicates (W6) run side by side without colliding; the promotion
    chain is read from the SAME root, keeping each replicate self-contained.

    ``control`` selects the MORE-TRAINING null (W9): ``"with"`` trains it
    alongside an untagged round and hands it to the promotion gate; ``"only"``
    (with a tag) trains just the control arm, to backfill rounds already run.
    The control starts from the same incumbent with the same iterations and
    seed at equal N, but its cohort is a random draw from ALREADY-SEEN cases —
    no arriving batch — so it isolates the value of the new corrections from
    the value of extra optimisation steps.
    """
    if rehearsal not in ("uniform", "burden_weighted"):
        raise ValueError(f"rehearsal must be uniform|burden_weighted, got {rehearsal!r}")
    if corrector not in CORRECTORS:
        raise ValueError(f"corrector must be one of {sorted(CORRECTORS)}, got {corrector!r}")
    if control not in ("none", "with", "only"):
        raise ValueError(f"control must be none|with|only, got {control!r}")
    if control == "only" and not tag:
        raise ValueError("control='only' is an arm — pass a tag (e.g. _control)")
    t0 = time.time()
    cache_dir = repo / "data" / "cache"
    part = json.loads((repo / "manifests" / "partition.json").read_text())
    cases = json.loads((repo / "manifests" / "cases.json").read_text())["cases"]
    by_id = {c["case_id"]: c for c in cases}
    rounds_root = out_root if out_root is not None else repo / "outputs" / "rounds"
    out_dir = rounds_root / f"round{k}{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = part["initial_pool"]
    test_ids = part["sequestered_test"]

    def load_folded(cid: str, poisoned: bool):
        key = f"{cid}__poisoned" if poisoned else cid
        img, seg = load_case(cache_dir, key)
        return img, fold_labels(seg)

    record: dict = {"round": k, "seed": seed, "iters": iters}
    if tag:
        record["arm"] = tag.lstrip("_")
    if force_admit:
        record["force_admit"] = True

    if k == 0:
        train_ids = sorted(pool)
        record["training"] = {"ids": train_ids, "mix": {"n_new": 0, "n_rehearsal": 0,
                              "note": "round 0 = initial pool baseline"}}
        data = {cid: load_folded(cid, False) for cid in train_ids}
        model, losses = train(data, iters=iters, seed=seed, device=device)
    else:
        batch_ids = part["arrival_batches"][k - 1]
        poisoned_round = (k - 1) == part["poisoned_batch_index"] and serve_poisoned
        record["arrival"] = {"batch_index": k - 1, "n": len(batch_ids),
                             "served_poisoned": poisoned_round}
        inc_path = incumbent_pointer(rounds_root, k)
        record["incumbent"] = str(inc_path.relative_to(repo))
        init_state = torch.load(inc_path, map_location=device, weights_only=True)
        inc_round = json.loads((inc_path.parent / "round.json").read_text())
        incumbent_eval = inc_round["eval"]

        # 1. predict the arriving batch with the INCUMBENT, score the deltas
        import nibabel as nib

        inc = make_model(device)
        inc.load_state_dict(init_state)
        deltas = []
        corr_dir = out_dir / "corrected"
        corr_log = []
        for n_c, cid in enumerate(batch_ids):
            img, _ = load_case(cache_dir, cid)
            pred = predict(inc, img, device=device)
            rec_c = by_id[cid]
            if CORRECTORS[corrector] is not None:
                # The simulated reviewer sees the served reference, spends a
                # budget, and approves the rest. From here on the loop sees
                # ONLY (auto, corrected) — never the reference.
                _, served = load_folded(cid, poisoned_round)
                corrected, c_rec = simulate_correction(
                    pred.astype(np.int16), served.astype(np.int16), CACHE_MM,
                    rng=np.random.default_rng([seed, k, n_c]), **CORRECTORS[corrector])
                corr_dir.mkdir(parents=True, exist_ok=True)
                # what training consumes: the corrected mask, or (W17) the target in
                # which unreviewed voxels carry IGNORE. The screen below always
                # scores the corrected mask — a reviewer's edits are the delta.
                target = c_rec.pop("target", None)
                np.savez_compressed(corr_dir / f"{cid}.npz", labels=(
                    corrected if target is None else target).astype(np.uint8))
                corr_log.append({"case_id": cid, **c_rec})
                d = score_case(pred.astype(np.int16), corrected, CACHE_MM)
                deltas.append({
                    "case_id": cid, "apl_mm": d.apl_mm,
                    "surface_dice": d.surface_dice, "kind": d.kind,
                    "level_kinds": d.kinds(), "level_offsets": relabel_offsets(d),
                })
                continue
            if poisoned_round:
                ref_p = repo / "data" / "poisoned" / f"{cid}_seg.nii.gz"
            else:
                ref_p = data_root / rec_c["seg"]
            native_ref = np.asanyarray(nib.load(str(ref_p)).dataobj)
            native_pred = _to_native(pred, native_ref.shape)
            d = score_case(native_pred, fold_labels(native_ref.astype(np.int16)),
                           tuple(rec_c["spacing_mm"]))
            deltas.append({
                "case_id": cid, "apl_mm": d.apl_mm,
                "surface_dice": d.surface_dice, "kind": d.kind,
                "level_kinds": d.kinds(),
                "level_offsets": relabel_offsets(d),
            })
        record["deltas"] = deltas
        if corr_log:
            tot = sum(c["burden_total_mm"] for c in corr_log)
            record["corrector"] = {
                "name": corrector, **CORRECTORS[corrector],
                "grid_mm": CACHE_MM[0],
                "burden_spent_frac": round(sum(c["burden_spent_mm"] for c in corr_log) / tot, 4)
                if tot else 0.0,
                "levels_fixed": sum(len(c["fixed"]) for c in corr_log),
                "levels_left_small": sum(len(c["left_small"]) for c in corr_log),
                "levels_left_over_budget": sum(len(c["left_over_budget"]) for c in corr_log),
                "per_case": corr_log,
            }
        (out_dir / "deltas.json").write_text(json.dumps(deltas, indent=2) + "\n")

        # 2. CURATE — the refusals (W3)
        screen = screen_batch(deltas)
        if screen.batch_refused:
            # Who is shifted? Ask the sequestered references — measured only
            # when the signature fires, cached beside the incumbent it describes.
            arb_p = inc_path.parent / "arbiter_profile.json"
            if arb_p.is_file():
                arbiter = json.loads(arb_p.read_text())["summary"]
            else:
                prof = offset_profile_on_test(inc, test_ids, cache_dir, data_root,
                                              by_id, device=device)
                arb_p.write_text(json.dumps(prof, indent=2) + "\n")
                arbiter = prof["summary"]
            screen = screen_batch(deltas, arbiter=arbiter)
        if force_admit and screen.batch_refused:
            record["screen_overridden"] = screen.refusal_reason
            from .gates import BatchScreen

            screen = BatchScreen(sorted(batch_ids), [], None, screen.evidence)
        record["curation"] = {
            "policy": "delta-screen v3 (enumeration signature, sequestered arbiter) + sequestration",
            "admitted": screen.admitted,
            "refused": screen.refused,
            "refusal_reason": screen.refusal_reason,
            "evidence": screen.evidence,
        }
        seen = list(pool)
        for j in range(k - 1):
            prev_rec = rounds_root / f"round{j + 1}" / "round.json"
            if prev_rec.is_file():
                seen += json.loads(prev_rec.read_text())["curation"].get("admitted", [])

        if screen.batch_refused:
            # No training toward poisoned references. The incumbent stands;
            # the round records the refusal and carries the incumbent's eval.
            record["training"] = {"ids": [], "mix": {"n_new": 0, "n_rehearsal": 0,
                                  "note": "batch refused — incumbent stands"}}
            record["loss_final"] = None
            record["eval"] = incumbent_eval
            record["promotion"] = {
                "promoted": False,
                "reasons": [f"batch refused at curation: {screen.refusal_reason}"],
                "evidence": screen.evidence,
            }
            torch.save(init_state, out_dir / "model.pt")  # the standing incumbent
            record["mlflow_run_id"] = log_round(repo, record, None)
            record["dvc"] = dvc_track_model(repo, out_dir / "model.pt")
            record["seconds"] = round(time.time() - t0, 1)
            (out_dir / "round.json").write_text(json.dumps(record, indent=2) + "\n")
            return record

        # 3. retrain with rehearsal (candidate arm)
        weights = None
        if rehearsal == "burden_weighted" and rehearsal_frac > 0:
            # W8: the incumbent's burden on every already-seen case (native grid,
            # true references) — where the deployed model still costs a reviewer
            # the most is where rehearsal is aimed.
            weights = {}
            for cid in sorted(set(seen)):
                img_s, _ = load_case(cache_dir, cid)
                ref_s = np.asanyarray(nib.load(str(data_root / by_id[cid]["seg"])).dataobj)
                d_s = score_case(_to_native(predict(inc, img_s, device=device), ref_s.shape),
                                 fold_labels(ref_s.astype(np.int16)),
                                 tuple(by_id[cid]["spacing_mm"]))
                weights[cid] = d_s.apl_mm
            (out_dir / "rehearsal_weights.json").write_text(json.dumps(weights, indent=2) + "\n")
        train_ids, mix = make_training_list(screen.admitted, seen, weights=weights,
                                            rehearsal_frac=rehearsal_frac, seed=seed + k)
        assert_sequestration(train_ids, test_ids)

        def control_cohort() -> list[str]:
            import random

            past = sorted(set(seen) - set(screen.admitted))
            return sorted(random.Random(seed + 1000 + k).sample(
                past, min(len(train_ids), len(past))))

        if control == "only":
            train_ids = control_cohort()
            assert_sequestration(train_ids, test_ids)
            mix = {"n_new": 0, "n_rehearsal": len(train_ids),
                   "note": "more-training null: equal N drawn from already-seen cases only"}
            record["arm"] = "control"
        record["training"] = {"ids": train_ids, "mix": mix}
        def corrected_labels(cid: str):
            """What the simulated reviewer approved for this case, newest first —
            this round's, else the round it was admitted in. None = full labels."""
            if CORRECTORS[corrector] is None:
                return None
            for j in range(k, 0, -1):
                c_p = rounds_root / f"round{j}{tag if j == k else ''}" / "corrected" / f"{cid}.npz"
                if c_p.is_file():
                    return np.load(c_p)["labels"]
            return None

        data = {}
        for cid in train_ids:
            is_new_poisoned = (poisoned_round and control != "only"
                               and cid in set(screen.admitted))
            img_c, lab_c = load_folded(cid, is_new_poisoned)
            approved = corrected_labels(cid)
            data[cid] = (img_c, lab_c if approved is None else approved)
        model, losses = train(data, iters=iters, seed=seed + k, device=device,
                              init_state=init_state)

        control_ev = None
        if control == "with" and not tag:
            c_ids = control_cohort()
            assert_sequestration(c_ids, test_ids)
            c_model, _ = train({cid: load_folded(cid, False) for cid in c_ids},
                               iters=iters, seed=seed + k, device=device,
                               init_state=init_state)
            control_ev = evaluate_on_test(c_model, test_ids, cache_dir, data_root, by_id,
                                          device=device)
            c_dir = rounds_root / f"round{k}_control"
            c_dir.mkdir(parents=True, exist_ok=True)
            (c_dir / "eval_per_case.json").write_text(
                json.dumps(control_ev["per_case"], indent=2) + "\n")
            record["control"] = {"ids": c_ids, "eval": control_ev["aggregate"]}
            del c_model

    torch.save(model.state_dict(), out_dir / "model.pt")
    record["loss_final"] = round(float(np.mean(losses[-50:])), 4)

    ev = evaluate_on_test(model, test_ids, cache_dir, data_root, by_id, device=device)
    record["eval"] = ev["aggregate"]
    (out_dir / "eval_per_case.json").write_text(json.dumps(ev["per_case"], indent=2) + "\n")

    # 4. PROMOTE — only past the nulls (round 0 is the founding incumbent;
    #    ablation arms never enter the promotion chain)
    if k > 0 and not tag:
        forgetting = check_forgetting(incumbent_eval, ev["aggregate"])
        inc_per_case = json.loads((inc_path.parent / "eval_per_case.json").read_text())
        decision = decide_promotion_banded(
            incumbent_eval, ev["aggregate"], inc_per_case, ev["per_case"],
            forgetting=forgetting,
            control_per_case=control_ev["per_case"] if control_ev else None)
        record["promotion"] = {"promoted": decision.promoted,
                               "reasons": decision.reasons,
                               "evidence": decision.evidence}
    elif k > 0 and tag:
        record["forgetting_vs_incumbent"] = check_forgetting(
            incumbent_eval, ev["aggregate"]).__dict__
    save_losses(out_dir, losses)
    record["mlflow_run_id"] = log_round(repo, record, losses)
    record["dvc"] = dvc_track_model(repo, out_dir / "model.pt")
    record["seconds"] = round(time.time() - t0, 1)
    (out_dir / "round.json").write_text(json.dumps(record, indent=2) + "\n")
    return record
