"""The round engine: one deployment round, end to end, from one seed.

A round is the unit of the whole demo:

  predict (auto) -> score correction deltas (frozen ruler, NATIVE grid)
  -> curate the retrain cohort -> retrain with rehearsal -> evaluate on the
  sequestered test set -> record everything in outputs/rounds/round_k/round.json

Round 0 is the special case: no arrivals yet, just the initial-pool baseline
and its evaluation — the incumbent every later round is measured against.

W2 NOTE — curation here is a deliberate stub: `curate_accept_all` admits the
whole batch and says so in the round record. The three refusals and the
two-null promotion gate are W3; the engine records enough per-round evidence
(deltas, mix, eval) that W3 slots in as a policy swap, not a rewrite.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from .delta import score_case
from .model import fold_labels, make_model, make_training_list, predict, train
from .preprocess import load_case

REGIONS = {"cervical": range(1, 8), "thoracic": range(8, 20), "lumbar": range(20, 26),
           "sacrum": (26,)}


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


def curate_accept_all(batch_ids: list[str]) -> dict:
    """W2 stub. Admits everything and is honest about it in the record."""
    return {
        "policy": "accept_all (W2 stub — refusals and nulls land in W3)",
        "admitted": sorted(batch_ids),
        "refused": [],
    }


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
) -> dict:
    """Execute round k and write outputs/rounds/round{k}/round.json."""
    t0 = time.time()
    cache_dir = repo / "data" / "cache"
    part = json.loads((repo / "manifests" / "partition.json").read_text())
    cases = json.loads((repo / "manifests" / "cases.json").read_text())["cases"]
    by_id = {c["case_id"]: c for c in cases}
    out_dir = repo / "outputs" / "rounds" / f"round{k}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = part["initial_pool"]
    test_ids = part["sequestered_test"]

    def load_folded(cid: str, poisoned: bool):
        key = f"{cid}__poisoned" if poisoned else cid
        img, seg = load_case(cache_dir, key)
        return img, fold_labels(seg)

    record: dict = {"round": k, "seed": seed, "iters": iters}

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
        prev = repo / "outputs" / "rounds" / f"round{k - 1}"
        init_state = torch.load(prev / "model.pt", map_location=device,
                                weights_only=True)

        # 1. predict the arriving batch with the INCUMBENT, score the deltas
        import nibabel as nib

        inc = make_model(device)
        inc.load_state_dict(init_state)
        deltas = []
        for cid in batch_ids:
            img, _ = load_case(cache_dir, cid)
            pred = predict(inc, img, device=device)
            rec_c = by_id[cid]
            if poisoned_round:
                ref_p = repo / "data" / "poisoned" / f"{cid}_seg.nii.gz"
            else:
                ref_p = data_root / rec_c["seg"]
            native_ref = np.asanyarray(nib.load(str(ref_p)).dataobj)
            native_pred = _to_native(pred, native_ref.shape)
            d = score_case(native_pred, fold_labels(native_ref.astype(np.int16)),
                           tuple(rec_c["spacing_mm"]))
            deltas.append({"case_id": cid, "apl_mm": d.apl_mm,
                           "surface_dice": d.surface_dice, "kind": d.kind})
        record["deltas"] = deltas
        (out_dir / "deltas.json").write_text(json.dumps(deltas, indent=2) + "\n")

        # 2. curate (W2 stub) + 3. retrain with rehearsal
        curation = curate_accept_all(batch_ids)
        record["curation"] = curation
        seen = list(pool)
        for j in range(k - 1):
            seen += part["arrival_batches"][j]
        train_ids, mix = make_training_list(curation["admitted"], seen,
                                            rehearsal_frac=rehearsal_frac, seed=seed + k)
        record["training"] = {"ids": train_ids, "mix": mix}
        data = {}
        for cid in train_ids:
            is_new_poisoned = poisoned_round and cid in set(curation["admitted"])
            data[cid] = load_folded(cid, is_new_poisoned)
        model, losses = train(data, iters=iters, seed=seed + k, device=device,
                              init_state=init_state)

    torch.save(model.state_dict(), out_dir / "model.pt")
    record["loss_final"] = round(float(np.mean(losses[-50:])), 4)

    ev = evaluate_on_test(model, test_ids, cache_dir, data_root, by_id, device=device)
    record["eval"] = ev["aggregate"]
    (out_dir / "eval_per_case.json").write_text(json.dumps(ev["per_case"], indent=2) + "\n")
    record["seconds"] = round(time.time() - t0, 1)
    (out_dir / "round.json").write_text(json.dumps(record, indent=2) + "\n")
    return record
