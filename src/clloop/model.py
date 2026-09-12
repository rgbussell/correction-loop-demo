"""The demo's segmentation model: a small MONAI 3-D U-Net, trained on patches.

Everything here is deliberately modest — 27 classes (background + the VerSe
vertebral ids folded to C1..L6 + sacrum), 96³ patches at 3 mm, a few hundred
iterations per round. The loop's honesty lives in the ruler and the gates, not
in model size, and a small model keeps every round's retraining in the
minutes range on one consumer GPU.

Rehearsal is a DATA policy, not a model trick: `make_training_list` mixes the
new cohort with a declared fraction of the previously-seen pool, and the mix
is recorded in the round record. The no-rehearsal ablation (G4) is the same
function with rehearsal_frac=0.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from monai.losses import DiceCELoss
from monai.networks.nets import UNet

N_CLASSES = 27  # 0 bg, 1-7 C, 8-19 T, 20-25 L1-L6, 26 sacrum
PATCH = (96, 96, 96)


def fold_labels(seg: np.ndarray) -> np.ndarray:
    """VerSe ids -> the demo's 27-class map (cocygis -> bg, T13 -> T12)."""
    out = seg.copy()
    out[out == 27] = 0
    out[out == 28] = 19
    out[out > 28] = 0
    return out


def make_model(device: str = "cuda") -> torch.nn.Module:
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=N_CLASSES,
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
        num_res_units=2,
    ).to(device)


def _pad_to(arr: np.ndarray, shape: tuple[int, int, int], value=0.0) -> np.ndarray:
    pads = [(0, max(0, s - a)) for a, s in zip(arr.shape, shape)]
    return np.pad(arr, pads, constant_values=value)


def _random_fg_patch(
    img: np.ndarray, seg: np.ndarray, rng: random.Random
) -> tuple[np.ndarray, np.ndarray]:
    """A PATCH-sized crop centred near a random foreground voxel (or anywhere
    when the case has no foreground)."""
    img = _pad_to(img, PATCH)
    seg = _pad_to(seg, PATCH)
    fg = np.argwhere(seg > 0)
    if len(fg):
        c = fg[rng.randrange(len(fg))]
    else:
        c = np.array([rng.randrange(s) for s in seg.shape])
    starts = [
        int(np.clip(ci - p // 2, 0, s - p))
        for ci, p, s in zip(c, PATCH, seg.shape)
    ]
    sl = tuple(slice(s, s + p) for s, p in zip(starts, PATCH))
    return img[sl], seg[sl]


def make_training_list(
    new_ids: list[str],
    seen_ids: list[str],
    *,
    rehearsal_frac: float,
    seed: int,
) -> tuple[list[str], dict]:
    """The round's training cases: the new cohort + a rehearsal draw of the past.

    n_rehearsal = rehearsal_frac / (1 - rehearsal_frac) * len(new) — i.e. the
    declared fraction OF THE FINAL MIX is rehearsal, capped by what exists.
    Returns (ids, mix_record); the record goes into the round file verbatim.
    """
    rng = random.Random(seed)
    pool = [c for c in seen_ids if c not in set(new_ids)]
    if rehearsal_frac <= 0 or not pool:
        n_reh = 0
    else:
        n_reh = min(len(pool), round(rehearsal_frac / (1 - rehearsal_frac) * len(new_ids)))
    rehearsal = sorted(rng.sample(pool, n_reh)) if n_reh else []
    ids = sorted(new_ids) + rehearsal
    return ids, {
        "n_new": len(new_ids),
        "n_rehearsal": len(rehearsal),
        "rehearsal_frac_requested": rehearsal_frac,
        "rehearsal_frac_actual": len(rehearsal) / len(ids) if ids else 0.0,
        "rehearsal_ids": rehearsal,
    }


def train(
    cases: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    iters: int = 400,
    batch: int = 2,
    lr: float = 1e-3,
    seed: int = 1337,
    device: str = "cuda",
    init_state: dict | None = None,
    log_every: int = 50,
) -> tuple[torch.nn.Module, list[float]]:
    """Patch-based training. `cases` maps id -> (image, folded labels)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)
    model = make_model(device)
    if init_state is not None:
        model.load_state_dict(init_state)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
    ids = sorted(cases)
    losses: list[float] = []
    model.train()
    for it in range(iters):
        xs, ys = [], []
        for _ in range(batch):
            cid = ids[rng.randrange(len(ids))]
            img, seg = cases[cid]
            xp, yp = _random_fg_patch(img, seg, rng)
            xs.append(xp[None])
            ys.append(yp[None])
        x = torch.from_numpy(np.stack(xs)).float().to(device)
        y = torch.from_numpy(np.stack(ys)).long().to(device)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        sched.step()
        losses.append(float(loss.item()))
        if log_every and (it + 1) % log_every == 0:
            print(f"    iter {it + 1}/{iters}  loss {np.mean(losses[-log_every:]):.4f}",
                  flush=True)
    return model, losses


@torch.no_grad()
def predict(
    model: torch.nn.Module, img: np.ndarray, *, device: str = "cuda", overlap: int = 16
) -> np.ndarray:
    """Sliding-window argmax prediction on the cached grid."""
    model.eval()
    shape = img.shape
    padded = _pad_to(img, tuple(max(s, p) for s, p in zip(shape, PATCH)))
    out_logits = np.zeros((N_CLASSES, *padded.shape), dtype=np.float32)
    counts = np.zeros(padded.shape, dtype=np.float32)
    steps = [
        list(range(0, max(1, s - p) + 1, p - overlap)) + [max(0, s - p)]
        for s, p in zip(padded.shape, PATCH)
    ]
    seen = set()
    for i in steps[0]:
        for j in steps[1]:
            for k in steps[2]:
                if (i, j, k) in seen:
                    continue
                seen.add((i, j, k))
                sl = (slice(i, i + PATCH[0]), slice(j, j + PATCH[1]), slice(k, k + PATCH[2]))
                x = torch.from_numpy(padded[sl][None, None]).float().to(device)
                logits = torch.softmax(model(x), dim=1)[0].cpu().numpy()
                out_logits[(slice(None), *sl)] += logits
                counts[sl] += 1.0
    out_logits /= np.maximum(counts, 1.0)
    pred = out_logits.argmax(0).astype(np.uint8)
    return pred[: shape[0], : shape[1], : shape[2]]
