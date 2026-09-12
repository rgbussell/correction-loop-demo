"""Preprocessing cache: every case resampled once to a small, fixed grid.

The demo's models train at 3 mm isotropic — deliberately coarse. The point of
the demo is loop mechanics measured honestly, not segmentation SOTA, and a
coarse grid makes every round's training a matter of minutes on one consumer
GPU. Each case is resampled once (image: linear, HU-clipped and normalized;
labels: nearest-neighbour) and cached as a compressed npz keyed by case id.

The DELTA RULER never sees these: correction deltas are scored on the native
grid (W1). The cache is a training convenience only, and predictions are
resampled back to the native grid before any scoring.
"""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

TARGET_MM = 3.0
HU_CLIP = (-1024.0, 1500.0)


def resample_case(
    img_path: Path, seg_path: Path, target_mm: float = TARGET_MM
) -> tuple[np.ndarray, np.ndarray, tuple, tuple]:
    """(image_norm, labels, native_shape, native_spacing) on the target grid."""
    img_nii = nib.load(str(img_path))
    seg_nii = nib.load(str(seg_path))
    spacing = np.array(img_nii.header.get_zooms()[:3], dtype=float)
    img = np.asanyarray(img_nii.dataobj).astype(np.float32)
    seg = np.asanyarray(seg_nii.dataobj).astype(np.int16)
    if seg.shape != img.shape:
        raise ValueError(f"{img_path.name}: image {img.shape} vs seg {seg.shape}")

    zoom = spacing / target_mm
    img_r = ndimage.zoom(img, zoom, order=1)
    seg_r = ndimage.zoom(seg, zoom, order=0)

    img_r = np.clip(img_r, *HU_CLIP)
    img_r = (img_r - HU_CLIP[0]) / (HU_CLIP[1] - HU_CLIP[0])  # [0, 1]
    return img_r, seg_r.astype(np.uint8), img.shape, tuple(spacing)


def build_cache(
    data_root: Path,
    cases: list[dict],
    cache_dir: Path,
    *,
    poisoned_dir: Path | None = None,
    poisoned_ids: set[str] | None = None,
) -> dict:
    """Cache every case. Poisoned cases get a SECOND entry (suffix ``__poisoned``)
    whose labels come from the poisoned copy — the engine chooses which to serve
    when that batch arrives; the clean entry remains for scoring."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for rec in cases:
        cid = rec["case_id"]
        out = cache_dir / f"{cid}.npz"
        if not out.exists():
            img, seg, shape, spacing = resample_case(
                data_root / rec["image"], data_root / rec["seg"]
            )
            np.savez_compressed(
                out, image=img, labels=seg,
                native_shape=np.array(shape), native_spacing=np.array(spacing),
            )
        manifest[cid] = out.name
        if poisoned_ids and cid in poisoned_ids and poisoned_dir is not None:
            outp = cache_dir / f"{cid}__poisoned.npz"
            if not outp.exists():
                img, seg, shape, spacing = resample_case(
                    data_root / rec["image"], poisoned_dir / f"{cid}_seg.nii.gz"
                )
                np.savez_compressed(
                    outp, image=img, labels=seg,
                    native_shape=np.array(shape), native_spacing=np.array(spacing),
                )
            manifest[f"{cid}__poisoned"] = outp.name
    (cache_dir / "cache_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_case(cache_dir: Path, cid: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(cache_dir / f"{cid}.npz")
    return z["image"].astype(np.float32), z["labels"].astype(np.int64)
