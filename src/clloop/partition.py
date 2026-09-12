"""Case scan + stream partition for the correction-loop demo.

The demo simulates a deployment: a model trained on an initial pool receives
ARRIVAL BATCHES of new cases over successive rounds, and a SEQUESTERED test set
measures every round's candidate. The partition is therefore not a random
split — it is a declared, seeded design with three properties:

1. **Sequestration.** The test cases are chosen once, recorded in the manifest,
   and never eligible for curation. The refusal that enforces this lives in the
   loop engine (W3); the manifest is the ground truth it checks against.
2. **A declared distribution shift.** Arrival batches are ordered so the
   cervical-containing fraction RISES round over round while the initial pool
   is thoraco-lumbar-dominant. The loop then has something genuine to learn
   (new anatomy arriving) and the forgetting gate something genuine to guard
   (the anatomy the initial model knew).
3. **A poisoned batch.** One declared batch carries an enumeration off-by-one
   applied to its reference masks (label v -> v+1 along the vertebral chain).
   A loop that cannot be seen refusing a bad batch is theater; this is the bad
   batch.

The scan reads each reference labelmap once (np.unique) plus the image header —
a couple of minutes for the full dataset — and the partition is reproducible
from (dataset, seed) alone.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

# VerSe enumeration: C1-C7 = 1-7, T1-T12 = 8-19, L1-L5 = 20-24, L6 = 25,
# sacrum = 26, cocygis = 27, T13 = 28.
CERVICAL = set(range(1, 8))
THORACIC = set(range(8, 20)) | {28}
LUMBAR = set(range(20, 26))

# Deterministic series preference when a case has several reconstructions.
# "" = the bare-name series (e.g. GL247.nii.gz), present for ~1/3 of cases.
SERIES_PREFERENCE = ("_CT-iso", "", "_CT-ax", "_CT_ax", "_CT-sag", "_CT_sag")


@dataclass(frozen=True)
class CaseRecord:
    """One VerSe case as the partition sees it: identity + coverage, no voxels."""

    case_id: str
    image: str  # path relative to the dataset root
    seg: str
    centroids: str
    levels: tuple[int, ...]  # vertebral labels present (from the reference labelmap)
    fov_group: str  # cervical-containing | thoraco-lumbar | other
    n_levels: int
    shape: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]


def fov_group(levels: set[int]) -> str:
    """Coarse field-of-view group. Three groups, deliberately: they are the
    shift axis AND a chart's categorical series, and three is the all-pairs-safe
    palette budget."""
    if levels & CERVICAL:
        return "cervical-containing"
    if levels & LUMBAR:
        return "thoraco-lumbar"
    return "other"


def _pick_series(case_dir: Path) -> tuple[Path, Path, Path | None] | None:
    """(image, seg, centroids|None) for the preferred series.

    VerSe file naming is heterogeneous (verse*/GL* ids, CT-iso/CT-ax/CT_ax
    tags, and bare-name series) and the centroid sidecar exists for only a
    minority of cases in some mirrors — so the centroid file is OPTIONAL and
    level coverage is derived from the segmentation volume instead.
    """
    for tag in SERIES_PREFERENCE:
        img = case_dir / f"{case_dir.name}{tag}.nii.gz"
        seg = case_dir / f"{case_dir.name}{tag}_seg.nii.gz"
        if img.is_file() and seg.is_file():
            ctds = sorted(case_dir.glob(f"{case_dir.name}{tag}*ctd.json"))
            return img, seg, (ctds[0] if ctds else None)
    return None


def scan_dataset(root: Path) -> list[CaseRecord]:
    """One record per case directory. Levels come from the segmentation volume
    (np.unique on the labelmap) — the reference the loop will actually train
    on — not from the optional centroid sidecar."""
    import nibabel as nib
    import numpy as np

    records: list[CaseRecord] = []
    case_dirs = sorted(
        d for d in root.rglob("*") if d.is_dir() and _pick_series(d) is not None
    )
    for case_dir in case_dirs:
        img_p, seg_p, ctd_p = _pick_series(case_dir)
        seg_arr = np.asanyarray(nib.load(str(seg_p)).dataobj)
        levels = tuple(int(v) for v in np.unique(seg_arr) if 0 < int(v) <= 28)
        if not levels:
            continue
        hdr = nib.load(str(img_p))
        shape = tuple(int(s) for s in hdr.shape[:3])
        spacing = tuple(round(float(z), 4) for z in hdr.header.get_zooms()[:3])
        records.append(
            CaseRecord(
                case_id=case_dir.name,
                image=str(img_p.relative_to(root)),
                seg=str(seg_p.relative_to(root)),
                centroids=str(ctd_p.relative_to(root)) if ctd_p else "",
                levels=levels,
                fov_group=fov_group(set(levels)),
                n_levels=len(levels),
                shape=shape,  # type: ignore[arg-type]
                spacing_mm=spacing,  # type: ignore[arg-type]
            )
        )
    return records


@dataclass
class Partition:
    """The declared stream design. All case lists are case_ids."""

    seed: int
    sequestered_test: list[str]
    initial_pool: list[str]
    arrival_batches: list[list[str]] = field(default_factory=list)
    poisoned_batch_index: int = 2  # 0-based; batch 3 of 4 in the default design
    poison_transform: str = "enumeration_shift_plus1"
    design_notes: dict = field(default_factory=dict)

    def all_streamed(self) -> list[str]:
        return [c for b in self.arrival_batches for c in b]

    def assert_disjoint(self) -> None:
        groups = [self.sequestered_test, self.initial_pool, *self.arrival_batches]
        seen: set[str] = set()
        for g in groups:
            overlap = seen & set(g)
            if overlap:
                raise ValueError(f"partition overlap: {sorted(overlap)[:5]}")
            seen |= set(g)

    def to_json(self) -> dict:
        d = asdict(self)
        d["n"] = {
            "sequestered_test": len(self.sequestered_test),
            "initial_pool": len(self.initial_pool),
            "arrival_batches": [len(b) for b in self.arrival_batches],
        }
        return d


def design_partition(
    records: list[CaseRecord],
    *,
    seed: int = 1337,
    n_test: int = 24,
    n_pool: int = 30,
    n_batches: int = 4,
) -> Partition:
    """Seeded stream design with the declared cervical-rising shift.

    - TEST: stratified by fov_group (proportional), so every round's eval covers
      both the incumbent anatomy and the arriving anatomy.
    - POOL: drawn thoraco-lumbar-heavy (the "deployed" distribution).
    - BATCHES: remaining cases ordered so the cervical-containing fraction rises
      monotonically across batches — cervical cases are spread by quantile of a
      seeded score biased against early batches.
    """
    rng = random.Random(seed)
    by_id = {r.case_id: r for r in records}
    ids = sorted(by_id)

    # --- stratified sequestered test -------------------------------------
    test: list[str] = []
    by_group: dict[str, list[str]] = {}
    for cid in ids:
        by_group.setdefault(by_id[cid].fov_group, []).append(cid)
    for group, members in sorted(by_group.items()):
        take = max(1, round(n_test * len(members) / len(ids)))
        picked = rng.sample(members, min(take, len(members)))
        test.extend(picked)
    test = sorted(test[:n_test])

    remaining = [c for c in ids if c not in set(test)]

    # --- thoraco-lumbar-heavy initial pool -------------------------------
    tl = [c for c in remaining if by_id[c].fov_group == "thoraco-lumbar"]
    non_tl = [c for c in remaining if by_id[c].fov_group != "thoraco-lumbar"]
    n_tl = min(len(tl), round(n_pool * 0.85))
    pool = rng.sample(tl, n_tl) + rng.sample(non_tl, min(n_pool - n_tl, len(non_tl)))
    pool = sorted(pool)

    # --- arrival batches with a rising cervical fraction ------------------
    stream = [c for c in remaining if c not in set(pool)]
    cerv = sorted(c for c in stream if by_id[c].fov_group == "cervical-containing")
    rest = sorted(c for c in stream if by_id[c].fov_group != "cervical-containing")
    rng.shuffle(cerv)
    rng.shuffle(rest)
    batches: list[list[str]] = [[] for _ in range(n_batches)]
    # Triangular weights 1..n_batches put few cervical cases early, many late.
    weights = list(range(1, n_batches + 1))
    total_w = sum(weights)
    idx = 0
    for b, w in enumerate(weights):
        take = round(len(cerv) * w / total_w)
        batches[b].extend(cerv[idx : idx + take])
        idx += take
    batches[-1].extend(cerv[idx:])  # remainder to the last batch
    # Fill to equal size with the rest, round-robin from batch 0.
    per = len(stream) // n_batches
    ri = 0
    for b in range(n_batches):
        while len(batches[b]) < per and ri < len(rest):
            batches[b].append(rest[ri])
            ri += 1
    for j, cid in enumerate(rest[ri:]):  # leftovers
        batches[j % n_batches].append(cid)
    batches = [sorted(b) for b in batches]

    part = Partition(
        seed=seed,
        sequestered_test=test,
        initial_pool=pool,
        arrival_batches=batches,
        design_notes={
            "shift": "cervical-containing fraction rises across batches (triangular weights)",
            "pool_bias": "initial pool drawn ~85% thoraco-lumbar (the deployed distribution)",
            "test_stratification": "proportional by fov_group",
            "poison": "batch index 2 references get enumeration label v->v+1 (built copies; "
            "originals untouched)",
        },
    )
    part.assert_disjoint()
    return part
