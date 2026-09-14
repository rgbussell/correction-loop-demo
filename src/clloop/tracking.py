"""Experiment tracking: every training run logs to MLflow, every model to DVC.

Two disciplines, deliberately separate:

- **MLflow** answers *how did this training go?* — one run per round/arm, the
  full per-iteration loss curve as a metric series, the training mix and seed
  as params, the sequestered-set evaluation and the promotion decision as
  metrics/tags. Local file store (``mlruns/``, gitignored); inspect with
  ``mlflow ui --backend-store-uri mlruns``.
- **DVC** answers *which bytes were this model?* — each round's ``model.pt``
  is dvc-added after its round completes and pushed to the configured remote.
  The ``.dvc`` pointer files are committed; the remote lives in
  ``.dvc/config.local`` (untracked), so a public clone carries the full
  version history without exposing anyone's storage. Configure your own
  remote with ``dvc remote add -d <name> <url> --local``.

The loss curve is additionally written to the round directory as
``losses.json`` so the repo's own artifacts are complete without either tool.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

EXPERIMENT = "correction-loop-demo"


def _dvc_bin() -> str | None:
    """Absolute dvc path — the interpreter's own bin dir first (systemd units
    have a bare PATH; a bare "dvc" was a live crash), then PATH."""
    cand = Path(sys.executable).parent / "dvc"
    if cand.is_file():
        return str(cand)
    return shutil.which("dvc")


def _mlflow(repo: Path):
    import mlflow

    mlflow.set_tracking_uri(f"file://{repo / 'mlruns'}")
    mlflow.set_experiment(EXPERIMENT)
    return mlflow


def log_round(
    repo: Path,
    record: dict,
    losses: list[float] | None,
    *,
    curve_stride: int = 10,
) -> str | None:
    """Log one completed round/arm to MLflow. Returns the run id.

    ``losses`` may be None for a refused round (nothing was trained — the
    refusal itself is logged, with the reason as a tag)."""
    mlflow = _mlflow(repo)
    k = record["round"]
    arm = record.get("arm", "")
    name = f"round{k}" + (f"-{arm}" if arm else "")
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params({
            "round": k,
            "arm": arm or "main",
            "seed": record.get("seed"),
            "iters": record.get("iters"),
            "n_new": record.get("training", {}).get("mix", {}).get("n_new"),
            "n_rehearsal": record.get("training", {}).get("mix", {}).get("n_rehearsal"),
            "force_admit": record.get("force_admit", False),
        })
        if losses:
            for i in range(0, len(losses), curve_stride):
                mlflow.log_metric("train_loss", float(losses[i]), step=i)
            mlflow.log_metric("train_loss", float(losses[-1]), step=len(losses) - 1)
        ev = record.get("eval", {})
        for key in ("apl_mm_median", "apl_mm_total", "surface_dice_median"):
            if ev.get(key) is not None:
                mlflow.log_metric(f"test_{key}", float(ev[key]))
        for reg in ("cervical", "thoracic", "lumbar", "sacrum"):
            v = ev.get(f"dice_{reg}_median")
            if v is not None:
                mlflow.log_metric(f"test_dice_{reg}", float(v))
        promo = record.get("promotion")
        if promo is not None:
            mlflow.set_tag("promoted", str(promo.get("promoted")))
            if promo.get("reasons"):
                mlflow.set_tag("refusal_reasons", "; ".join(promo["reasons"])[:500])
        cur = record.get("curation", {})
        if cur.get("refusal_reason"):
            mlflow.set_tag("batch_refused", cur["refusal_reason"][:500])
        return run.info.run_id


def save_losses(out_dir: Path, losses: list[float]) -> None:
    (out_dir / "losses.json").write_text(json.dumps([round(x, 5) for x in losses]) + "\n")


def dvc_track_model(repo: Path, model_path: Path, *, push: bool = True) -> dict:
    """`dvc add` the model and (optionally) push it to the configured remote.

    Returns a status dict; a missing remote downgrades push to a recorded
    no-op rather than an error, so a public clone without credentials still
    runs the loop end to end."""
    model_path = model_path.resolve()
    repo = repo.resolve()
    status: dict = {"path": str(model_path.relative_to(repo))}
    dvc = _dvc_bin()
    if dvc is None:
        status["added"] = False
        status["add_error"] = "dvc binary not found — model NOT version-tracked"
        return status
    add = subprocess.run(
        [dvc, "add", str(model_path)], cwd=repo, capture_output=True, text=True
    )
    status["added"] = add.returncode == 0
    if add.returncode != 0:
        status["add_error"] = add.stderr.strip()[-300:]
        return status
    if push:
        p = subprocess.run(
            [dvc, "push", str(model_path) + ".dvc"],
            cwd=repo, capture_output=True, text=True,
        )
        status["pushed"] = p.returncode == 0
        if p.returncode != 0:
            status["push_error"] = p.stderr.strip()[-300:]
    return status
