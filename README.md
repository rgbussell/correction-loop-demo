# correction-loop-demo

**A continual-learning loop that learns from human-correction deltas — and can
be seen refusing a bad round.** Open spine-CT demonstration (VerSe 2020,
CC BY-SA 4.0). Code MIT.

> Status: **W0–W1 complete** — partition + audited poison + the frozen delta ruler,
> validated 13/13 on the real poisoned pairs. Narrated build report:
> [`docs/build-report.html`](docs/build-report.html)

## The idea

Production segmentation models are corrected by humans every day, and the
corrections are usually thrown away. This repo builds the loop that keeps
them: each round a model meets a batch of new cases, its outputs are
"corrected" (reference masks stand in for the human), the **correction
delta** becomes the training signal, and a retrained candidate is promoted
only if it beats **two null controls** — a matched random cohort at equal N,
and doing nothing. One arrival batch is **deliberately poisoned** with a
wrong-level enumeration error; the loop must refuse it, visibly.

## Quickstart

```bash
scripts/download_verse.sh              # fetch VerSe 2020 (or point at a local copy)
pip install -e .[dev]
python scripts/build_partition.py      # scan + design the stream (seed 1337)
python scripts/build_poisoned_batch.py # materialize the audited poison
pytest -q                              # 5 partition tests
python tools/report/build_report.py    # regenerate docs/build-report.html
```

Data never enters git — only manifests (case inventory, partition design,
poison audit) are committed.

## Attribution

Sekuboyina et al., *VerSe: A Vertebrae Labelling and Segmentation Benchmark
for Multi-detector CT Images*, Medical Image Analysis 2021. Dataset
CC BY-SA 4.0, obtained from the authors' published mirrors
(https://github.com/anjany/verse). This demo shows a loop *pattern* on open
data; its numbers make no claim about any clinical system.
