# correction-loop-demo

**A continual-learning loop that learns from human-correction deltas — and can
be seen refusing a bad round.**

Open spine-CT demonstration: VerSe 2020 data (CC BY-SA 4.0), MIT code, one
consumer GPU. Five deployment rounds, one deliberately poisoned batch, three
kinds of verdict — *promoted*, *refused at the gate*, *refused at the door*.

📊 **The full narrated build report, with every figure and number:**
[`docs/build-report.html`](docs/build-report.html)

📄 **What held up and what did not, for a reader who will not run the code:** [`RESULTS.md`](RESULTS.md)

To reuse the controls rather than read about them: [`docs/method-transfer-brief.md`](docs/method-transfer-brief.md)
maps each one to the experiment you would run on a new training recipe, with a kill condition per experiment.

---

## The idea

Production segmentation models are corrected by humans every day, and those
corrections are usually thrown away. This repo builds the loop that keeps
them. Each round:

1. a batch of new cases **arrives**; the incumbent model predicts them;
2. the **correction delta** between prediction and corrected reference is
   scored by a frozen ruler — *added path length* (the contour a human must
   redraw), *surface Dice* (deliberately label-blind), and a
   boundary/relabel/missing/spurious **taxonomy**;
3. the batch is **curated** past three refusals — sequestration (a held-out
   case in a training cohort *raises*), an enumeration-signature screen
   (systematically mis-labeled references are refused whole), and a
   forgetting gate;
4. the model **retrains** with a declared ~25% rehearsal mix;
5. the candidate is **promoted only if it beats two trained nulls** on a
   sequestered test set — the do-nothing incumbent and a matched
   more-training control — judged on a paired bootstrap interval rather than
   a fixed bar. A do-nothing candidate must fail; that degeneracy check is a
   test.

## What actually happened (the unscripted part)

The first live poisoned round is the best exhibit in the repo. The admission
screen's original rule **missed** the poison — the weak model's boundary
noise diluted the case-level verdicts it depended on — and the batch was
admitted on evidence of 37/39 relabel levels agreeing on offset +1. **The
promotion gate caught it**: training on the poison made the candidate 40%
*worse* against the do-nothing null, it was refused, and the incumbent
stood. The screen was then revised to read the level-wise fingerprint
(pinned by a regression test built from the real round record), and in the
definitive run it refuses the poison *before any training*.

That is the whole argument for layered nulls, played out in the artifacts:
the elegant detector failed, the outcome-based null did not, and the fix is
a regression test.

| round | verdict | evidence |
|---|---|---|
| 0 | baseline | thor/lumbar Dice 0.69/0.68 · cervical 0 (unseen) |
| 1 | **promoted** | +10.0% APL vs do-nothing · 26% rehearsal |
| 2 | **promoted** | +9.0% |
| 3 (poisoned) | **refused, no training** | 39 relabel levels · 95% offset consensus on +1 |
| — counterfactual | (forced) | +40.2% burden — the damage the refusal prevented |
| 4 | **promoted** | cervical 0 → 0.54 · thoraco-lumbar held · +8.5% |

Honest results are kept in: the no-rehearsal ablation at round 4 shows **no
regional collapse** at this gentle scale — rehearsal's measured value here is
a 5.5% APL margin, and the report says exactly that instead of dramatizing.
The gains above are point estimates. Re-judged later with intervals, the
round-4 burden gain does not exclude zero in any seed; those promotions stand
on a separate unserved-region clause, and `RESULTS.md` §3 says so.

## The finding that matters most

Every number above learns from the **full reference mask** — an oracle
standing in for a human correction. Replace it with a reviewer who has a
budget, fixes what is badly wrong, and approves the rest, and **the loop
promotes nothing at all**, in three seeds of three. Burden-ranked triage
fixed 8% of cervical levels against 66% of lumbar, and the model's empty
cervical output — approved as background — taught the blind spot back to it.
Masking the unreviewed voxels out of the loss is a partial repair, not a fix.

*A partial correction is not a label.* What the reviewer did not touch has to
reach training as **unknown**, not as **approved**. That is the transferable
result here, and it is a negative one: see `RESULTS.md` §4 and the report's
Step 9.

## Quickstart

```bash
scripts/download_verse.sh                 # VerSe 2020 from the authors' mirrors
pip install -e .[train,dev]
python scripts/build_partition.py         # seeded stream design (test/pool/batches)
python scripts/build_poisoned_batch.py    # the audited enumeration poison
pytest -q                                 # 74 pass + 1 xfail: a pinned known defect
scripts/w4_chain2.sh                      # the full 5-round run (one command)
python tools/report/build_report.py       # regenerate docs/build-report.html
mlflow ui --backend-store-uri mlruns      # every training curve
```

Data never enters git. Models are DVC-tracked (pointers committed; configure
your own remote with `dvc remote add -d <name> <url> --local`). Every
round/arm logs one MLflow run with its full training curve.

## Design commitments

- **The ruler is frozen.** The correction-delta scorer passed a null suite
  (identity scores exactly zero; a pure relabel is seen despite perfect
  shapes; APL is monotone in damage extent) before scoring anything, and its
  sha256 is recorded. Its one optimization was re-validated to byte-identical
  verdicts on all 13 poisoned pairs, then re-frozen — declared, never silent.
- **Refusals raise; nothing filters silently.** Every gate has a
  planted-defect test proving it can be seen firing.
- **Determinism.** Rounds 0–2 reproduced byte-identical training losses
  across two independent chain executions.
- **Small on purpose.** A modest 3-D U-Net at 3 mm; the demo measures loop
  mechanics, not segmentation SOTA, and says so.

## Attribution & limits

Data: Sekuboyina et al., *VerSe: A Vertebrae Labelling and Segmentation
Benchmark for Multi-detector CT Images*, Medical Image Analysis 2021 —
CC BY-SA 4.0, obtained from the authors' published mirrors
(<https://github.com/anjany/verse>). Methods draw on the public literature:
added path length (Vaassen et al. 2020), surface Dice (Nikolov et al. 2018),
rehearsal for continual learning, and the MONAI stack.

This demo shows a loop *pattern* on open data. Its numbers make no claim
about any clinical system, and nothing here is medical software.
