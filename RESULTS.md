# Results — what this loop showed, and what it did not

Written for a reader who will not run the code. Every number traces to a file in
`manifests/` or `outputs/`; the file is named beside it. Where an earlier claim
of this project turned out to be wrong, it is corrected here in the open rather
than quietly replaced.

**Setting.** VerSe 2020 spine CT. A small 3-D U-Net at 3 mm. Thirty cases to
start, four arriving batches of 13–14 that shift toward cervical anatomy, one of
them deliberately poisoned (every vertebra label shifted by one). 24 sequestered
test cases, 9 of them with cervical anatomy. Three full replicates (training
seeds 1337, 2027, 3117). The measure is **correction burden**: millimetres of
contour a person would have to redraw (added path length at 2 mm tolerance),
totalled over the test set. None of this is a clinical result.

---

## 1. What held up

**No poisoned model was ever deployed — but not always for the reason intended.**
With full references the poisoned batch was refused *at the door*, before training,
in 3 of 3 seeds (`manifests/w15_screen_v3.json`); force-admitting it cost **+40.2%**
burden (`outputs/rounds/round3_counterfactual/`). Under a budgeted reviewer (§4) the
door failed in one seed of three: only part of the shifted labelling reaches the
correction delta, the signal thinned to 9 levels, and the rate test added in §3
attributed it to the model (p = 0.0499 against α = 0.01) and **admitted the
poison** — a batch the older, cruder rule would have refused. The promotion gate
then refused every candidate trained on it (−11.4%, −15.9%, −7.0%, each also
tripping the forgetting gate). Layering worked; the screen did not. This false negative is pinned as a
known defect in `tests/test_gates.py` and is **not fixed**: the known false positive
sits at p = 0.25 and this false negative at p = 0.05, too close to separate by
re-tuning a threshold on two points.

**A loop that learned worse never deployed worse.** Across every arm in this
repository, no promoted model regressed the sequestered burden. Candidates that
were worse (seed 2027 round 2: −12.4%) or that forgot a region were refused with
a recorded reason.

**End-to-end burden fell in every seed with full references:** −25.1%, −26.4%,
−19.3% (`manifests/w6_seed_band.json`; the last after the promotion fix in §3).
Cervical Dice went from ~0.03 to 0.49–0.58. *Read §2 before quoting this.*

## 2. What the second null took away

The design promised that every promotion would beat **two** nulls. For most of
this project only one — do-nothing — had ever been trained. The second is the
**more-training null**: same starting model, same steps, same number of cases,
but drawn only from cases already seen. It asks whether a gain came from the new
corrections or merely from more optimisation (`manifests/w9_control_rejudge.json`).

- **Round 1 is mostly an optimisation gain.** More steps on old data alone
  recovered **96%, 44% and 58%** of the round-1 gain in the three seeds; only 1
  of 3 round-1 candidates beat that control beyond noise. The starting model was
  under-trained, and part of the end-to-end reduction in §1 is that — not
  learning from corrections.
- **After round 1 the corrections do the work.** In 6 of 6 later rounds the
  control ended *worse* than the model it started from, and all three
  cervical-arrival candidates beat it beyond noise (+10.8%, +22.6%, +11.0%).
  Caveat: an equal-size subset of old data overfits when trained further, so
  this control is easier to beat late than a control rehearsing *all* old data.
  That stronger control has not been run.

The predictions written down before the control arms were trained were wrong in
both directions. They are kept as written in the program log.

## 3. The gates were themselves wrong twice — measured, then fixed

**The admission screen refused a clean batch** in one seed. The screen reads a
consistent label offset in (reference − prediction) as "the references are
shifted"; here the *model* was shifted. Matching on direction would have been
unsafe — +1 is both the poison and the commonest model bias on clean batches.
The fix compares *rates*: the batch's rate of that offset against the model's
own rate on known-clean references. The poison ran 7–48× the model's rate; the
false positive ran 1.5×. Clean batches refused: 1 → 0 of 9; poison still refused
3 of 3. Cost, stated: the screen now reads the sequestered references. It
selects no training case, but it is a coupling.

**The promotion bar sat inside noise.** The same data under three training seeds
spreads total burden by CV 2.9% (5.6% end to end); the bar was a fixed 2.0%
(`manifests/w16_promotion_rejudge.json`). The gate now promotes when a paired
bootstrap interval over test cases excludes zero, or when a region the model did
not serve (median Dice < 0.20) rises by ≥ 0.20 with its own interval excluding
zero, burden not worse, nothing forgotten. One verdict moved — a candidate that
took cervical from 0.06 to 0.58 and had been refused at +1.95% — and none was
lost.

**What that re-judgement exposed:** with intervals, the burden gain of *every*
cervical-arrival round is indistinguishable from zero over do-nothing
(e.g. +8.5% [−3.6%, +19.9%]). Those three promotions stand on the unserved-region
clause, which rests on **9 test cases**. The earlier statements "every promoted
step beat the do-nothing null" and "the gate never promoted noise" hold as point
estimates only. The rule was fixed before any verdict was computed under it, but
after the aggregate gains had been seen; that ordering is the honest limit.

## 4. The result that matters most for practice: the oracle was doing the work

Everything above learns from the **full reference mask** — an oracle standing in
for a human correction. A real reviewer has a budget, fixes what is badly wrong,
and approves the rest. Simulated (`manifests/w7_corrector.json`, `manifests/w17_reviewed_only.json`): fix errors largest-first until half the
case's burden is spent, leave anything under 5% alone.

| what the loop trains on | promotions per seed | end-to-end per seed | final candidate cervical |
|---|---|---|---|
| full reference (oracle) | 3, 2, 2 | −25.1%, −26.4%, −19.3% | 0.54, 0.49, 0.58 |
| budgeted review, **unfixed output approved as truth** | 0, 0, 0 | 0%, 0%, 0% | 0.00, 0.00, 0.09 |
| same review, unreviewed voxels **masked from the loss** | 1, 1, 0 | −7.1%, −10.5%, 0% | 0.15, 0.00, 0.20 |
| masked + budget split **per anatomical region** | 0, 0, 1 | 0%, 0%, −5.1% | 0.30, 0.35, 0.40 |

(Seeds 1337, 2027, 3117; `manifests/w18_budgeted_seed_band.json`.)

- **Replicated 3 of 3:** when the model's unfixed output is approved as truth, the
  loop promotes *nothing*. The pre-registered kill condition for that row was met.
  A burden-ranked reviewer fixed 8% of cervical levels against 66% of lumbar, and
  the model's empty cervical output — approved as background — taught its blind
  spot back to it.
- **Masking the unreviewed voxels is a partial repair at best.** Three claims were
  written down before the two extra seeds were run, to be made only if all three
  seeds agreed. One held (the row above). Two did not: the poison was *not* refused
  at the door in every arm (§1), and the masked arm did *not* promote in every seed
  (1, 1, 0). By the rule fixed in advance, the two masked arms are **not detectably
  different from each other at this scale**. It is true that in every seed exactly
  one masked arm promoted once while the unmasked arm never did — but that grouping
  was noticed after the fact and is offered as a lead, not a finding.
- **The per-region budget does what it says and is not enough.** Reviewer effort on
  cervical went from 8% to 63% of levels fixed (seed 1337) and the final candidates
  reached cervical 0.30–0.40, yet only one of the nine clean-batch candidates deployed.
- **Partial review costs thoracic, every time.** Across all nine budgeted chains,
  candidates regressed thoracic Dice by 0.04–0.20 and tripped the forgetting gate
  in 22 of 30 trained rounds. The mechanism is not established here.
- A fourth arm modelling inter-reviewer boundary variability could not be
  evaluated: one 3 mm voxel of jitter is ~3× real variability, the model sprayed
  foreground across the volume, and scoring exceeded the memory budget
  (`manifests/w7_jitter_scatter.json`). That is a failure of the arm's design.

**For anyone building this for real:** a partial correction is not a label. What
the reviewer did not touch must reach training as *unknown*, not as *approved*;
and a reviewer who triages by size will starve small structures unless the
budget is allocated to prevent it.

## 5. Choosing *which* cases to rehearse: better in round 1, worth nothing by round 5

Rehearsal draws a quarter of each training set from cases already seen, and it
draws them uniformly. The obvious improvement is to draw the ones the current
model is worst at. The rule was written down before the run
(`manifests/w8_selection.json`): draw without replacement with probability
proportional to the incumbent's own correction burden on each seen case — same
number of cases, same seeds, same gates. **WIN** required lower deployed burden
in 3 of 3 seeds with the paired interval excluding zero in at least 2;
**HARMFUL** required higher in 3 of 3. The recorded expectation was a null.

It is a null. Deployed burden moved **−7.0%, −6.0%, +3.2%** (seeds 1337, 2027,
3117; positive means the weighted arm deployed *lower* burden) — better in 1 of
3 seeds, beyond noise in none.

The instructive part is *why*, because for one round the weighting plainly
worked. Judged at round 1, where both arms start from the same model and the
comparison is clean, the weighted arm was ahead in all three seeds — **+11.0%
[+6.8%, +15.4%], +2.2%, +5.1%** — significantly so in one. Then it stopped. It
promoted round 1 and never promoted again in any seed (uniform: 3, 2, 2). Its
later candidates were not idle: every weighted round-4 candidate *did* learn
cervical anatomy, to Dice 0.43–0.48. Every one was also refused, by two gates
at once. They lost to the do-nothing null on total burden — **−4.2%, −9.5%,
−7.9%**, with the interval excluding zero in two of the three seeds, so worse
than deploying nothing rather than merely unproven — and they regressed lumbar
by **0.09–0.14**, past the forgetting tolerance of 0.05. Learning the new
region did not pay for what it cost elsewhere: in the one seed where the
unserved-region clause did open on cervical, the gate recorded that it opened
"at a net burden cost", and it did not rescue the candidate. The arm that
learned the new anatomy fastest is the arm that never deployed it — final
deployed cervical Dice **0.13, 0.18, 0.21**, against **0.54, 0.49, 0.58**
under uniform rehearsal. The poison was still refused in 3 of 3 seeds.

Over-rehearsing the cases a model finds hardest is, mechanically,
under-rehearsing everything else, and the cost lands on the region that was
already fine. One round could not see that; five rounds and a forgetting gate
could. **A rehearsal policy measured in a single round is not evidence about
that policy in a loop.**

## 6. Limits

- **Scale.** Small network, 3 mm grid, 24 test cases. Effect sizes illustrate a
  method; they do not estimate anything.
- **Three seeds** give a min–max envelope, not a confidence interval.
- **Burden is not time.** Added path length has not been calibrated against a
  stopwatch here.
- **The forgetting gate** compares medians to a fixed 0.05 tolerance and has not
  had the interval treatment the promotion gate received.
- **Rehearsal** was shown to help by 5.2% burden in one gentle round; the
  catastrophic-forgetting case is cited from the literature, not reproduced.
  Only one alternative to uniform rehearsal was tried (§5), at one draw size.
- **Simulated reviewers** are simulations. No human corrected anything in this
  repository.

## 7. Where to look

`docs/build-report.html` — the narrated build, figures, and the corrections in
context (Step 8). `docs/method-transfer-brief.md` — each control as an experiment
with a kill condition. `tests/` — every gate has a planted-defect test, and the
results above are pinned to the committed artifacts so they cannot drift.
