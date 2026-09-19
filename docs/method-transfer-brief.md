# Method transfer brief — from the loop's controls to segmentation-development experiments

**Purpose.** The demo proved a *shape*: a learning loop whose admissions and
promotions are decided by measurements that were themselves tested against
nulls. This brief turns each demonstrated control into the experiment you would
run when the goal is no longer "show the loop" but "develop a better
segmentation training recipe" — and gives each experiment a condition under
which it is abandoned. Everything here refers to public artifacts in this
repository. It makes no clinical claim and transfers no result: a control that
worked on 24 open CT cases with a small network is a *method* to re-test, not a
finding to import.

**How to read a row.** *Demonstrated* = what this repo actually showed, with the
artifact. *Experiment* = the question to ask of a new recipe. *Kill condition* =
the observation that says stop, written before the run. An experiment with no
kill condition is a demonstration, not a test.

---

## 1. The ruler: correction burden, not overlap

**Demonstrated.** Added path length (APL, mm of contour a human would redraw at a
2 mm tolerance) plus a label-blind surface Dice plus a per-level error taxonomy
(`src/clloop/delta.py`). Before it scored anything it passed its nulls —
identical masks score exactly zero; a pure relabel scores near-1 surface Dice
with non-zero APL; APL is monotone in injected boundary drift
(`manifests/w1_null_panel.json`) — and was then frozen by hash.

**Experiment.** Rank two training recipes by sequestered APL and by volumetric
Dice. Ask whether the rankings *agree*.

**Kill condition.** If APL and Dice rank every recipe pair identically across
three seeds, the ruler adds cost without information for this task: report that
and select on Dice. If they disagree, APL decides and the disagreement is the
finding — Dice is insensitive to exactly the thin boundary errors that dominate
editing time.

**Limit to carry over.** APL here uses a first-order path-length approximation
and oracle references. It measures *redraw length*, not *minutes*; the mapping
to reviewer time is untested in this repo.

## 2. Frozen means frozen

**Demonstrated.** The ruler's hash is recorded before round 1 and re-frozen
exactly once, declared, when it was made faster (the null suite re-run as the
proof it changed nothing).

**Experiment.** Any change to the metric code during recipe development re-runs
the full null panel and re-scores *every* prior recipe under the new hash.

**Kill condition.** If a metric change re-orders previously decided comparisons,
every decision made under the old hash is void until re-judged. A ruler that
drifts with the thing it measures is not a ruler.

## 3. Sequestration as a refusal, not a filter

**Demonstrated.** `assert_sequestration` *raises* when a held-out case reaches a
training cohort; there is no skip flag. A planted leak is a test.

**Experiment.** Run the recipe pipeline with one test case deliberately injected
into the training list.

**Kill condition.** If the pipeline completes, stop everything: no number it has
ever produced can be trusted, because the estimator could touch the evaluator.
Silent de-duplication is the wrong fix — it hides the process defect that put
the case there.

**Limit, found here.** Screen v3 (below) *reads* the sequestered references to
arbitrate admissions. It selects no training case, but it is a coupling between
the test set and the loop, and it is stated rather than hidden. A stricter design
holds out a separate arbiter split.

## 4. Admission screen: fingerprints live at the mechanism's granularity

**Demonstrated.** A batch whose references carry a systematic enumeration shift
(every label +1) is refused before training. Three versions, each forced by a
real failure:
- v1 judged by *case* verdicts and missed a live poison — boundary noise diluted
  9 of 13 poisoned cases to "mixed" while 37 of 39 relabelled *levels* agreed on
  +1. The promotion gate caught it (−40% vs do-nothing). Layering worked; the
  screen did not.
- v2 judged by levels and then refused a *clean* batch in one seed: the offset
  was the model's own bias. The delta is (reference − prediction); alone it
  cannot say which side moved.
- v3 compares the batch's rate of the modal offset with the incumbent's rate of
  that offset on known-clean references. The poison ran 7–48× the incumbent's
  rate; the false positive ran 1.5× (`manifests/w15_screen_v3.json`).

**Experiment.** For each label-noise mechanism you believe exists in your
incoming data (enumeration shift, truncated field of view, protocol-specific
under-segmentation), build the poisoned batch *and* a clean batch scored by a
model that has the same bias, and require the screen to separate them.

**Kill condition.** If a screen cannot separate "references are wrong" from
"model is wrong" on the planted pair, it is not an admission control — remove it
and rely on the promotion gate, rather than keep a control that refuses clean
data. A screen with no demonstrated false-positive test is unfinished.

## 5. Promotion against the do-nothing null — with uncertainty

**Demonstrated.** A candidate must beat the incumbent on the sequestered set. The
original fixed 2.0% bar was then *measured* to sit inside training noise: the
same data under three seeds spreads total burden by CV 2.9%, 5.6% end to end
(`manifests/w16_promotion_rejudge.json`). The gate now uses a paired-bootstrap
interval over test cases. The degeneracy check is a test: the incumbent judged
against itself must not promote.

**Experiment.** Before comparing recipes, train the *same* recipe under ≥3 seeds
and record the spread. That spread, not intuition, sets the smallest difference
worth claiming.

**Kill condition.** If the between-recipe difference is smaller than the
same-recipe seed spread, the comparison is undecided — do not ship the "winner".
Either add test cases, add seeds, or stop.

**Uncomfortable result to carry over.** Re-judged with intervals, the burden gain
of every cervical-arrival round in this demo is indistinguishable from zero on
24 test cases. Those promotions stand on a second, explicit clause (next row),
not on burden.

## 6. Crediting gain composition: the unserved-region clause

**Demonstrated.** Total burden is blind to *where* the gain is. A candidate that
took cervical Dice from 0.06 to 0.58 was refused at +1.95% total. The banded
gate credits a region the incumbent did not serve (median Dice < 0.20) when it
rises by ≥ 0.20, its paired interval excludes zero, burden is not worse, and no
served region regresses.

**Experiment.** When a new recipe targets a sub-population (an anatomy, a scanner
class, a pathology), pre-declare that stratum and its threshold, and judge it
separately from the aggregate.

**Kill condition.** If the stratum is declared *after* seeing results, the clause
is void for that comparison — that is a forking path, not a criterion. In this
repo the aggregate gains had been seen before the rule was written; the rule was
fixed before any verdict was computed under it, and that ordering is recorded.
The clause rests on 9 cervical test cases; treat it as a pattern, not evidence of
size.

## 7. The more-training null

**Demonstrated.** A control arm starts from the same incumbent with the same
iterations and seed at equal N, but draws its cohort only from already-seen
cases. It asks: is the gain from the *new corrections*, or from *more
optimisation steps*? Result (`manifests/w9_control_rejudge.json`): in round 1 the
control alone recovered 96%, 44% and 58% of the gain across three seeds — the
baseline was under-trained and much of that "learning" was optimisation. After
round 1 the control ended *worse* than its incumbent in 6 of 6 rounds, and all
three cervical-arrival candidates beat it beyond noise. The predictions recorded
before training the arms were wrong in both directions.

**Experiment.** Every "fine-tuning on new data improved the model" claim gets
this arm. It costs one extra training run.

**Kill condition.** If the control captures most of the gain, the incumbent was
under-trained and the claim about the new data is unsupported: fix the baseline's
training budget first, then re-ask. Attributing an optimisation gain to data is
the commonest way a learning loop flatters itself.

**Limit.** An equal-N subset of old data overfits when trained further, so beating
this control in later rounds is a weaker statement than beating a control that
rehearses *all* seen data for the same number of steps. That stronger control has
not been run here.

## 8. Rehearsal against forgetting

**Demonstrated.** A 25% rehearsal mix, with a no-rehearsal ablation run beside it
so forgetting is *shown*, and a forgetting gate that fails any candidate
regressing a region the incumbent knew by more than 0.05 median Dice.

**Experiment.** Under a shifting case mix, sweep the rehearsal fraction (0, 10,
25, 50%) and plot old-region retention against new-region gain.

**Kill condition.** If 0% rehearsal shows no forgetting at your stream's shift
rate, rehearsal is unnecessary cost — drop it. If no fraction prevents
forgetting, the problem is capacity or schedule, and rehearsal is the wrong tool.

**Limit.** The forgetting gate compares *medians* against a fixed 0.05 tolerance;
it has not been given the interval treatment the promotion gate received, and a
0.0375 lumbar drop passed it in one replicate. That asymmetry is a known gap.

## 9. Seed band before story

**Demonstrated.** Three full-chain replicates. End-to-end claims replicated
(burden −19 to −26%, poison refused 3 of 3); *per-round paths did not* — one seed
refused a clean batch, another refused a real improvement.

**Experiment.** No recipe comparison is reported from one seed.

**Kill condition.** If the sign of the recipe difference flips across seeds, the
honest result is "no detectable difference at this budget".

---

## What does not transfer

- **The oracle corrector.** References here stand in for human corrections; real
  reviewers are budgeted, inconsistent, and leave small errors alone. Until the
  realistic-corrector arm is run, "learning from corrections" means "learning
  from ground truth that arrived late".
- **Scale.** A small network, low-resolution volumes, 24 test cases, 9 of them
  cervical. Effect sizes are illustrative.
- **Burden ≠ time.** APL is a proxy for editing effort that has not been
  calibrated against a stopwatch in this repository.

## The order to run them

1 → 2 → 3 are prerequisites (no ruler, no experiment). Then 9 (know your noise),
then 5 and 7 together (both nulls), then 4, 6, 8 as the data stream demands.
