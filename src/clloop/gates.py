"""The three refusals and the two-null promotion gate.

The dangerous part of a learning loop is not the training, it is the
admission and the promotion. Each control here is a REFUSAL — it raises or
returns a reason-coded rejection, never a silent filter — and each has a
planted-defect test proving it can be seen firing.

1. **Sequestration** (`assert_sequestration`): a held-out test case reaching
   any training cohort raises. No skip flag, no override. If the process that
   selects training data can touch the data that measures it, the estimator
   and the evaluator collapse into each other and every round looks fine
   forever.

2. **Batch admission** (`screen_batch`): the arriving batch's correction
   deltas are screened for the enumeration-poison signature before any
   training — a batch where most relabel-class levels share ONE consistent
   label offset is a systematic labeling error in the REFERENCES, not a
   model failure, and training toward it would teach the error. Detected at
   the cost of reading deltas already computed, not a training run.

3. **Forgetting** (`check_forgetting`): a candidate that regresses any
   region the incumbent genuinely knew fails, regardless of how much it
   improved elsewhere. This is what the rehearsal mix exists to prevent;
   the gate is what proves the rehearsal worked.

4. **Promotion** (`decide_promotion`): a candidate is promoted only if it
   beats BOTH nulls on the sequestered test — the do-nothing incumbent, and
   (when a control arm was trained) the matched random-control candidate.
   The gate's own degeneracy check: a do-nothing candidate must NOT pass —
   a gate that promotes the incumbent over itself is measuring noise.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


class SequestrationBreach(RuntimeError):
    """A sequestered test case reached a training cohort."""


def assert_sequestration(train_ids: list[str], sequestered: list[str]) -> None:
    overlap = sorted(set(train_ids) & set(sequestered))
    if overlap:
        raise SequestrationBreach(
            f"{len(overlap)} sequestered case(s) in a training cohort: "
            f"{overlap[:5]}{'…' if len(overlap) > 5 else ''} — fix the cohort, "
            "never filter here."
        )


# ------------------------------------------------------------- admission
@dataclass
class BatchScreen:
    admitted: list[str]
    refused: list[str]
    refusal_reason: str | None
    evidence: dict = field(default_factory=dict)

    @property
    def batch_refused(self) -> bool:
        return self.refusal_reason is not None


def _shift_is_the_models(
    deltas: list[dict], modal_offset: int, modal_n: int, arbiter: dict,
    alpha: float, evidence: dict,
) -> bool:
    """The who-is-shifted discriminator (screen v3). Mutates ``evidence``.

    The signature alone cannot say WHO is shifted: the delta is (reference −
    auto), so an incumbent with its own enumeration bias makes clean
    references look shifted (s3117 round 2: 9/12 offsets at −1, refused, and
    the batch was clean). The sequestered references are known-clean, so the
    incumbent's rate of the SAME offset there is the model's share of the
    signal. One-sided Fisher exact test on levels: is the batch's rate of
    modal-offset levels higher than the incumbent's own rate on the
    sequestered set? If not, the shift is the model's — and training toward
    clean references is the cure, not the poison — so the batch is admitted.

    Direction alone is NOT the test: +1 is also the commonest model bias on
    clean batches, and the poison is +1. Measured across the seed band the
    poison runs 7–48x the incumbent's own rate; the false positive ran 1.5x.
    Levels within a case are correlated (a shift runs through a whole spine),
    which makes the test anticonservative — it errs toward REFUSING, the
    safe direction, and the promotion gate still stands behind an admit.
    """
    from scipy.stats import fisher_exact

    batch_levels = sum(sum(d.get("level_kinds", {}).values()) for d in deltas)
    arb_levels = int(arbiter["n_levels"])
    arb_n = int(arbiter["offset_histogram"].get(str(modal_offset), 0))
    if batch_levels < modal_n or arb_levels <= 0:
        return False  # no usable denominator — the refusal stands
    _, p = fisher_exact(
        [[modal_n, batch_levels - modal_n], [arb_n, arb_levels - arb_n]],
        alternative="greater",
    )
    models = p >= alpha
    evidence["arbiter"] = {
        "batch_rate": round(modal_n / batch_levels, 4),
        "incumbent_rate_on_sequestered": round(arb_n / arb_levels, 4),
        "batch_counts": [modal_n, batch_levels],
        "sequestered_counts": [arb_n, arb_levels],
        "p_batch_exceeds_incumbent": float(f"{p:.3g}"),
        "alpha": alpha,
        "shift_attributed_to": "model" if models else "references",
    }
    return models


def screen_batch(
    deltas: list[dict],
    *,
    min_relabel_levels: int = 8,
    min_offset_consensus: float = 0.7,
    arbiter: dict | None = None,
    arbiter_alpha: float = 0.01,
) -> BatchScreen:
    """Screen an arriving batch by its correction-delta profile.

    The enumeration-poison signature is LEVEL-wise: across the batch's
    relabel-class levels, the (reference label − auto label) offsets
    concentrate on one value. Scattered relabels are the model's problem;
    a CONSISTENT offset across many levels is the references' problem, and a
    batch whose references are systematically mis-enumerated is refused whole.

    REVISED after the first live poisoned round (2026-09-12): the original
    rule also required >=50% of CASE verdicts to be relabel-class, and a weak
    incumbent's boundary noise diluted 9/13 poisoned cases to `mixed` — the
    screen admitted the batch on evidence of 37/39 relabel levels agreeing on
    +1 (consensus 0.949). The promotion gate caught it (the candidate scored
    -40% vs do-nothing and was refused), which is what the layered design is
    for — but the fingerprint lives at the LEVEL, not the case verdict, so
    the rule now reads: >= ``min_relabel_levels`` relabel-class levels whose
    offsets reach ``min_offset_consensus``. Case-kind fractions remain in the
    evidence for the dashboard, not in the decision.

    REVISED again after the seed band (v3): the signature fired on a CLEAN
    batch because the incumbent itself was shifted. When ``arbiter`` (the
    incumbent's offset profile on the sequestered references:
    ``{"n_levels", "offset_histogram"}``) is supplied, a fired signature is
    refused only if the batch's rate exceeds the incumbent's own — see
    ``_shift_is_the_models``. Without an arbiter the v2 rule stands.

    Requires each delta dict to carry ``level_offsets``: the list of
    (final_label − auto_label) values for its relabel-class levels, and (for
    the arbiter) ``level_kinds``, whose counts sum to the levels scored.
    """
    ids = [d["case_id"] for d in deltas]
    all_offsets: list[int] = []
    n_relabel_cases = 0
    for d in deltas:
        offs = d.get("level_offsets", [])
        all_offsets.extend(offs)
        if d.get("kind") == "relabel":
            n_relabel_cases += 1

    evidence = {
        "n_cases": len(deltas),
        "n_relabel_cases": n_relabel_cases,
        "n_relabel_levels": len(all_offsets),
        "offset_histogram": dict(Counter(all_offsets)),
    }
    if not all_offsets:
        return BatchScreen(sorted(ids), [], None, evidence)

    relabel_frac = n_relabel_cases / len(deltas)
    modal_offset, modal_n = Counter(all_offsets).most_common(1)[0]
    consensus = modal_n / len(all_offsets)
    evidence.update({
        "relabel_case_frac": round(relabel_frac, 3),
        "modal_offset": int(modal_offset),
        "offset_consensus": round(consensus, 3),
    })
    if len(all_offsets) >= min_relabel_levels and consensus >= min_offset_consensus:
        if arbiter is not None and _shift_is_the_models(
                deltas, int(modal_offset), modal_n, arbiter, arbiter_alpha, evidence):
            return BatchScreen(sorted(ids), [], None, evidence)
        return BatchScreen(
            [], sorted(ids),
            f"enumeration-shift signature: {len(all_offsets)} relabel-class "
            f"levels across the batch and {consensus:.0%} of their offsets "
            f"agree on {modal_offset:+d} — a systematic reference labeling "
            "error, refused before any training",
            evidence,
        )
    return BatchScreen(sorted(ids), [], None, evidence)


# ------------------------------------------------------------- forgetting
@dataclass
class ForgettingVerdict:
    passed: bool
    regressions: list[str]
    detail: dict = field(default_factory=dict)


def check_forgetting(
    incumbent_eval: dict,
    candidate_eval: dict,
    *,
    max_drop: float = 0.05,
    known_floor: float = 0.20,
) -> ForgettingVerdict:
    """Candidate must not regress any region the incumbent genuinely knew.

    "Knew" = incumbent median Dice >= known_floor (a region the incumbent
    scored 0.0 on cannot be forgotten). "Regress" = median Dice drop >
    max_drop. Regions absent from the test references (None) are skipped —
    absence is no evidence.
    """
    regressions, detail = [], {}
    for reg in ("cervical", "thoracic", "lumbar", "sacrum"):
        inc = incumbent_eval.get(f"dice_{reg}_median")
        cand = candidate_eval.get(f"dice_{reg}_median")
        if inc is None or cand is None:
            continue
        detail[reg] = {"incumbent": inc, "candidate": cand, "drop": round(inc - cand, 4)}
        if inc >= known_floor and (inc - cand) > max_drop:
            regressions.append(
                f"{reg}: {inc:.3f} -> {cand:.3f} (drop {inc - cand:.3f} > {max_drop})"
            )
    return ForgettingVerdict(not regressions, regressions, detail)


# ------------------------------------------------------------- promotion
@dataclass
class PromotionDecision:
    promoted: bool
    reasons: list[str]
    evidence: dict = field(default_factory=dict)


def decide_promotion(
    incumbent_eval: dict,
    candidate_eval: dict,
    *,
    control_eval: dict | None = None,
    forgetting: ForgettingVerdict | None = None,
    min_apl_gain_frac: float = 0.02,
) -> PromotionDecision:
    """Promote only past BOTH nulls (and the forgetting gate when supplied).

    Null 1 — do-nothing: the candidate's total correction burden on the
    sequestered test must undercut the incumbent's by at least
    ``min_apl_gain_frac`` (a do-nothing candidate has gain exactly 0 and
    must fail — that is the gate's degeneracy check, tested).
    Null 2 — matched random control: when a control arm was trained at equal
    N, the candidate must not lose to it; losing means the curation policy
    subtracted value.
    """
    reasons, evidence = [], {}
    inc_apl = float(incumbent_eval["apl_mm_total"])
    cand_apl = float(candidate_eval["apl_mm_total"])
    gain = (inc_apl - cand_apl) / inc_apl if inc_apl > 0 else 0.0
    evidence["apl_gain_frac_vs_do_nothing"] = round(gain, 4)
    if gain < min_apl_gain_frac:
        reasons.append(
            f"fails do-nothing null: APL gain {gain:+.1%} < {min_apl_gain_frac:.0%} "
            f"({inc_apl / 1000:.0f} m -> {cand_apl / 1000:.0f} m)"
        )

    if control_eval is not None:
        ctrl_apl = float(control_eval["apl_mm_total"])
        evidence["apl_vs_random_control"] = {
            "candidate_m": round(cand_apl / 1000, 1),
            "control_m": round(ctrl_apl / 1000, 1),
        }
        if cand_apl > ctrl_apl:
            reasons.append(
                f"loses to matched random control: {cand_apl / 1000:.0f} m vs "
                f"{ctrl_apl / 1000:.0f} m — the curation policy subtracted value"
            )

    if forgetting is not None and not forgetting.passed:
        reasons.append("forgetting gate: " + "; ".join(forgetting.regressions))
        evidence["forgetting"] = forgetting.detail

    return PromotionDecision(not reasons, reasons, evidence)
