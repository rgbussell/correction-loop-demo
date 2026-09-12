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


def screen_batch(
    deltas: list[dict],
    *,
    min_relabel_frac: float = 0.5,
    min_offset_consensus: float = 0.7,
) -> BatchScreen:
    """Screen an arriving batch by its correction-delta profile.

    The enumeration-poison signature is level-wise: across the batch's
    relabel-class levels, the (reference label − auto label) offsets
    concentrate on one value. Scattered relabels are the model's problem;
    a CONSISTENT offset across cases is the references' problem, and a batch
    whose references are systematically mis-enumerated is refused whole.

    Requires each delta dict to carry ``level_offsets``: the list of
    (final_label − auto_label) values for its relabel-class levels.
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
    if relabel_frac >= min_relabel_frac and consensus >= min_offset_consensus:
        return BatchScreen(
            [], sorted(ids),
            f"enumeration-shift signature: {relabel_frac:.0%} of cases are "
            f"relabel-class and {consensus:.0%} of relabel offsets agree on "
            f"{modal_offset:+d} — a systematic reference labeling error, "
            "refused before any training",
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
