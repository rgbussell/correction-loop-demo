# Correction-loop demo — goals, uses, critique (2026-09-13)

**Audience:** humans who want the point without reading the full HTML report.  
**Artifacts:** `docs/build-report.html`, `README.md`, program ledger `paai/.claude/programs/cl-demo.yaml` (rev bumped for Phase 3).  
**Not clinical software.** Numbers demonstrate *loop mechanics* on open VerSe CT.

---

## 1. Goals (what we set out to show)

| # | Goal | Practical use if true |
|---|------|------------------------|
| G1 | Run a **multi-round** correction→retrain→promote loop on open spine CT | Template for keeping human fixes instead of throwing them away |
| G2 | Score learning as **correction burden** (APL + surface Dice + taxonomy), not Dice alone | Choose methods that reduce editing labour |
| G3 | **Refuse** bad data/rounds via layered gates (admission + promotion nulls) | Stop poison / silent degradation from becoming the new teacher |
| G4 | Use **rehearsal** so new anatomy can arrive without erasing old coverage | Continual improvement when the case mix shifts (e.g. cervical arriving after lumbar) |
| G5 | Make the whole story **auditable and public** (tests, manifests, report) | External readers can verify the pattern without private IP |

**Non-goals (declared):** SOTA segmentation accuracy; clinical performance claims; any employer or proprietary data/code.

---

## 2. What ran (data summary, readable)

**Setup:** VerSe 2020 · ~107 usable cases · small MONAI 3-D U-Net @ 3 mm · frozen correction-delta ruler · sequestered test never used for training.

| Round | What arrived | Decision | Why it matters |
|-------|--------------|----------|----------------|
| 0 | Lumbar/thoracic-heavy baseline | baseline | Starts blind to cervical (Dice ~0) |
| 1 | New batch + ~25% rehearsal | **promoted** (+~10% APL vs do-nothing) | Loop can improve from correction deltas |
| 2 | Further shift | **promoted** (+~9%) | Improvement repeats |
| 3 | **Poisoned** references (enumeration +1) | **refused before training** (definitive chain) | Bad batch does not become teacher |
| 3∅ | Counterfactual: force-train on poison | would be **−40% APL** | Refusal had measurable value |
| 4 | Cervical-heavy batch | **promoted**; cervical Dice ~0→0.54; TL held | Learns new region without obvious TL collapse |
| 4− | No-rehearsal ablation | no regional collapse; rehearsal worth ~5.5% APL | Honest: catastrophic forgetting *not* reproduced at this gentle scale |

---

## 3. Extent goals were met

| Goal | Verdict | Extent |
|------|---------|--------|
| G1 Multi-round loop | **Met** | 5-round chain from one script; 28/28 tests |
| G2 Labour-centric ruler | **Met** | Frozen APL/sDice/taxonomy; null suite; sha recorded |
| G3 Visible refusal | **Met (strong)** | Unscripted miss on first screen → promotion gate saved; screen fixed + regression test; definitive run refuses at the door |
| G4 Rehearsal / shift | **Partially met** | Cervical learning shown; rehearsal margin measured; *not* a stress test of catastrophic forgetting |
| G5 Public audit trail | **Met** | Public GitHub repo + HTML report + manifests |

**Overall:** Phase-1 demo goal (**show the loop shape, including “no”**) is achieved. Phase-2 (uncertainty bands, noisy corrector, per-round matched controls) and Phase-3 (use the stack to *pick better segmentation recipes*) are the honest remainder.

---

## 4. Critique (what is weak or easy to over-read)

1. **Oracle corrector.** References stand in for humans — signal is cleaner than real edits. Do not claim production labour reduction yet (that is W7 / private program).
2. **Single seed.** Per-round +8–10% APL gains have **no uncertainty band** until W6. Easy to overfit a narrative to one chain.
3. **Matched random control not on every promote.** Do-nothing null is strong; equal-N random cohort is thinner in the happy path (W9).
4. **Toy capacity.** Coarse 3 mm U-Net — absolute Dice/APL are **not** transferable to clinical nnU-Net systems.
5. **First poison screen failed.** Important honesty: layered defense worked, but the “elegant” detector needed a post-hoc fix. Treat admission rules as empirical, not sacred.
6. **Rehearsal story is mild.** No-rehearsal did not collapse regions; citing catastrophic forgetting from *other* work is fine only if labeled literature, not this demo’s result.
7. **Goal drift risk.** A beautiful loop demo ≠ “we improved segmentation methods.” That requires **recipe A vs B judged by the ruler** (W13) or the private correction-learning arms.

---

## 5. Practical next uses (segmentation method development)

Use this demo as a **test harness philosophy**, not a scoreboard:

- Gate candidate losses/curricula on **sequestered APL**, with Dice monitoring only.
- Require **poison / bad-batch refusal** tests in any continual or correction-learning pipeline.
- Pre-register kill conditions before comparing replay policies (boundary-weighted vs random).
- Keep public pattern evidence here; keep clinical corpora in the private program.

Ledger follow-ups (additive, cl-demo Phase 3): **W12** method-transfer brief · **W13** ruler-gated recipe smoke · **W14** hand-off checklist to private correction-learning.

---

*Chief of Staff review, 2026-09-13. Critique is about evidence quality, not authorship.*
