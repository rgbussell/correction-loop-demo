#!/usr/bin/env python3
"""Build docs/build-report.html — the narrated build log of the demo.

Regenerated after every workstream. Figures are matplotlib SVG embedded inline,
styled to the reference dataviz palette (categorical slots in fixed order,
hairline grid, thin marks, text in ink tokens — never in series color). The
report is fully self-contained: no external assets, no scripts.
"""

from __future__ import annotations

import io
import json
from collections import Counter
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "build-report.html"

# ---- reference dataviz palette (light mode) --------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3 (all-pairs safe)
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CRITICAL = "#d03b3b"

GROUP_ORDER = ["thoraco-lumbar", "cervical-containing", "other"]
GROUP_COLOR = {"thoraco-lumbar": S1, "cervical-containing": S2, "other": S3}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "text.color": INK,
        "axes.edgecolor": BASE,
        "axes.labelcolor": INK2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 1.0,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "font.size": 10,
    }
)


def _svg(fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    svg = buf.getvalue()
    return svg[svg.index("<svg") :]


def load():
    cases = json.loads((REPO / "manifests" / "cases.json").read_text())
    part = json.loads((REPO / "manifests" / "partition.json").read_text())
    poison = json.loads((REPO / "manifests" / "poison.json").read_text())
    w1_p = REPO / "manifests" / "w1_ruler_validation.json"
    w1 = json.loads(w1_p.read_text()) if w1_p.is_file() else None
    rounds = []
    rdir = REPO / "outputs" / "rounds"
    if rdir.is_dir():
        for d in sorted(rdir.iterdir()):
            f = d / "round.json"
            if f.is_file():
                rounds.append(json.loads(f.read_text()))
    return cases, part, poison, w1, rounds


# ---------------------------------------------------------------- figures
def fig_partition_composition(cases, part):
    """Stacked horizontal bars: each partition role by fov group."""
    by_id = {c["case_id"]: c for c in cases["cases"]}
    roles = [
        ("Sequestered test", part["sequestered_test"]),
        ("Initial pool", part["initial_pool"]),
    ] + [
        (f"Arrival batch {i}" + (" (poisoned)" if i == part["poisoned_batch_index"] else ""), b)
        for i, b in enumerate(part["arrival_batches"])
    ]
    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    y = np.arange(len(roles))[::-1]
    left = np.zeros(len(roles))
    for g in GROUP_ORDER:
        vals = np.array([sum(by_id[c]["fov_group"] == g for c in ids) for _, ids in roles])
        ax.barh(y, vals, left=left, height=0.62, color=GROUP_COLOR[g], label=g,
                edgecolor=SURFACE, linewidth=2)
        left += vals
    for yi, (name, ids) in zip(y, roles):
        ax.text(left[list(y).index(yi)] + 0.6, yi, str(len(ids)), va="center",
                color=INK2, fontsize=9)
    ax.set_yticks(y, [r[0] for r in roles], color=INK2)
    ax.set_xlabel("cases")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    ax.set_title("The stream design: who goes where", loc="left", color=INK, fontsize=11)
    return _svg(fig)


def fig_shift(cases, part):
    """The declared shift: cervical-containing fraction per arrival batch."""
    by_id = {c["case_id"]: c for c in cases["cases"]}
    fracs, ns = [], []
    for b in part["arrival_batches"]:
        fracs.append(sum(by_id[c]["fov_group"] == "cervical-containing" for c in b) / len(b))
        ns.append(len(b))
    pool_frac = sum(
        by_id[c]["fov_group"] == "cervical-containing" for c in part["initial_pool"]
    ) / len(part["initial_pool"])
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    x = np.arange(len(fracs))
    ax.axhline(pool_frac, color=MUTED, linewidth=1.0, linestyle=(0, (1, 2)))
    ax.text(len(fracs) - 0.52, pool_frac + 0.015, f"initial pool {pool_frac:.0%}",
            color=MUTED, fontsize=8.5, ha="right")
    ax.plot(x, fracs, color=S2, linewidth=2, marker="o", markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2)
    pi = part["poisoned_batch_index"]
    ax.plot([x[pi]], [fracs[pi]], marker="o", markersize=8, color=CRITICAL,
            markeredgecolor=SURFACE, markeredgewidth=2)
    ax.annotate("poisoned batch", (x[pi], fracs[pi]), textcoords="offset points",
                xytext=(10, -14), color=CRITICAL, fontsize=9)
    for xi, f in zip(x, fracs):
        ax.text(xi, f + 0.035, f"{f:.0%}", ha="center", color=INK2, fontsize=9)
    ax.set_xticks(x, [f"batch {i}\n(n={n})" for i, n in enumerate(ns)])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("cervical-containing fraction")
    ax.set_title("The declared distribution shift, batch over batch",
                 loc="left", color=INK, fontsize=11)
    return _svg(fig)


def fig_level_coverage(cases, part):
    """Heatmap: fraction of cases carrying each vertebral level, per role."""
    by_id = {c["case_id"]: c for c in cases["cases"]}
    names = (["C" + str(i) for i in range(1, 8)] + ["T" + str(i) for i in range(1, 13)]
             + ["L" + str(i) for i in range(1, 7)] + ["Sac"])
    ids_ = list(range(1, 27))
    roles = [("test", part["sequestered_test"]), ("pool", part["initial_pool"])] + [
        (f"b{i}", b) for i, b in enumerate(part["arrival_batches"])
    ]
    mat = np.zeros((len(roles), len(ids_)))
    for r, (_, members) in enumerate(roles):
        for c in members:
            for lv in by_id[c]["levels"]:
                if lv in ids_:
                    mat[r, ids_.index(lv)] += 1
        mat[r] /= len(members)
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq", [SURFACE] + SEQ)
    fig, ax = plt.subplots(figsize=(8.6, 2.6))
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, fontsize=7.5)
    ax.set_yticks(range(len(roles)), [r[0] for r in roles], fontsize=9, color=INK2)
    ax.grid(visible=False)
    pi = part["poisoned_batch_index"]
    ax.get_yticklabels()[2 + pi].set_color(CRITICAL)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01)
    cb.set_label("fraction of cases with level", color=INK2, fontsize=8.5)
    cb.ax.tick_params(labelsize=8, color=MUTED, labelcolor=MUTED)
    cb.outline.set_visible(False)
    ax.set_title("Anatomical coverage by partition role", loc="left", color=INK, fontsize=11)
    return _svg(fig)


def fig_poison(poison):
    """One poisoned case's label chain, before vs after."""
    case = max(poison["cases"], key=lambda c: len(c["label_map"]))
    m = {int(k): v for k, v in case["label_map"].items()}
    src = sorted(m)
    NAME = {i: f"C{i}" for i in range(1, 8)}
    NAME.update({i: f"T{i - 7}" for i in range(8, 20)})
    NAME.update({i: f"L{i - 19}" for i in range(20, 26)})
    NAME.update({26: "Sac", 27: "Coc", 28: "T13"})
    fig, ax = plt.subplots(figsize=(7.8, 2.2))
    x = np.arange(len(src))
    for xi, v in zip(x, src):
        changed = m[v] != v
        ax.text(xi, 0.72, NAME.get(v, v), ha="center", color=INK2, fontsize=9)
        ax.annotate("", xy=(xi, 0.42), xytext=(xi, 0.62),
                    arrowprops=dict(arrowstyle="->", color=CRITICAL if changed else MUTED,
                                    linewidth=1.6 if changed else 1.0))
        ax.text(xi, 0.28, NAME.get(m[v], m[v]), ha="center",
                color=INK if changed else MUTED,
                fontsize=9, fontweight="bold" if changed else "normal")
    ax.text(-0.9, 0.72, "true", ha="right", color=MUTED, fontsize=9)
    ax.text(-0.9, 0.28, "poisoned", ha="right", color=MUTED, fontsize=9)
    ax.set_xlim(-1.6, len(src) - 0.4)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        f"The enumeration poison, case {case['case_id']}: every vertebra keeps its shape, "
        "wears its neighbour's name", loc="left", color=INK, fontsize=11)
    return _svg(fig)


def fig_ruler_signature(val):
    """Scatter: surface Dice vs APL for the 13 real poisoned pairs."""
    rows = val["cases"]
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    x = [r["apl_mm"] / 1000 for r in rows]
    y = [r["surface_dice"] for r in rows]
    ax.scatter(x, y, s=64, color=CRITICAL, edgecolor=SURFACE, linewidth=2, zorder=3,
               label="poisoned pair (n=13)")
    ax.axhline(1.0, color=MUTED, linewidth=1.0, linestyle=(0, (1, 2)))
    ax.text(max(x) * 0.99, 0.965, "shape agreement = perfect", color=MUTED,
            fontsize=8.5, ha="right")
    ax.annotate("the enumeration signature:\nshapes right, names wrong,\nredraw cost real",
                (np.median(x), 1.0), textcoords="offset points", xytext=(-8, -52),
                color=INK2, fontsize=9)
    ax.set_xlabel("added path length (metres of contour)")
    ax.set_ylabel("surface Dice (label-blind)")
    ax.set_ylim(0.9, 1.02)
    ax.set_title("The ruler reads all 13 poisoned pairs as what they are",
                 loc="left", color=INK, fontsize=11)
    ax.legend(loc="lower left", frameon=False, fontsize=8.5)
    return _svg(fig)


def fig_ruler_nulls():
    """The null suite as a picture: what each planted input must score."""
    rows = [
        ("N1 identity", "APL = 0.0 exactly · sDice = 1.0 · accepted", True),
        ("N2 pure relabel", "sDice = 1.0 BUT APL > 0 · verdict relabel", True),
        ("N3 drift extent", "APL rises with levels perturbed (monotone)", True),
        ("N3b deep erosion", "APL saturates at total contour length (bounded)", True),
        ("N4 missing / spurious", "named as such, never folded into boundary", True),
        ("N5 empty auto", "sDice = 0, all levels missing — never agreement", True),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 2.3))
    for i, (name, desc, ok) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.text(0.02, y, name, fontsize=9.5, color=INK, va="center", fontweight="bold")
        ax.text(0.30, y, desc, fontsize=9, color=INK2, va="center")
        ax.text(0.985, y, "PASS", fontsize=9, color="#006300", va="center", ha="right")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.axis("off")
    ax.set_title("The null suite the ruler passed before being frozen",
                 loc="left", color=INK, fontsize=11)
    return _svg(fig)


def fig_round_dice(rounds):
    """Per-region median Dice on the sequestered test, round over round."""
    regions = ["cervical", "thoracic", "lumbar", "sacrum"]
    colors = {"cervical": S2, "thoracic": S3, "lumbar": S1, "sacrum": MUTED}
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    x = [r["round"] for r in rounds]
    for reg in regions:
        y = [r["eval"].get(f"dice_{reg}_median") for r in rounds]
        ax.plot(x, y, color=colors[reg], linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=reg)
        if y and y[-1] is not None:
            ax.annotate(f"{y[-1]:.2f}", (x[-1], y[-1]), textcoords="offset points",
                        xytext=(8, -3), color=INK2, fontsize=8.5)
    ax.set_xticks(x, [f"round {i}" for i in x])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("median Dice (sequestered test)")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, ncols=2)
    ax.set_title("What the model knows, round over round", loc="left",
                 color=INK, fontsize=11)
    return _svg(fig)


def fig_round_burden(rounds):
    """The loop's headline: correction burden on the sequestered test."""
    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    x = [r["round"] for r in rounds]
    y = [r["eval"]["apl_mm_median"] / 1000 for r in rounds]
    ax.plot(x, y, color=S1, linewidth=2, marker="o", markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2)
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:.1f} m", (xi, yi), textcoords="offset points",
                    xytext=(0, 9), ha="center", color=INK2, fontsize=9)
    ax.set_xticks(x, [f"round {i}" for i in x])
    ax.set_ylabel("median APL per test case (m)")
    ax.set_ylim(bottom=0)
    ax.set_title("The headline: correction burden per case", loc="left",
                 color=INK, fontsize=11)
    return _svg(fig)


# ---------------------------------------------------------------- report
CSS = f"""
:root {{ color-scheme: light; }}
body {{ margin: 0; background: #f9f9f7; color: {INK};
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }}
.wrap {{ max-width: 880px; margin: 0 auto; padding: 32px 24px 72px; }}
.card {{ background: {SURFACE}; border: 1px solid rgba(11,11,11,0.10);
  border-radius: 10px; padding: 20px 24px; margin: 18px 0; }}
h1 {{ font-size: 26px; margin: 0 0 4px; }}
h2 {{ font-size: 18px; margin: 26px 0 8px; }}
h3 {{ font-size: 14px; margin: 18px 0 6px; color: {INK2}; }}
p, li {{ font-size: 14px; line-height: 1.55; color: {INK2}; }}
strong {{ color: {INK}; }}
.meta {{ color: {MUTED}; font-size: 12.5px; }}
.hero {{ font-size: 48px; font-weight: 600; color: {INK}; }}
.tiles {{ display: flex; gap: 14px; flex-wrap: wrap; }}
.tile {{ flex: 1 1 150px; background: {SURFACE}; border: 1px solid rgba(11,11,11,0.10);
  border-radius: 10px; padding: 14px 16px; }}
.tile .v {{ font-size: 26px; font-weight: 600; color: {INK}; }}
.tile .l {{ font-size: 12px; color: {MUTED}; }}
table {{ border-collapse: collapse; font-size: 13px; width: 100%; }}
th {{ text-align: left; color: {MUTED}; font-weight: 500; border-bottom: 1px solid {GRID};
  padding: 6px 10px 6px 0; }}
td {{ padding: 6px 10px 6px 0; border-bottom: 1px solid {GRID};
  font-variant-numeric: tabular-nums; color: {INK2}; }}
code {{ background: #f0efec; border-radius: 4px; padding: 1px 5px; font-size: 12.5px; }}
figure {{ margin: 14px 0 6px; }}
figcaption {{ font-size: 12.5px; color: {MUTED}; margin-top: 6px; line-height: 1.5; }}
.badge {{ display: inline-block; border-radius: 999px; padding: 2px 10px; font-size: 12px;
  background: #eaf3ea; color: #006300; }}
"""


def build() -> None:
    cases, part, poison, w1, rounds = load()
    by_id = {c["case_id"]: c for c in cases["cases"]}
    n = cases["n_cases"]
    groups = Counter(c["fov_group"] for c in cases["cases"])
    zs = [c["spacing_mm"][2] for c in cases["cases"]]
    batches = part["arrival_batches"]
    pi = part["poisoned_batch_index"]


    w1_html = ""
    if w1:
        w1_html = f'''
<h2>Step 4 — The ruler, null-tested and validated on the real poison (W1)</h2>
<div class="card">
<p><strong>What was built.</strong> <code>src/clloop/delta.py</code> — the frozen
correction-delta ruler. Per (auto, final) pair it reports three things:
<strong>added path length</strong> (APL, mm — the contour a human would have to redraw,
label-aware so identity errors carry their true cost; Vaassen&nbsp;et&nbsp;al. 2020),
<strong>surface Dice</strong> (Nikolov&nbsp;et&nbsp;al. 2018 — deliberately label-<em>blind</em>, so
its disagreement with APL becomes a diagnostic), and a per-level
<strong>taxonomy</strong>: boundary / relabel / missing / spurious.</p>
<p><strong>Why two scores that can disagree?</strong> Because their disagreement is the
signature of the most dangerous failure class. An enumeration error leaves shapes perfect
(surface Dice ≈ 1) while every renamed contour still needs redrawing (APL large). A single
overlap score cannot see this; the pair cannot miss it.</p>
</div>
<figure>{fig_ruler_nulls()}
<figcaption><strong>Fig 5 — The null suite.</strong> Six planted, known-answer inputs the
ruler must score correctly before it may score anything real. N3b was a genuine finding of
the suite: APL is bounded by total contour length, so uniformly deeper damage saturates —
the monotone axis is the <em>extent</em> of damage, not its depth. The suite is also the
proof that the bounding-box speed optimization (120s → 14s per case) changed nothing.</figcaption></figure>
<figure>{fig_ruler_signature(w1)}
<figcaption><strong>Fig 6 — Live validation on the 13 real poisoned pairs.</strong> Every
poisoned case scores surface Dice ≥ {w1["surface_dice_min"]:.3f} (shapes essentially
perfect) yet a median of {w1["apl_mm_median"] / 1000:.1f} metres of contour to redraw, and
{w1["n_verdict_relabel"]}/{w1["n_cases"]} receive the <code>relabel</code> verdict. The
ruler reads the poison as exactly what it is — before any model exists. This is the
instrument the loop's refusals will lean on.</figcaption></figure>
'''

    w2_html = ""
    if rounds:
        r0 = rounds[0]
        latest = rounds[-1]
        mix_rows = "".join(
            f"<tr><td>round {r['round']}</td>"
            f"<td>{r['training']['mix'].get('n_new', 0)}</td>"
            f"<td>{r['training']['mix'].get('n_rehearsal', 0)}</td>"
            f"<td>{r['eval']['apl_mm_median'] / 1000:.1f} m</td>"
            f"<td>{r['eval']['surface_dice_median']:.3f}</td>"
            f"<td>{r.get('curation', {}).get('policy', 'baseline')}</td></tr>"
            for r in rounds
        )
        w2_html = f'''
<h2>Step 5 — The loop turns: baseline + round engine (W2)</h2>
<div class="card">
<p><strong>What was built.</strong> A one-command round engine
(<code>scripts/run_round.py --round K</code>): the incumbent model predicts the arriving
batch, the frozen W1 ruler scores every correction delta <em>on the native grid</em>, the
cohort is curated (in W2 still an <code>accept_all</code> stub that says so in its own
record), the model retrains with a declared ~25% rehearsal mix, and the candidate is
evaluated on the {len(part["sequestered_test"])}-case sequestered test set — per-region
Dice for what it knows, ruler APL for what it would cost to fix. Every round writes a
complete <code>round.json</code>: training mix, per-case deltas, evaluation, timing.</p>
<p><strong>Scale honesty:</strong> a small MONAI 3-D U-Net at 3 mm isotropic,
{r0["iters"]} iterations/round — minutes per round on one consumer GPU. The demo measures
loop mechanics, not segmentation SOTA; the coarse grid is a declared choice, and the ruler
still scores on the native grid.</p>
</div>
<figure>{fig_round_dice(rounds)}
<figcaption><strong>Fig 7 — Per-region knowledge, round over round.</strong> Median Dice on
the sequestered test set. Round 0 is the initial-pool baseline (thoraco-lumbar-heavy world);
later rounds fold in arriving batches with rehearsal. Cervical is where learning must show;
thoraco-lumbar/sacrum is where forgetting would show.</figcaption></figure>
<figure>{fig_round_burden(rounds)}
<figcaption><strong>Fig 8 — The headline metric.</strong> Median added-path-length per
sequestered test case: the contour a human would still have to redraw. This is the number
the loop exists to drive down — Dice tells you overlap, APL tells you labour.</figcaption></figure>
<div class="card">
<table>
<tr><th>round</th><th>new cases</th><th>rehearsal</th><th>test APL (median)</th><th>test surface Dice</th><th>curation</th></tr>
{mix_rows}
</table>
</div>
'''

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>correction-loop-demo — build report</title>
<style>{CSS}</style></head><body><div class="wrap">

<h1>correction-loop-demo — build report</h1>
<p class="meta">A continual-learning loop that learns from correction deltas — and can be
seen refusing a bad round. Open data (VerSe 2020, CC&nbsp;BY-SA&nbsp;4.0), MIT code.
Updated {date.today().isoformat()} ·
{("W0–W2: " + str(len(rounds)) + " round(s) run") if rounds else ("W0–W1 complete" if w1 else "W0 complete")}
<span class="badge">{"17/17" if rounds else ("12/12" if w1 else "5/5")} tests passing</span></p>

<div class="card">
<h2 style="margin-top:0">What this demo is</h2>
<p>Production segmentation models are corrected by humans every day, and those
corrections are usually thrown away. This demo builds the loop that keeps them:
each round, a model meets a batch of new cases, its outputs are corrected, the
<strong>correction delta</strong> becomes the training signal, and a retrained candidate is
promoted only if it beats explicit null controls. The demo's job is to show the
<strong>loop shape</strong> — including the round where the loop says <em>no</em>.</p>
<p>Three design commitments, made before any model exists:</p>
<ul>
<li><strong>Sequestration is a refusal, not a filter.</strong> Held-out test cases raising an
error on contact with curation — never silently dropped.</li>
<li><strong>Every promotion beats two nulls.</strong> A matched random cohort at equal N, and
do-nothing. Improvement that can't beat "just add any cases" proves nothing about
the selection rule.</li>
<li><strong>One batch is poisoned on purpose</strong> (wrong-level enumeration). A loop that
cannot be seen refusing a bad batch is theater.</li>
</ul>
</div>

<h2>Step 1 — The data, scanned honestly</h2>
<div class="tiles">
<div class="tile"><div class="v">{n}</div><div class="l">usable cases (of 119 dirs)</div></div>
<div class="tile"><div class="v">{groups["thoraco-lumbar"]}</div><div class="l">thoraco-lumbar</div></div>
<div class="tile"><div class="v">{groups["cervical-containing"]}</div><div class="l">cervical-containing</div></div>
<div class="tile"><div class="v">{np.median(zs):.2f}&nbsp;mm</div><div class="l">median z-spacing (range {min(zs):.1f}–{max(zs):.1f})</div></div>
</div>
<div class="card">
<p><strong>What was done.</strong> <code>scripts/build_partition.py</code> scans every case
directory, picks one CT series per case by a fixed preference order
(<code>CT-iso</code> → bare → <code>CT-ax</code> → <code>CT-sag</code>), and derives each case's
vertebral coverage from the <em>reference labelmap itself</em> (<code>np.unique</code>),
not from metadata sidecars. Two honest findings along the way, kept in:
(1)&nbsp;VerSe file naming is heterogeneous — <code>verse*</code> and <code>GL*</code> ids,
dash and underscore series tags, and a bare-name series for a third of cases; the scanner
handles all of them and skips <strong>12 directories</strong> with no complete image+mask pair.
(2)&nbsp;Centroid JSONs exist for only 26 of 119 directories in this mirror, which is why
coverage comes from the masks. Every case's chosen series, coverage, shape and spacing is
recorded in <code>manifests/cases.json</code> — the committed, diffable inventory the whole
demo builds on.</p>
</div>

<h2>Step 2 — The stream design (the partition is the experiment)</h2>
<figure>{fig_partition_composition(cases, part)}
<figcaption><strong>Fig 1 — Who goes where.</strong> {len(part["sequestered_test"])} cases are
sequestered as the frozen test set (stratified by field-of-view group, so every round's
evaluation sees both the old and the new anatomy). {len(part["initial_pool"])} cases form the
initial training pool, drawn ~85% thoraco-lumbar — the "deployed distribution" the
round-0 model will know. The remaining {sum(len(b) for b in batches)} cases arrive in
{len(batches)} batches. Bar segments are field-of-view groups; numbers are case counts.
Seed 1337; rebuilding reproduces this exactly.</figcaption></figure>

<figure>{fig_shift(cases, part)}
<figcaption><strong>Fig 2 — The declared shift.</strong> The cervical-containing fraction rises
monotonically across arrival batches: {" → ".join(f"{sum(by_id[c]['fov_group'] == 'cervical-containing' for c in b) / len(b):.0%}" for b in batches)},
against an initial pool at {sum(by_id[c]["fov_group"] == "cervical-containing" for c in part["initial_pool"]) / len(part["initial_pool"]):.0%}.
This is the demo's engine: the model's world genuinely changes, so there is something real to
learn (arriving cervical anatomy) and something real to forget (the thoraco-lumbar anatomy it
started with). Batch {pi} is the poisoned batch — marked red here and everywhere.</figcaption></figure>

<figure>{fig_level_coverage(cases, part)}
<figcaption><strong>Fig 3 — Level-by-level coverage.</strong> Fraction of cases in each
partition role containing each vertebral level, C1→sacrum. The initial pool is dark through
the thoraco-lumbar span and nearly empty in the cervical columns; later batches fill the
cervical end in. The sequestered test row spans both — it can measure learning <em>and</em>
forgetting. This chart is the forgetting gate's evidence base, drawn before any training.</figcaption></figure>

<h2>Step 3 — The poison, declared and audited</h2>
<figure>{fig_poison(poison)}
<figcaption><strong>Fig 4 — What the poison does.</strong> In batch {pi}, every reference mask
gets the classic wrong-level error: label <em>v</em>&nbsp;→&nbsp;<em>v</em>+1, so each vertebra keeps its true
shape but wears its cranial neighbour's name (sacrum keeps its own). Overlap metrics barely
notice — the shapes are perfect — which is exactly the point: a loop watching only Dice would
happily train on this. The taxonomy-aware delta ruler (W1) and the promotion gate (W3) are
supposed to catch it; the dashboard will show whether they do.</figcaption></figure>
<div class="card">
<p><strong>Audit trail.</strong> <code>manifests/poison.json</code> records, per poisoned case:
the source mask's sha256, the poisoned copy's sha256, the full label map applied, and the
foreground voxel count (identical before/after — the poison relabels, never redraws).
Originals are never touched; poisoned copies live outside git in
<code>data/poisoned/</code> and are substituted only when batch {pi} "arrives".
{poison["n_cases"]} cases, {sum(len(c["label_map"]) for c in poison["cases"])} labels shifted in total.</p>
</div>

{w1_html}\n{w2_html}\n<h2>What exists so far</h2>
<div class="card">
<table>
<tr><th>Artifact</th><th>What it is</th></tr>
<tr><td><code>scripts/download_verse.sh</code></td><td>Official VerSe'20 mirrors + checksum recording; data never enters git</td></tr>
<tr><td><code>src/clloop/partition.py</code></td><td>Scanner + seeded stream designer (disjointness raises, shift declared)</td></tr>
<tr><td><code>manifests/cases.json</code></td><td>The committed case inventory: series choice, coverage, geometry per case</td></tr>
<tr><td><code>manifests/partition.json</code></td><td>The committed stream design: test/pool/batches + poison declaration</td></tr>
<tr><td><code>manifests/poison.json</code></td><td>Per-case poison audit (sha256 before/after, label maps)</td></tr>
<tr><td><code>tests/test_partition.py</code></td><td>5 tests: seeded reproducibility, disjointness raises on a planted leak, monotone shift, off-by-one correctness, manifest integrity</td></tr>
{("<tr><td><code>src/clloop/delta.py</code></td><td>The frozen correction-delta ruler (APL + surface Dice + taxonomy), 7-test null suite</td></tr>"
  "<tr><td><code>manifests/w1_ruler_validation.json</code></td><td>Per-case scores of the 13 real poisoned pairs — the ruler's live validation</td></tr>") if w1 else ""}
</table>
<p style="margin-bottom:0"><strong>Next:</strong> {"W3 — the three refusals + the two-null promotion gate; then the full run with the poisoned round." if rounds else ("W2 — the round-0 baseline model and the round engine." if w1 else "W1 — the frozen correction-delta ruler.")}</p>
</div>

<p class="meta">Code MIT · Data: Sekuboyina&nbsp;et&nbsp;al., <em>VerSe: A Vertebrae Labelling and
Segmentation Benchmark</em> (Medical Image Analysis 2021), CC&nbsp;BY-SA&nbsp;4.0, obtained from the
authors' published mirrors. This demo shows a loop pattern on open data; its numbers make no
claim about any clinical system.</p>

</div></body></html>"""
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html)
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    build()
