"""Regenerate every figure of the S2WISH NeurIPS-template technical report from the repo's raw files.

Run from anywhere:  python docs/publication/neurips/make_figures.py
Output: vector PDFs in docs/publication/neurips/figures/ (pdf.fonttype 42, no in-figure titles).
Nothing here hardcodes a number that exists in a raw file; the only literals are layout choices and the
FACTS cross-checks, which are parsed from docs/publication/FACTS_2026-10-16.md.

Data sources
  fig_platform_runs.pdf    experiments/platform_submissions/*.json (repeats_*, 2026-10-15_submission_history,
                           dated per-run files), deduplicated by Codabench run id
  fig_programme_forest.pdf experiments/runs/preregistration/exp_00{42,46,52,53,57}_*.json
  fig_rho_value.pdf        experiments/runs/e6/e6.json (same file sections_v2.e6() reads)
  fig_excess_map.pdf       experiments/pattern_mining/meta_2026-10-16/m10_excess_map.csv
  fig_platform_replay.pdf  experiments/pattern_mining/meta_2026-10-16/m3_grid.csv
  fig_budget_curves.pdf    experiments/runs/phase2/phase2_table.txt (L0, R2), experiments/runs/e5/real.json (v0, R2),
                           experiments/runs/preregistration/exp_0044_tempering_replay_2026-10-14.json (v1_3, R2 baseline)
"""
from __future__ import annotations

import csv
import glob
import json
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "figures"
PLAT = ROOT / "experiments" / "platform_submissions"
PREREG = ROOT / "experiments" / "runs" / "preregistration"
RUNS = ROOT / "experiments" / "runs"
META = ROOT / "experiments" / "pattern_mining" / "meta_2026-10-16"
FACTS = ROOT / "docs" / "publication" / "FACTS_2026-10-16.md"

FULL_W, HALF_W, MAX_H = 5.5, 2.7, 2.4
BUDGETS = [0, 1, 3, 7, 15, 31]

# Print palette: dataviz reference categorical slots 1-3 (blue, orange, aqua) + neutral inks.
# Every series also carries a distinct marker / line style, so identity never rests on colour alone.
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#d9d8d4"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.02, "figure.constrained_layout.w_pad": 0.02,
})


def J(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def one(pattern):
    hits = sorted(glob.glob(str(PREREG / pattern)))
    assert len(hits) == 1, (pattern, hits)
    return J(hits[0])


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", OUT / name)


def fm(v, f="+.2f"):
    """Format with a typographic minus sign."""
    return format(v, f).replace("-", "−")


def ygrid(ax):
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ FACTS cross-check helpers
def facts_programme():
    """Parse the programme table rows '| 0042 ... | +1.68, CI [+1.08, +2.30] | ...' from FACTS."""
    out = {}
    num = r"([+\-−]?\d+\.\d+)"
    for line in FACTS.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\|\s*(\d{4})\b.*?" + num + r",\s*CI \[" + num + r",\s*" + num + r"\]", line)
        if m:
            out[m.group(1)] = tuple(float(x.replace("−", "-")) for x in m.group(2, 3, 4))
    return out


# ------------------------------------------------------------------ figure 1: platform runs
# Plain-language column names (archive file -> column). Archive internals, checked 2026-10-18:
#   v1_3_named_facets.zip has the same model.py as v1_2.zip and the same parameters except facets "named" vs true,
#   both of which the code treats as "all fields"; it is therefore a functionally identical re-package of v1_2 and is
#   plotted in the item-group column with an open marker. v1_5 = metadata/IRT hybrid, v1_6 = sparse-label damping of
#   early updates, v1_7 = adaptive cap on early jumps, v1_8 = mixture prior (docs/research_notes/OPTIMIZATION_LOG.md,
#   model.py diffs). The archive-to-name map is also printed in the report's appendix.
ARCHIVE_COLUMN = OrderedDict([
    ("v1.zip", "first\nmodel"), ("v1_1.zip", "hedged-\ncentre"), ("v1_2.zip", "item-group\nmodel"),
    ("v1_3_named_facets.zip", "item-group\nmodel"), ("v1_3_named_facets_noguard.zip", "final\nmodel"),
    ("v1_5_hybrid_experimental.zip", "metadata\nhybrid"), ("v1_6_damped_facets.zip", "damped\nupdates"),
    ("v1_7_adaptive_facets.zip", "capped\nupdates"), ("v1_8_ext10.zip", "mixture\nprior"),
])
# Archives without a column here (an explored variant the paper does not report) are not plotted.
REPACKAGE = {"v1_3_named_facets.zip"}
COLUMN_GROUP = [("development ladder", 0, 3), ("rejected variants", 4, 7)]
ARCHIVE_LABEL = ARCHIVE_COLUMN  # kept for the print-out below


def platform_runs():
    """Return {run_key: (archive_basename, alc)} deduplicated by Codabench run id."""
    runs = {}

    def add(rid, archive, alc):
        archive = Path(archive).name
        if rid in runs:
            assert runs[rid][0] == archive and abs(runs[rid][1] - alc) < 5e-7, (rid, runs[rid], archive, alc)
        runs[rid] = (archive, float(alc))

    for f in sorted(PLAT.glob("repeats_*.json")) + sorted(PLAT.glob("*_repeats.json")):
        d = J(f)
        for r in d["runs"]:
            add(r["id"], d["zip"], r["alc"])
    for r in J(PLAT / "2026-10-15_submission_history.json")["runs"]:
        add(r["id"], r["file"], r["alc"])
    for f in sorted(PLAT.glob("2026-*.json")):
        d = J(f)
        arch = d.get("archive") or d.get("zip")
        if arch is None:
            continue  # submission history (no single archive), handled above
        if "runs" in d and "zip" in d and "pairs" not in d:
            continue  # pure repeat file, handled above
        if "runs" in d:
            for r in d["runs"]:
                add(r["id"], arch, r["alc"])
        if "codabench_id" in d:
            add(d["codabench_id"], arch, d["alc"])
        for r in d.get("other_runs", []):
            add(r["id"], arch, r["alc"])
        if "runs" not in d and "codabench_id" not in d and "pairs" in d:
            # per-pair feedback file without a run id: run ALC = mean of pair ALCs.
            alc = float(np.mean([p["alc"] for p in d["pairs"]]))
            match = [k for k, (a, v) in runs.items() if a == Path(arch).name and abs(v - alc) < 5e-7]
            if match:
                continue  # already logged under its Codabench id (e.g. v1_1 = 955381)
            add(f"{Path(arch).name}@{d['date']}", arch, alc)
    return runs


def fig_platform_runs():
    runs = platform_runs()
    by = OrderedDict((k, []) for k in ARCHIVE_LABEL)
    for rid, (arch, alc) in runs.items():
        if arch not in ARCHIVE_COLUMN:
            continue
        by[arch].append(alc)
    by = OrderedDict((k, v) for k, v in by.items() if v)

    # FACTS checks (per archive, before any column grouping)
    ft = FACTS.read_text(encoding="utf-8")
    v13 = by["v1_3_named_facets_noguard.zip"]; v11 = by["v1_1.zip"]
    assert len(v13) == 4 and f"Mean **{np.mean(v13):.4f}**" in ft, np.mean(v13)
    assert f"v1_1: {len(v11)} runs, mean {np.mean(v11):.4f}" in ft, (len(v11), np.mean(v11))

    # group archives into plain-named columns; keep (value, is_repackage) per run
    cols = OrderedDict()
    for arch, vals in by.items():
        cols.setdefault(ARCHIVE_COLUMN[arch], []).extend((v, arch in REPACKAGE) for v in vals)
    names = list(cols)
    assert len(names) == 8, names

    fig, ax = plt.subplots(figsize=(FULL_W, 2.25))
    rng = np.random.default_rng(7)
    for i, name in enumerate(names):
        pts = sorted(cols[name])
        v = np.array([p[0] for p in pts]); rep = np.array([p[1] for p in pts])
        x = i + (rng.uniform(-0.13, 0.13, len(v)) if len(v) > 1 else np.zeros(1))
        ax.scatter(x[~rep], v[~rep], s=18, marker="o", facecolor=C1, edgecolor="white", linewidth=0.6, zorder=3)
        ax.scatter(x[rep], v[rep], s=18, marker="o", facecolor="white", edgecolor=C1, linewidth=1.0, zorder=3)
        m = v.mean()
        ax.plot([i - 0.28, i + 0.28], [m, m], color=INK, lw=1.4, zorder=4, solid_capstyle="butt")
        ax.text(i, 0.1415, f"n={len(v)}", ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=6.8)
    ax.set_xlim(-0.6, len(names) - 0.4)
    lo = min(min(v) for v in by.values()); hi = max(max(v) for v in by.values())
    top = max(0.262, hi + 0.024)
    ax.set_ylim(0.140, top)
    assert lo > 0.1445
    # group brackets along the top of the axes
    for label, a, b in COLUMN_GROUP:
        yb = top - 0.004
        ax.plot([a - 0.35, b + 0.35], [yb, yb], color=INK2, lw=0.6)
        ax.text((a + b) / 2, yb + 0.001, label, ha="center", va="bottom", fontsize=6.5, color=INK2)
        if a > 0:
            ax.axvline(a - 0.5, color=GRID, lw=0.6, zorder=0)
    ax.set_ylabel("Brier ALC (lower is better)")
    ygrid(ax)
    h1 = ax.scatter([], [], s=18, marker="o", facecolor=C1, edgecolor="white", label="one formative run")
    h1b = ax.scatter([], [], s=18, marker="o", facecolor="white", edgecolor=C1, linewidth=1.0,
                     label="identical re-package")
    h3, = ax.plot([], [], color=INK, lw=1.4, label="column mean")
    fig.legend(handles=[h1, h1b, h3], loc="outside upper center", ncol=3, handletextpad=0.3, columnspacing=1.2)
    save(fig, "fig_platform_runs.pdf")
    n = sum(len(v) for v in by.values())
    print("  figure 1 runs:", n, {k.replace("\n", " "): len(v) for k, v in cols.items()})
    return n


# ------------------------------------------------------------------ teaser: the ladder in one picture
def fig_teaser():
    """Local ALC (one subject per benchmark, 4 LOBO folds, 10 seeds) for the official baseline, Beta-binomial
    shrinkage and the final model, with the local pair-constant floor a(1-a). Sources:
    experiments/runs/review_round1/ladder_intervals.json (R2 alc) and floor_ceiling.json (local_R2 mean)."""
    lad = J(RUNS / "review_round1" / "ladder_intervals.json")["R2"]["alc"]
    floor = J(RUNS / "review_round1" / "floor_ceiling.json")["local_R2"]["mean"]
    base, v0, fin = lad["baseline"], lad["v0"], lad["final"]
    assert (round(base, 4), round(v0, 4), round(fin, 4), round(floor, 4)) == (0.2451, 0.2114, 0.2088, 0.1797), \
        (base, v0, fin, floor)
    gain_shrink, gain_total = 1e3 * (base - v0), 1e3 * (base - fin)
    print(f"  teaser: shrinkage {gain_shrink:.1f} of {gain_total:.1f} (x1e-3); floor {floor:.4f}")

    rows = [("official baseline", base, MUTED),
            ("Beta-binomial shrinkage", v0, C2),
            ("final model", fin, C1)]
    FS = 8  # >= 8 pt at print size (the figure is placed at its native width)
    fig, ax = plt.subplots(figsize=(FULL_W, 1.65))
    y = np.arange(len(rows))[::-1]
    x0 = 0.170
    for yi, (lab, v, col) in zip(y, rows):
        ax.barh(yi, v - x0, left=x0, height=0.5, color=col, edgecolor="none")
        ax.text(v + 0.0007, yi, f"{v:.4f}", va="center", ha="left", fontsize=FS, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=FS)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=FS)
    ax.axvline(floor, color=INK, lw=0.9, ls=(0, (3, 2)))
    ax.text(floor + 0.0006, -0.68, f"pair-constant floor {floor:.4f} (item knowledge needed below)",
            ha="left", va="center", fontsize=FS, color=INK)
    # arrow in the gap between the baseline and shrinkage bars: the shrinkage share of the gain
    yb = (y[0] + y[1]) / 2
    ax.annotate("", xy=(v0, yb), xytext=(base, yb),
                arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.8, shrinkA=0, shrinkB=0))
    # its label sits inside the baseline bar, directly above the arrow
    ax.text(base - 0.0015, y[0], f"shrinkage: {gain_shrink:.1f} of {gain_total:.1f}" + r" $\times10^{-3}$",
            ha="right", va="center", fontsize=FS, color="white")
    # small bracket in the gap between the shrinkage and final-model bars: what the refinements add
    yr = (y[1] + y[2]) / 2
    ax.annotate("", xy=(fin, yr), xytext=(v0, yr),
                arrowprops=dict(arrowstyle="|-|", color=INK2, lw=0.8, shrinkA=0, shrinkB=0,
                                mutation_scale=2.5))
    ax.text(v0 + 0.0010, yr, f"refinements: {gain_total - gain_shrink:.1f}", ha="left", va="center",
            fontsize=FS, color=INK)
    ax.set_xlim(x0, 0.2560); ax.set_ylim(-0.95, y.max() + 0.45)
    ax.set_xlabel("local ALC, one subject per benchmark (lower is better)", fontsize=FS)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    save(fig, "fig_teaser.pdf")


# ------------------------------------------------------------------ figure 2: programme forest
def fig_programme_forest():
    facts = facts_programme()
    e42 = one("exp_0042_*.json")["results"]["R2"]["lexical"]["d"]
    e46 = one("exp_0046_gpirt_2026*.json")["decisions"]["R2"]["gpirt"]
    e52 = one("exp_0052_*.json")["regimes"]["R2"]["brier"]["lambda_corrected"]["d"]
    e53 = one("exp_0053_*.json")["regimes"]["R2"]["brier"]["Tfit"]["d"]
    e57 = one("exp_0057_*.json")
    rows = [
        ("0042", "lexical residual smoothing", e42["pt"], e42["lo"], e42["hi"]),
        ("0046", "gp-IRT (our reimplementation)", e46["mean_delta_alc_x1e3"], *e46["pair_bootstrap_95ci_x1e3"]),
        ("0052", "shrinkage correction", e52["pt"], e52["lo"], e52["hi"]),
        ("0053", "fitted temperature", e53["pt"], e53["lo"], e53["hi"]),
        ("0057", "mixture prior", e57["delta_alc_x1e3"], *e57["pair_bootstrap_95ci_x1e3"]),
    ]
    disagree = []
    for k, _, pt, lo, hi in rows:
        got = tuple(round(v, 2) for v in (pt, lo, hi))
        if k not in facts or any(abs(a - b) > 0.0051 for a, b in zip(got, facts[k])):
            disagree.append((k, got, facts.get(k)))
    for k, got, want in disagree:
        print(f"  WARNING FACTS mismatch EXP-{k}: raw {got} vs FACTS {want}")
        assert want is not None and all(abs(a - b) <= 0.0101 for a, b in zip(got, want)), (k, got, want)

    fig, ax = plt.subplots(figsize=(FULL_W, 1.5))
    y = np.arange(len(rows))[::-1]
    for yi, (_, lab, pt, lo, hi) in zip(y, rows):
        ax.plot([lo, hi], [yi, yi], color=INK, lw=1.0, solid_capstyle="butt")
        ax.plot([lo, lo], [yi - .12, yi + .12], color=INK, lw=0.8)
        ax.plot([hi, hi], [yi - .12, yi + .12], color=INK, lw=0.8)
        ax.scatter([pt], [yi], s=22, marker="s", color=C1, edgecolor="white", linewidth=0.5, zorder=3)
        ax.text(hi + 0.25, yi, f"{fm(pt)} [{fm(lo)}, {fm(hi)}]", va="center", fontsize=6.5, color=INK2)
    ax.axvline(0, color=INK2, lw=0.7)
    bar = -1.0
    ax.axvline(bar, color=C2, lw=0.9, ls=(0, (3, 2)))
    ax.text(bar - 0.15, y.max() + 0.55, "adoption bar −1.0", ha="right", va="center", fontsize=6.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([r[1] for r in rows], fontsize=7)
    ax.set_ylim(-0.6, len(rows) - 0.2)
    ax.set_xlim(-3.2, max(r[4] for r in rows) + 5.6)
    ax.set_xlabel("paired ΔALC ×10⁻³ against the final model (+ = worse)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    save(fig, "fig_programme_forest.pdf")
    return rows


# ------------------------------------------------------------------ figure 3: value of rho
def fig_rho_value():
    e = J(RUNS / "e6" / "e6.json")
    A = e["arms"]
    rhos = [str(r) for r in e["config"]["rhos"]] + ["1.0"]
    pts = [(float(r), A[f"item({r})"]["d"]["pt"], A[f"item({r})"]["d"]["lo"], A[f"item({r})"]["d"]["hi"]) for r in rhos]
    x = np.array([p[0] for p in pts]); yv = np.array([p[1] for p in pts])
    c = float(np.sum(x * x * yv) / np.sum(x ** 4))  # least-squares c in Δ = c·ρ² (as sections_v2.e6)
    print(f"  E6 fit c = {c:.2f}")
    fig, ax = plt.subplots(figsize=(HALF_W, 2.1))
    xx = np.linspace(0, 1, 101)
    # E6 arm: research engine cfg facets="named", guard=0 (printed below; see e6.json "config")
    print("  E6 engine cfg:", {k: e["config"]["cfg"][k] for k in ("facets", "guard")})
    ax.plot(xx, c * xx ** 2, color=INK2, lw=1.0, ls=(0, (3, 2)), label=f"{fm(c, '.0f')}·$\\rho_S^2$ fit")
    ax.errorbar(x, yv, yerr=[yv - np.array([p[2] for p in pts]), np.array([p[3] for p in pts]) - yv],
                fmt="o", ms=4, color=C1, mec="white", mew=0.5, ecolor=INK, elinewidth=0.8, capsize=2,
                label="measured, 95% CI", zorder=3)
    ax.axhline(0, color=INK2, lw=0.6)
    for i, (xi, yi) in enumerate(zip(x, yv)):
        # the first two points (rho .2, .3) sit close together near zero: label the first below-left
        off, ha, va = ((-4, -3), "right", "top") if i == 0 else ((5, 2), "left", "bottom")
        ax.annotate(fm(yi, "+.1f"), (xi, yi), xytext=off, textcoords="offset points", ha=ha, va=va,
                    fontsize=6.5, color=INK2)
    ax.set_xlim(-0.03, 1.12); ax.set_ylim(min(p[2] for p in pts) - 6, 6)
    ax.set_xticks([0, 0.2, 0.3, 0.5, 0.7, 1.0]); ax.set_xticklabels(["0", ".2", ".3", ".5", ".7", "1"])
    ax.set_xlabel("signal quality $\\rho_S$ (Spearman)")
    ax.set_ylabel("ΔALC ×10⁻³ (− = better)")
    ygrid(ax)
    ax.legend(loc="lower left", handlelength=1.6)
    save(fig, "fig_rho_value.pdf")
    return c


# ------------------------------------------------------------------ figure 4: excess map
def fig_excess_map():
    rows = list(csv.DictReader(open(META / "m10_excess_map.csv", encoding="utf-8")))
    M = 1e3 * np.array([[float(r[f"B{b}"]) for b in BUDGETS] for r in rows])
    N = [int(r["n_slots"]) for r in rows]
    lab = [r["acc_bin"].replace("a<=", "≤").replace("-", "–") for r in rows]
    fig, ax = plt.subplots(figsize=(HALF_W, 1.9))
    vmax = M.max()
    im = ax.imshow(M, cmap="Blues", vmin=min(0, M.min()), vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, fm(M[i, j], ".1f"), ha="center", va="center", fontsize=6.5,
                    color="white" if M[i, j] > 0.55 * vmax else INK)
    ax.set_xticks(range(len(BUDGETS))); ax.set_xticklabels([str(b) for b in BUDGETS])
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{l} ({n})" for l, n in zip(lab, N)])
    ax.set_xlabel("labels revealed (budget)"); ax.set_ylabel("pair accuracy a (slots)")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("excess ×10⁻³", fontsize=7); cb.ax.tick_params(labelsize=6.5, length=2); cb.outline.set_linewidth(0.4)
    save(fig, "fig_excess_map.pdf")
    return M.sum(), sum(N)


# ------------------------------------------------------------------ figure 5: platform replay
def fig_platform_replay():
    rows = list(csv.DictReader(open(META / "m3_grid.csv", encoding="utf-8")))
    g = [r for r in rows if r["tau_mult"] and float(r["centre_shift"]) == 0.0 and not r["variant"].startswith("v1_8")]
    tms = sorted({float(r["tau_mult"]) for r in g})
    styles = [(0.5, MUTED, "v", ":"), (1.0, INK, "o", "-"), (2.0, C1, "s", "--"), (3.0, C2, "^", "-.")]
    npairs = {int(r["n_pairs"]) for r in rows}
    assert len(npairs) == 1
    fig, ax = plt.subplots(figsize=(HALF_W, 2.1))
    for sm, col, mk, ls in styles:
        ys = [1e3 * float(next(r["d_alc"] for r in g if float(r["tau_mult"]) == t and float(r["scale_mult"]) == sm))
              for t in tms]
        ax.plot(tms, ys, color=col, marker=mk, ms=3.5, lw=1.0, ls=ls, mec="white", mew=0.4, label=f"prior scale ×{sm:g}")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xticks(tms); ax.set_xticklabels(["" if t == 1.25 else f"{t:g}" for t in tms])
    ax.minorticks_off()
    ax.set_xlabel("τ multiplier (log scale)")
    ax.set_ylabel("ΔALC ×10⁻³ vs shipped")
    ygrid(ax)
    ax.legend(loc="upper right", handlelength=2.2, ncol=1)
    save(fig, "fig_platform_replay.pdf")
    return npairs.pop()


# ------------------------------------------------------------------ figure 6: budget curves
def phase2_r2_l0():
    reg = None
    for line in (RUNS / "phase2" / "phase2_table.txt").read_text(encoding="utf-8").splitlines():
        m = re.match(r"### (R\d)", line)
        if m:
            reg = m.group(1)
        elif reg == "R2" and line.startswith("| L0 "):
            c = [x.strip() for x in line.strip("|").split("|")]
            return [float(x) for x in c[1:7]]
    raise RuntimeError("L0 R2 row not found")


def fig_budget_curves():
    l0 = phase2_r2_l0()
    v0 = J(RUNS / "e5" / "real.json")["R2"]["v0"]["curve"]
    v13 = one("exp_0044_*.json")["results"]["R2"]["baseline"]
    ft = FACTS.read_text(encoding="utf-8")
    assert f"Local R2 ALC is **{v13['alc']:.6f}**" in ft, v13["alc"]
    fig, ax = plt.subplots(figsize=(HALF_W, 2.1))
    xs = np.arange(len(BUDGETS))
    for ys, lab, col, mk, ls in [(l0, "official empirical mean", MUTED, "^", ":"),
                                 (v0, "v0 (Beta-binomial)", C2, "s", "--"),
                                 (v13["curve"], "v1_3", C1, "o", "-")]:
        ax.plot(xs, ys, color=col, marker=mk, ms=3.5, lw=1.1, ls=ls, mec="white", mew=0.4, label=lab)
    ax.set_xticks(xs); ax.set_xticklabels([str(b) for b in BUDGETS])
    ax.set_xlabel("labels revealed (budget)"); ax.set_ylabel("Brier, local R2")
    ygrid(ax)
    ax.legend(loc="upper right", handlelength=2.2, bbox_to_anchor=(1.0, 1.02))
    save(fig, "fig_budget_curves.pdf")


if __name__ == "__main__":
    fig_teaser()
    n = fig_platform_runs()
    fig_programme_forest()
    fig_rho_value()
    tot, slots = fig_excess_map()
    print(f"  excess total {tot:.2f}e-3 over {slots} slots")
    print("  replay pairs", fig_platform_replay())
    fig_budget_curves()
    print("TOTAL formative runs in figure 1:", n)
