"""Stage-2: all charts + their CSVs for internal_findings.md. Reads saved outputs only (runs in a few seconds).

python experiments/analysis/patterns/internal_plots.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from internal_stats import FOLDS, FS, NP, W, heterogeneity, load, primary

HERE = Path(__file__).resolve().parent
RUNS = HERE.parents[2] / "experiments" / "runs"
OK = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7", "grey": "#999999"}
FAMC = {"offset_prior": OK["verm"], "version": OK["grey"], "hyper_refit": OK["orange"], "shrink_temper": OK["blue"],
        "subject_prior": OK["green"], "item_facets": OK["purple"], "item_text": OK["sky"], "guard": OK["yellow"],
        "item_pooling": OK["black"], "acquisition": "#8c6d31", "control": "#bbbbbb"}
FOLDM = {"multi_swebench": "o", "real_webagents": "s", "researchcodebench": "^", "matharena": "D"}
FOLDC = {"multi_swebench": OK["blue"], "real_webagents": OK["green"], "researchcodebench": OK["orange"],
         "matharena": OK["verm"]}
SHORT = {"multi_swebench": "mswe", "real_webagents": "rwa", "researchcodebench": "rcb", "matharena": "math"}
BUD = [0, 1, 3, 7, 15, 31]
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 150})


def save(fig, name, df):
    fig.savefig(HERE / f"internal_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    df.to_csv(HERE / f"internal_{name}.csv", index=False)


def L(p):
    return json.loads((RUNS / p).read_text(encoding="utf-8"))


d = load()
p = primary(d)
p = pd.concat([p, p.apply(heterogeneity, axis=1)], axis=1)
dep = p[p.kind.isin(["deployable", "control"])].dropna(subset=FS).drop_duplicates(subset=["pt"] + FS).copy()
dep["n_neg"] = (dep[FS] < 0).sum(1)
dep["worst"] = dep[FS].max(1)


def ppred(r):
    x = np.array([r[c] for c in FS]); m = x.mean(); s = x.std(ddof=1)
    return float(stats.t.cdf(-m / (s * np.sqrt(1 + 1 / 4)), 3))


dep["P_new_neg"] = dep.apply(ppred, axis=1)
dep["passes_bar"] = (dep.pt <= -1) & (dep.hi < 0) & (dep.n_neg >= 3) & (dep.worst <= 0.5)
famorder = ["subject_prior", "item_facets", "shrink_temper", "acquisition", "item_pooling", "guard", "item_text",
            "offset_prior", "hyper_refit", "version", "control"]
dep["fo"] = dep.family.map({f: i for i, f in enumerate(famorder)})
dep = dep.sort_values(["fo", "pt"])

# ---------------------------------------------------------------- 1 forest
fig, ax = plt.subplots(figsize=(8, 12))
y = np.arange(len(dep))[::-1]
for yi, (_, r) in zip(y, dep.iterrows()):
    c = FAMC[r.family]
    if not pd.isna(r.hlo):
        ax.plot([r.hlo, r.hhi], [yi, yi], color=c, alpha=0.25, lw=5, solid_capstyle="butt")
    ax.plot([r.lo, r.hi], [yi, yi], color=c, lw=1.6)
    ax.plot(r.pt, yi, "|", color="k", ms=9)
    for f in FOLDS:
        ax.plot(r["f_" + f], yi, FOLDM[f], mfc="none", mec=FOLDC[f], ms=4, mew=0.8)
ax.axvline(0, color="k", lw=0.6); ax.axvline(-1, color=OK["green"], ls="--", lw=0.8)
ax.set_yticks(y); ax.set_yticklabels([f"{r.label} [{r.regime}]" for _, r in dep.iterrows()], fontsize=6.5)
ax.set_xlim(-12, 14)
ax.set_xlabel("paired Δ ALC vs own baseline (×1e-3; negative = better)")
med = np.median(dep.se_fold / dep.se_pair)
ax.set_title(f"All {len(dep)} tested real-data interventions: fold points (open markers) scatter far outside\n"
             f"the pair CIs (thin bars); between-fold SE = {med:.1f}× pair SE; median I² = {dep.I2.median():.2f}")
from matplotlib.lines import Line2D
h = [Line2D([], [], marker=FOLDM[f], ls="", mfc="none", mec=FOLDC[f], label=SHORT[f]) for f in FOLDS]
h += [Line2D([], [], color=FAMC[f], lw=3, label=f) for f in famorder]
h += [Line2D([], [], color=OK["green"], ls="--", label="adoption bar −1e-3")]
ax.legend(handles=h, fontsize=6, loc="lower right", ncol=2)
ax.annotate("wide pale band = hierarchical (fold-resampled) CI", (0.01, 0.005), xycoords="axes fraction", fontsize=6)
save(fig, "forest", dep[["label", "family", "regime", "pt", "lo", "hi", "hlo", "hhi"] + FS +
                         ["se_pair", "se_fold", "Q", "I2", "n_neg", "worst", "P_new_neg", "passes_bar"]])

# ---------------------------------------------------------------- 2 SE ratio / I2 by family
fig, ax = plt.subplots(figsize=(7, 4.2))
fams = [f for f in famorder if f in set(dep.family)]
for i, f in enumerate(fams):
    s = dep[dep.family == f]
    ax.scatter(np.full(len(s), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(s)), s.I2, color=FAMC[f],
               edgecolor="k", lw=0.4, s=30)
ax.set_xticks(range(len(fams))); ax.set_xticklabels(fams, rotation=35, ha="right")
ax.set_ylabel("I² of the 4 fold deltas vs pair noise (0 = homogeneous)")
ax.axhline(0.5, color=OK["grey"], ls=":", lw=0.8)
sp = dep[dep.family == "subject_prior"]
ax.set_title("Subject-ability priors are the only fold-homogeneous family;\n"
             f"every other family has I² > 0.45 (median overall {dep.I2.median():.2f})")
for _, r in sp.iterrows():
    if r.I2 < 0.2:
        ax.annotate(f"{r.arm} I²={r.I2:.2f}", (0, r.I2), xytext=(0.4, r.I2 + 0.05), fontsize=6,
                    arrowprops=dict(arrowstyle="-", lw=0.4))
save(fig, "i2_by_family", dep[["label", "family", "I2", "Q", "se_pair", "se_fold"]])

# ---------------------------------------------------------------- 3 matharena axis
r2 = dep[(dep.kind == "deployable") & (dep.regime == "R2")].copy()
r2["others"] = r2[FS[:3]].mean(1)
X = r2[FS].to_numpy(); U, S, Vt = np.linalg.svd(X, full_matrices=False)
share = S[0] ** 2 / np.sum(S ** 2)
rr = np.corrcoef(r2.others, r2.f_matharena)[0, 1]
fig, ax = plt.subplots(figsize=(6.2, 5.2))
for f in fams:
    s = r2[r2.family == f]
    if len(s):
        ax.scatter(s.others, s.f_matharena, color=FAMC[f], edgecolor="k", lw=0.4, s=34, label=f)
lim = 12; ax.plot([-lim, lim], [lim, -lim], color=OK["grey"], ls=":", lw=0.8)
ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel("mean Δ ALC on the three hard folds: mswe, rwa, rcb (×1e-3)")
ax.set_ylabel("Δ ALC on matharena, the only easy fold (×1e-3)")
ax.set_title(f"What helps matharena hurts the hard folds: r = {rr:.2f} over {len(r2)} R2 interventions;\n"
             f"PC1 (math vs rest) carries {share:.0%} of fold-delta variance")
for _, r in r2.iterrows():
    if abs(r.f_matharena) > 6 or abs(r.others) > 4:
        ax.annotate(r.arm, (r.others, r.f_matharena), fontsize=6, xytext=(3, 2), textcoords="offset points")
ax.legend(fontsize=6, loc="upper right")
save(fig, "fold_axis", r2[["label", "family", "others", "f_matharena"] + FS])

# ---------------------------------------------------------------- 4 fold composition: re-weight matharena (easy fold)
rows = []
for _, r in r2.iterrows():
    mo, ma = r.others, r.f_matharena
    w_star = mo / (mo - ma) if np.sign(mo) != np.sign(ma) and mo != ma else np.nan
    rows.append(dict(label=r.label, family=r.family, pt=r.pt, others=mo, matharena=ma, w_flip=w_star,
                     d_w25=0.25 * ma + 0.75 * mo, d_w50=0.5 * ma + 0.5 * mo, d_w75=0.75 * ma + 0.25 * mo))
fl = pd.DataFrame(rows)
fl = fl[(fl.pt.abs() > 0.3) | (fl.others.abs() > 1) | (fl.matharena.abs() > 1)].sort_values("matharena")
fig, ax = plt.subplots(figsize=(7.5, 7))
yy = np.arange(len(fl))[::-1]
for yi, (_, r) in zip(yy, fl.iterrows()):
    c = OK["blue"] if r.matharena < r.others else OK["verm"]
    ax.annotate("", xy=(r.d_w50, yi), xytext=(r.d_w25, yi), arrowprops=dict(arrowstyle="->", color=c, lw=1.3))
    ax.plot(r.d_w25, yi, "o", color="k", ms=3.5)
    ax.plot(r.d_w75, yi, "x", color=c, ms=4)
ax.axvline(0, color="k", lw=0.5); ax.axvline(-1, color=OK["green"], ls="--", lw=0.8)
ax.set_yticks(yy); ax.set_yticklabels([f"{l} (math {m:+.1f})" for l, m in zip(fl.label, fl.matharena)], fontsize=6.3)
ax.set_xlabel("Δ ALC (×1e-3) with matharena weight 1/4 (dot, = today) → 1/2 (arrow) → 3/4 (x)")
nb = int(((fl.d_w25 > 0) & (fl.d_w50 < 0)).sum()); nw = int(((fl.d_w25 < 0) & (fl.d_w50 > 0)).sum())
ax.set_title("Doubling the easy fold's weight reverses the sign of "
             f"{nb + nw} decisions ({nb} toward adoption, {nw} away):\n"
             "level-raising priors (v0, b_zero, E7 hyper) gain; G-centred mixtures (X1-bma, EXP-0045) lose")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], color=OK["blue"], label="better with more easy weight"),
                   Line2D([], [], color=OK["verm"], label="worse with more easy weight")], fontsize=7, loc="lower right")
save(fig, "flip_weight", fl)

# ---------------------------------------------------------------- 5 base-rate shift sweeps (E5 + phase3)
sw = L("e5/sweep.json")
deltas = [-3, -2, -1, 0, 1, 2, 3]
arms = ["v0", "v1", "X1-bma", "X1-ext10", "X1-ext20", "X1-t1", "X1-Gext10"]
rows = []
for a in arms:
    for dl in deltas:
        rows.append(dict(arm=a, delta=dl, d_vs_v1_1=1e3 * (sw["alc"][str(dl)]["R2"][a] - sw["alc"][str(dl)]["R2"]["v1_1"])))
ex = L("e5/extreme.json")
for a in arms:
    rows.append(dict(arm=a, delta=9, d_vs_v1_1=1e3 * (ex["0.98"]["alc"]["R2"][a] - ex["0.98"]["alc"]["R2"]["v1_1"])))
swd = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(7, 4.6))
cols = [OK["grey"], OK["black"], OK["verm"], OK["sky"], OK["blue"], OK["green"], OK["purple"]]
for a, c in zip(arms, cols):
    s = swd[(swd.arm == a) & (swd.delta <= 3)]
    ax.plot(s.delta, s.d_vs_v1_1, "-o", color=c, ms=3, label=a)
    e = swd[(swd.arm == a) & (swd.delta == 9)].d_vs_v1_1.iloc[0]
    ax.plot(4.2, e, "o", color=c, ms=4)
ax.axvspan(1, 2, color=OK["yellow"], alpha=0.25)
ax.text(1.05, ax.get_ylim()[1] * 0.85, "platform shift\n(acc .41→.72 ≈ +1.3 logit)", fontsize=6.5)
ax.axhline(0, color="k", lw=0.5)
ax.set_xticks(deltas + [4.2]); ax.set_xticklabels([str(x) for x in deltas] + ["rate .98\n(synthetic)"])
ax.set_xlabel("logit shift δ applied to the real folds (positive = easier benchmark)")
ax.set_ylabel("ALC − ALC(v1_1), R2 (×1e-3; negative = beats v1_1)")
v = swd.set_index(["arm", "delta"]).d_vs_v1_1
ax.set_title("E5 tail priors only beat v1_1 at δ ≥ +3 or near-ceiling rates; at the platform's\n"
             f"likely +1…+2 shift they tie (X1-ext20 {v['X1-ext20', 1]:+.1f}/{v['X1-ext20', 2]:+.1f}; v0 "
             f"{v['v0', 1]:+.1f}/{v['v0', 2]:+.1f})")
ax.legend(fontsize=6.5, ncol=2)
save(fig, "baserate_sweep", swd)

# ---------------------------------------------------------------- 6 budget profile heatmap
bp = p[p.kind.isin(["deployable", "control", "oracle_bound", "offline_bound"])].dropna(subset=[f"b{b}" for b in BUD])
bp = bp.drop_duplicates(subset=["pt"]).copy()
B = bp[[f"b{b}" for b in BUD]].to_numpy() * W
den = np.abs(B).sum(1, keepdims=True); den[den == 0] = 1
Sh = B / den
bp["share_B0B1"] = np.abs(Sh[:, :2]).sum(1)
bp["share_B7plus"] = np.abs(Sh[:, 3:]).sum(1)
bp["com"] = (np.abs(Sh) * np.arange(6)).sum(1)
for i, b in enumerate(BUD):
    bp[f"sh{b}"] = Sh[:, i]
bp["fo"] = bp.family.map({f: i for i, f in enumerate(famorder + ["oracle_item", "oracle_offline"])})
bp = bp.sort_values(["fo", "com"])
fig, ax = plt.subplots(figsize=(6.5, 11))
im = ax.imshow(bp[[f"sh{b}" for b in BUD]].to_numpy(), cmap="PuOr_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(6)); ax.set_xticklabels([f"B{b}" for b in BUD])
ax.set_yticks(range(len(bp))); ax.set_yticklabels([f"{r.family}: {r.arm} ({r.pt:+.2f})" for _, r in bp.iterrows()],
                                                  fontsize=6)
plt.colorbar(im, ax=ax, label="signed share of |weighted Δ| (orange = harm, purple = gain)", shrink=0.5)
fam_com = bp.groupby("family").agg(B0B1=("share_B0B1", "median"), B7p=("share_B7plus", "median"), n=("pt", "size"))
ax.set_xlabel("budget (revealed labels per pair)")
ax.set_title("Priors live at B0–B1, item signals at B7+: median |Δ| share in B0–B1 is\n"
             f"{fam_com.loc['subject_prior','B0B1']:.0%} (subject), {fam_com.loc['offset_prior','B0B1']:.0%} (offset) vs "
             f"{fam_com.loc['item_facets','B0B1']:.0%} (facets), {fam_com.loc['oracle_item','B0B1']:.0%} (item oracle)")
save(fig, "budget_profile", bp[["label", "family", "regime", "pt"] + [f"sh{b}" for b in BUD] +
                              ["share_B0B1", "share_B7plus", "com"]])
fam_com.reset_index().to_csv(HERE / "internal_budget_profile_family.csv", index=False)

# ---------------------------------------------------------------- 7 v1_3 loss by fold x budget + item-oracle gap
e1 = L("e1/e1_results.json"); e4 = L("e4/real.json"); e6 = L("e6/e6.json")
rows = []
for f in FOLDS:
    base = np.array(e1["R2"]["per_fold"][f]["v1_1|curve"])  # 5 seeds
    fac = np.array([e["per_fold"][f] for e in e4["R2"]["facets"]["d_budget"]]) / 1e3
    facn = np.array([e["per_fold"][f] for e in e4["R2"]["facets_named"]["d_budget"]]) / 1e3
    orc = np.array([e["per_fold"][f] for e in e6["arms"]["item(1.0)"]["d_budget"]]) / 1e3
    yb = e1["R2"]["per_fold"][f]["v1_1|B0|murphy"]["ybar"]
    for i, b in enumerate(BUD):
        v13 = base[i] + fac[i]; o = base[i] + facn[i] + orc[i]
        rows.append(dict(fold=f, budget=b, v1_3_brier=v13, item_oracle_brier=o, weighted=W[i] * v13,
                         oracle_gap_weighted=W[i] * (v13 - o), base_rate=yb, unc=yb * (1 - yb)))
lo = pd.DataFrame(rows)
fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.6)); fig.subplots_adjust(wspace=0.75)
M = lo.pivot(index="fold", columns="budget", values="v1_3_brier").loc[FOLDS]
G = lo.pivot(index="fold", columns="budget", values="oracle_gap_weighted").loc[FOLDS]
for ax, Mx, lab, cm in [(axs[0], M, "v1_3 Brier", "YlOrBr"), (axs[1], G * 1e3, "weighted gap to perfect-item oracle (×1e-3 ALC)", "Blues")]:
    im = ax.imshow(Mx.to_numpy(), cmap=cm, aspect="auto")
    ax.set_xticks(range(6)); ax.set_xticklabels([f"B{b}" for b in BUD]); ax.set_yticks(range(4))
    ax.set_yticklabels([f"{SHORT[f]} (rate {lo[lo.fold == f].base_rate.iloc[0]:.2f})" for f in FOLDS])
    for i in range(4):
        for j in range(6):
            ax.text(j, i, f"{Mx.to_numpy()[i, j]:.3f}" if Mx is M else f"{Mx.to_numpy()[i, j]:.1f}", ha="center",
                    va="center", fontsize=6.5)
    plt.colorbar(im, ax=ax, label=lab, shrink=0.8); ax.set_xlabel("budget")
gap_tot = G.to_numpy().sum()
gB7 = G[[7, 15, 31]].to_numpy().sum() / gap_tot
fsh = G.sum(1) / G.to_numpy().sum()
fig.suptitle(f"v1_3's {gap_tot*1e3/4:.0f}e-3 item-recoverable gap is spread over B1–B15 ({gB7:.0%} at B7+), and rcb+math "
             f"hold {fsh['researchcodebench']+fsh['matharena']:.0%} of it;\n"
             f"at B0 every fold scores worse than its own "
             f"base-rate Brier (mswe {M.loc['multi_swebench',0]:.3f} vs {lo[lo.fold=='multi_swebench'].unc.iloc[0]:.3f}); "
             f"rcb B1 {M.loc['researchcodebench',1]:.3f} > B0 {M.loc['researchcodebench',0]:.3f}", fontsize=8.5)
save(fig, "v1_3_loss", lo)

# ---------------------------------------------------------------- 8 seed-set level shift vs paired stability
pairs = pd.DataFrame([
    dict(what="v1_1 level R2", s5=208.8601, s10=209.7378),
    dict(what="v1 level R2", s5=208.9666, s10=209.7378 + 0.086),
    dict(what="v1_1 level R1 (same draws)", s5=215.0533, s10=215.0533),
])
pdl = pd.DataFrame([
    dict(what="Δ v1−v1_1 R2", s5=0.106, s10=0.086),
    dict(what="Δ subject prior (E7) R2, 1 vs 10 seeds", s5=-1.297, s10=-1.016),
    dict(what="Δ hyper refit (E7) R2, 1 vs 10 seeds", s5=3.649, s10=3.551),
])
fig, ax = plt.subplots(figsize=(6.5, 3.4))
xx = np.arange(3)
ax.bar(xx - 0.2, pairs.s10 - pairs.s5, 0.35, color=OK["verm"], label="level change (×1e-3 ALC)")
ax.bar(3 + xx - 0.2, pdl.s10 - pdl.s5, 0.35, color=OK["blue"], label="paired-Δ change (×1e-3)")
ax.axhline(0, color="k", lw=0.5); ax.axhline(-1, color=OK["green"], ls="--", lw=0.7)
ax.set_xticks(list(xx - 0.2) + list(3 + xx - 0.2))
ax.set_xticklabels(list(pairs.what) + list(pdl.what), rotation=30, ha="right", fontsize=6.5)
ax.set_ylabel("10-seed minus fewer-seed value (×1e-3)")
ax.set_title("R2 seeds 0–4 vs 0–9 move the ALC level by +0.88e-3 (≈ the adoption bar),\n"
             "while paired deltas move ≤ 0.28e-3: absolute ALCs across reports are not comparable")
ax.legend(fontsize=6.5)
save(fig, "seed_level", pd.concat([pairs.assign(kind="level"), pdl.assign(kind="paired")]))

# ---------------------------------------------------------------- 9 Jev vs PI
J = []
for line in (RUNS.parent / "decisions.jsonl").read_text(encoding="utf-8").splitlines():
    x = json.loads(line)
    J.append(dict(id=x["id"], jev=x["jev"]["verdict"]["choice"], conf=x["jev"]["verdict"]["confidence"],
                  within_noise=x["jev"].get("within_noise"), pi=x.get("pi_decision"), agree=x.get("agree")))
J = pd.DataFrame(J)
later = {"P2-item-pooling": "PI", "P4-v1-claim": "PI", "P3-v1_1-replace": "both/unclear",
         "CB2-attribution": "Jev", "TRIB-04-e6-citation": "open", "TRIB-06-adoption-gate": "open"}
J["later_right"] = J.id.map(later).fillna("")
fig, ax = plt.subplots(figsize=(7, 4))
c = J.agree.map({True: OK["blue"], False: OK["verm"]})
ax.scatter(J.within_noise, J.conf, c=c, s=45, edgecolor="k", lw=0.4)
for _, r in J[~J.agree.astype(bool)].iterrows():
    ax.annotate(f"{r.id}: Jev {r.jev} / PI {r.pi} → {r.later_right}", (r.within_noise, r.conf), fontsize=5.8,
                xytext=(-150, 5), textcoords="offset points", arrowprops=dict(arrowstyle="-", lw=0.3))
ax.set_xlabel("Jev 'within noise' score"); ax.set_ylabel("Jev verdict confidence")
agr = J.agree.astype(bool).mean()
ax.set_title(f"Jev and PI agree on {J.agree.astype(bool).sum()}/{len(J)} ({agr:.0%}); of 4 resolved disagreements\n"
             "the PI was later right on 2, Jev on 1, 1 split — Jev over-eliminates (5 of 6 dissents are ELIMINATE/MODIFY)")
ax.scatter([], [], c=OK["blue"], label="agree"); ax.scatter([], [], c=OK["verm"], label="disagree"); ax.legend(fontsize=7)
save(fig, "jev_pi", J)

print(dep[["label", "family", "pt", "I2", "P_new_neg", "passes_bar"]].round(3).to_string())
print(fam_com.round(2))
print(fl.round(2).to_string()); print("fold gap shares", (G.sum(1)/G.to_numpy().sum()).round(3).to_dict())
print(lo.round(4).to_string())
print("gap total (fold-mean x1e3):", gap_tot * 1e3 / 4, "B7+ share", gB7)
