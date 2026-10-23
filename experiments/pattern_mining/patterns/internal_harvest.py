"""Harvest every saved real-data paired delta record into one catalogue.

Reads experiments/runs/*/*.json only (no recomputation). Units: Delta ALC x 1e-3 (lower is better).
Output: experiments/analysis/patterns/internal_catalogue.csv
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "experiments" / "runs"
OUT = Path(__file__).resolve().parent / "internal_catalogue.csv"
FOLDS = ["multi_swebench", "real_webagents", "researchcodebench", "matharena"]
BUDGETS = [0, 1, 3, 7, 15, 31]
W = [0.1, 0.2, 0.2, 0.2, 0.2, 0.1]


def load(p):
    return json.loads((RUNS / p).read_text(encoding="utf-8"))


rows = []


def budget_list(b):
    """Return (pt list of 6, per-fold dict of lists or None)."""
    if b is None:
        return None, None
    if isinstance(b, dict):  # {"0": rec, "1": rec ...}
        b = [b[str(k)] if str(k) in b else b[k] for k in BUDGETS]
    pts, pf = [], {f: [] for f in FOLDS}
    have_pf = True
    for e in b:
        if isinstance(e, (int, float)):
            pts.append(float(e)); have_pf = False
        else:
            pts.append(float(e["pt"]))
            if isinstance(e.get("per_fold"), dict) and all(f in e["per_fold"] for f in FOLDS):
                for f in FOLDS:
                    pf[f].append(float(e["per_fold"][f]))
            else:
                have_pf = False
    return pts, (pf if have_pf else None)


def add(source, family, arm, baseline, regime, rec, budget=None, kind="deployable", engine="", note=""):
    pf = rec.get("per_fold") if isinstance(rec.get("per_fold"), dict) else {}
    if budget is None:
        budget = rec.get("budget", rec.get("per_budget"))
    bpt, bpf = budget_list(budget)
    r = dict(source=source, family=family, arm=arm, baseline=baseline, regime=regime, kind=kind,
             engine=engine, note=note, pt=rec["pt"], lo=rec.get("lo"), hi=rec.get("hi"),
             hlo=rec.get("hlo"), hhi=rec.get("hhi"))
    for f in FOLDS:
        r["f_" + f] = pf.get(f)
    for i, b in enumerate(BUDGETS):
        r[f"b{b}"] = bpt[i] if bpt else None
        for f in FOLDS:
            r[f"b{b}_{f}"] = bpf[f][i] if bpf else None
    rows.append(r)


# ---------------- phase 3 (v1 -> v1_1), baseline = v1 ("ref")
s1 = load("phase3_v1_1/step1.json")
fam1 = {"a_t1": "offset_prior", "a_mix20": "offset_prior", "a_mix40": "offset_prior", "b_zero": "offset_prior",
        "b_half": "offset_prior", "c_k0": "offset_prior", "c_k50": "offset_prior", "v0": "version"}
for reg in ["R2", "R1"]:
    for arm, v in s1["real"][reg].items():
        if "d_ref" in v:
            add("phase3/step1", fam1.get(arm, "offset_prior"), arm, "v1", reg, v["d_ref"], v.get("d_ref_budget"), engine="v1")
s1b = load("phase3_v1_1/step1b.json")
for reg in ["R2", "R1"]:
    for arm in ["bh_k50", "bh_mix20"]:
        v = s1b["real"][reg][arm]
        add("phase3/step1b", "offset_prior", arm, "v1", reg, v["d_ref"], v.get("d_ref_budget"), engine="v1")
fin = load("phase3_v1_1/final.json")
for reg in ["R2", "R1"]:
    v = fin["real"][reg]["v1_1"]
    add("phase3/final", "version", "v1_1 (vs v1)", "v1", reg, v["d_ref"], v.get("d_ref_budget"), engine="v1")
s3 = load("phase3_v1_1/step3.json")
for reg in ["R2", "R1"]:
    for arm, v in s3[reg].items():
        if "d_ref" in v:
            add("phase3/step3", "shrink_temper", arm, "v1_1-ish ref", reg, v["d_ref"], v.get("d_ref_budget"), engine="v1_1")
# step2* item pooling, real only, vs v1 (R1, R3)
for fn in ["step2", "step2b", "step2c", "step2d"]:
    d = load(f"phase3_v1_1/{fn}.json")
    for reg in ["R1", "R3"]:
        for arm, v in d["real"][reg].items():
            if arm in ("v1", "L2"):
                continue
            if any(r["arm"] == arm and r["regime"] == reg and r["source"].startswith("phase3/step2") for r in rows):
                continue
            add(f"phase3/{fn}", "item_pooling", arm, "v1", reg, v["d_v1"], v.get("d_v1_budget"), engine="v1")
s4 = load("phase3_v1_1/step4.json")
for arm in ["A2", "A3", "A5", "A8"]:
    v = s4["R1"][f"v1_1|{arm}"]
    add("phase3/step4", "acquisition", arm, "v1_1|A1", "R1", v["d_A1"], v.get("d_A1_budget"), engine="v1_1")

# ---------------- E1 metadata prior (vs v1_1)
e1 = load("e1/e1_results.json")
for reg in ["R2", "R1"]:
    add("e1", "subject_prior", "metadata_prior", "v1_1", reg, e1[reg]["metadata_dALC"], engine="v1_1")

# ---------------- E2c text students + Jev (vs v1_1)
for enc in ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]:
    d = load(f"e2/e2c_{enc}.json")
    for reg in ["R2", "R1"]:
        for arm in ["v1_1+stuA", "v1_1+stuB"]:
            add(f"e2c/{enc}", "item_text", f"{arm.split('+')[1]}:{enc.split('-')[1]}", "v1_1", reg,
                d["e2e"][reg][arm]["d_v1_1"], engine="v1_1")
        if enc == "all-MiniLM-L6-v2":
            add("e2c", "oracle_offline", "jev (LLM, offline)", "v1_1", reg, d["e2e"][reg]["v1_1+jev"]["d_v1_1"],
                kind="offline_bound", engine="v1_1")

# ---------------- E4 facets / guard (vs v1_1)
e4 = load("e4/real.json")
fam4 = {"facets": "item_facets", "both": "item_facets", "facets_named": "item_facets", "shuffled": "control",
        "guard": "guard", "guard2": "guard"}
for reg in ["R2", "R1"]:
    for arm, v in e4[reg].items():
        if "d" in v:
            add("e4", fam4[arm], arm, "v1_1", reg, v["d"], v.get("d_budget"), engine="v1_1",
                kind="control" if arm == "shuffled" else "deployable")

# ---------------- E5 X1 robust priors (vs v1_1)
e5 = load("e5/real.json")
for reg in ["R2", "R1"]:
    for arm, v in e5[reg].items():
        if "d_v1_1" in v:
            add("e5", "version" if arm in ("v0", "v1") else "offset_prior", arm, "v1_1", reg, v["d_v1_1"],
                v.get("budget_v1_1"), engine="v1_1")

# ---------------- E6 oracles (named-only engine base)
e6 = load("e6/e6.json")
for arm, v in e6["arms"].items():
    if "d" in v:
        add("e6", "oracle_item", arm, "named-engine", "R2", v["d"], v.get("d_budget"), kind="oracle_bound",
            engine="named")

# ---------------- E7 enlarged-pool priors (named-only engine base)
e7 = load("e7/e7_primary_s10.json")
fam7 = {"hyper": "hyper_refit", "subject": "subject_prior", "subject+hyper": "subject_prior",
        "subject_allyears": "subject_prior", "subject_relkept": "subject_prior", "control_perm": "control",
        "SUPP_evalperm": "control", "SUPP_subject_2way": "subject_prior", "SUPP_hyper_tau_only": "shrink_temper",
        "SUPP_hyper_offset_only": "hyper_refit", "SUPP_hyper_sb_only": "hyper_refit", "SUPP_name_only": "subject_prior"}
for arm, v in e7["arms"].items():
    if "d" in v:
        add("e7", fam7[arm], arm, "named-engine", "R2", v["d"], v.get("d_budget"), engine="named",
            kind="control" if fam7[arm] == "control" else "deployable")

# ---------------- EXP-0044 tempering (vs v1_3, all-facet engine)
x44 = load("preregistration/exp_0044_tempering_replay_2026-10-14.json")
for reg in ["R2", "R1"]:
    for arm, v in x44["results"][reg].items():
        if "d" in v:
            add("EXP-0044", "shrink_temper", arm, "v1_3", reg, v["d"], v.get("d_budget"), engine="all")

# ---------------- EXP-0045 nested meta-learned prior (vs v1_3)
x45 = load("preregistration/exp_0045_nested_alc_2026-10-15.json")
for reg in ["R2", "R1"]:
    v = x45["results"][reg]["x3"]
    add("EXP-0045", "offset_prior", "nested x3 prior", "v1_3", reg, v["d"], v.get("d_budget"), engine="all")

# ---------------- correlated hierarchical (R3, fold-balanced; deltas in ALC units -> x1e3)
for p in sorted((RUNS / "correlated_hierarchical").glob("*.json")):
    d = json.loads(p.read_text(encoding="utf-8"))
    rec = {"pt": 1e3 * d["fold_balanced_delta"],
           "per_fold": {f: 1e3 * x for f, x in zip(FOLDS, d["benchmark_fold_deltas"])}}
    ci = d.get("paired_pair_bootstrap_95pct_ci")
    if ci:
        rec["lo"], rec["hi"] = 1e3 * ci[0], 1e3 * ci[1]
    sc = d["scales"]
    arm = f"corrhier b{sc['benchmark']} s{sc['system']} x{sc['system_benchmark']} i{sc['item']}" + (
        " seed1" if "seed1" in p.name else "")
    add("correlated_hierarchical/" + p.name, "corr_hier", arm, "baseline(R3)", "R3", rec, engine="corrhier",
        note="fold-balanced mean (not pair-weighted)")
    for a, bl in (d.get("blends") or {}).items():
        if a in ("0",):
            continue
        rec = {"pt": 1e3 * bl["fold_balanced_delta_vs_baseline"],
               "per_fold": {f: 1e3 * x for f, x in zip(FOLDS, bl["benchmark_fold_deltas"])},
               "lo": 1e3 * bl["paired_pair_bootstrap_95pct_ci"][0], "hi": 1e3 * bl["paired_pair_bootstrap_95pct_ci"][1]}
        add("correlated_hierarchical/" + p.name, "corr_hier", arm + f" blend{a}", "baseline(R3)", "R3", rec,
            engine="corrhier", note="exploratory blend; fold-balanced")

cols = list(rows[0].keys())
with OUT.open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if v is None else v) for k, v in r.items()})
print(len(rows), "records ->", OUT)
