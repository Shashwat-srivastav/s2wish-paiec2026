"""Phase-4 red team of submission v1 (report: docs/research_notes/16_RED_TEAM_V1.md).

Sections (python redteam.py --section A1,A2,...):
  A1  fold leakage: per-fold params vs shipped all-benchmark params; draw-size leakage
  A2  draw-regime sensitivity (pair-counted vs distinct-item cap, large / many-small groups)
  A3  synthetic base-rate shift of the held-out benchmark (Rasch resample, logit offset)
  A4  stateful execution / order invariance of the packaged model.py
  A5  item-pooling identity: false merges (content-only key, random merges), no-overlap
  A6  calibration: per-budget reliability + simple shrink/cap variants (report only)
  A7  runtime robustness of packaged model.py + labeling.py
  A8  reviewer-from-hell: fold-level bootstrap, sign test
  L4  explicit item-pooling verdict: per-budget delta L2i vs L2 with CIs (R1/R2 + harsh regimes)
All engine numbers use ladder.py predictors (verified == packaged model.py to 2e-16) with
hyperparameters re-fit per LOBO fold (ladder.hyper_for). Writes only to paiec/experiments.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from paiec import paths
from paiec.evaluation import episodes as E
from paiec.models import ladder as Lad
from paiec.evaluation.simulator import _hseed

ROOT = paths.ROOT
V1 = ROOT / "submissions" / "v1"
OUT = paths.RUNS / "phase4_redteam"
FOLDS = E.EVAL_FOLDS
W = E.W
RES = {}


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# Model zoo on the vectorised engine
# ---------------------------------------------------------------------------
def v1_variant(ep, H, b, alpha=1.0, sb_scale=1.0, clip=(0.005, 0.995), item=True, sB_scale=1.0):
    li, lk, ly = ep.labels(b)
    HH = dict(H, sB=H["sB"] * sB_scale)
    q = Lad.pooled_pair(ep, HH, li, ly, "none")
    if not item:
        p = q[ep.t_inst]
    else:
        H2 = dict(HH, sigma_beta=H["sigma_beta"] * sb_scale)
        p, _ = Lad.item_level(ep, H2, q, li, lk, ly, False)
        qt = q[ep.t_inst]
        p = qt + alpha * (p - qt)
    return np.clip(p, clip[0], clip[1])


MODELS = {
    "v0": lambda ep, H, b: Lad.predict_budget(ep, H, "L1", b)[0],
    "v1": lambda ep, H, b: v1_variant(ep, H, b),
    "L2": lambda ep, H, b: v1_variant(ep, H, b, item=False),
    "v1_a85": lambda ep, H, b: v1_variant(ep, H, b, alpha=0.85),
    "v1_a70": lambda ep, H, b: v1_variant(ep, H, b, alpha=0.70),
    "v1_sb80": lambda ep, H, b: v1_variant(ep, H, b, sb_scale=0.8),
    "v1_clip02": lambda ep, H, b: v1_variant(ep, H, b, clip=(0.02, 0.98)),
    "v1_clip05": lambda ep, H, b: v1_variant(ep, H, b, clip=(0.05, 0.95)),
    "v1_wide15": lambda ep, H, b: v1_variant(ep, H, b, sB_scale=1.5),
    "v1_wide20": lambda ep, H, b: v1_variant(ep, H, b, sB_scale=2.0),
}


def run_models(ep, H, names, keep_probs=False):
    out = {}
    for nm in names:
        probs = {b: MODELS[nm](ep, H, b) for b in E.BUDGETS}
        br, ec = ep.score(probs)
        out[nm] = dict(br=br, ec=ec, alc=br @ W, pair=ep.inst_pair.copy())
        if keep_probs:
            out[nm]["probs"] = probs
    return out


_H = {}


def H_fold(B):
    if B not in _H:
        _H[B] = Lad.hyper_for(B)
    return _H[B]


def H_shipped():
    prm = json.load(open(V1 / "v1_params.json", encoding="utf-8"))
    H = dict(prm)
    H["name_table"] = {k: {"all": v} for k, v in prm["name_table"].items()}
    return H


def pair_mean(x, pair):
    u, inv = np.unique(pair, return_inverse=True)
    return np.bincount(inv, x) / np.bincount(inv)


def boot_ci(fold_vals, nboot=2000, seed=0, hier=False):
    """fold_vals: list of per-pair arrays. Stratified pair bootstrap (folds fixed) or
    hierarchical (resample folds, then pairs)."""
    rng = np.random.default_rng(seed)
    pt = float(np.mean([v.mean() for v in fold_vals]))
    bs = np.empty(nboot)
    F = len(fold_vals)
    for k in range(nboot):
        fs = rng.integers(0, F, F) if hier else range(F)
        bs[k] = np.mean([fold_vals[f][rng.integers(0, len(fold_vals[f]), len(fold_vals[f]))].mean()
                         for f in fs])
    return pt, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def delta_summary(res_by_fold, a, b, budget=None, hier=False):
    """Per-pair mean of (a - b) ALC (or one budget's Brier), per fold, + CI."""
    vals, per_fold = [], {}
    for B, r in res_by_fold.items():
        xa = r[a]["alc"] if budget is None else r[a]["br"][:, budget]
        xb = r[b]["alc"] if budget is None else r[b]["br"][:, budget]
        d = pair_mean(xa - xb, r[a]["pair"])
        vals.append(d)
        per_fold[B] = float(d.mean())
    pt, lo, hi = boot_ci(vals, hier=hier)
    return dict(pt=pt * 1e3, lo=lo * 1e3, hi=hi * 1e3, per_fold={k: v * 1e3 for k, v in per_fold.items()})


def fmt(d):
    pf = " ".join(f"{k[:5]}={v:+.1f}" for k, v in d["per_fold"].items())
    return f"{d['pt']:+.2f} [{d['lo']:+.2f},{d['hi']:+.2f}] ({pf})"


def alc_fold_mean(res_by_fold, m):
    return float(np.mean([pair_mean(r[m]["alc"], r[m]["pair"]).mean() for r in res_by_fold.values()]))


# ---------------------------------------------------------------------------
# Draw regimes (A2)
# ---------------------------------------------------------------------------
_ITEMSETS = {}


def item_sets(bench):
    if bench not in _ITEMSETS:
        D = E.compact(bench)
        P = len(D["pair_sid"])
        cell = np.unique(D["row_pair"].astype(np.int64) * 10**7 + D["row_item"])
        pp, ii = cell // 10**7, cell % 10**7
        bounds = np.searchsorted(pp, np.arange(P + 1))
        _ITEMSETS[bench] = [ii[bounds[p]:bounds[p + 1]] for p in range(P)]
    return _ITEMSETS[bench]


def draw_generic(groups, limit, seed, distinct):
    """groups: {(sid, bench): np.array(item ids)}. Same preference rule as simulator.draw_groups;
    cost = |items| (pair-counted) or |items \\ already drawn items of that bench| (distinct)."""
    rng = random.Random(_hseed("draw", seed))
    keys = sorted(groups)
    tiebreak = {k: rng.random() for k in keys}
    bench_n, subj_n = {}, {}
    used = {}
    remaining, chosen, pool = limit, [], set(keys)
    while pool:
        k = min(pool, key=lambda g: (bench_n.get(g[1], 0), subj_n.get(g[0], 0), tiebreak[g]))
        pool.discard(k)
        its = groups[k]
        if distinct:
            u = used.setdefault(k[1], set())
            cost = len(set(its.tolist()) - u)
        else:
            cost = len(its)
        if cost > remaining:
            continue
        chosen.append(k)
        remaining -= cost
        if distinct:
            used[k[1]].update(its.tolist())
        bench_n[k[1]] = bench_n.get(k[1], 0) + 1
        subj_n[k[0]] = subj_n.get(k[0], 0) + 1
    return chosen


def regime_draw(B, seed, kind, max_tries=60):
    """B pair indices chosen in one draw of the given kind; [] if B never drawn."""
    D = E.compact(B)
    sets_B = item_sets(B)
    others = [b for b in E.BENCHES if b != B]
    pub_src = others if kind != "withB" else list(E.BENCHES)
    pub = np.concatenate([E.distinct_sizes(E.compact(b)) for b in pub_src])
    for t in range(max_tries):
        kk = "R1g" if kind in ("noleak", "withB") else kind  # paired randomness for the two R1 controls
        rng = np.random.default_rng(_hseed("rt", kk, B, seed, t) % 2**32)
        groups = {(f"B{p}", B): s for p, s in enumerate(sets_B)}
        if kind in ("real_pair", "real_distinct"):
            K = int(rng.integers(1, 5))
            for k in range(K - 1):
                src = others[int(rng.integers(len(others)))]
                ss = item_sets(src)
                ns = int(rng.integers(3, 41))
                pick = rng.choice(len(ss), size=min(ns, len(ss)), replace=False)
                for j in pick:
                    groups[(f"ph{k}_{j}", f"ph{k}")] = ss[j]
        elif kind == "large":
            K = int(rng.integers(2, 5))
            for k in range(K - 1):
                for j in range(int(rng.integers(3, 41))):
                    groups[(f"ph{k}_{j}", f"ph{k}")] = np.arange(int(rng.integers(400, 901)))
        elif kind == "many_small":
            K = int(rng.integers(7, 14))
            for k in range(K - 1):
                for j in range(int(rng.integers(3, 11))):
                    groups[(f"ph{k}_{j}", f"ph{k}")] = np.arange(int(rng.integers(80, 151)))
        elif kind in ("noleak", "withB"):  # R1 with phantom sizes from other benchmarks only / all
            K = int(rng.integers(1, 5))
            for k in range(K - 1):
                for j in range(int(rng.integers(3, 41))):
                    groups[(f"ph{k}_{j}", f"ph{k}")] = np.arange(int(rng.choice(pub)))
        else:
            raise ValueError(kind)
        distinct = kind == "real_distinct"
        chosen = draw_generic(groups, 1000, _hseed("rtd", kk, B, seed, t) % 2**31, distinct)
        picked = sorted(int(g[0][1:]) for g in chosen if g[1] == B)
        if picked:
            return picked
    return []


def build_regime(D, B, kind, seeds):
    if kind in ("R1", "R2", "R3"):
        return E.build(D, kind, seeds)
    ip, isd, ig = [], [], []
    for g, s in enumerate(seeds):
        pk = regime_draw(B, s, kind)
        ip += pk; isd += [s] * len(pk); ig += [g] * len(pk)
    ep = E.Episodes(D, ip, isd, ig)
    ep.draw_info = [(int((np.asarray(ig) == g).sum()), 0) for g in range(len(seeds))]
    return ep


def eval_regime(kind, names, seeds, H_of=H_fold, D_of=None, folds=FOLDS):
    res = {}
    npairs = {}
    for B in folds:
        D = E.compact(B) if D_of is None else D_of(B)
        ep = build_regime(D, B, kind, seeds)
        res[B] = run_models(ep, H_of(B), names)
        npairs[B] = float(np.mean([x[0] for x in ep.draw_info])) if getattr(ep, "draw_info", None) else float(ep.n)
    return res, npairs


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def sec_A1():
    log("== A1 fold leakage ==")
    shp = H_shipped()
    rows = {}
    for B in FOLDS:
        H = H_fold(B)
        rows[B] = {k: round(float(H[k]), 3) for k in ("G", "sB", "tau", "sigma_beta")}
        rows[B]["train"] = H["train"]
        assert B not in H["train"]
    rows["shipped(all5)"] = {k: round(float(shp[k]), 3) for k in ("G", "sB", "tau", "sigma_beta")}
    log(json.dumps(rows, indent=1))
    seeds = list(range(40))
    r_fold, _ = eval_regime("R1", ["v0", "v1"], seeds)
    seeds = list(range(200))
    r_wb, npw = eval_regime("withB", ["v0", "v1"], seeds)
    r_ship, _ = eval_regime("R1", ["v0", "v1"], list(range(40)), H_of=lambda B: shp)
    r_nl, npl = eval_regime("noleak", ["v0", "v1"], seeds)
    out = dict(params=rows,
               perfold_v1=alc_fold_mean(r_fold, "v1"), perfold_v0=alc_fold_mean(r_fold, "v0"),
               shipped_v1=alc_fold_mean(r_ship, "v1"),
               d_perfold=delta_summary(r_fold, "v1", "v0"),
               d_shipped=delta_summary(r_ship, "v1", "v0"),
               leak_gain=delta_summary({B: {"a": r_ship[B]["v1"], "b": r_fold[B]["v1"]} for B in FOLDS}, "a", "b"),
               noleak_draw=delta_summary(r_nl, "v1", "v0"), noleak_npairs=npl,
               withB_draw200=delta_summary(r_wb, "v1", "v0"), withB_npairs=npw)
    for k, v in out.items():
        log(k, fmt(v) if isinstance(v, dict) and "pt" in v else v)
    RES["A1"] = out
    return r_fold


def sec_A2(r1=None):
    log("== A2 draw regimes ==")
    seeds = list(range(40))
    names = ["v0", "L2", "v1"]
    out = {}
    for kind in ["R1", "noleak", "real_pair", "real_distinct", "large", "many_small", "R2", "R3"]:
        sd = list(range(200)) if kind not in ("R1", "R2", "R3") else (seeds if kind == "R1" else [0, 1])
        t0 = time.time()
        res, npairs = eval_regime(kind, names, sd)
        o = dict(npairs=npairs, v0=alc_fold_mean(res, "v0"), v1=alc_fold_mean(res, "v1"),
                 L2=alc_fold_mean(res, "L2"),
                 d_v1_v0=delta_summary(res, "v1", "v0"), d_v1_v0_hier=delta_summary(res, "v1", "v0", hier=True),
                 d_v1_L2=delta_summary(res, "v1", "L2"),
                 d_v1_L2_budget={b: delta_summary(res, "v1", "L2", budget=j) for j, b in enumerate(E.BUDGETS)},
                 d_v1_v0_budget={b: delta_summary(res, "v1", "v0", budget=j) for j, b in enumerate(E.BUDGETS)})
        out[kind] = o
        log(f"[{kind}] {time.time()-t0:.0f}s npairs={ {k[:5]: round(v,1) for k,v in npairs.items()} } "
            f"v0={o['v0']:.4f} L2={o['L2']:.4f} v1={o['v1']:.4f}")
        log("   v1-v0", fmt(o["d_v1_v0"]), " hier", fmt(o["d_v1_v0_hier"]))
        log("   v1-L2", fmt(o["d_v1_L2"]))
        log("   v1-L2 B1/B3/B7:", " | ".join(fmt(o["d_v1_L2_budget"][b]) for b in (1, 3, 7)))
        if kind == "R1":
            RES["_R1res"] = res
    RES["A2"] = out


_RASCH = {}


def synth_D(B, delta, seed):
    D = E.compact(B)
    if B not in _RASCH:
        th, be, _ = Lad.rasch(D)
        _RASCH[B] = (th, be)
    th, be = _RASCH[B]
    eta = th[D["row_pair"]] + be[D["item_key"][D["row_item"]]] + delta
    rng = np.random.default_rng(_hseed("synth", B, delta, seed) % 2**32)
    y = (rng.random(len(eta)) < 1 / (1 + np.exp(-eta))).astype(np.int8)
    D2 = dict(D)
    D2["row_y"] = y
    return D2


def sec_A3():
    log("== A3 synthetic base-rate shift ==")
    names = ["v0", "v1", "v1_wide15", "v1_wide20"]
    out = {}
    for delta in [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]:
        o = {}
        for kind, sd in (("withB", list(range(100))), ("R2", [0])):
            res, _ = eval_regime(kind, names, sd, D_of=lambda B: synth_D(B, delta, 0))
            o[kind] = {nm: alc_fold_mean(res, nm) for nm in names}
            o[kind]["d"] = delta_summary(res, "v1", "v0")
            o[kind]["d_w15"] = delta_summary(res, "v1_wide15", "v0")
            o[kind]["d_w15_v1"] = delta_summary(res, "v1_wide15", "v1")
            o[kind]["d_B0"] = delta_summary(res, "v1", "v0", budget=0)
        o["base_rate"] = {B: float(synth_D(B, delta, 0)["row_y"].mean()) for B in FOLDS}
        out[delta] = o
        log(f"delta={delta:+} base={ {k[:5]: round(v,2) for k,v in o['base_rate'].items()} }")
        for kind in ("withB", "R2"):
            log(f"   {kind} v1-v0 {fmt(o[kind]['d'])}  B0 {o[kind]['d_B0']['pt']:+.1f}  "
                f"wide1.5-v0 {o[kind]['d_w15']['pt']:+.2f} wide1.5-v1 {o[kind]['d_w15_v1']['pt']:+.2f}")
    RES["A3"] = out


def rekey(B, mode, m=None):
    """Return D with item_key replaced (content-only / random merges of m items)."""
    D = dict(E.compact(B))
    ids = D["item_ids"]
    if mode == "content":
        import pandas as pd
        it = pd.read_parquet(E.DATA / B / "items.parquet", columns=["item_id", "content"])
        cmap = dict(zip(it.item_id, it.content.map(E._s)))
        keys = [hashlib.sha1(cmap[i].encode("utf-8", "replace")).hexdigest() for i in ids]
    elif mode == "random":
        rng = np.random.default_rng(_hseed("merge", B, m) % 2**32)
        old = D["item_key"]
        perm = rng.permutation(old.max() + 1)
        keys = [str(perm[k] // m) for k in old]
    else:
        raise ValueError(mode)
    u = sorted(set(keys))
    ix = {k: j for j, k in enumerate(u)}
    D["item_key"] = np.array([ix[k] for k in keys], np.int32)
    D["key_jev"] = np.full(len(u), np.nan)
    return D, len(u)


def sec_A5():
    log("== A5 item-pooling identity ==")
    names = ["v0", "L2", "v1"]
    out = {}
    cases = [("content", None), ("random", 2), ("random", 5), ("random", 20)]
    for mode, m in cases:
        tag = f"{mode}{m or ''}"
        nk = {}

        def D_of(B):
            D2, n = rekey(B, mode, m)
            nk[B] = (len(E.compact(B)["key_jev"]), n)
            return D2
        o = {}
        for kind, sd in (("withB", list(range(200))), ("R3", [0])):
            res, _ = eval_regime(kind, names, sd, D_of=D_of)
            o[kind] = dict(v1_L2=delta_summary(res, "v1", "L2"), v1_v0=delta_summary(res, "v1", "v0"))
        o["keys"] = dict(nk)
        out[tag] = o
        log(f"[{tag}] keys(orig->new)={nk}")
        for kind in ("withB", "R3"):
            log(f"   {kind} v1-L2 {fmt(o[kind]['v1_L2'])}   v1-v0 {fmt(o[kind]['v1_v0'])}")
    RES["A5"] = out


def reliability(ep, probs, b):
    p = probs[b][ep.r_t]
    y = ep.r_y
    bins = np.minimum((p * 10).astype(int), 9)
    rows = []
    for k in range(10):
        m = bins == k
        if m.sum():
            rows.append((k / 10, int(m.sum()), float(p[m].mean()), float(y[m].mean())))
    return rows


def sec_A6():
    log("== A6 calibration ==")
    names = ["v1", "v1_a85", "v1_a70", "v1_sb80", "v1_clip02", "v1_clip05"]
    out = {}
    for kind, sd in (("withB", list(range(200))), ("R2", [0, 1]), ("R3", [0, 1])):
        res = {}
        rel = {b: [] for b in (0, 1, 7, 31)}
        for B in FOLDS:
            ep = build_regime(E.compact(B), B, kind, sd)
            res[B] = run_models(ep, H_fold(B), names, keep_probs=True)
            for b in rel:
                rel[b] += [(B,) + r for r in reliability(ep, res[B]["v1"]["probs"], b)]
            for nm in names:
                res[B][nm].pop("probs")
        o = {nm: delta_summary(res, nm, "v1") for nm in names[1:]}
        o["ece_alcw"] = {nm: float(np.mean([(r[nm]["ec"] @ W).mean() for r in res.values()])) for nm in names}
        o["ece_by_budget_v1"] = [float(np.mean([r["v1"]["ec"][:, j].mean() for r in res.values()])) for j in range(6)]
        # pooled reliability over folds (row-weighted)
        agg = {}
        for b, rows in rel.items():
            tab = {}
            for (_, lo, n, pm, ym) in rows:
                t = tab.setdefault(lo, [0, 0.0, 0.0])
                t[0] += n; t[1] += pm * n; t[2] += ym * n
            agg[b] = [(lo, t[0], round(t[1] / t[0], 3), round(t[2] / t[0], 3)) for lo, t in sorted(tab.items())]
        o["reliability_v1"] = agg
        out[kind] = o
        log(f"[{kind}] ECE(ALCw) {o['ece_alcw']}  ECE by budget v1 {np.round(o['ece_by_budget_v1'],3)}")
        for nm in names[1:]:
            log(f"   {nm}-v1 {fmt(o[nm])}")
        for b in (7, 31):
            log(f"   reliability B{b}: " + "; ".join(f"{lo:.1f}:n={n} p={pm} y={ym}" for lo, n, pm, ym in agg[b] if lo >= 0.6 or lo <= 0.1))
    RES["A6"] = out


def sec_A8():
    for kind, sd in (("R1", list(range(40))), ("withB", list(range(200)))):
        _sec_A8(kind, sd)


def _sec_A8(kind, seeds):
    log(f"== A8 reviewer from hell ({kind}, {len(seeds)} draws) ==")
    res = RES.get("_R1res") if kind == "R1" else None
    if res is None:
        res, _ = eval_regime(kind, ["v0", "L2", "v1"], seeds)
    fold_d = {B: float(pair_mean(r["v1"]["alc"] - r["v0"]["alc"], r["v1"]["pair"]).mean()) * 1e3 for B, r in res.items()}
    vals = np.array(list(fold_d.values()))
    import itertools
    allres = [np.mean(c) for c in itertools.product(vals, repeat=len(vals))]
    tcrit = 3.182  # t_{0.975, 3}
    se = vals.std(ddof=1) / math.sqrt(len(vals))
    o = dict(per_fold=fold_d, mean=float(vals.mean()),
             t_ci=(float(vals.mean() - tcrit * se), float(vals.mean() + tcrit * se)),
             fold_boot_exact_frac_ge0=float(np.mean(np.array(allres) >= 0)),
             fold_boot_exact_ci=(float(np.percentile(allres, 2.5)), float(np.percentile(allres, 97.5))),
             hier=delta_summary(res, "v1", "v0", hier=True), strat=delta_summary(res, "v1", "v0"),
             sign_test_p_one_sided=float(sum(math.comb(4, k) for k in range(int((vals < 0).sum()), 5)) / 16),
             leave_one_fold_out={B: float(np.mean([v for k, v in fold_d.items() if k != B])) for B in fold_d},
             item_hier=delta_summary(res, "v1", "L2", hier=True), offset_hier=delta_summary(res, "L2", "v0", hier=True))
    # seed (draw) variability: sd of draw-level mean delta
    for k, v in o.items():
        log(k, fmt(v) if isinstance(v, dict) and "pt" in v else v)
    # draw-level spread: mean delta per draw group
    RES["A8_" + kind] = o


def _two_subject(D, B, n=300):
    """Episodes with exactly 2 pairs of B per draw (formative: a 2-subject benchmark)."""
    P = len(D["pair_sid"])
    rng = np.random.default_rng(_hseed("two", B) % 2**32)
    ip, isd, ig = [], [], []
    for g in range(n):
        a, b = rng.choice(P, 2, replace=False)
        ip += [int(a), int(b)]; isd += [g, g]; ig += [g, g]
    return E.Episodes(D, ip, isd, ig)


def sec_A9():
    """Reproduce the first formative failure modes (8 pairs / 7 benchmarks ~ R2):
    B1 > B0 on average, non-monotone curves (0.36 at B3/B7), a 2-subject benchmark at
    0.27-0.31 for all budgets. Formative numbers are used as targets to *reproduce*, never fit."""
    log("== A9 formative failure-mode reproduction ==")
    names = ["v0", "L2", "v1", "v1_wide15"]
    out = {}
    rng = np.random.default_rng(7)
    for tag, delta in (("real", None), ("synth-1", -1), ("synth0", 0), ("synth+1", 1), ("synth+2", 2)):
        per = {}  # fold -> model -> br [n,6]
        y1 = {}
        for B in FOLDS:
            D = E.compact(B) if delta is None else synth_D(B, delta, 0)
            ep = E.build(D, "R2", [0, 1, 2, 3, 4])
            r = run_models(ep, H_fold(B), names)
            per[B] = {nm: r[nm]["br"] for nm in names}
            y1[B] = np.array([a[0, 1] if len(a) else -1 for a in ep.acq])
        o = {}
        for nm in names:
            allbr = np.concatenate([per[B][nm] for B in FOLDS])
            d10 = allbr[:, 1] - allbr[:, 0]
            yy = np.concatenate([y1[B] for B in FOLDS])
            # 8-pair formative-like samples: each pair from a random fold (fold-balanced)
            S = 20000
            fsel = rng.integers(0, len(FOLDS), (S, 8))
            sims = np.empty((S, 6)); bump = np.zeros(S, bool); stuck = np.zeros(S, bool)
            for f, B in enumerate(FOLDS):
                m = fsel == f
                idx = rng.integers(0, len(per[B][nm]), m.sum())
                pick = per[B][nm][idx]
                tmp = np.zeros((S, 8, 6)); tmp[m] = pick
                if f == 0:
                    acc = tmp
                else:
                    acc = acc + tmp
            sims = acc.mean(1)
            bump = (acc[:, :, [2, 3]].max(2) >= 0.36).any(1)
            stuck = (acc.min(2) > 0.27).any(1)
            o[nm] = dict(
                curve=[float(x) for x in np.mean([per[B][nm].mean(0) for B in FOLDS], 0)],
                mean_B1_minus_B0=float(np.mean([(per[B][nm][:, 1] - per[B][nm][:, 0]).mean() for B in FOLDS])),
                frac_pairs_B1_gt_B0=float((d10 > 0).mean()),
                B1_minus_B0_if_y1=float(d10[yy == 1].mean()) if (yy == 1).any() else None,
                B1_minus_B0_if_y0=float(d10[yy == 0].mean()) if (yy == 0).any() else None,
                frac_pairs_bump036=float((allbr[:, [2, 3]].max(1) >= 0.36).mean()),
                frac_pairs_all_budgets_gt027=float((allbr.min(1) > 0.27).mean()),
                frac_pairs_alc_gt025=float(((allbr @ W) > 0.25).mean()),
                P8_meanB1_gt_meanB0=float((sims[:, 1] > sims[:, 0]).mean()),
                P8_any_bump036=float(bump.mean()), P8_any_stuck027=float(stuck.mean()),
                P8_alc_mean=float((sims @ W).mean()), P8_alc_sd=float((sims @ W).std()))
        out[tag] = o
        log(f"[{tag}]")
        for nm in names:
            x = o[nm]
            log(f"   {nm:9s} curve={np.round(x['curve'],4).tolist()} B1-B0={x['mean_B1_minus_B0']*1e3:+.1f}e-3 "
                f"(y1=1:{(x['B1_minus_B0_if_y1'] or 0)*1e3:+.1f} y1=0:{(x['B1_minus_B0_if_y0'] or 0)*1e3:+.1f}) "
                f"fracB1>B0={x['frac_pairs_B1_gt_B0']:.2f} bump={x['frac_pairs_bump036']:.3f} "
                f"stuck={x['frac_pairs_all_budgets_gt027']:.3f} | 8-pair: P(B1>B0)={x['P8_meanB1_gt_meanB0']:.2f} "
                f"P(bump)={x['P8_any_bump036']:.2f} P(stuck)={x['P8_any_stuck027']:.2f} ALC={x['P8_alc_mean']:.4f}+-{x['P8_alc_sd']:.4f}")
    # 2-subject benchmark: does cross-pair pooling between the two subjects hurt?
    two = {}
    for B in FOLDS:
        ep = _two_subject(E.compact(B), B)
        r = run_models(ep, H_fold(B), ["v0", "L2", "v1"])
        d = r["v1"]["br"] - r["L2"]["br"]
        two[B] = dict(v1_L2_by_budget=(d.mean(0) * 1e3).round(2).tolist(),
                      frac_v1_worse_than_L2_by_1e2=float(((r["v1"]["alc"] - r["L2"]["alc"]) > 0.01).mean()),
                      stuck027_v1=float((r["v1"]["br"].min(1) > 0.27).mean()),
                      stuck027_v0=float((r["v0"]["br"].min(1) > 0.27).mean()),
                      v1_v0_alc=float((r["v1"]["alc"] - r["v0"]["alc"]).mean() * 1e3))
        log(f"   two-subject {B[:5]}: {two[B]}")
    out["two_subject"] = two
    RES["A9"] = out


# ---------------------------------------------------------------------------
# Packaged-model tests (A4, A7)
# ---------------------------------------------------------------------------
def _load_pkg(path=V1):
    from paiec.evaluation import simulator as sim
    return sim.load_model_module(path / "model.py"), sim.load_model_module(path / "labeling.py")


def _real_case(bench="real_webagents", n_pairs=5, seed=0, extra_bench="researchcodebench"):
    from paiec.evaluation import simulator as sim
    rows = [r for r in sim.load_benchmark_dir(str(E.DATA / bench)) if r["y"] in (0, 1)]
    sids = sorted({r["subject_id"] for r in rows})[: n_pairs * 3]
    rows = [r for r in rows if r["subject_id"] in set(sids)]
    pairs = [p for p in sim.prepare_pairs(rows, seed=seed)][:n_pairs]
    labeled = [[p_.acq[k]["input"], p_.acq[k]["y"]] for p_ in pairs for k in range(min(31, len(p_.acq)))]
    if extra_bench:
        rx = [r for r in sim.load_benchmark_dir(str(E.DATA / extra_bench)) if r["y"] in (0, 1)]
        sx = sorted({r["subject_id"] for r in rx})[:2]
        px = sim.prepare_pairs([r for r in rx if r["subject_id"] in set(sx)], seed=seed)
        labeled += [[p_.acq[k]["input"], p_.acq[k]["y"]] for p_ in px for k in range(31)]
        del rx
    seen, targets = set(), []
    for p_ in pairs:
        for r in p_.eval_rows:
            k = json.dumps(r["input"], sort_keys=True)
            if k not in seen:
                seen.add(k); targets.append(r["input"])
    return pairs, labeled, targets


def sec_A4(path=V1, tag="A4"):
    log(f"== {tag} stateful execution / order ({path.name}) ==")
    from paiec.evaluation import simulator as sim
    mod, lab = _load_pkg(path)
    pairs, labeled, targets = _real_case()
    targets = targets[:250]
    o = {"n_labeled": len(labeled), "n_targets": len(targets)}
    base = [mod.predict(copy.deepcopy(t), copy.deepcopy(labeled)) for t in targets]
    # (a) permuted labeled order
    md = 0.0
    for s in range(5):
        L = copy.deepcopy(labeled); random.Random(s).shuffle(L)
        pr = [mod.predict(copy.deepcopy(t), L) for t in targets]
        md = max(md, max(abs(a - b) for a, b in zip(base, pr)))
    o["perm_labeled_maxdiff"] = md
    # (b) reversed target order + repeated calls
    rev = [mod.predict(copy.deepcopy(t), copy.deepcopy(labeled)) for t in reversed(targets)][::-1]
    rep = [mod.predict(copy.deepcopy(t), copy.deepcopy(labeled)) for t in targets]
    o["target_order_maxdiff"] = max(abs(a - b) for a, b in zip(base, rev))
    o["repeat_maxdiff"] = max(abs(a - b) for a, b in zip(base, rep))
    # (c) mutation of arguments
    L = copy.deepcopy(labeled); T = copy.deepcopy(targets[0]); L0 = copy.deepcopy(L); T0 = copy.deepcopy(T)
    mod.predict(T, L)
    o["mutates_args"] = not (L == L0 and T == T0)
    # (d) isolated fresh process
    tmp = Path(os.environ.get("PAIEC_SCRATCH", str(OUT))) / "redteam_a4_case.json"
    json.dump({"targets": targets[:60], "labeled": labeled}, open(tmp, "w", encoding="utf-8"))
    code = ("import json,sys;sys.path.insert(0,r'%s');import model;"
            "d=json.load(open(r'%s',encoding='utf-8'));"
            "print(json.dumps([model.predict(t,d['labeled']) for t in d['targets']]))") % (path, tmp)
    outp = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    fresh = json.loads(outp.stdout)
    o["fresh_process_maxdiff"] = max(abs(a - b) for a, b in zip(base[:60], fresh))
    tmp.unlink()
    # (e) model state after interleaving other benchmarks/budgets: call with empty labeled then re-run
    _ = [mod.predict(copy.deepcopy(t), []) for t in targets[:20]]
    again = [mod.predict(copy.deepcopy(t), copy.deepcopy(labeled)) for t in targets[:20]]
    o["after_interleave_maxdiff"] = max(abs(a - b) for a, b in zip(base[:20], again))
    # (f) full streaming run: pair order permuted -> per-pair ALC identical?
    def run(prs):
        r = sim.run_direct(prs, mod.predict, lab.acquisition_function, isolation="checkpoint")
        return {k: v["alc"] for k, v in r["per_pair"].items()}
    small = pairs[:3]
    a1 = run(small); a2 = run(list(reversed(small)))
    o["pair_order_alc_maxdiff"] = max(abs(a1[k] - a2[k]) for k in a1)
    # (g) the labeled list contains duplicate entries (same label twice) -> double counting?
    dup = labeled + labeled[:31]
    o["duplicate_entries_change"] = max(abs(mod.predict(copy.deepcopy(t), copy.deepcopy(dup)) - b)
                                        for t, b in zip(targets[:30], base[:30]))
    # (h) item-key identity probes: 10 other subjects all failed item X (y=0) but pass 5 other items
    subj = lambda n: {"normalized_name": n, "provider": "P", "release_date": "2025-01-01", "access_date": "",
                      "harness": "", "reasoning_effort": "", "harness_version": "", "subject_features_extra": ""}
    itX = {"item_content": "Solve the task.", "item_features": "k=1", "interactors": "", "benchmark_id": "hb"}
    oth = lambda j: dict(itX, item_content=f"other {j}")
    def lab_with(xform):
        L = []
        for s in range(10):
            L.append([[subj(f"m{s}"), xform(dict(itX))], 0])
            L += [[[subj(f"m{s}"), oth(j)], 1] for j in range(5)]
        return L
    tgt = [subj("target"), dict(itX)]
    probes = {
        "exact": lambda d: d,
        "trailing_ws": lambda d: dict(d, item_content=d["item_content"] + " "),
        "crlf": lambda d: dict(d, item_content=d["item_content"].replace(".", ".\r\n")),
        "features_changed": lambda d: dict(d, item_features="k=2"),
        "interactors_None": lambda d: dict(d, interactors=None),
        "bench_int_vs_str": lambda d: d,
        "key_order": lambda d: dict(reversed(list(d.items()))),
    }
    ph = {k: mod.predict(copy.deepcopy(tgt), lab_with(f)) for k, f in probes.items()}
    ph["no_item_evidence(L2)"] = mod.predict(copy.deepcopy(tgt), [e for e in lab_with(probes["exact"]) if e[0][1]["item_content"] != "Solve the task."])
    o["item_key_probes"] = {k: round(v, 4) for k, v in ph.items()}
    for k, v in o.items():
        log(k, v)
    RES[tag] = o


def sec_A7(path=V1, tag="A7"):
    log(f"== {tag} runtime robustness ({path.name}) ==")
    import tracemalloc
    mod, lab = _load_pkg(path)
    subj = {"normalized_name": "Ünïcödé 模型 🚀", "provider": "X", "release_date": "2025-13-45",
            "access_date": "", "harness": "", "reasoning_effort": "", "harness_version": "",
            "subject_features_extra": ""}
    item = {"item_content": "∑ ∫ \x00 \ud800? " * 3, "item_features": "", "interactors": "", "benchmark_id": "b1"}
    good = [[dict(subj), dict(item)], 1]
    weird_labels = [True, False, "1", "0", 1.0, 0.0, 0.5, "abc", None, float("nan"), float("inf"), -1, 2,
                    np.int64(1), np.float32(0.0), [1], {"y": 1}, b"1"]
    weird_entries = [None, 5, "str", [], [1], [[subj], 1], [[subj, item]], [[subj, item], 1, 2],
                     [[None, None], 1], [["a", "b"], 1], {"a": 1}, [[subj, {"benchmark_id": None}], 1],
                     [[{}, {}], 1], (([subj, item]), 1)]
    weird_inputs = [None, [], [subj], [subj, item, 3], [None, item], [subj, None], "x", [subj, {}],
                    [{}, item], [subj, dict(item, benchmark_id=5)], {"a": 1}, [subj, dict(item, item_content=None)]]
    weird_labeled = [None, [], "abc", 5, {"a": 1}, iter([good]), (good,), [good] * 3]
    fails, n = [], 0

    def chk(inp, L, tagx):
        nonlocal n
        n += 1
        try:
            p = mod.predict(inp, L)
            f = float(p)
            if not (isinstance(p, float) and math.isfinite(f) and 0 <= f <= 1):
                fails.append((tagx, repr(p)))
        except BaseException as e:  # noqa: BLE001
            fails.append((tagx, repr(e)))
    for y in weird_labels:
        chk([dict(subj), dict(item)], [[[dict(subj), dict(item)], y]], f"label={y!r}")
    for e in weird_entries:
        chk([dict(subj), dict(item)], [e, good], f"entry={str(e)[:40]}")
    for inp in weird_inputs:
        chk(inp, [good], f"input={str(inp)[:40]}")
    for L in weird_labeled:
        chk([dict(subj), dict(item)], L, f"labeled={str(L)[:40]}")
    # labeling hook
    hook_fails = []
    for ctx in [None, {}, {"labels_remaining": "x"}, {"labels_remaining": 3}, {"labels_remaining": 0}, "bad"]:
        try:
            r = lab.acquisition_function([subj, item], context=ctx)
            if ctx is not None and not isinstance(r, bool):
                hook_fails.append((ctx, r))
        except BaseException as e:  # noqa: BLE001
            hook_fails.append((ctx, repr(e)))
    # latency / memory: 10k labeled
    lat = {}
    rng = np.random.default_rng(0)

    def big(nsub, nper, nitems=200):
        L = []
        for s in range(nsub):
            sd = dict(subj, normalized_name=f"m{s}")
            for k in range(nper):
                L.append([[sd, dict(item, item_content=f"q{int(rng.integers(nitems))}")], int(rng.integers(2))])
        return L
    for nsub, nper in ((1, 10000), (300, 34), (1000, 10), (2000, 5)):
        L = big(nsub, nper)
        tracemalloc.start()
        t0 = time.perf_counter()
        for _ in range(3):
            p = mod.predict([dict(subj), dict(item, item_content="q1")], L)
        dt = (time.perf_counter() - t0) / 3
        peak = tracemalloc.get_traced_memory()[1] / 2**20
        tracemalloc.stop()
        lat[f"{nsub}x{nper}"] = dict(ms=round(dt * 1e3, 1), peak_MB=round(peak, 1), p=round(p, 4))
        del L
    o = dict(n_cases=n, fails=fails, hook_fails=hook_fails, latency=lat)
    for k, v in o.items():
        log(k, v)
    RES[tag] = o


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items() if not str(k).startswith("_")}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", default="A1,A2,A3,A5,A6,A8,A4,A7")
    ap.add_argument("--out", default=str(OUT / "redteam_results.json"))
    ap.add_argument("--pkg", default=str(V1))
    ap.add_argument("--tag", default="", help="suffix for result keys, e.g. _v1_1")
    a = ap.parse_args()
    secs = a.section.split(",")
    fn = {"A1": sec_A1, "A2": sec_A2, "A3": sec_A3, "A4": sec_A4, "A5": sec_A5, "A6": sec_A6,
          "A8": sec_A8, "A9": sec_A9, "A7": lambda: sec_A7(Path(a.pkg), "A7" + a.tag),
          "A4": lambda: sec_A4(Path(a.pkg), "A4" + a.tag)}
    prev = {}
    if Path(a.out).exists():
        prev = json.load(open(a.out, encoding="utf-8"))
    for s in secs:
        t0 = time.time()
        fn[s]()
        log(f"-- {s} done in {time.time()-t0:.0f}s")
        prev.update(_jsonable({k: v for k, v in RES.items() if not k.startswith("_")}))
        json.dump(prev, open(a.out, "w", encoding="utf-8"), indent=1, default=str)
