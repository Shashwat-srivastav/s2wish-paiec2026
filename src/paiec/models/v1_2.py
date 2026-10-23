"""E4 candidate engine v1_2 = v1_1 + (a) item-group facet random effect + (b) degenerate-key guard.

Pre-registration: docs/preregistrations/E4_facets_preregistration.md. Report: docs/research_notes/23_E4_FACETS.md.
v1_1.py is imported, never edited. Config = v1_1.BASE2 plus switches:

  facets     False | "all" | "named"   item_features facet effect (own-subject labels only)
             logit p = c_n + sum_f (gamma_f[v] - s_nf) + beta_res, gamma ~ N(0, pi sB^2),
             beta_res ~ N(0, (1-pi) sB^2); pi BMA over FACET_PIS with prior FACET_PW by the pair's
             marginal likelihood on facet cells; centring s_nf = label-weighted mean gamma of field f.
             Exact fallback to v1_1 for any target without a labelled facet cell (B0 always).
  guard      0 (off) | 1 (G1: rho_k >= GUARD_RHO when the target key has no cross-subject label)
             | 2 (G2: ignore same-key labels unless >= 2 cross-subject labels)
  shuffle    False | int seed   permute the key -> feature map across keys (control)

Run: python -m paiec.models.v1_2 --step check,real,merge
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

from paiec import paths
from paiec.evaluation import episodes as E
from paiec.models import ladder as Lad
from paiec.models import v1_1 as V

GX, GW = Lad.GH_X, Lad.GH_W
LGW = np.log(GW)
sig, log_sig = Lad.sig, Lad.log_sig
OUT = paths.RUNS / "e4"

FACET_PIS = (1e-3, 0.1, 0.3)
FACET_PW = np.log([0.4, 0.4, 0.2])
GUARD_RHO = 0.5
NAMED = ("competition", "lang", "website", "paper")

BASE = dict(V.BASE2, facets=False, guard=0, shuffle=False)


def cfg_of(**kw):
    c = dict(BASE)
    c.update(kw)
    return c


# ---------------------------------------------------------------------------
# Facet encoding (runtime-visible item_features of each key)
# ---------------------------------------------------------------------------
_KF = {}


def key_features(B, D):
    """{key: {field: value}} from items.parquet item_features (keys embed the features)."""
    if B in _KF:
        return _KF[B]
    import pandas as pd
    it = pd.read_parquet(E.DATA / B / "items.parquet", columns=["item_id", "item_features"])
    f = dict(zip(it.item_id, it.item_features.map(E._s)))
    kf = {}
    for iid, k in zip(D["item_ids"], D["item_key"]):
        if int(k) in kf:
            continue
        d = {}
        for kv in (f.get(iid) or "").split(";"):
            if "=" in kv:
                a, v = kv.split("=", 1)
                d[a.strip()] = v.strip()
        kf[int(k)] = d
    _KF[B] = kf
    return kf


def encode(kf, K, mode="all", shuffle=False):
    """CSR facet encoding: ptr[K+1], code[E] (global field-value id), code_field[NC]."""
    keys = np.arange(K)
    src = keys
    if shuffle is not False and shuffle is not None:
        src = np.random.default_rng(int(shuffle)).permutation(K)
    fid, cid = {}, {}
    ptr = np.zeros(K + 1, np.int64)
    codes = []
    for k in keys:
        d = kf.get(int(src[k]), {})
        for a in sorted(d):
            if mode == "named" and a not in NAMED:
                continue
            fi = fid.setdefault(a, len(fid))
            codes.append(cid.setdefault((fi, d[a]), len(cid)))
        ptr[k + 1] = len(codes)
    code_field = np.zeros(len(cid), np.int64)
    for (fi, _), c in cid.items():
        code_field[c] = fi
    return dict(ptr=ptr, code=np.array(codes, np.int64), code_field=code_field,
                NC=max(len(cid), 1), F=max(len(fid), 1))


def _expand(keys, fe):
    """Entries of a key array: (row index, code)."""
    ptr = fe["ptr"]
    cnt = ptr[keys + 1] - ptr[keys]
    rows = np.repeat(np.arange(len(keys)), cnt)
    if not len(rows):
        return rows, rows
    start = np.repeat(ptr[keys], cnt)
    within = np.arange(len(rows)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    return rows, fe["code"][start + within]


# ---------------------------------------------------------------------------
# Item level with degenerate-key guard (guard=0 is v1_1.item_level exactly)
# ---------------------------------------------------------------------------
def item_level(ep, H, q, li, lk, ly, rho, pool=False, guard=0):
    if guard == 0:
        return V.item_level(ep, H, q, li, lk, ly, rho, pool)[0]
    sb = H["sigma_beta"]
    c = Lad._invert_c(q, sb)
    K = len(ep.D["key_jev"])
    g_t = ep.inst_group[ep.t_inst] * K + ep.t_key
    out = q[ep.t_inst].copy()
    if not len(li):
        return out
    g_l = ep.inst_group[li] * K + lk
    uk, inv = np.unique(g_l, return_inverse=True)
    e = c[li][:, None] + sb * GX[None, :]
    contrib = np.where(ly[:, None] > 0.5, log_sig(e), log_sig(-e))
    acc = np.zeros((len(uk), len(GX)))
    np.add.at(acc, inv, contrib)
    pos = np.minimum(np.searchsorted(uk, g_t), len(uk) - 1)
    has = uk[pos] == g_t
    # cross-subject label count on the target key = group count - target instance's own count
    ncnt = np.bincount(inv, minlength=len(uk))
    ik_l = li.astype(np.int64) * K + lk
    uik, cik = np.unique(ik_l, return_counts=True)
    ik_t = ep.t_inst.astype(np.int64) * K + ep.t_key
    p2 = np.minimum(np.searchsorted(uik, ik_t), len(uik) - 1)
    own = np.where(uik[p2] == ik_t, cik[p2], 0)
    ncross = np.where(has, ncnt[pos], 0) - own
    if guard == 2:
        has = has & (ncross >= 2)
    if has.any():
        a = acc[pos[has]] + LGW[None, :]
        amax = a.max(1, keepdims=True)
        wgt = np.exp(a - amax)
        wgt /= wgt.sum(1)[:, None]
        p_same = (wgt * sig(c[ep.t_inst[has]][:, None] + sb * GX[None, :])).sum(1)
        if rho > 0:
            qq = np.clip(q[li], 1e-12, 1 - 1e-12)
            lind = np.zeros(len(uk))
            np.add.at(lind, inv, np.where(ly > 0.5, np.log(qq), np.log1p(-qq)))
            aall = acc + LGW[None, :]
            amx = aall.max(1)
            lr = np.log(np.exp(aall - amx[:, None]).sum(1)) + amx - lind
            if pool in ("mom", "momd"):
                rg = V.mom_rho(li, lk, ep.inst_group[li], rho,
                               y=ly if pool == "momd" else None, qi=q if pool == "momd" else None)
                rho_k = np.array([rg[int(g)] for g in uk // K])
            else:
                rho_k = V.pooled_rho(lr, ncnt, uk // K, rho, pool)
            rt = rho_k[pos[has]]
            if guard == 1:
                rt = np.where(ncross[has] == 0, np.maximum(rt, GUARD_RHO), rt)
            ps = sig(np.log1p(-rt) - np.log(rt) + lr[pos[has]])
            p_same = ps * p_same + (1 - ps) * q[ep.t_inst[has]]
        out[has] = p_same
    return out


# ---------------------------------------------------------------------------
# Facet adjustment
# ---------------------------------------------------------------------------
def _lse(a, axis):
    m = a.max(axis, keepdims=True)
    return (np.log(np.exp(a - m).sum(axis, keepdims=True)) + m).squeeze(axis)


def facet_adjust(ep, H, q, p_base, li, lk, ly, fe, stats=None):
    if not len(li) or not len(fe["code"]):
        return p_base
    sb = H["sigma_beta"]
    c = Lad._invert_c(q, sb)
    K = len(ep.D["key_jev"])
    NC, F = fe["NC"], fe["F"]
    # label entries -> (instance, code) cells: own-subject only
    lr_, lcode = _expand(lk, fe)
    if not len(lr_):
        return p_base
    linst = li[lr_]
    ly_e = ly[lr_] > 0.5
    cell_id = linst.astype(np.int64) * NC + lcode
    cells, cinv = np.unique(cell_id, return_inverse=True)
    nl = np.bincount(cinv, minlength=len(cells)).astype(float)
    c_inst = cells // NC
    c_field = fe["code_field"][cells % NC]
    cf_id = c_inst * F + c_field
    ucf, cfinv = np.unique(cf_id, return_inverse=True)
    # targets eligible: key not labelled anywhere in the group (else item level handles it)
    gk_l = np.unique(ep.inst_group[li].astype(np.int64) * K + lk)
    gk_t = ep.inst_group[ep.t_inst].astype(np.int64) * K + ep.t_key
    elig = ~np.isin(gk_t, gk_l)
    tr, tcode = _expand(ep.t_key, fe)
    keep = elig[tr]
    tr, tcode = tr[keep], tcode[keep]
    tcell = ep.t_inst[tr].astype(np.int64) * NC + tcode
    pos = np.minimum(np.searchsorted(cells, tcell), len(cells) - 1)
    hit = cells[pos] == tcell
    tr, pos = tr[hit], pos[hit]
    if not len(tr):
        return p_base
    used = np.bincount(tr, minlength=len(p_base)) > 0
    lq = np.log(q / (1 - q))
    outs, lmls = [], []
    for pi in FACET_PIS:
        sg, sr = math.sqrt(pi) * sb, math.sqrt(1 - pi) * sb
        # m[n, j] = E_x sig(c_n + sg g_j + sr x)
        mt = sig(c[:, None, None] + sg * GX[None, :, None] + sr * GX[None, None, :]) @ GW
        mt = np.clip(mt, 1e-12, 1 - 1e-12)
        ll = np.where(ly_e[:, None], np.log(mt[linst]), np.log1p(-mt[linst]))
        lw = np.zeros((len(cells), len(GX)))
        np.add.at(lw, cinv, ll)
        lw += LGW[None, :]
        lml_c = _lse(lw, 1)
        w = np.exp(lw - lml_c[:, None])
        gm = w @ (sg * GX)
        shift = (np.bincount(cfinv, nl * gm, len(ucf)) / np.bincount(cfinv, nl, len(ucf)))[cfinv]
        # predictive for an unseen item in each cell (depends only on the cell)
        e = (c[c_inst] - shift)[:, None, None] + sg * GX[None, :, None] + sr * GX[None, None, :]
        pt = np.einsum("cj,cjx,x->c", w, sig(e), GW)
        pt = np.clip(pt, 1e-12, 1 - 1e-12)
        off_c = np.log(pt / (1 - pt)) - lq[c_inst]
        offs = np.bincount(tr, off_c[pos], len(p_base))
        o = p_base.copy()
        o[used] = sig(lq[ep.t_inst[used]] + offs[used])
        outs.append(o)
        lmls.append(np.bincount(c_inst, lml_c, ep.n))
    L = np.array(lmls) + FACET_PW[:, None]                     # (n_pi, n_inst)
    wts = np.exp(L - L.max(0)); wts /= wts.sum(0)
    mix = (wts[:, ep.t_inst] * np.array(outs)).sum(0)
    if stats is not None:
        stats.setdefault("used_frac", []).append(float(used.mean()))
        stats.setdefault("w_pi", []).append(wts.mean(1).tolist())
    return np.where(used, mix, p_base)


def predict(ep, H, b, cfg, fe=None, stats=None):
    li, lk, ly = ep.labels(b)
    q, empty = V.pooled_pair(ep, H, li, ly, cfg)
    if cfg.get("item", True):
        p = item_level(ep, H, q, li, lk, ly, cfg["rho"], cfg.get("rho_pool", False), cfg.get("guard", 0))
    else:
        p = q[ep.t_inst]
    k = cfg["kappa"]
    if k != 1.0:
        e = empty[ep.t_inst]
        p = np.where(e, 0.5 + k * (p - 0.5), p)
    if cfg.get("facets") and fe is not None:
        p = facet_adjust(ep, H, q, p, li, lk, ly, fe, stats)
    return np.clip(p, 0.005, 0.995)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def fe_for(B, D, cfg):
    if not cfg.get("facets"):
        return None
    K = len(D["key_jev"])
    return encode(key_features(B, D), K, cfg["facets"], cfg.get("shuffle", False))


def run_cfgs(B, D, ep, H, cfgs, stats=None):
    out = {}
    for nm, cfg in cfgs.items():
        fe = fe_for(B, D, cfg)
        st = {} if stats is not None else None
        probs = {b: predict(ep, H, b, cfg, fe, st) for b in E.BUDGETS}
        if stats is not None:
            stats[nm] = st
        br, ec = ep.score(probs)
        out[nm] = dict(br=br, ec=ec, alc=br @ E.W, pair=ep.inst_pair.copy())
    return out


T3 = 3.182446305284263      # t_{0.975, 3}


def fold_stats(d):
    pf = np.array(list(d["per_fold"].values()))
    m, s = pf.mean(), pf.std(ddof=1)
    return dict(t3=[m - T3 * s / 2, m + T3 * s / 2], pi=[m - T3 * s * math.sqrt(1.25), m + T3 * s * math.sqrt(1.25)],
                worst_fold=float(pf.max()), n_neg=int((pf < 0).sum()))


def summarise(res, names, ref):
    o = {}
    for nm in names:
        o[nm] = dict(alc=V.mean_alc(res, nm), curve=V.mean_curve(res, nm).tolist())
        if nm != ref:
            d = V.delta(res, nm, ref)
            d.update(fold_stats(d))
            o[nm]["d"] = d
            o[nm]["d_budget"] = [V.delta(res, nm, ref, budget=j) for j in range(6)]
    return o


def log(*a):
    print(*a, flush=True)


def fmt(d):
    return (f"{d['pt']:+.3f} pair[{d['lo']:+.2f},{d['hi']:+.2f}] hier[{d['hlo']:+.2f},{d['hhi']:+.2f}] "
            f"t3[{d['t3'][0]:+.2f},{d['t3'][1]:+.2f}] PI[{d['pi'][0]:+.2f},{d['pi'][1]:+.2f}] "
            + " ".join(f"{k[:5]}={v:+.2f}" for k, v in d["per_fold"].items()))


ARMS = {
    "v1_1": BASE,
    "facets": cfg_of(facets="all"),
    "guard": cfg_of(guard=1),
    "both": cfg_of(facets="all", guard=1),
    "shuffled": cfg_of(facets="all", shuffle=0),
    "facets_named": cfg_of(facets="named"),
    "guard2": cfg_of(guard=2),
}
SEEDS = {"R2": list(range(10)), "R1": list(range(100))}


def evaluate(cfgs, regimes, seeds, D_of=None, tag=""):
    out, stats = {}, {}
    t0 = time.time()
    for R in regimes:
        res = {}
        for B in E.EVAL_FOLDS:
            D = E.compact(B) if D_of is None else D_of(B)
            ep = E.build(D, R, seeds[R])
            st = {}
            res[B] = run_cfgs(B, D, ep, V.H_fold(B), cfgs, st)
            stats[(R, B)] = {nm: dict(used_frac=float(np.mean(s["used_frac"])) if s.get("used_frac") else 0.0,
                                      w_pi=np.mean(s["w_pi"], 0).tolist() if s.get("w_pi") else None)
                             for nm, s in st.items()}
            log(f"  [{tag} {R} {B}] n_inst={ep.n} {time.time()-t0:.0f}s")
        out[R] = res
    return out, stats


def check():
    """Implementation checks 1-3 of the pre-registration."""
    worst = {"off": 0.0, "empty": 0.0, "b0": 0.0}
    for B in E.EVAL_FOLDS:
        D = E.compact(B)
        H = V.H_fold(B)
        for R, sd in (("R2", [0, 1]), ("R1", [0, 1, 2])):
            ep = E.build(D, R, sd)
            K = len(D["key_jev"])
            fe0 = encode({}, K)
            fe = fe_for(B, D, ARMS["facets"])
            for b in E.BUDGETS:
                ref = V.predict(ep, H, b, V.BASE2)
                worst["off"] = max(worst["off"], float(np.abs(predict(ep, H, b, BASE) - ref).max()))
                worst["empty"] = max(worst["empty"], float(np.abs(predict(ep, H, b, ARMS["facets"], fe0) - ref).max()))
                if b == 0:
                    worst["b0"] = max(worst["b0"], float(np.abs(predict(ep, H, b, ARMS["both"], fe) - ref).max()))
        log(f"  check {B}: {worst}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(worst, open(OUT / "check.json", "w"), indent=1)
    assert all(v < 1e-12 for v in worst.values()), worst
    return worst


def _dump(tag, out):
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / f"{tag}.json", "w", encoding="utf-8"), indent=1, default=str)


def real(arms=None, tag="real"):
    cfgs = {k: ARMS[k] for k in (arms or ARMS)}
    names = list(cfgs)
    res, stats = evaluate(cfgs, ("R2", "R1"), SEEDS, tag=tag)
    out = {"stats": {f"{R}|{B}": v for (R, B), v in stats.items()}}
    for R, rr in res.items():
        o = summarise(rr, names, "v1_1")
        out[R] = o
        log(f"[{tag} {R}]")
        for nm in names:
            x = o[nm]
            log(f"   {nm:13s} ALC={x['alc']:.5f}" + (f"  d {fmt(x['d'])}" if "d" in x else ""))
            if "d_budget" in x:
                log("      budget: " + " ".join(f"B{b}={d['pt']:+.3f}" for b, d in zip(E.BUDGETS, x["d_budget"])))
                log("      matharena budget: " + " ".join(f"B{b}={d['per_fold']['matharena']:+.3f}"
                                                         for b, d in zip(E.BUDGETS, x["d_budget"])))
    _dump(tag, out)
    return out


def merge(ms=(5, 20), tag="merge"):
    from paiec.analysis.redteam import rekey
    cfgs = {"v1_1": BASE, "guard": ARMS["guard"], "guard2": ARMS["guard2"]}
    out = {}
    for m in ms:
        res, _ = evaluate(cfgs, ("R2", "R1"), {"R2": list(range(10)), "R1": list(range(50))},
                          D_of=lambda B, m=m: rekey(B, "random", m)[0], tag=f"m{m}")
        out[f"m{m}"] = {}
        for R, rr in res.items():
            o = summarise(rr, list(cfgs), "v1_1")
            out[f"m{m}"][R] = o
            log(f"[merge m={m} {R}]")
            for nm in cfgs:
                x = o[nm]
                log(f"   {nm:8s} ALC={x['alc']:.5f}" + (f"  d {fmt(x['d'])}" if "d" in x else ""))
    _dump(tag, out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="check")
    a = ap.parse_args()
    for s in a.step.split(","):
        if s:
            globals()[s]()
