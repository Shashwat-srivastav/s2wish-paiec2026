"""E1 — where does the budget-0 / budget-1 loss of v1_1 come from?

Pre-registered in docs/preregistrations/E1_E2_preregistration.md (procedure + fixed decision table).
Report: docs/research_notes/19_E1_B0_DECOMPOSITION.md.  Run: ``python -m paiec.analysis.e1_b0_decomp``.

Predictors (each differs from v1_1 = submissions/v1_1 engine ``v1_1.BASE2`` in exactly one thing):
  v1_1            the champion (posterior over offset mu_b x pair ability theta, item level, momd guard)
  oracle_offset   mu_b fixed at the held-out benchmark's full-data mean pair logit
  oracle_subject  theta_pair fixed at (full-data pair logit - benchmark mean); mu_b still inferred
  oracle_item     item difficulty = full-data Rasch beta (centred); pair level c solves
                  mean_k sig(c + beta_k) = q over the benchmark's item keys
  oracle_all      all three oracles (the floor)
  metadata_prior  theta ~ N(m_meta, tau_meta^2), m_meta = ridge(release clipped to training range,
                  provider, size tokens) fit on the training benchmarks only (LOBO residual tau)

Leakage conditions (report 12): folds are leave-one-benchmark-out, all hyperparameters and the
metadata ridge are fit per fold on training benchmarks only, no feature uses the benchmark name
(pooling groups are opaque ints), item keys are the runtime content dict (duplicates merged).
Oracle quantities intentionally use the held-out benchmark's full data (that is what an oracle is).
"""
from __future__ import annotations

import json
import math
import re
import sys
import time

import numpy as np

from paiec import paths
from paiec.evaluation import episodes as E
from paiec.models import ladder as Lad
from paiec.models import v1_1 as V

OUT = paths.RUNS / "e1"
SEEDS = {"R2": list(range(5)), "R1": list(range(100))}     # == report 17 (phase3_v1_1)
R1_CHUNK = 25                                             # seeds per episode batch (memory only)
GH_X, GH_W, LOG_GH_W, MU = V.GH_X, V.GH_W, V.LOG_GH_W, V.MU
sig, log_sig = V.sig, V.log_sig
CFG = V.BASE2
NAMES = ["v1_1", "oracle_offset", "oracle_subject", "oracle_item", "oracle_all", "metadata_prior",
         "oracle_item_lpo", "oracle_all_lpo"]
# *_lpo = SUPPLEMENTARY sensitivity (not pre-registered, not used by the decision table): item beta
# re-estimated without the target pair's own responses, to size the in-sample optimism of the item oracle.
BUD01 = (0, 1)

SMALL = {"mini", "nano", "flash", "lite", "haiku", "small", "turbo", "fast", "air", "scout", "tiny"}
BIG = {"pro", "opus", "max", "ultra", "large", "plus", "maverick", "deep", "speciale"}


def log(*a):
    print(*a, flush=True)
    with open(OUT / "e1.log", "a", encoding="utf-8") as f:
        print(*a, file=f)


# ---------------------------------------------------------------------------
# Full-data oracle quantities for the held-out benchmark
# ---------------------------------------------------------------------------
_OR = {}


def oracle_stats(B):
    if B not in _OR:
        st = Lad.bench_stats(B)
        D = E.compact(B)
        _, be, _ = Lad.rasch(D)
        be = be - be.mean()
        # beta distribution used to define c (equal weight per key present in the data)
        present = np.unique(D["item_key"][D["row_item"]])
        P = len(D["pair_sid"])
        lpo = np.zeros((P, len(be)))
        pools = []
        for p in range(P):
            keep = D["row_pair"] != p
            Dp = dict(D, row_pair=D["row_pair"][keep], row_item=D["row_item"][keep], row_y=D["row_y"][keep])
            _, bp, _ = Lad.rasch(Dp)
            pres = np.unique(D["item_key"][Dp["row_item"]])
            bp = bp - bp[pres].mean()
            lpo[p] = bp
            pools.append(bp[pres])
        _OR[B] = dict(mu=float(st["mean"]), theta=st["y"] - st["mean"], y=st["y"], beta=be,
                      beta_pool=be[present], beta_lpo=lpo, pool_lpo=pools)
    return _OR[B]


def invert_c_emp(q, beta):
    """c with mean_k sig(c + beta_k) = q (Newton, vectorised over q)."""
    q = np.clip(q, 1e-4, 1 - 1e-4)
    c = np.log(q / (1 - q))
    for _ in range(30):
        pr = sig(c[:, None] + beta[None, :])
        f = pr.mean(1) - q
        d = (pr * (1 - pr)).mean(1)
        c -= np.clip(f / np.maximum(d, 1e-9), -3, 3)
    return c


# ---------------------------------------------------------------------------
# Metadata prior (report 14 rec. 3), training benchmarks only
# ---------------------------------------------------------------------------
def _name_tokens(nm):
    n = re.sub(r"^[nc]:", "", nm or "")
    t = re.sub(r"[-_/()]", " ", n).split()
    small = float(any(x in SMALL for x in t))
    big = float(any(x in BIG for x in t))
    m = re.findall(r"(\d+(?:\.\d+)?)b\b", n.replace("-", " "))
    lp = math.log10(float(m[0])) if m and float(m[0]) > 0 else 0.0
    return [small, big, lp, 1.0 if m else 0.0]


def meta_X(names, release, provider, M):
    rows = []
    for nm, r, pv in zip(names, release, provider):
        miss = not np.isfinite(r)
        rc = 0.0 if miss else float(np.clip(r, M["rel_lo"], M["rel_hi"]) - 2025.0)
        rows.append([rc, 1.0 if miss else 0.0] + [1.0 if pv == p else 0.0 for p in M["provs"]]
                    + _name_tokens(nm))
    return np.array(rows, float).reshape(len(rows), 2 + len(M["provs"]) + 4)


def fit_meta(train, tau, alpha=3.0):
    multi = [s for s in (Lad.bench_stats(b) for b in train) if s["P"] >= 5]
    rel = np.concatenate([s["release"] for s in multi])
    pv = np.concatenate([s["provider"] for s in multi])
    u, c = np.unique(pv, return_counts=True)
    M = dict(rel_lo=float(np.nanmin(rel)), rel_hi=float(np.nanmax(rel)),
             provs=[p for p, k in zip(u, c) if k >= 3 and p != ""])
    Xs, Ys, Ws, Vs, Bs = [], [], [], [], []
    for j, s in enumerate(multi):
        Xs.append(meta_X(s["names"], s["release"], s["provider"], M))
        Ys.append(s["y"] - s["mean"]); Ws.append(np.full(s["P"], 1.0 / s["P"])); Vs.append(s["v"])
        Bs.append(np.full(s["P"], j))
    X, Y, W, Vv, Bv = map(np.concatenate, (Xs, Ys, Ws, Vs, Bs))
    M["coef"] = Lad._ridge(X, Y, W, alpha).tolist()
    res = []
    for b in np.unique(Bv):
        tr, te = Bv != b, Bv == b
        cb = Lad._ridge(X[tr], Y[tr], W[tr], alpha)
        pred = np.hstack([X[te], np.ones((te.sum(), 1))]) @ cb
        res.append(np.mean((Y[te] - pred) ** 2 - Vv[te]))
    M["tau_meta"] = math.sqrt(float(np.clip(np.mean(res), 0.3 * tau ** 2, 1.5 * tau ** 2)))
    M["loo_mse_minus_v"] = [float(x) for x in res]
    return M


def meta_m(ep, M):
    p = ep.inst_pair
    X = meta_X(ep.D["pair_name"][p], ep.D["pair_release"][p], ep.D["pair_provider"][p], M)
    return np.hstack([X, np.ones((len(p), 1))]) @ np.array(M["coef"])


# ---------------------------------------------------------------------------
# Generalised v1_1 pair level (same math as v1_1.pooled_pair with optional oracles)
# ---------------------------------------------------------------------------
def pooled_pair_x(ep, H, li, ly, mu_fix=None, theta_fix=None, m=None, tau=None):
    n = ep.n
    S = np.bincount(li, ly, n) if len(li) else np.zeros(n)
    N = np.bincount(li, minlength=n).astype(float) if len(li) else np.zeros(n)
    grid = MU if mu_fix is None else np.array([mu_fix])
    if theta_fix is not None:
        eta = grid[None, :, None] + theta_fix[:, None, None]
        lw = np.zeros(1)
    else:
        t = np.full(n, H["tau"] if tau is None else tau)
        mm = np.zeros(n) if m is None else m
        eta = grid[None, :, None] + mm[:, None, None] + t[:, None, None] * GH_X[None, None, :]
        lw = LOG_GH_W
    ll = S[:, None, None] * log_sig(eta) + (N - S)[:, None, None] * log_sig(-eta) + lw
    mx = ll.max(2, keepdims=True)
    wt = np.exp(ll - mx)
    Ln = wt.sum(2)
    logL = np.log(Ln) + mx[:, :, 0]
    Eq = (wt * sig(eta)).sum(2) / Ln
    if mu_fix is not None:
        return Eq[:, 0]
    groups, ginv = np.unique(ep.inst_group, return_inverse=True)
    lp = np.zeros((len(groups), len(MU)))
    np.add.at(lp, ginv, logL)
    lp += V.log_prior(H, CFG["prior"])[None, :]
    lp -= lp.max(1, keepdims=True)
    post = np.exp(lp); post /= post.sum(1, keepdims=True)
    return (post[ginv] * Eq).sum(1)


def predict_x(ep, H, b, name, B, M):
    li, lk, ly = ep.labels(b)
    O = oracle_stats(B)
    th = O["theta"][ep.inst_pair]
    if name == "v1_1":
        return V.predict(ep, H, b, CFG)
    if name == "oracle_offset":
        q = pooled_pair_x(ep, H, li, ly, mu_fix=O["mu"])
    elif name == "oracle_subject":
        q = pooled_pair_x(ep, H, li, ly, theta_fix=th)
    elif name == "metadata_prior":
        q = pooled_pair_x(ep, H, li, ly, m=meta_m(ep, M), tau=M["tau_meta"])
    elif name in ("oracle_item", "oracle_item_lpo"):
        q = V.pooled_pair(ep, H, li, ly, CFG)[0]
    elif name in ("oracle_all", "oracle_all_lpo"):
        q = sig(O["y"][ep.inst_pair])
    else:
        raise ValueError(name)
    if name in ("oracle_item", "oracle_all"):
        c = invert_c_emp(q, O["beta_pool"])
        p = sig(c[ep.t_inst] + O["beta"][ep.t_key])
    elif name in ("oracle_item_lpo", "oracle_all_lpo"):
        c = np.zeros(ep.n)
        for pp in np.unique(ep.inst_pair):
            m = ep.inst_pair == pp
            c[m] = invert_c_emp(q[m], O["pool_lpo"][pp])
        p = sig(c[ep.t_inst] + O["beta_lpo"][ep.inst_pair[ep.t_inst], ep.t_key])
    else:
        p, _ = V.item_level(ep, H, q, li, lk, ly, CFG["rho"], CFG.get("rho_pool", False))
    return np.clip(p, 0.005, 0.995)


# ---------------------------------------------------------------------------
# Scoring: per-instance Brier, pair-equal row weights, Murphy terms
# ---------------------------------------------------------------------------
def inst_brier(ep, p):
    cnt = np.bincount(ep.r_inst, minlength=ep.n)
    pr = p[ep.r_t]
    br = np.bincount(ep.r_inst, (pr - ep.r_y) ** 2, ep.n) / cnt
    pbar = np.bincount(ep.r_inst, pr, ep.n) / cnt
    ybar = np.bincount(ep.r_inst, ep.r_y, ep.n) / cnt
    return br, (pbar - ybar) ** 2


class Acc:
    """Accumulates rows across episode chunks of one fold (pair-equal weights at the end)."""

    def __init__(self):
        self.pair, self.br, self.citl, self.rows = [], {}, {}, {}

    def add(self, ep, probs):
        self.pair.append(ep.inst_pair.copy())
        cnt = np.bincount(ep.r_inst, minlength=ep.n)
        for nm, pb in probs.items():
            for b, p in pb.items():
                br, ct = inst_brier(ep, p)
                self.br.setdefault((nm, b), []).append(br)
                self.citl.setdefault((nm, b), []).append(ct)
                if b in BUD01:
                    pr = p[ep.r_t]
                    P = len(ep.D["pair_sid"])
                    g = ep.inst_pair[ep.r_inst] * 10 + np.minimum((pr * 10).astype(int), 9)
                    w0 = 1.0 / cnt[ep.r_inst]
                    a = np.stack([np.bincount(g, w0 * v, P * 10) for v in (1.0, pr, ep.r_y, (pr - ep.r_y) ** 2)])
                    self.rows[(nm, b)] = self.rows.get((nm, b), 0) + a

    def finish(self):
        pair = np.concatenate(self.pair)
        up, pinv = np.unique(pair, return_inverse=True)
        n_inst = np.bincount(pinv)
        out = {"pair": pair}
        for k, v in self.br.items():
            out[("br",) + k] = np.concatenate(v)
            out[("citl",) + k] = np.concatenate(self.citl[k])
        for k, a in self.rows.items():
            a = a.reshape(4, -1, 10)                       # stat x pair x bin
            wp = np.zeros(a.shape[1]); wp[up] = 1.0 / n_inst / len(up)
            a = a * wp[None, :, None]
            out[("murphy",) + k] = murphy(a.sum(1))
        return out


def murphy(a):
    """a: 4 x 10 pair-equal weighted bin sums of (w, w p, w y, w (p-y)^2)."""
    W, sp, sy, sb = a
    ok = W > 0
    pk, yk = sp[ok] / W[ok], sy[ok] / W[ok]
    tot = W.sum()
    yb = float(sy.sum() / tot)
    rel = float((W[ok] * (pk - yk) ** 2).sum() / tot)
    res = float((W[ok] * (yk - yb) ** 2).sum() / tot)
    unc = yb * (1 - yb)
    brier = float(sb.sum() / tot)
    pb = float(sp.sum() / tot)
    return dict(REL=rel, RES=res, UNC=unc, brier=brier, resid=brier - (rel - res + unc),
                bench_citl=(pb - yb) ** 2, pbar=pb, ybar=yb)


def pm(x, pair):
    return V.pair_mean(x, pair)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_regime(R):
    folds = {}
    t0 = time.time()
    for B in E.EVAL_FOLDS:
        H = V.H_fold(B)
        M = fit_meta(H["train"], H["tau"])
        D = E.compact(B)
        acc = Acc()
        seeds = SEEDS[R]
        chunks = [seeds] if R == "R2" else [seeds[i:i + R1_CHUNK] for i in range(0, len(seeds), R1_CHUNK)]
        for ch in chunks:
            ep = E.build(D, R, ch)
            probs = {}
            for nm in NAMES:
                buds = E.BUDGETS if nm in ("v1_1", "metadata_prior") else BUD01
                probs[nm] = {b: predict_x(ep, H, b, nm, B, M) for b in buds}
            acc.add(ep, probs)
            del ep, probs
        folds[B] = dict(res=acc.finish(), meta=dict(provs=M["provs"], tau_meta=M["tau_meta"], tau=H["tau"],
                                                   rel_hi=M["rel_hi"], coef=M["coef"]))
        log(f"  [{R}] {B}: {time.time() - t0:.0f}s, instances={len(folds[B]['res']['pair'])}")
    return folds


def summarise(R, folds):
    o = {"per_fold": {}, "mean": {}}
    for B, f in folds.items():
        r, pair = f["res"], f["res"]["pair"]
        d = {"n_pairs": int(len(np.unique(pair))), "n_instances": int(len(pair)), "meta_fit": f["meta"]}
        for nm in NAMES:
            for b in BUD01:
                d[f"{nm}|B{b}"] = float(pm(r[("br", nm, b)], pair).mean())
                d[f"{nm}|B{b}|offset_term"] = float(pm(r[("citl", nm, b)], pair).mean())
                d[f"{nm}|B{b}|rest"] = d[f"{nm}|B{b}"] - d[f"{nm}|B{b}|offset_term"]
                d[f"{nm}|B{b}|murphy"] = r[("murphy", nm, b)]
        for nm in ("v1_1", "metadata_prior"):
            alc = sum(E.W[j] * r[("br", nm, b)] for j, b in enumerate(E.BUDGETS))
            d[f"{nm}|ALC"] = float(pm(alc, pair).mean())
            d[f"{nm}|curve"] = [float(pm(r[("br", nm, b)], pair).mean()) for b in E.BUDGETS]
        o["per_fold"][B] = d
    keys = [k for k in o["per_fold"][E.EVAL_FOLDS[0]] if k not in ("meta_fit",)]
    for k in keys:
        vals = [o["per_fold"][B][k] for B in E.EVAL_FOLDS]
        if isinstance(vals[0], dict):
            o["mean"][k] = {kk: float(np.mean([v[kk] for v in vals])) for kk in vals[0]}
        elif isinstance(vals[0], list):
            o["mean"][k] = np.mean(vals, 0).tolist()
        else:
            o["mean"][k] = float(np.mean(vals))
    # headroom shares (on fold means, and per fold)
    o["shares"] = {}
    for b in BUD01:
        sh = {}
        for scope, src in [("mean", o["mean"])] + [(B, o["per_fold"][B]) for B in E.EVAL_FOLDS]:
            base, floor = src[f"v1_1|B{b}"], src[f"oracle_all|B{b}"]
            sh[scope] = {X: (base - src[f"oracle_{X}|B{b}"]) / (base - floor) for X in ("offset", "subject", "item")}
            sh[scope]["headroom"] = base - floor
            fl = src[f"oracle_all_lpo|B{b}"]
            sh[scope]["SUPP_lpo"] = {"item": (base - src[f"oracle_item_lpo|B{b}"]) / (base - fl),
                                     "offset": (base - src[f"oracle_offset|B{b}"]) / (base - fl),
                                     "subject": (base - src[f"oracle_subject|B{b}"]) / (base - fl),
                                     "headroom": base - fl}
        o["shares"][f"B{b}"] = sh
    # metadata delta (paired over pairs, x1e-3)
    res = {B: {nm: dict(alc=sum(E.W[j] * f["res"][("br", nm, bb)] for j, bb in enumerate(E.BUDGETS)),
                        br=np.stack([f["res"][("br", nm, bb)] for bb in E.BUDGETS], 1),
                        pair=f["res"]["pair"]) for nm in ("v1_1", "metadata_prior")}
           for B, f in folds.items()}
    dm = V.delta(res, "metadata_prior", "v1_1")
    dm["per_budget"] = [V.delta(res, "metadata_prior", "v1_1", budget=j) for j in range(6)]
    pf = list(dm["per_fold"].values())
    dm["folds_negative"] = int(sum(x < 0 for x in pf))
    dm["max_fold"] = float(max(pf))
    dm["meets_threshold"] = bool(dm["pt"] <= -1.0 and dm["hi"] < 0 and dm["folds_negative"] >= 3 and dm["max_fold"] <= 1.0)
    o["metadata_dALC"] = dm
    return o


def decide(S):
    sh = S["R2"]["shares"]["B0"]["mean"]
    meta_ok = S["R2"]["metadata_dALC"]["meets_threshold"]
    rows = []
    if sh["offset"] >= 0.50:
        rows.append(("share_offset >= 0.50 at B0", "Close the metadata question for good; next investment is a base-rate prior (H5)."))
    if sh["subject"] >= 0.20 and meta_ok:
        rows.append(("share_subject >= 0.20 and metadata meets threshold", "Add metadata prior to a v1_2 candidate (red-team first)."))
    if sh["subject"] >= 0.20 and not meta_ok:
        rows.append(("share_subject >= 0.20 but metadata fails threshold", "Record as limitation; no change."))
    if sh["item"] >= 0.20:
        rows.append(("share_item >= 0.20", "Raises the value of E2 (Jev / distilled prior)."))
    if not rows:
        rows.append(("none of the above", "No change; stop optimising B0."))
    return dict(regime="R2", shares_B0=sh, metadata_meets_threshold=meta_ok,
                fired=[r[0] for r in rows], actions=[r[1] for r in rows])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "e1.log").write_text("", encoding="utf-8")
    log(f"E1 start; seeds R2={SEEDS['R2']} R1=0..{SEEDS['R1'][-1]}; cfg={CFG}")
    S = {}
    for R in ("R2", "R1"):
        S[R] = summarise(R, run_regime(R))
        m = S[R]["mean"]
        for b in BUD01:
            log(f"[{R} B{b}] " + " ".join(f"{nm}={m[f'{nm}|B{b}']:.4f}" for nm in NAMES))
            sh = S[R]["shares"][f"B{b}"]["mean"]
            log(f"   shares: offset={sh['offset']:.3f} subject={sh['subject']:.3f} item={sh['item']:.3f} headroom={sh['headroom']:.4f}")
        log(f"[{R}] ALC v1_1={m['v1_1|ALC']:.4f} meta={m['metadata_prior|ALC']:.4f}  dALC {V.fmt(S[R]['metadata_dALC'])}"
            f" meets={S[R]['metadata_dALC']['meets_threshold']}")
    S["decision"] = decide(S)
    log("DECISION:", json.dumps(S["decision"], indent=1))
    json.dump(S, open(OUT / "e1_results.json", "w", encoding="utf-8"), indent=1, default=str)
    return S


if __name__ == "__main__":
    main()
