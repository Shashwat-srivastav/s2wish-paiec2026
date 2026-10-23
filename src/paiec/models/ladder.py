"""Phase-2 baseline ladder: predictors, training-fold hyperparameters, LOBO evaluation.

Model (logit scale; pair-mean = marginal over items):
    logit q_pair = mu_b + theta_pair,  mu_b ~ t3(G, s_B),  theta ~ N(m_pair, tau_pair^2)
    item level (conditional scale): logit p_{pair,i} = c_pair + beta_i,
        beta_i ~ N(-gamma*z_jev_i, sigma_r^2),  c_pair solves E_beta[sig(c+beta)] = q_pair
Posterior integration: exact grid over mu_b (121 pts) x Gauss-Hermite over theta (20)
per pooling group; Gauss-Hermite (20) posterior over beta_i given all other labels on the
same runtime item dict. No plug-in.

Rungs (cumulative unless noted):
  L0 empirical mean (official external/official_baselines/empirical_mean)
  L1 v0 Beta-binomial m=.5,k=2 (as shipped)            L1f BB with m,k fit on training folds
  L2 + pooled benchmark offset (t3 prior), theta~N(0,tau^2)
  L3 + subject name link: m_pair = lambda * a_name (training-benchmark name table)
  L4 + item difficulty pooled across pairs (same runtime item dict)
  L5 + Jev difficulty prior (OFFLINE ONLY: Jev not callable at runtime)
  L6 + subject metadata prior (release clipped, provider) jointly with name
  L6nj = L6 without Jev (shippable)
Hyperparameters are fit ONLY on training benchmarks (all public benchmarks except the
held-out one; inner folds additionally drop the inner held-out benchmark).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from paiec import paths
from paiec.evaluation import episodes as E

GH_X, GH_W = np.polynomial.hermite_e.hermegauss(20)
GH_W = GH_W / GH_W.sum()
MU_GRID = np.linspace(-7.0, 5.0, 121)
OUT = paths.RUNS / "phase2"


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def log_sig(x):
    return -np.logaddexp(0.0, -x)


# ---------------------------------------------------------------------------
# Benchmark-local statistics (cached; depend only on that benchmark's own data)
# ---------------------------------------------------------------------------
_BSTATS = {}


def rasch(D, iters=150):
    cp = D["row_pair"].astype(np.int64)
    ck = D["item_key"][D["row_item"]].astype(np.int64)
    P, K = len(D["pair_sid"]), len(D["key_jev"])
    cell = cp * K + ck
    u, inv = np.unique(cell, return_inverse=True)
    S = np.bincount(inv, D["row_y"].astype(float))
    N = np.bincount(inv).astype(float)
    up, uk = u // K, u % K
    th = np.zeros(P); be = np.zeros(K); s2 = 4.0
    for it in range(iters):
        pr = sig(th[up] + be[uk])
        g = np.bincount(up, S - N * pr, P) - th / 100.0
        h = np.bincount(up, N * pr * (1 - pr), P) + 1 / 100.0
        th += np.clip(g / h, -1, 1)
        pr = sig(th[up] + be[uk])
        g = np.bincount(uk, S - N * pr, K) - be / s2
        hb = np.bincount(uk, N * pr * (1 - pr), K) + 1 / s2
        be += np.clip(g / hb, -1, 1)
        m = be.mean(); be -= m; th += m
        if it % 10 == 9:
            s2 = float(np.mean(be ** 2 + 1 / hb))
    return th, be, math.sqrt(s2)


def bench_stats(bench):
    if bench in _BSTATS:
        return _BSTATS[bench]
    D = E.compact(bench)
    P = len(D["pair_sid"])
    S = np.bincount(D["row_pair"], D["row_y"].astype(float), P)
    n = np.bincount(D["row_pair"], minlength=P).astype(float)
    q = (S + 0.5) / (n + 1)
    y = np.log(q / (1 - q))
    v = 1 / ((n + 1) * q * (1 - q))
    th, be, sb = rasch(D)
    jz = D["key_jev"]
    ok = np.isfinite(jz)
    gamma = float("nan")
    if ok.sum() > 50:
        z = (jz[ok] - jz[ok].mean()) / (jz[ok].std() + 1e-9)
        gamma = float(np.dot(be[ok] - be[ok].mean(), z) / np.dot(z, z))  # easiness per SD
    st = dict(bench=bench, P=P, acc=S / n, y=y, v=v, mean=float(y.mean()),
              tau2=float(y.var(ddof=1) - v.mean()) if P >= 5 else float("nan"),
              sigma_beta=sb, gamma=gamma, names=D["pair_name"], release=D["pair_release"],
              provider=D["pair_provider"])
    _BSTATS[bench] = st
    return st


# ---------------------------------------------------------------------------
# Hyperparameters from training benchmarks only
# ---------------------------------------------------------------------------
def _name_table(stats):
    tab = {}
    for st in stats:
        if st["P"] < 5:  # 1-subject benchmark: deviation is identically 0, uninformative
            continue
        dev = st["y"] - st["mean"]
        acc = {}
        for nm, d in zip(st["names"], dev):
            if nm:
                acc.setdefault(nm, []).append(d)
        for nm, ds in acc.items():
            tab.setdefault(nm, {})[st["bench"]] = float(np.mean(ds))
    return tab


def name_ability(tab, nm, exclude=()):
    d = tab.get(nm) if nm else None
    if not d:
        return None
    vals = [v for b, v in d.items() if b not in exclude]
    return float(np.mean(vals)) if vals else None


def _meta_x(names, release, provider, tab, provs, rel_lo, rel_hi, exclude_bench=None):
    rows = []
    for nm, r, pv in zip(names, release, provider):
        a = name_ability(tab, nm, exclude=(exclude_bench,) if exclude_bench else ())
        rm = not np.isfinite(r)
        rc = 0.0 if rm else float(np.clip(r, rel_lo, rel_hi) - 2025.0)
        x = [a if a is not None else 0.0, 1.0 if a is not None else 0.0, rc, 1.0 if rm else 0.0]
        x += [1.0 if pv == p else 0.0 for p in provs]
        rows.append(x)
    return np.array(rows, float).reshape(len(rows), 4 + len(provs))


def _ridge(X, y, w, alpha):
    Xa = np.hstack([X, np.ones((len(X), 1))])
    R = alpha * np.eye(Xa.shape[1]); R[-1, -1] = 1e-6
    A = Xa.T @ (Xa * w[:, None]) + R
    return np.linalg.solve(A, Xa.T @ (w * y))


def fit_hyper(train, alpha=3.0):
    stats = [bench_stats(b) for b in train]
    means = np.array([s["mean"] for s in stats])
    G, sB = float(means.mean()), float(max(means.std(ddof=1), 0.5))
    tau = math.sqrt(np.mean([s["tau2"] for s in stats if np.isfinite(s["tau2"])]))
    sbeta = math.sqrt(np.mean([s["sigma_beta"] ** 2 for s in stats]))
    gam = np.array([s["gamma"] for s in stats if np.isfinite(s["gamma"])])
    gamma = float(gam.mean()) if len(gam) else 0.0
    # fitted Beta-binomial (benchmark-balanced moment match)
    accs = [s["acc"] for s in stats if s["P"] >= 5]
    m_bb = float(np.mean([a.mean() for a in accs]))
    eqq = np.mean([np.mean(a * (1 - a)) for a in accs])
    emq = np.mean([np.mean((a - m_bb) ** 2) for a in accs])
    k_bb = float(eqq / emq)
    # name link: lambda (through origin) + out-of-benchmark residual variance
    tab = _name_table(stats)
    multi = [s for s in stats if s["P"] >= 5]
    A, Y, Wt, V = [], [], [], []
    for s in multi:
        dev = s["y"] - s["mean"]
        for nm, d, v in zip(s["names"], dev, s["v"]):
            a = name_ability(tab, nm, exclude=(s["bench"],))
            if a is not None:
                A.append(a); Y.append(d); Wt.append(1.0 / s["P"]); V.append(v)
    A, Y, Wt, V = map(np.array, (A, Y, Wt, V))
    lam = float(np.clip(np.sum(Wt * A * Y) / max(np.sum(Wt * A * A), 1e-9), 0, 1)) if len(A) else 0.0
    tau_name = math.sqrt(float(np.clip(np.sum(Wt * ((Y - lam * A) ** 2 - V)) / np.sum(Wt),
                                       0.3 * tau ** 2, 1.5 * tau ** 2))) if len(A) else tau
    # metadata ridge (name ability, release clipped, provider); LOBO residuals for tau
    rel_all = np.concatenate([s["release"] for s in multi])
    rel_lo, rel_hi = float(np.nanmin(rel_all)), float(np.nanmax(rel_all))
    pv_all = np.concatenate([s["provider"] for s in multi])
    u, c = np.unique(pv_all, return_counts=True)
    provs = [p for p, k in zip(u, c) if k >= 3]
    Xs, Ys, Ws, Vs, Bs = [], [], [], [], []
    for s in multi:
        Xs.append(_meta_x(s["names"], s["release"], s["provider"], tab, provs, rel_lo, rel_hi, s["bench"]))
        Ys.append(s["y"] - s["mean"]); Ws.append(np.full(s["P"], 1.0 / s["P"])); Vs.append(s["v"])
        Bs.append(np.full(s["P"], len(Bs)))
    X, Yv, Wv, Vv, Bv = map(np.concatenate, (Xs, Ys, Ws, Vs, Bs))
    coef = _ridge(X, Yv, Wv, alpha)
    res = []
    for b in np.unique(Bv):
        tr, te = Bv != b, Bv == b
        cb = _ridge(X[tr], Yv[tr], Wv[tr], alpha)
        pred = np.hstack([X[te], np.ones((te.sum(), 1))]) @ cb
        res.append(np.mean((Yv[te] - pred) ** 2 - Vv[te]))
    tau_meta = math.sqrt(float(np.clip(np.mean(res), 0.3 * tau ** 2, 1.5 * tau ** 2)))
    return dict(train=list(train), G=G, sB=sB, tau=tau, sigma_beta=sbeta, gamma=gamma,
                sigma_r=math.sqrt(max(sbeta ** 2 - gamma ** 2, 0.25)), m_bb=m_bb, k_bb=k_bb,
                lam=lam, tau_name=tau_name, name_table=tab, provs=provs, rel_lo=rel_lo,
                rel_hi=rel_hi, meta_coef=coef.tolist(), tau_meta=tau_meta, T=1.0)


# ---------------------------------------------------------------------------
# Predictors (vectorised over an Episodes batch)
# ---------------------------------------------------------------------------
def t3_logpdf(x, loc, scale):
    z = (x - loc) / scale
    return -2.0 * np.log1p(z * z / 3.0)


def pair_prior(ep, H, mode):
    D = ep.D
    p = ep.inst_pair
    n = ep.n
    if mode == "none":
        return np.zeros(n), np.full(n, H["tau"])
    names, rel, prov = D["pair_name"][p], D["pair_release"][p], D["pair_provider"][p]
    if mode == "name":
        m = np.zeros(n); t = np.full(n, H["tau"])
        for j, nm in enumerate(names):
            a = name_ability(H["name_table"], nm)
            if a is not None:
                m[j] = H["lam"] * a; t[j] = H["tau_name"]
        return m, t
    X = _meta_x(names, rel, prov, H["name_table"], H["provs"], H["rel_lo"], H["rel_hi"])
    m = np.hstack([X, np.ones((n, 1))]) @ np.array(H["meta_coef"])
    return m, np.full(n, H["tau_meta"])


def pooled_pair(ep, H, li, ly, prior_mode):
    """Posterior predictive pair-mean q per instance (exact grid x GH)."""
    n = ep.n
    S = np.bincount(li, ly, n) if len(li) else np.zeros(n)
    N = np.bincount(li, minlength=n).astype(float) if len(li) else np.zeros(n)
    m, t = pair_prior(ep, H, prior_mode)
    eta = MU_GRID[None, :, None] + m[:, None, None] + t[:, None, None] * GH_X[None, None, :]
    ll = S[:, None, None] * log_sig(eta) + (N - S)[:, None, None] * log_sig(-eta)
    ll = ll + np.log(GH_W)[None, None, :]
    mx = ll.max(2, keepdims=True)
    w = np.exp(ll - mx)
    Ln = w.sum(2)                                   # n x G (scaled)
    logL = np.log(Ln) + mx[:, :, 0]
    Eq = (w * sig(eta)).sum(2) / Ln                 # n x G
    groups, ginv = np.unique(ep.inst_group, return_inverse=True)
    lp = np.zeros((len(groups), len(MU_GRID)))
    np.add.at(lp, ginv, logL)
    lp += t3_logpdf(MU_GRID, H["G"], H["sB"])[None, :]
    lp -= lp.max(1, keepdims=True)
    post = np.exp(lp); post /= post.sum(1, keepdims=True)
    return (post[ginv] * Eq).sum(1)


def _invert_c(q, s):
    """c with E_{x~N(0,1)} sig(c + s x) = q (Newton, vectorised)."""
    q = np.clip(q, 1e-4, 1 - 1e-4)
    c = np.log(q / (1 - q)) * math.sqrt(1 + math.pi * s * s / 8)
    for _ in range(12):
        pr = sig(c[:, None] + s * GH_X[None, :])
        f = pr @ GH_W - q
        d = (pr * (1 - pr)) @ GH_W
        c -= f / np.maximum(d, 1e-9)
    return c


def item_level(ep, H, q, li, lk, ly, use_jev):
    """Item posterior from all labels on the same runtime item dict in the same group."""
    sb = H["sigma_beta"]
    c = _invert_c(q, sb)
    K = len(ep.D["key_jev"])
    if use_jev:
        jz = ep.D["key_jev"]
        ok = np.isfinite(jz)
        z = np.zeros(K)
        z[ok] = (jz[ok] - jz[ok].mean()) / (jz[ok].std() + 1e-9)
        mu_b = H["gamma"] * z
        sr = H["sigma_r"]
    else:
        mu_b = np.zeros(K); sr = sb
    g_t = ep.inst_group[ep.t_inst] * K + ep.t_key
    out = np.empty(len(ep.t_key))
    LL = {}
    if len(li):
        g_l = ep.inst_group[li] * K + lk
        uk, inv = np.unique(g_l, return_inverse=True)
        bnodes = mu_b[uk % K][:, None] + sr * GH_X[None, :]
        e = c[li][:, None] + bnodes[inv]
        contrib = np.where(ly[:, None] > 0.5, log_sig(e), log_sig(-e))
        acc = np.zeros((len(uk), len(GH_X)))
        np.add.at(acc, inv, contrib)
        pos = np.searchsorted(uk, g_t)
        pos = np.minimum(pos, len(uk) - 1)
        has = uk[pos] == g_t
    else:
        has = np.zeros(len(g_t), bool)
    # targets without item evidence
    nh = ~has
    if use_jev:
        bn = mu_b[ep.t_key[nh]][:, None] + sr * GH_X[None, :]
        out[nh] = sig(c[ep.t_inst[nh]][:, None] + bn) @ GH_W
    else:
        out[nh] = q[ep.t_inst[nh]]
    if has.any():
        a = acc[pos[has]] + np.log(GH_W)[None, :]
        a = np.exp(a - a.max(1, keepdims=True)); a /= a.sum(1, keepdims=True)
        bn = mu_b[ep.t_key[has]][:, None] + sr * GH_X[None, :]
        out[has] = (a * sig(c[ep.t_inst[has]][:, None] + bn)).sum(1)
    return out, float(has.mean())


RUNGS = ["L0", "L1", "L1f", "L2", "L3", "L4", "L5", "L6", "L6nj"]
SPEC = {  # prior_mode, item, jev
    "L2": ("none", False, False), "L3": ("name", False, False), "L4": ("name", True, False),
    "L5": ("name", True, True), "L6": ("meta", True, True), "L6nj": ("meta", True, False),
    "L2i": ("none", True, False), "L6p": ("meta", False, False),
}


def predict_budget(ep, H, rung, b):
    li, lk, ly = ep.labels(b)
    n = ep.n
    S = np.bincount(li, ly, n) if len(li) else np.zeros(n)
    N = np.bincount(li, minlength=n).astype(float) if len(li) else np.zeros(n)
    overlap = 0.0
    if rung == "L0":
        q = np.where(N > 0, S / np.maximum(N, 1), 0.5)
        return q[ep.t_inst], overlap
    if rung == "L1":
        q = np.clip((1.0 + S) / (2.0 + N), 0.02, 0.98)
        return q[ep.t_inst], overlap
    if rung == "L1f":
        q = (H["k_bb"] * H["m_bb"] + S) / (H["k_bb"] + N)
        p = q[ep.t_inst]
    else:
        pm, item, jev = SPEC[rung]
        q = pooled_pair(ep, H, li, ly, pm)
        if item or jev:
            p, overlap = item_level(ep, H, q, li, lk, ly, jev)
        else:
            p = q[ep.t_inst]
    T = H.get("T", 1.0)
    if T != 1.0:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        p = sig(T * np.log(p / (1 - p)))
    return np.clip(p, 0.005, 0.995), overlap


def run_rung(ep, H, rung):
    probs, ov = {}, {}
    for b in E.BUDGETS:
        probs[b], ov[b] = predict_budget(ep, H, rung, b)
    br, ec = ep.score(probs)
    return br, ec, ov, probs


def fit_temperature(train_eval, H_by_inner, rung, seeds):
    """T minimising pair-equal, ALC-weighted Brier on inner R1 episodes."""
    cache = []
    for b, H in H_by_inner.items():
        ep = E.build(E.compact(b), "R1", seeds)
        H1 = dict(H, T=1.0)
        lp = {}
        for bb in E.BUDGETS:
            p, _ = predict_budget(ep, H1, rung, bb)
            lp[bb] = np.log(p / (1 - p))
        cache.append((ep, lp))
    grid = np.linspace(0.5, 1.4, 19)
    best = None
    for T in grid:
        tot = 0.0
        for ep, lp in cache:
            br, _ = ep.score({bb: np.clip(sig(T * lp[bb]), 0.005, 0.995) for bb in E.BUDGETS})
            tot += float((br @ E.W).mean())
        if best is None or tot < best[1]:
            best = (float(T), tot)
    return best[0]


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------
SEEDS = {"R1": list(range(40)), "R2": list(range(5)), "R3": list(range(5))}
CAL_RUNGS = ["L2", "L4", "L6nj", "L6"]


def hyper_for(B):
    return fit_hyper([b for b in E.BENCHES if b != B])


def run_all(regimes=("R1", "R2", "R3"), folds=E.EVAL_FOLDS, out=OUT / "phase2_results.npz"):
    res = {}
    temps = {}
    for B in folds:
        H = hyper_for(B)
        inner = {b: fit_hyper([x for x in E.BENCHES if x not in (B, b)])
                 for b in E.EVAL_FOLDS if b != B}
        for r in CAL_RUNGS:
            temps[(B, r)] = fit_temperature(None, inner, r, list(range(100, 115)))
        print(B, "T", {r: temps[(B, r)] for r in CAL_RUNGS}, flush=True)
        D = E.compact(B)
        for R in regimes:
            ep = E.build(D, R, SEEDS[R])
            for r in RUNGS + [x + "+T" for x in CAL_RUNGS] + ["L2i", "L6p"]:
                base = r[:-2] if r.endswith("+T") else r
                HH = dict(H, T=temps[(B, base)]) if r.endswith("+T") else H
                t0 = time.time()
                br, ec, ov, _ = run_rung(ep, HH, base)
                res[(R, B, r)] = dict(br=br, ec=ec, ov=np.array([ov[b] for b in E.BUDGETS]),
                                      pair=ep.inst_pair, seed=ep.inst_seed,
                                      sec=(time.time() - t0) / max(len(ep.t_key), 1))
            print(B, R, "done", ep.n, "instances", flush=True)
    flat = {}
    for (R, B, r), v in res.items():
        for k, a in v.items():
            flat[f"{R}|{B}|{r}|{k}"] = np.asarray(a)
    np.savez(out, **flat)
    json.dump({f"{B}|{r}": t for (B, r), t in temps.items()}, open(OUT / "phase2_temps.json", "w"), indent=1)
    return res


def load_results(path=OUT / "phase2_results.npz"):
    z = np.load(path)
    res = {}
    for key in z.files:
        R, B, r, k = key.split("|")
        res.setdefault((R, B, r), {})[k] = z[key]
    return res


def summarize(res, R, rungs, folds=E.EVAL_FOLDS, nboot=2000):
    """Fold-balanced means; sd over seeds; paired bootstrap over pairs (within fold)."""
    rng = np.random.default_rng(0)
    rows = {}
    for r in rungs:
        per_fold = []
        seeds_vals = {}
        for B in folds:
            v = res[(R, B, r)]
            alc = v["br"] @ E.W
            ecew = v["ec"] @ E.W
            per_fold.append(np.r_[v["br"].mean(0), alc.mean(), ecew.mean(), v["ec"].mean(0)])
            for s in np.unique(v["seed"]):
                m = v["seed"] == s
                seeds_vals.setdefault(int(s), []).append(np.r_[v["br"][m].mean(0), alc[m].mean(), ecew[m].mean()])
        mean = np.mean(per_fold, 0)
        sv = [np.mean(x, 0) for x in seeds_vals.values() if len(x) == len(folds)]
        sd = np.std(sv, 0, ddof=1) if len(sv) > 1 else np.zeros(8)
        rows[r] = dict(mean=mean, sd=sd, per_fold={B: pf[6] for B, pf in zip(folds, per_fold)})
    def delta(r, ref):
        fold_d = []
        for B in folds:
            a, b = res[(R, B, r)], res[(R, B, ref)]
            d = (a["br"] @ E.W) - (b["br"] @ E.W)
            pairs = a["pair"]
            up, inv = np.unique(pairs, return_inverse=True)
            fold_d.append(np.bincount(inv, d) / np.bincount(inv))
        point = np.mean([x.mean() for x in fold_d])
        boots = np.array([np.mean([x[rng.integers(0, len(x), len(x))].mean() for x in fold_d])
                          for _ in range(nboot)])
        return point, np.percentile(boots, 2.5), np.percentile(boots, 97.5)
    return rows, delta


def table(res, R, order, refs):
    rows, delta = summarize(res, R, order)
    head = "| rung | B0 | B1 | B3 | B7 | B15 | B31 | ALC | ECE(ALC-wtd) | ΔALC vs ref [95% CI] |"
    lines = [head, "|" + "---|" * 10]
    for r in order:
        m, s = rows[r]["mean"], rows[r]["sd"]
        cells = [f"{m[j]:.4f}±{s[j]:.4f}" if j in (6,) else f"{m[j]:.4f}" for j in range(8)]
        ref = refs.get(r)
        if ref:
            p, lo, hi = delta(r, ref)
            dtxt = f"{p*1000:+.1f} [{lo*1000:+.1f},{hi*1000:+.1f}] vs {ref}"
        else:
            dtxt = "—"
        lines.append(f"| {r} | " + " | ".join(cells) + f" | {dtxt} |")
    return "\n".join(lines), rows


# ---------------------------------------------------------------------------
# v1 packaging + fidelity check (engine vs packaged model.py through simulator)
# ---------------------------------------------------------------------------
V1 = paths.SUBMISSIONS / "v1"


def build_v1(prior_mode="meta", item_pool=True, T=None):
    """Fit on ALL public benchmarks (no held-out data exists at submission time)."""
    H = fit_hyper(list(E.BENCHES))
    if T is None:  # temperature from LOBO inner episodes over all eval folds
        inner = {b: fit_hyper([x for x in E.BENCHES if x != b]) for b in E.EVAL_FOLDS}
        rung = {("meta", True): "L6nj", ("name", True): "L4", ("none", False): "L2", ("none", True): "L2i",
                ("meta", False): "L6p"}[(prior_mode, item_pool)]
        T = fit_temperature(None, inner, rung, list(range(200, 215)))
    tab = {nm: float(np.mean(list(d.values()))) for nm, d in H["name_table"].items()}
    params = {k: H[k] for k in ("G", "sB", "tau", "sigma_beta", "lam", "tau_name", "provs",
                                "rel_lo", "rel_hi", "meta_coef", "tau_meta")}
    params.update(name_table=tab, T=float(T), prior_mode=prior_mode, item_pool=bool(item_pool),
                  clip=[0.005, 0.995], provs=[str(p) for p in H["provs"]],
                  _fit_on=list(E.BENCHES), _source="src/paiec/models/ladder.py build_v1")
    V1.mkdir(parents=True, exist_ok=True)
    with open(V1 / "v1_params.json", "w", encoding="utf-8") as fh:
        json.dump(params, fh, indent=1)
    return params


def check_v1(bench="real_webagents", seed=0, n_pairs=6):
    """Run packaged model.py via simulator.run_direct (official client semantics) on a
    real episode and compare with the vectorised engine on the same split/order."""
    from paiec.evaluation import simulator as sim
    from paiec.evaluation.simulator import load_model_module
    rows_all = sim.load_benchmark_dir(str(E.DATA / bench))
    rows_all = [r for r in rows_all if r["y"] in (0, 1)]
    D = E.compact(bench)
    sids = [str(s) for s in D["pair_sid"][:n_pairs]]
    rows = [r for r in rows_all if str(r["subject_id"]) in set(sids)]
    del rows_all
    pairs = sim.prepare_pairs(rows, seed=seed)
    sys.path.insert(0, str(V1))
    mod = load_model_module(V1 / "model.py")
    lab = load_model_module(V1 / "labeling.py")
    t0 = time.time()
    out = sim.run_direct(pairs, mod.predict, lab.acquisition_function, isolation="call")
    wall = time.time() - t0
    calls = out["client_stats"]["prediction_calls"]
    # engine with identical params
    prm = json.load(open(V1 / "v1_params.json"))
    H = dict(prm)
    H["name_table"] = {k: {"all": v} for k, v in prm["name_table"].items()}
    rung = {("meta", True): "L6nj", ("name", True): "L4", ("none", False): "L2", ("none", True): "L2i",
            ("meta", False): "L6p"}[(prm["prior_mode"], prm["item_pool"])]
    ep = E.Episodes(D, list(range(n_pairs)), [seed] * n_pairs, [0] * n_pairs)
    br, ec, _, _ = run_rung(ep, H, rung)
    sim_alc = np.array([out["per_pair"][(s, bench)]["alc"] for s in sids])
    eng_alc = br @ E.W
    return dict(sim_alc=sim_alc.round(5).tolist(), eng_alc=eng_alc.round(5).tolist(),
                max_abs_diff=float(np.abs(sim_alc - eng_alc).max()), calls=calls,
                ms_per_call=1000 * wall / max(calls, 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--build-v1", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--mode", default="meta")
    ap.add_argument("--no-item", action="store_true")
    a = ap.parse_args()
    if a.build_v1:
        print(json.dumps({k: v for k, v in build_v1(a.mode, not a.no_item).items() if k != "name_table"}, indent=1))
    if a.check:
        for b in ("real_webagents", "researchcodebench"):
            print(b, check_v1(b))
    if a.run:
        run_all()
    if a.table:
        res = load_results()
        order = ["L0", "L1", "L1f", "L2", "L3", "L4", "L5", "L6", "L6nj", "L2i", "L6p",
                 "L2+T", "L4+T", "L6nj+T", "L6+T"]
        refs = {"L1": "L0", "L1f": "L1", "L2": "L1f", "L3": "L2", "L4": "L3", "L5": "L4", "L6": "L5",
                "L6nj": "L4", "L2i": "L2", "L6p": "L2", "L2+T": "L2", "L4+T": "L4", "L6nj+T": "L6nj",
                "L6+T": "L6"}
        for R in ("R1", "R2", "R3"):
            txt, rows = table(res, R, order, refs)
            print(f"\n### {R}\n" + txt)
            print("per-fold ALC:", {r: {B[:8]: round(v, 4) for B, v in rows[r]["per_fold"].items()}
                                    for r in ["L1", "L1f", "L2", "L4", "L6nj", "L6nj+T", "L6"]})
