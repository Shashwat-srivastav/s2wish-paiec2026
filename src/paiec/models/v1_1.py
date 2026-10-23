"""Phase-3 repair engine for submission v1_1 (report: docs/research_notes/17_V1_1.md).

Generalises the v1 (L2i) predictor of ladder.py with the candidate repairs, each a
config switch (default config == v1 exactly; checked by ``--check-equiv``):

  prior      offset prior on mu_b. ('t', nu, center, scale_mode) or mixture
             ('mix', pi, wide_scale): (1-pi) t3(G,sB) + pi t3(0, wide_scale)
             center: 'G' (v1), 'zero', 'half' (G/2); scale_mode 'sB' (v1) or
             'learned' = max(0.5, rms(training means - center)).
  kappa      no-evidence shrink: if the pooling group has no labels at all, predict
             0.5 + kappa (p - 0.5). kappa=1 is v1.
  w_od       pair-likelihood tempering (beta-binomial style N_eff = w N). 1 = v1.
  tau_scale  inflation of the pair-ability sd. 1 = v1.
  rho        false-merge guard: prior probability that an item key is a merge of
             unrelated items. Posterior P(shared difficulty | outcomes) by exact
             marginal likelihoods (shared beta vs independent betas); the item-level
             prediction is averaged with the pair mean. 0 = v1.

Acquisition (Phase 3): ``simulate_policy`` replays the streaming protocol (fixed
random candidate order, accept/reject, permanent skips, nested caps, pairs advanced
in sorted order within a checkpoint, forced acceptance when items_remaining <=
labels_remaining) on top of the vectorised Episodes, and rewrites ``ep.acq`` with
the accepted sequence so the standard scorer is reused.

All hyperparameters are re-fit per LOBO fold (ladder.hyper_for).
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

GH_X, GH_W = Lad.GH_X, Lad.GH_W
LOG_GH_W = np.log(GH_W)
MU = Lad.MU_GRID
sig, log_sig = Lad.sig, Lad.log_sig
OUT = paths.RUNS / "phase3_v1_1"

V1_CFG = dict(prior=("t", 3, "G", "sB"), kappa=1.0, w_od=1.0, tau_scale=1.0, rho=0.0, item=True)


def cfg_of(**kw):
    c = dict(V1_CFG)
    c.update(kw)
    return c


# ---------------------------------------------------------------------------
# Offset prior
# ---------------------------------------------------------------------------
def _t_logpdf_norm(x, loc, scale, nu):
    z = (x - loc) / scale
    lp = -(nu + 1) / 2 * np.log1p(z * z / nu)
    lp = lp - np.log(np.exp(lp - lp.max()).sum()) - lp.max()
    return lp


def train_means(H):
    return np.array([Lad.bench_stats(b)["mean"] for b in H["train"]])


def log_prior(H, prior):
    kind = prior[0]
    if kind == "t":
        _, nu, center, smode = prior
        G = float(H["G"])
        c = {"G": G, "zero": 0.0, "half": G / 2}[center] if isinstance(center, str) else float(center)
        if smode == "sB":
            s = float(H["sB"])
        elif smode == "learned":
            m = H.get("_means")
            m = train_means(H) if m is None else m
            s = float(max(0.5, math.sqrt(np.mean((m - c) ** 2))))
        else:
            s = float(smode)
        if kind == "t" and nu == 3 and center == "G" and smode == "sB":
            return Lad.t3_logpdf(MU, G, s)  # exact v1 (unnormalised; constant irrelevant)
        return _t_logpdf_norm(MU, c, s, nu)
    if kind == "mix":
        pi, wide = prior[1], prior[2]
        if len(prior) > 3 and prior[3] == "half":
            a = log_prior(H, ("t", 3, "half", "learned"))
        else:
            a = _t_logpdf_norm(MU, float(H["G"]), float(H["sB"]), 3)
        b = _t_logpdf_norm(MU, 0.0, float(wide), 3)
        return np.logaddexp(np.log1p(-pi) + a, np.log(pi) + b)
    raise ValueError(prior)


# ---------------------------------------------------------------------------
# Predictor (vectorised over an Episodes batch)
# ---------------------------------------------------------------------------
def pooled_pair(ep, H, li, ly, cfg):
    n = ep.n
    w = cfg["w_od"]
    S = (np.bincount(li, ly, n) if len(li) else np.zeros(n)) * w
    N = (np.bincount(li, minlength=n).astype(float) if len(li) else np.zeros(n)) * w
    t = np.full(n, H["tau"] * cfg["tau_scale"])
    eta = MU[None, :, None] + t[:, None, None] * GH_X[None, None, :]
    ll = S[:, None, None] * log_sig(eta) + (N - S)[:, None, None] * log_sig(-eta) + LOG_GH_W
    mx = ll.max(2, keepdims=True)
    wt = np.exp(ll - mx)
    Ln = wt.sum(2)
    logL = np.log(Ln) + mx[:, :, 0]
    Eq = (wt * sig(eta)).sum(2) / Ln
    groups, ginv = np.unique(ep.inst_group, return_inverse=True)
    lp = np.zeros((len(groups), len(MU)))
    np.add.at(lp, ginv, logL)
    lp += log_prior(H, cfg["prior"])[None, :]
    lp -= lp.max(1, keepdims=True)
    post = np.exp(lp); post /= post.sum(1, keepdims=True)
    q = (post[ginv] * Eq).sum(1)
    if len(li):
        gN = np.bincount(ginv, np.bincount(li, minlength=n).astype(float), len(groups))
    else:
        gN = np.zeros(len(groups))
    empty = gN[ginv] == 0
    return q, empty


def pooled_rho(lr, nobs, grp, rho0, pool, strength=10.0, iters=8):
    """Per-key merge prior. pool=False: rho0. pool=True: benchmark-level merge rate
    by EM over the group's keys with >= 2 outcomes, Beta(rho0*s, (1-rho0)*s) prior."""
    out = np.full(len(lr), rho0)
    if not pool:
        return out
    m = nobs >= 2
    if not m.any():
        return out
    a0, b0 = rho0 * strength, (1 - rho0) * strength
    ug, gi = np.unique(grp[m], return_inverse=True)
    K = np.bincount(gi, minlength=len(ug)).astype(float)
    r = np.full(len(ug), rho0)
    for _ in range(iters):
        ps = sig(np.log1p(-r[gi]) - np.log(r[gi]) + lr[m])
        r = (a0 + np.bincount(gi, 1 - ps, len(ug))) / (a0 + b0 + K)
        r = np.clip(r, 1e-4, 1 - 1e-4)
    pos = np.searchsorted(ug, grp)
    pos = np.minimum(pos, len(ug) - 1)
    hit = ug[pos] == grp
    out[hit] = r[pos[hit]]
    return out


def mom_rho(inst, key, grp, rho0, k0=2.0, y=None, qi=None):
    """Per-group merge rate by moments: within-subject same-key label pairs can only be
    distinct items (false merges); cross-subject same-key pairs are true matches or
    merges. rho = (within rate / cross rate), shrunk to rho0 with k0 pseudo-pairs.
    Returns {group: rho}."""
    out = {}
    for g in np.unique(grp):
        m = grp == g
        ii, kk = inst[m], key[m]
        cell, cinv, c_sk = np.unique(ii.astype(np.int64) * 10**7 + kk, return_inverse=True, return_counts=True)
        _, t_k = np.unique(kk, return_counts=True)
        _, n_s = np.unique(ii, return_counts=True)
        n_w = float((c_sk * (c_sk - 1) / 2).sum())
        n_wpairs = n_w
        if y is not None:   # disagreement-weighted: genuine duplicate items agree
            s_sk = np.bincount(cinv, y[m], len(cell))
            qc = np.clip(qi[(cell // 10**7).astype(np.int64)], 0.02, 0.98)
            n_w = float(min((s_sk * (c_sk - s_sk) / (2 * qc * (1 - qc))).sum(), n_wpairs))
        n_x = float((t_k * (t_k - 1) / 2).sum()) - n_wpairs
        P_w = float((n_s * (n_s - 1) / 2).sum())
        P_x = float(n_s.sum() ** 2 - (n_s ** 2).sum()) / 2
        est = n_w * P_x / P_w if P_w > 0 else 0.0           # merges expected among cross pairs
        out[int(g)] = float(np.clip((est + k0 * rho0) / (n_x + k0), rho0, 1 - 1e-4))
    return out


def item_level(ep, H, q, li, lk, ly, rho, pool=False):
    """v1 item posterior; with rho > 0 a shared-vs-independent difficulty mixture."""
    sb = H["sigma_beta"]
    c = Lad._invert_c(q, sb)
    K = len(ep.D["key_jev"])
    g_t = ep.inst_group[ep.t_inst] * K + ep.t_key
    out = q[ep.t_inst].copy()
    if not len(li):
        return out, 0.0
    g_l = ep.inst_group[li] * K + lk
    uk, inv = np.unique(g_l, return_inverse=True)
    e = c[li][:, None] + sb * GH_X[None, :]
    contrib = np.where(ly[:, None] > 0.5, log_sig(e), log_sig(-e))
    acc = np.zeros((len(uk), len(GH_X)))
    np.add.at(acc, inv, contrib)
    pos = np.minimum(np.searchsorted(uk, g_t), len(uk) - 1)
    has = uk[pos] == g_t
    if has.any():
        a = acc[pos[has]] + LOG_GH_W[None, :]
        amax = a.max(1, keepdims=True)
        wgt = np.exp(a - amax)
        Ls = wgt.sum(1)
        wgt /= Ls[:, None]
        p_same = (wgt * sig(c[ep.t_inst[has]][:, None] + sb * GH_X[None, :])).sum(1)
        if rho > 0:
            qq = np.clip(q[li], 1e-12, 1 - 1e-12)
            lind = np.zeros(len(uk))
            np.add.at(lind, inv, np.where(ly > 0.5, np.log(qq), np.log1p(-qq)))
            aall = acc + LOG_GH_W[None, :]
            amx = aall.max(1)
            lr = np.log(np.exp(aall - amx[:, None]).sum(1)) + amx - lind      # log L_same / L_indep
            if pool in ("mom", "momd"):
                # subject id == instance within a group (one instance per pair per draw)
                rg = mom_rho(li, lk, ep.inst_group[li], rho,
                             y=ly if pool == "momd" else None, qi=q if pool == "momd" else None)
                rho_k = np.array([rg[int(g)] for g in uk // K])
            else:
                rho_k = pooled_rho(lr, np.bincount(inv, minlength=len(uk)), uk // K, rho, pool)
            ps = sig(np.log1p(-rho_k[pos[has]]) - np.log(rho_k[pos[has]]) + lr[pos[has]])
            p_same = ps * p_same + (1 - ps) * q[ep.t_inst[has]]
        out[has] = p_same
    return out, float(has.mean())


def predict(ep, H, b, cfg, labels=None):
    li, lk, ly = ep.labels(b) if labels is None else labels
    q, empty = pooled_pair(ep, H, li, ly, cfg)
    if cfg.get("item", True):
        p, _ = item_level(ep, H, q, li, lk, ly, cfg["rho"], cfg.get("rho_pool", False))
    else:
        p = q[ep.t_inst]
    k = cfg["kappa"]
    if k != 1.0:
        e = empty[ep.t_inst]
        p = np.where(e, 0.5 + k * (p - 0.5), p)
    return np.clip(p, 0.005, 0.995)


def run_cfgs(ep, H, cfgs):
    out = {}
    for nm, cfg in cfgs.items():
        if cfg == "v0":
            probs = {b: Lad.predict_budget(ep, H, "L1", b)[0] for b in E.BUDGETS}
        else:
            probs = {b: predict(ep, H, b, cfg) for b in E.BUDGETS}
        br, ec = ep.score(probs)
        out[nm] = dict(br=br, ec=ec, alc=br @ E.W, pair=ep.inst_pair.copy())
    return out


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
_H = {}


def H_fold(B):
    if B not in _H:
        H = Lad.hyper_for(B)
        H["_means"] = train_means(H)
        _H[B] = H
    return _H[B]


def pair_mean(x, pair):
    u, inv = np.unique(pair, return_inverse=True)
    return np.bincount(inv, x) / np.bincount(inv)


def boot(fold_vals, hier, nboot=2000, seed=0):
    rng = np.random.default_rng(seed)
    pt = float(np.mean([v.mean() for v in fold_vals]))
    F = len(fold_vals)
    bs = np.empty(nboot)
    for k in range(nboot):
        fs = rng.integers(0, F, F) if hier else range(F)
        bs[k] = np.mean([fold_vals[f][rng.integers(0, len(fold_vals[f]), len(fold_vals[f]))].mean() for f in fs])
    return pt, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def delta(res, a, b, budget=None):
    vals, pf = [], {}
    for B, r in res.items():
        xa = r[a]["alc"] if budget is None else r[a]["br"][:, budget]
        xb = r[b]["alc"] if budget is None else r[b]["br"][:, budget]
        d = pair_mean(xa - xb, r[a]["pair"])
        vals.append(d); pf[B] = float(d.mean()) * 1e3
    pt, lo, hi = boot(vals, False)
    _, hlo, hhi = boot(vals, True)
    return dict(pt=pt * 1e3, lo=lo * 1e3, hi=hi * 1e3, hlo=hlo * 1e3, hhi=hhi * 1e3, per_fold=pf)


def fmt(d, folds=True):
    s = f"{d['pt']:+.2f} [{d['lo']:+.2f},{d['hi']:+.2f}] hier[{d['hlo']:+.2f},{d['hhi']:+.2f}]"
    if folds:
        s += " (" + " ".join(f"{k[:5]}={v:+.1f}" for k, v in d["per_fold"].items()) + ")"
    return s


def mean_alc(res, m):
    return float(np.mean([pair_mean(r[m]["alc"], r[m]["pair"]).mean() for r in res.values()]))


def mean_curve(res, m):
    return np.mean([np.array([pair_mean(r[m]["br"][:, j], r[m]["pair"]).mean() for j in range(6)])
                    for r in res.values()], 0)


SEEDS = {"R1": list(range(100)), "R2": list(range(5))}


def evaluate(cfgs, regimes=("R2", "R1"), D_of=None, seeds=None):
    out = {}
    for R in regimes:
        res = {}
        for B in E.EVAL_FOLDS:
            D = E.compact(B) if D_of is None else D_of(B)
            ep = E.build(D, R, (seeds or SEEDS)[R])
            res[B] = run_cfgs(ep, H_fold(B), cfgs)
        out[R] = res
    return out


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# Step 1: easy-benchmark B0 risk (red-team A3)
# ---------------------------------------------------------------------------
STEP1 = {
    "v0": "v0",
    "v1": V1_CFG,
    "a_t1": cfg_of(prior=("t", 1, "G", "sB")),
    "a_mix20": cfg_of(prior=("mix", 0.2, 2.5)),
    "a_mix40": cfg_of(prior=("mix", 0.4, 2.5)),
    "b_zero": cfg_of(prior=("t", 3, "zero", "learned")),
    "b_half": cfg_of(prior=("t", 3, "half", "learned")),
    "c_k0": cfg_of(kappa=0.0),
    "c_k50": cfg_of(kappa=0.5),
}
STEP1B = {
    "v0": "v0", "v1": V1_CFG,
    "b_half": cfg_of(prior=("t", 3, "half", "learned")),
    "a_mix40": cfg_of(prior=("mix", 0.4, 2.5)),
    "bh_k50": cfg_of(prior=("t", 3, "half", "learned"), kappa=0.5),
    "bh_mix20": cfg_of(prior=("mix", 0.2, 2.5, "half")),
}


def step1b():
    return step1(STEP1B, "step1b")


DELTAS = [-3, -2, -1, 0, 1, 2, 3]


def _summ(res, names, ref="v1"):
    o = {}
    for nm in names:
        o[nm] = dict(alc=mean_alc(res, nm), curve=mean_curve(res, nm).tolist())
        if nm != ref:
            o[nm]["d_ref"] = delta(res, nm, ref)
            o[nm]["d_ref_budget"] = [delta(res, nm, ref, budget=j) for j in range(6)]
    return o


def step1(cfgs=STEP1, tag="step1"):
    from paiec.analysis.redteam import synth_D
    names = list(cfgs)
    out = {"real": {}, "sweep": {}}
    t0 = time.time()
    real = evaluate(cfgs)
    for R, res in real.items():
        out["real"][R] = _summ(res, names)
        log(f"[real {R}] {time.time()-t0:.0f}s")
        for nm in names:
            x = out["real"][R][nm]
            log(f"   {nm:9s} ALC={x['alc']:.4f}" + (f"  d_v1 {fmt(x['d_ref'])}" if "d_ref" in x else ""))
    for d in DELTAS:
        res = evaluate(cfgs, D_of=lambda B: synth_D(B, d, 0), seeds={"R2": [0], "R1": list(range(50))})
        out["sweep"][d] = {R: {nm: mean_alc(r, nm) for nm in names} for R, r in res.items()}
        out["sweep"][d]["d_v1"] = {R: {nm: delta(r, nm, "v1")["pt"] for nm in names if nm != "v1"} for R, r in res.items()}
        log(f"[delta {d:+}] {time.time()-t0:.0f}s " + " | ".join(
            f"{R}: " + " ".join(f"{nm}={v:.4f}" for nm, v in out['sweep'][d][R].items()) for R in res))
    # worst-case regret: per delta, regret = ALC - min over candidates (excluding v0? include all)
    for R in ("R2", "R1"):
        reg = {nm: [out["sweep"][d][R][nm] - min(out["sweep"][d][R].values()) for d in DELTAS] for nm in names}
        out[f"regret_{R}"] = {nm: dict(max=max(v) * 1e3, mean=float(np.mean(v)) * 1e3, by_delta=[x * 1e3 for x in v])
                              for nm, v in reg.items()}
        log(f"regret {R}: " + " ".join(f"{nm}=max{v['max']:.2f}/mean{v['mean']:.2f}" for nm, v in out[f'regret_{R}'].items()))
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / f"{tag}.json", "w", encoding="utf-8"), indent=1)
    return out



# ---------------------------------------------------------------------------
# Step 2: false-merge guard (red-team A5)
# ---------------------------------------------------------------------------
BASE = cfg_of(prior=("t", 3, "half", "learned"))   # step-1 decision


def step2(rhos=(0.05, 0.1, 0.2), tag="step2", pooled=(), extra=None):
    from paiec.analysis.redteam import rekey
    cfgs = {"v1": BASE, "L2": dict(BASE, item=False)}   # 'v1' here = step-1 base predictor
    for r in rhos:
        cfgs[f"rho{r:g}"] = dict(BASE, rho=r)
    for r in pooled:
        cfgs[f"pool{r:g}"] = dict(BASE, rho=r, rho_pool=True)
    cfgs.update(extra or {})
    names = list(cfgs)
    out = {}
    t0 = time.time()
    for mode, m in (("real", None), ("random", 2), ("random", 5), ("random", 20)):
        tagm = "real" if mode == "real" else f"merge{m}"
        D_of = None if mode == "real" else (lambda B, m=m: rekey(B, "random", m)[0])
        res = evaluate(cfgs, regimes=("R1", "R3"), D_of=D_of, seeds={"R1": list(range(100)), "R3": [0, 1]})
        o = {}
        for R, rr in res.items():
            o[R] = {nm: dict(alc=mean_alc(rr, nm)) for nm in names}
            for nm in names:
                if nm != "v1":
                    o[R][nm]["d_v1"] = delta(rr, nm, "v1")
                if nm != "L2":
                    o[R][nm]["d_L2"] = delta(rr, nm, "L2")
                    o[R][nm]["d_L2_budget"] = [delta(rr, nm, "L2", budget=j) for j in range(6)]
            log(f"[{tagm} {R}] {time.time()-t0:.0f}s")
            for nm in names:
                x = o[R][nm]
                log(f"   {nm:8s} ALC={x['alc']:.4f}" + (f" vs L2 {fmt(x['d_L2'], False)}" if 'd_L2' in x else "")
                    + (f" vs v1 {fmt(x['d_v1'], False)}" if 'd_v1' in x else ""))
        out[tagm] = o
    json.dump(out, open(OUT / f"{tag}.json", "w", encoding="utf-8"), indent=1)
    return out


# ---------------------------------------------------------------------------
# Step 3: pair-level overdispersion (red-team A6)
# ---------------------------------------------------------------------------
def reliability(res_eps, nm, b):
    tab = {}
    for ep, probs in res_eps:
        p = probs[nm][b][ep.r_t]; y = ep.r_y
        bins = np.minimum((p * 10).astype(int), 9)
        for k in range(10):
            m = bins == k
            if m.any():
                t = tab.setdefault(k, [0, 0.0, 0.0]); t[0] += int(m.sum()); t[1] += float(p[m].sum()); t[2] += float(y[m].sum())
    return {k / 10: (n, round(sp / n, 3), round(sy / n, 3)) for k, (n, sp, sy) in sorted(tab.items())}


def step2b():
    return step2(rhos=(0.1,), pooled=(0.05, 0.1), tag="step2b")


def step2c():
    from paiec.analysis.redteam import rekey
    return step2(rhos=(), pooled=(), tag="step2c", extra={"mom05": dict(BASE, rho=0.05, rho_pool="mom"),
                                                          "momd05": dict(BASE, rho=0.05, rho_pool="momd")})


def step2d():
    return step2(rhos=(), pooled=(), tag="step2d", extra={"momd05": dict(BASE, rho=0.05, rho_pool="momd"),
                                                          "momd10": dict(BASE, rho=0.1, rho_pool="momd")})


BASE2 = dict(BASE, rho=0.05, rho_pool="momd")       # step-2 decision


def step3(base=None, tag="step3"):
    base = dict(BASE2 if base is None else base)
    cfgs = {"base": base}
    for w in (0.7, 0.85):
        cfgs[f"w{w:g}"] = dict(base, w_od=w)
    for ts in (0.8, 1.25):
        cfgs[f"tau{ts:g}"] = dict(base, tau_scale=ts)
    names = list(cfgs)
    out = {}
    for R, sd in (("R2", list(range(5))), ("R1", list(range(100)))):
        res, rel_eps = {}, []
        for B in E.EVAL_FOLDS:
            ep = E.build(E.compact(B), R, sd)
            H = H_fold(B)
            res[B] = run_cfgs(ep, H, cfgs)
            if R == "R2":
                rel_eps.append((ep, {nm: {7: predict(ep, H, 7, c), 31: predict(ep, H, 31, c)} for nm, c in cfgs.items()}))
        o = _summ(res, names, ref="base")
        if rel_eps:
            o["reliability_B7"] = {nm: reliability(rel_eps, nm, 7) for nm in names}
        out[R] = o
        log(f"[{R}]")
        for nm in names:
            x = o[nm]
            log(f"   {nm:8s} ALC={x['alc']:.4f}" + (f" vs base {fmt(x['d_ref'])}" if 'd_ref' in x else ""))
        if rel_eps:
            for nm in names:
                log(f"   rel B7 {nm}: " + "; ".join(f"{k}:{v}" for k, v in o['reliability_B7'][nm].items() if k >= 0.6))
    json.dump(out, open(OUT / f"{tag}.json", "w", encoding="utf-8"), indent=1, default=str)
    return out


# ---------------------------------------------------------------------------
# Step 4: acquisition policies under the streaming protocol (Phase 3)
# ---------------------------------------------------------------------------
POLICIES = ("A1", "A2", "A3", "A5", "A8")


class _Group:
    """Exact posterior state of one pooling group (same math as pooled_pair/item_level)."""

    def __init__(self, H, cfg, n):
        self.H, self.cfg, self.n = H, cfg, n
        self.sb = H["sigma_beta"]
        self.eta = MU[:, None] + H["tau"] * cfg["tau_scale"] * GH_X[None, :]          # 121 x 20
        self.lsp, self.lsn, self.seta = log_sig(self.eta), log_sig(-self.eta), sig(self.eta)
        self.lprior = log_prior(H, cfg["prior"])
        self.S = np.zeros(n); self.N = np.zeros(n)
        self.obs = {}                                                                 # key -> [(i, y)]
        self.cnode = Lad._invert_c(self.seta.ravel(), self.sb).reshape(self.eta.shape)
        self.refresh()

    def refresh(self):
        w = self.cfg["w_od"]
        S, N = self.S * w, self.N * w
        ll = S[:, None, None] * self.lsp + (N - S)[:, None, None] * self.lsn + LOG_GH_W   # n x 121 x 20
        mx = ll.max(2, keepdims=True)
        self.wt = np.exp(ll - mx)
        Ln = self.wt.sum(2)
        self.logL = np.log(Ln) + mx[:, :, 0]
        self.Ln = Ln
        self.Eq = (self.wt * self.seta).sum(2) / Ln
        lp = self.logL.sum(0) + self.lprior
        lp -= lp.max()
        post = np.exp(lp)
        self.post = post / post.sum()
        self.q = self.Eq @ self.post
        self.c = Lad._invert_c(self.q, self.sb)
        self._unseen = {}

    def add(self, i, key, y):
        self.S[i] += y; self.N[i] += 1
        self.obs.setdefault(int(key), []).append((i, y))
        self.refresh()

    def beta_post(self, key):
        """(GH weights over beta nodes, P(shared)) for a key given the group's labels on it."""
        ob = self.obs.get(int(key))
        if not ob:
            return None, 0.0
        acc = LOG_GH_W.copy()
        lind = 0.0
        for j, y in ob:
            e = self.c[j] + self.sb * GH_X
            acc = acc + (log_sig(e) if y > 0.5 else log_sig(-e))
            qj = min(max(self.q[j], 1e-12), 1 - 1e-12)
            lind += math.log(qj) if y > 0.5 else math.log1p(-qj)
        amax = acc.max()
        a = np.exp(acc - amax)
        Ls = a.sum()
        a /= Ls
        rho = self.cfg["rho"]
        ps = 1.0 if rho <= 0 else float(sig(math.log(1 - rho) + math.log(Ls) + amax - math.log(rho) - lind))
        return a, ps

    def pred(self, i, key):
        a, ps = self.beta_post(key)
        if a is None:
            return float(self.q[i]), None, 0.0
        p_same = float(a @ sig(self.c[i] + self.sb * GH_X))
        return ps * p_same + (1 - ps) * float(self.q[i]), a, ps

    def fisher(self, i, a, ps):
        s = sig(self.c[i] + self.sb * GH_X)
        wv = GH_W if a is None else ps * a + (1 - ps) * GH_W
        p = wv @ s
        d = wv @ (s * (1 - s))
        return d * d / max(p * (1 - p), 1e-12)

    def ebr(self, i, a, ps):
        """One-step expected reduction of the pair's expected Brier on unseen items
        (= Var_y of the updated pair mean, exact over the mu-grid x GH posterior)."""
        if a is None and ("e", i) in self._unseen:
            return self._unseen[("e", i)]
        l1 = self.seta
        if a is not None:
            known = sig(self.cnode[:, :, None] + self.sb * GH_X[None, None, :]) @ a
            l1 = ps * known + (1 - ps) * self.seta
        qs, py = [], []
        for ly in (l1, 1 - l1):
            Ln2 = (self.wt[i] * ly).sum(1)
            Eq2 = (self.wt[i] * ly * self.seta).sum(1) / Ln2
            r = Ln2 / self.Ln[i]
            py.append(float(self.post @ r))
            post2 = self.post * r
            post2 /= post2.sum()
            qs.append(float(post2 @ Eq2))
        v = py[0] * py[1] * (qs[0] - qs[1]) ** 2
        if a is None:
            self._unseen[("e", i)] = v
        return v


def _decide(policy, G, i, key, counts):
    if policy == "A1":
        return True
    p, a, ps = G.pred(i, key)
    if policy == "A2":
        return abs(p - 0.5) <= abs(G.q[i] - 0.5) + 1e-12
    if policy == "A3":
        return G.fisher(i, a, ps) >= G.fisher(i, None, 0.0) - 1e-15
    if policy == "A5":
        if a is None:
            return True
        return counts[i][min(int(p * 3), 2)] <= min(counts[i])
    if policy == "A8":
        return G.ebr(i, a, ps) >= G.ebr(i, None, 0.0) - 1e-15
    raise ValueError(policy)


def simulate_policy(ep, H, cfg, policy, max_labels=31):
    """Replay the streaming protocol; rewrite ep.acq with each instance's accepted
    sequence (ep.labels(b) then returns the nested prefix at cap b)."""
    orig = ep.acq
    new = [[] for _ in range(ep.n)]
    stats = dict(skipped=0, seen=0, forced=0)
    for g in np.unique(ep.inst_group):
        members = np.flatnonzero(ep.inst_group == g)       # sorted pair order == simulator order
        G = _Group(H, cfg, len(members))
        idx = [0] * len(members)
        counts = [[0, 0, 0] for _ in members]
        for b in E.BUDGETS[1:]:
            cap = min(b, max_labels)
            for li_, n in enumerate(members):
                acq = orig[n]
                while len(new[n]) < cap and idx[li_] < len(acq):
                    key, y = int(acq[idx[li_], 0]), int(acq[idx[li_], 1])
                    items_rem = len(acq) - idx[li_]
                    lab_rem = max_labels - len(new[n])
                    idx[li_] += 1
                    stats["seen"] += 1
                    if policy != "A1" and items_rem <= lab_rem:
                        ok = True
                        stats["forced"] += 1
                    else:
                        ok = _decide(policy, G, li_, key, counts)
                    if ok:
                        if policy == "A5":
                            p, a, _ = G.pred(li_, key)
                            if a is not None:
                                counts[li_][min(int(p * 3), 2)] += 1
                        new[n].append((key, y))
                        G.add(li_, key, y)
                    else:
                        stats["skipped"] += 1
    ep.acq = [np.array(x, np.int64).reshape(-1, 2) for x in new]
    stats["mean_labels31"] = float(np.mean([len(x) for x in new]))
    return stats


def step4(regimes=(("R1", list(range(100))), ("R2", [0, 1])), tag="step4", predictors=None):
    import copy
    predictors = predictors or {"v1": V1_CFG, "v1_1": BASE2}
    out = {}
    t0 = time.time()
    for R, sd in regimes:
        res, st = {}, {}
        for B in E.EVAL_FOLDS:
            H = H_fold(B)
            ep0 = E.build(E.compact(B), R, sd)
            res[B] = {}
            for pn, pc in predictors.items():
                for pol in POLICIES:
                    ep = copy.copy(ep0)
                    st[(B, pn, pol)] = simulate_policy(ep, H, pc, pol)
                    res[B].update(run_cfgs(ep, H, {f"{pn}|{pol}": pc}))
            log(f"  {R} {B} {time.time()-t0:.0f}s")
        names = [f"{pn}|{pol}" for pn in predictors for pol in POLICIES]
        o = {}
        for nm in names:
            pn, pol = nm.split("|")
            o[nm] = dict(alc=mean_alc(res, nm), curve=mean_curve(res, nm).tolist(),
                         skip_frac=float(np.mean([st[(B, pn, pol)]["skipped"] / max(st[(B, pn, pol)]["seen"], 1)
                                                  for B in E.EVAL_FOLDS])),
                         labels31=float(np.mean([st[(B, pn, pol)]["mean_labels31"] for B in E.EVAL_FOLDS])))
            ref = f"{pn}|A1"
            if nm != ref:
                o[nm]["d_A1"] = delta(res, nm, ref)
                o[nm]["d_A1_budget"] = [delta(res, nm, ref, budget=j) for j in range(6)]
        pns = list(predictors)
        for pn in pns[1:]:
            for pol in POLICIES:
                o[f"{pn}-{pns[0]}|{pol}"] = delta(res, f"{pn}|{pol}", f"{pns[0]}|{pol}")
        out[R] = o
        log(f"[{R}]")
        for nm in names:
            x = o[nm]
            log(f"   {nm:10s} ALC={x['alc']:.4f} skip={x['skip_frac']:.2f} L31={x['labels31']:.1f}"
                + (f" vs A1 {fmt(x['d_A1'])}" if "d_A1" in x else ""))
            if "d_A1_budget" in x:
                log("      by budget: " + " ".join(f"B{b}={d['pt']:+.2f}[{d['lo']:+.2f},{d['hi']:+.2f}]"
                                                  for b, d in zip(E.BUDGETS, x["d_A1_budget"])))
        for k, v in o.items():
            if "-" in k:
                log(f"   predictor {k}: {fmt(v)}")
    json.dump(out, open(OUT / f"{tag}.json", "w", encoding="utf-8"), indent=1, default=str)
    return out


def final():
    """Headline table for report 17: v0 / v1 / v1_1 (= BASE2) on real R1/R2 + A3 sweep."""
    out = step1({"v0": "v0", "v1": V1_CFG, "v1_1": BASE2}, "final")
    for R in ("R2", "R1"):
        o = out["real"][R]
        log(f"[{R}] curves " + " | ".join(f"{nm}: " + " ".join(f"{x:.4f}" for x in o[nm]["curve"]) for nm in o))
        for nm in ("v0", "v1_1"):
            log(f"   {nm}-v1 by budget: " + " ".join(f"B{b}={d['pt']:+.2f}[{d['lo']:+.2f},{d['hi']:+.2f}]h[{d['hlo']:+.2f},{d['hhi']:+.2f}]"
                                                   for b, d in zip(E.BUDGETS, o[nm]["d_ref_budget"])))
        for nm in ("v1", "v1_1"):
            dv0 = [out["sweep"][d][R][nm] - out["sweep"][d][R]["v0"] for d in DELTAS]
            log(f"   sweep {nm}-v0 by delta: " + " ".join(f"{d:+}:{x*1e3:+.2f}" for d, x in zip(DELTAS, dv0)) + f"  worst {max(dv0)*1e3:+.2f}")
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="")
    a = ap.parse_args()
    for s in a.step.split(","):
        if s:
            globals()[s]()
