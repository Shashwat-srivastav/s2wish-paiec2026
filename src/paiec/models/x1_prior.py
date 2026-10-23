"""E5 / X1: robust mixture offset prior for the hidden-benchmark offset mu_b.

Pre-registration: docs/preregistrations/E5_X1_preregistration.md; report: docs/research_notes/24_E5_X1.md.

Prior kind ``("x1", components)`` with components = tuple of (weight, centre, scale, nu):
  centre  'G' (training mean), 'half' (G/2), 'zero', or a float (logit)
  scale   'sB' (training sd of benchmark means), 'learned' (= max(0.5, rms(means - centre))), or float
  nu      Student-t degrees of freedom
Each component is a t density normalised on the 121-point mu grid; the prior is the weighted
mixture (weights renormalised). Under exact grid integration this *is* Bayesian model averaging
over the components: posterior component weight = w_k * marginal likelihood of the revealed labels.

Usage (drop-in for the v1_1 engine; no edit of v1_1.py needed):

    from paiec.models import v1_1 as V, x1_prior as X
    X.install()                       # wraps V.log_prior; old prior kinds unchanged
    cfg = dict(V.BASE2, prior=X.X1_SHIP)

``install`` patches the module-level name used by both ``pooled_pair`` (vectorised scorer) and
``_Group`` (streaming acquisition), so a v1_2 engine that imports ``log_prior`` from v1_1 by module
attribute picks it up; an engine with its own prior function can call ``x1_log_prior(H, prior)``.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

from paiec import paths
from paiec.evaluation import episodes as E
from paiec.models import v1_1 as V

MU = V.MU
OUT = paths.RUNS / "e5"
_ORIG = V.log_prior


def _centre(H, c):
    G = float(H["G"])
    if isinstance(c, str):
        return {"G": G, "half": G / 2, "zero": 0.0}[c]
    return float(c)


def _scale(H, s, c):
    if s == "sB":
        return float(H["sB"])
    if s == "learned":
        m = H.get("_means")
        m = V.train_means(H) if m is None else m
        return float(max(0.5, math.sqrt(np.mean((m - c) ** 2))))
    return float(s)


def x1_log_prior(H, prior):
    if prior[0] != "x1":
        return _ORIG(H, prior)
    comps = prior[1]
    wsum = sum(w for w, *_ in comps)
    out = None
    for w, c, s, nu in comps:
        cc = _centre(H, c)
        lp = math.log(w / wsum) + V._t_logpdf_norm(MU, cc, _scale(H, s, cc), nu)
        out = lp if out is None else np.logaddexp(out, lp)
    return out


def install():
    V.log_prior = x1_log_prior


def ext(core, eps, loc=3.0, scale=1.0, nu=3):
    """core: list of (w, c, s, nu) summing to 1; add eps/2 at -loc and +loc."""
    comps = [((1 - eps) * w, c, s, n) for w, c, s, n in core]
    return ("x1", tuple(comps + [(eps / 2, -loc, scale, nu), (eps / 2, loc, scale, nu)]))


CORE_G = [(1.0, "G", "sB", 3)]
CORE_H = [(1.0, "half", "learned", 3)]
CORE_BMA = [(0.5, "G", "sB", 3), (0.5, "half", "learned", 3)]

CANDS = {
    "v0": "v0",
    "v1": V.V1_CFG,
    "v1_1": V.BASE2,
    "X1-bma": dict(V.BASE2, prior=("x1", tuple(CORE_BMA))),
    "X1-ext10": dict(V.BASE2, prior=ext(CORE_H, 0.10)),
    "X1-ext20": dict(V.BASE2, prior=ext(CORE_H, 0.20)),
    "X1-bma-ext10": dict(V.BASE2, prior=ext(CORE_BMA, 0.10)),
    "X1-Gext10": dict(V.BASE2, prior=ext(CORE_G, 0.10)),
    "X1-t1": dict(V.BASE2, prior=("x1", ((1.0, "half", "learned", 1),))),
}
X1_SHIP = None  # set from report 24 decision

DELTAS = [-3, -2, -1, 0, 1, 2, 3]
SYN_SEEDS = [0, 1, 2]
T3 = 3.182446305284263  # t_{0.975, 3}


def log(*a):
    print(*a, flush=True)


def fold_stats(res, a, b, budget=None):
    pf = []
    for B, r in res.items():
        xa = r[a]["alc"] if budget is None else r[a]["br"][:, budget]
        xb = r[b]["alc"] if budget is None else r[b]["br"][:, budget]
        pf.append(float(V.pair_mean(xa - xb, r[a]["pair"]).mean()) * 1e3)
    pf = np.array(pf)
    m, sd = float(pf.mean()), float(pf.std(ddof=1))
    k = len(pf)
    return dict(mean=m, t3=[m - T3 * sd / math.sqrt(k), m + T3 * sd / math.sqrt(k)],
                pred=[m - T3 * sd * math.sqrt(1 + 1 / k), m + T3 * sd * math.sqrt(1 + 1 / k)],
                folds=dict(zip(res.keys(), pf.round(3).tolist())))


def real(names, seeds_r2=10, seeds_r1=100):
    out = {}
    res_all = V.evaluate({n: CANDS[n] for n in names}, seeds={"R2": list(range(seeds_r2)), "R1": list(range(seeds_r1))})
    for R, res in res_all.items():
        o = {}
        for n in names:
            o[n] = dict(alc=V.mean_alc(res, n), curve=V.mean_curve(res, n).tolist())
            if n != "v1_1":
                o[n]["d_v1_1"] = V.delta(res, n, "v1_1")
                o[n]["fold_v1_1"] = fold_stats(res, n, "v1_1")
                o[n]["budget_v1_1"] = [fold_stats(res, n, "v1_1", j)["mean"] for j in range(6)]
        out[R] = o
        log(f"[real {R}]")
        for n in names:
            x = o[n]
            s = f"  {n:13s} ALC={x['alc']:.5f}"
            if "d_v1_1" in x:
                f = x["fold_v1_1"]
                s += (f"  d_v1_1 {V.fmt(x['d_v1_1'])} t3[{f['t3'][0]:+.2f},{f['t3'][1]:+.2f}]"
                      f" pred[{f['pred'][0]:+.2f},{f['pred'][1]:+.2f}] byB " + " ".join(f"{v:+.2f}" for v in x["budget_v1_1"]))
            log(s)
    return out


def sweep_eval(names, D_of_seed, r1_seeds=50):
    """ALC (and budget curve) averaged over synthetic seeds, for R2 and R1."""
    acc = {R: {n: [] for n in names} for R in ("R2", "R1")}
    cur = {R: {n: [] for n in names} for R in ("R2", "R1")}
    for s in SYN_SEEDS:
        res = V.evaluate({n: CANDS[n] for n in names}, D_of=lambda B: D_of_seed(B, s),
                         seeds={"R2": [0], "R1": list(range(r1_seeds))})
        for R, r in res.items():
            for n in names:
                acc[R][n].append(V.mean_alc(r, n))
                cur[R][n].append(V.mean_curve(r, n))
    return ({R: {n: float(np.mean(v)) for n, v in d.items()} for R, d in acc.items()},
            {R: {n: np.mean(v, 0).tolist() for n, v in d.items()} for R, d in cur.items()})


def sweep(names):
    from paiec.analysis.redteam import synth_D
    out = {"alc": {}, "curve": {}}
    t0 = time.time()
    for d in DELTAS:
        a, c = sweep_eval(names, lambda B, s, d=d: synth_D(B, d, s))
        out["alc"][d], out["curve"][d] = a, c
        log(f"[delta {d:+}] {time.time()-t0:.0f}s " + " | ".join(
            f"{R}: " + " ".join(f"{n}={v:.4f}" for n, v in a[R].items()) for R in a))
    for R in ("R2", "R1"):
        reg = {n: [out["alc"][d][R][n] - min(out["alc"][d][R].values()) for d in DELTAS] for n in names}
        out[f"regret_{R}"] = {n: dict(max=max(v) * 1e3, mean=float(np.mean(v)) * 1e3, by_delta=[x * 1e3 for x in v])
                              for n, v in reg.items()}
        log(f"regret {R}: " + " ".join(f"{n}=max{v['max']:.2f}/mean{v['mean']:.2f}" for n, v in out[f'regret_{R}'].items()))
    return out


def _delta_for_rate(B, rate):
    from paiec.analysis.redteam import synth_D
    lo, hi = -9.0, 9.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if synth_D(B, mid, 0)["row_y"].mean() < rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def extreme(names):
    from paiec.analysis.redteam import synth_D
    out = {}
    for rate in (0.02, 0.98):
        dB = {B: _delta_for_rate(B, rate) for B in E.EVAL_FOLDS}
        a, c = sweep_eval(names, lambda B, s: synth_D(B, dB[B], s))
        out[rate] = dict(delta={k: round(v, 3) for k, v in dB.items()}, alc=a, curve=c)
        log(f"[rate {rate}] deltas {out[rate]['delta']}")
        for R in a:
            log(f"   {R}: " + " ".join(f"{n}={v:.4f}(B0 {c[R][n][0]:.3f},B1 {c[R][n][1]:.3f},B3 {c[R][n][2]:.3f})"
                                     for n, v in a[R].items()))
    return out


def check_equiv():
    """v1_1 through the wrapper and an x1 single-component spelling of v1_1 reproduce v1_1."""
    B = E.EVAL_FOLDS[0]
    H = V.H_fold(B)
    a = _ORIG(H, V.BASE2["prior"])
    b = x1_log_prior(H, ("x1", ((1.0, "half", "learned", 3),)))
    ok_prior = float(np.abs((a - a.max()) - (b - b.max())).max())
    ep = E.build(E.compact(B), "R2", [0])
    V.log_prior = _ORIG
    r0 = V.run_cfgs(ep, H, {"v1_1": V.BASE2})["v1_1"]["alc"].mean()
    install()
    r1 = V.run_cfgs(ep, H, {"v1_1": V.BASE2})["v1_1"]["alc"].mean()
    r2 = V.run_cfgs(ep, H, {"x": dict(V.BASE2, prior=("x1", ((1.0, "half", "learned", 3),)))})["x"]["alc"].mean()
    log(f"equiv: prior maxdiff={ok_prior:.2e} alc v1_1={r0:.8f} wrapped={r1:.8f} x1-spelling={r2:.8f}")
    return dict(prior_maxdiff=ok_prior, alc=[float(r0), float(r1), float(r2)])


def main(parts):
    install()
    names = list(CANDS)
    OUT.mkdir(parents=True, exist_ok=True)
    for p in parts:
        t0 = time.time()
        if p == "equiv":
            o = check_equiv()
        elif p == "real":
            o = real(names)
        elif p == "sweep":
            o = sweep(names)
        elif p == "extreme":
            o = extreme(names)
        else:
            raise ValueError(p)
        json.dump(o, open(OUT / f"{p}.json", "w", encoding="utf-8"), indent=1, default=str)
        log(f"== {p} done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="equiv,real,sweep,extreme")
    main(ap.parse_args().part.split(","))
