"""EXP-0045 (X3): learn four prior hyperparameters by minimising simulated ALC (nested LOBO).

Locked procedure: docs/preregistrations/2026-10-14_comprehensive_research_preregistration.md section 2.
For each outer fold B (EVAL_FOLDS), the training set is BENCHES minus B. Inner folds hold out each training
benchmark b in turn, fit the likelihood hyperparameters on BENCHES minus {B, b}, and score the 81-point grid
on simulated cold-start episodes of b (R2, seeds 0-9). The grid point with the lowest mean inner ALC is applied
(as relative multipliers / centre shift) to the outer-fold hyperparameters, and the outer fold is scored once,
paired against frozen v1_3 = v1_2.cfg_of(facets="all", guard=0) on identical episodes.

Grid: centre G/2 + {-0.5, 0, +0.5}; prior scale, tau, sigma_beta = training-fold value x {0.8, 1, 1.2}.

    python scripts/run_slot.py -- python -m paiec.analysis.exp_0045_alc_meta --step time
    python scripts/run_slot.py -- python -m paiec.analysis.exp_0045_alc_meta --step full
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time

import numpy as np

from paiec import paths
from paiec.evaluation import episodes as E
from paiec.models import ladder as Lad
from paiec.models import v1_1 as V1
from paiec.models import v1_2 as V

OUTDIR = paths.RUNS / "preregistration"
SEEDS = {"R2": list(range(10)), "R1": list(range(100))}
SHIFTS = (-0.5, 0.0, 0.5)
MULTS = (0.8, 1.0, 1.2)
GRID = list(itertools.product(SHIFTS, MULTS, MULTS, MULTS))   # (dc, m_scale, m_tau, m_sb)
NULL = (0.0, 1.0, 1.0, 1.0)
BASE_CFG = V.cfg_of(facets="all", guard=0)                     # frozen v1_3 in the engine


def log(*a):
    print(*a, flush=True)


def learned_scale(H):
    c = float(H["G"]) / 2
    m = H.get("_means")
    m = V1.train_means(H) if m is None else m
    return float(max(0.5, math.sqrt(np.mean((m - c) ** 2))))


def apply_point(H, point):
    """Return (H', cfg') for a grid point; NULL reproduces v1_3 exactly (checked)."""
    dc, ms, mt, mb = point
    if point == NULL:
        return H, BASE_CFG
    Hp = dict(H)
    Hp["tau"] = H["tau"] * mt
    Hp["sigma_beta"] = H["sigma_beta"] * mb
    cfg = dict(BASE_CFG, prior=("t", 3, float(H["G"]) / 2 + dc, learned_scale(H) * ms))
    return Hp, cfg


def score_point(B, D, ep, H, point, fe):
    Hp, cfg = apply_point(H, point)
    probs = {b: V.predict(ep, Hp, b, cfg, fe) for b in E.BUDGETS}
    br, _ = ep.score(probs)
    return br


def fold_alc(br, ep):
    return float(V1.pair_mean(br @ E.W, ep.inst_pair).mean())


_HCACHE = {}


def H_on(train):
    key = tuple(sorted(train))
    if key not in _HCACHE:
        H = Lad.fit_hyper(list(key))
        H["_means"] = V1.train_means(H)
        _HCACHE[key] = H
    return _HCACHE[key]


def check_null():
    """Grid NULL via the numeric prior path must equal the v1_3 config exactly."""
    worst = 0.0
    for B in E.EVAL_FOLDS:
        D = E.compact(B)
        ep = E.build(D, "R2", [0, 1])
        H = V1.H_fold(B)
        fe = V.fe_for(B, D, BASE_CFG)
        dc, ms, mt, mb = NULL
        Hp = dict(H)
        cfg = dict(BASE_CFG, prior=("t", 3, float(H["G"]) / 2, learned_scale(H)))
        for b in E.BUDGETS:
            a = V.predict(ep, H, b, BASE_CFG, fe)
            c = V.predict(ep, Hp, b, cfg, fe)
            worst = max(worst, float(np.abs(a - c).max()))
    log(f"null-equivalence max abs diff = {worst:.3e}")
    assert worst < 1e-10, worst
    return worst


def time_one():
    """Measure cost of one inner-grid evaluation (one grid point on one inner fold, R2 seeds 0-9)."""
    check_null()
    out = {}
    for b in E.BENCHES:
        B = next(x for x in E.EVAL_FOLDS if x != b)
        t0 = time.perf_counter()
        D = E.compact(b)
        ep = E.build(D, "R2", SEEDS["R2"])
        H = H_on([x for x in E.BENCHES if x not in (B, b)])
        fe = V.fe_for(b, D, BASE_CFG)
        setup = time.perf_counter() - t0
        ts = []
        for point in (NULL, (0.5, 1.2, 0.8, 1.2)):
            t1 = time.perf_counter()
            br = score_point(b, D, ep, H, point, fe)
            ts.append(time.perf_counter() - t1)
        out[b] = dict(n_inst=ep.n, setup_s=setup, point_s=ts, alc=fold_alc(br, ep))
        log(f"  inner {b}: n_inst={ep.n} setup={setup:.1f}s per-point={ts}")
    per_point = {b: float(np.mean(v["point_s"])) for b, v in out.items()}
    est = sum(len(GRID) * per_point[x] for Bo in E.EVAL_FOLDS for x in E.BENCHES if x != Bo)
    log(f"per-point seconds: {per_point}; estimated inner-grid total = {est/3600:.2f} h")
    rep = dict(per_point_s=per_point, details=out, estimated_inner_total_h=est / 3600)
    (OUTDIR / "exp_0045_timing.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return rep


def run_full(points=None):
    """Nested LOBO: inner grid search, then single outer evaluation (R2 primary, R1 secondary)."""
    check_null()
    points = GRID if points is None else points
    t0 = time.perf_counter()
    inner_alc = {}            # outer -> inner -> list of ALC per point
    # inner tables are shared across outer folds when (B, b) sets coincide; compute per (outer, inner)
    episodes_cache = {}
    for Bo in E.EVAL_FOLDS:
        inner_alc[Bo] = {}
        for b in [x for x in E.BENCHES if x != Bo]:
            if b not in episodes_cache:
                D = E.compact(b)
                episodes_cache[b] = (D, E.build(D, "R2", SEEDS["R2"]), V.fe_for(b, D, BASE_CFG))
            D, ep, fe = episodes_cache[b]
            H = H_on([x for x in E.BENCHES if x not in (Bo, b)])
            vals = []
            for p in points:
                vals.append(fold_alc(score_point(b, D, ep, H, p, fe), ep))
            inner_alc[Bo][b] = vals
            log(f"  [inner {Bo} <- {b}] {time.perf_counter()-t0:.0f}s best={min(vals):.6f} null={vals[points.index(NULL)]:.6f}")
    selected = {}
    for Bo in E.EVAL_FOLDS:
        M = np.array(list(inner_alc[Bo].values())).mean(0)
        k = int(np.argmin(M))
        if M[points.index(NULL)] <= M[k]:
            k = points.index(NULL)
        selected[Bo] = dict(point=list(points[k]), inner_mean_alc=float(M[k]),
                            inner_null_alc=float(M[points.index(NULL)]),
                            inner_gain_x1e3=float((M[k] - M[points.index(NULL)]) * 1e3),
                            inner_per_fold_x1e3={b: (v[k] - v[points.index(NULL)]) * 1e3
                                                 for b, v in inner_alc[Bo].items()})
        log(f"  selected {Bo}: {points[k]} inner gain {selected[Bo]['inner_gain_x1e3']:+.3f}e-3")
    # sensitivity (not decisional): selection ignoring the one-pair swe_rebench inner fold
    selected_sens = {}
    for Bo in E.EVAL_FOLDS:
        M = np.array([v for b, v in inner_alc[Bo].items() if b != "swe_rebench"]).mean(0)
        k = int(np.argmin(M))
        if M[points.index(NULL)] <= M[k]:
            k = points.index(NULL)
        selected_sens[Bo] = list(points[k])
        log(f"  sensitivity (no swe_rebench) {Bo}: {points[k]}")
    # outer: one evaluation per fold, paired vs v1_3 on identical episodes
    results, raw = {}, {}
    for R in ("R2", "R1"):
        res = {}
        for Bo in E.EVAL_FOLDS:
            D = E.compact(Bo)
            ep = E.build(D, R, SEEDS[R])
            H = V1.H_fold(Bo)
            fe = V.fe_for(Bo, D, BASE_CFG)
            res[Bo] = {}
            arms = [("v1_3", NULL), ("x3", tuple(selected[Bo]["point"]))]
            if tuple(selected_sens[Bo]) != tuple(selected[Bo]["point"]):
                arms.append(("x3_no_swe", tuple(selected_sens[Bo])))
            else:
                arms.append(("x3_no_swe", tuple(selected[Bo]["point"])))
            for nm, p in arms:
                br = score_point(Bo, D, ep, H, p, fe)
                res[Bo][nm] = dict(br=br, alc=br @ E.W, pair=ep.inst_pair.copy())
            log(f"  [outer {R} {Bo}] {time.perf_counter()-t0:.0f}s")
        o = V.summarise(res, ["v1_3", "x3", "x3_no_swe"], "v1_3")
        results[R] = o
        d = o["x3"]["d"]
        log(f"[{R}] x3 d={d['pt']:+.3f}e-3 pair[{d['lo']:+.3f},{d['hi']:+.3f}] t3[{d['t3'][0]:+.3f},{d['t3'][1]:+.3f}] "
            f"folds={d['per_fold']}")
    d = results["R2"]["x3"]["d"]
    adopt = bool(d["pt"] <= -1.0 and d["hi"] < 0 and d["n_neg"] >= 3 and d["worst_fold"] <= 0.5)
    rep = dict(experiment_id="EXP-0045", preregistered=True, dataset_revision="bc8204d811",
               grid=dict(shifts=SHIFTS, mults=MULTS, n_points=len(points)),
               inner_regime="R2 seeds 0-9", inner_alc=inner_alc, selected=selected, selected_sensitivity_no_swe=selected_sens,
               results=results, adopt_by_locked_rule=adopt,
               elapsed_seconds=time.perf_counter() - t0)
    (OUTDIR / "exp_0045_nested_alc_2026-10-15.json").write_text(json.dumps(rep, indent=1, default=float),
                                                                encoding="utf-8")
    log(f"adopt={adopt} elapsed={rep['elapsed_seconds']:.0f}s")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="time")
    a = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    {"time": time_one, "full": run_full, "check": check_null,
     "smoke": lambda: run_full([NULL, GRID[0]])}[a.step]()
