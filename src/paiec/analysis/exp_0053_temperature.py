"""EXP-0053: nested held-out temperature of the frozen v1_3 predictor (preregistration §5, 'Kelly / log utility').

No quantity here is "Kelly-optimal". Temperature acts on the final probability, p_T = sig(logit(p) / T)
(T > 1 softens, T < 1 sharpens), then the model's [0.005, 0.995] clip is reapplied.

Nested folds: for outer benchmark B, every inner benchmark b in EVAL_FOLDS - {B} is predicted with
hyperparameters fit on BENCHES - {B, b}; T_B minimises the benchmark-equal, pair-weighted, ALC-weighted row
log loss over those inner held-out episodes (inner seeds disjoint from the outer seeds). T_B is then applied
once to the outer B episodes and compared with T = 1 on identical episodes.

Run: python scripts/run_slot.py -- python -m paiec.analysis.exp_0053_temperature
"""
from __future__ import annotations

import json
import time

import numpy as np

from paiec.evaluation import episodes as E
from paiec.models import ladder as Lad
from paiec.models import v1_1 as V1
from paiec.models import v1_2 as V2
from paiec.analysis import exp_0052_shrinkage as M

INNER_SEEDS = {"R2": list(range(100, 110)), "R1": list(range(100, 150))}
TGRID = np.round(np.arange(0.50, 2.0001, 0.005), 4)
EPS = 1e-12


def logits(P):
    p = {b: np.clip(v, 1e-6, 1 - 1e-6) for b, v in P.items()}
    return {b: np.log(v / (1 - v)) for b, v in p.items()}


def temper(L, T):
    return {b: np.clip(1 / (1 + np.exp(-v / T)), 0.005, 0.995) for b, v in L.items()}


def logloss_curve(ep, L, grid):
    """Pair-weighted ALC-weighted mean row log loss for every T in grid -> [len(grid)], plus per budget."""
    cnt = np.bincount(ep.r_inst, minlength=ep.n)
    u, inv = np.unique(ep.inst_pair, return_inverse=True)
    pc = np.bincount(inv, minlength=len(u)).astype(float)
    out = np.zeros((len(grid), 6))
    for j, b in enumerate(E.BUDGETS):
        lt = L[b][ep.r_t]
        for g, T in enumerate(grid):
            p = np.clip(1 / (1 + np.exp(-lt / T)), 0.005, 0.995)
            ll = -(ep.r_y * np.log(p) + (1 - ep.r_y) * np.log(1 - p))
            inst = np.bincount(ep.r_inst, ll, ep.n) / cnt
            out[g, j] = float((np.bincount(inv, inst, len(u)) / pc).mean())
    return out


def ll_score(ep, P):
    """Per-instance log loss [n,6] for paired summaries (stored in the 'br' slot)."""
    cnt = np.bincount(ep.r_inst, minlength=ep.n)
    br = np.zeros((ep.n, 6))
    for j, b in enumerate(E.BUDGETS):
        p = P[b][ep.r_t]
        ll = -(ep.r_y * np.log(p) + (1 - ep.r_y) * np.log(1 - p))
        br[:, j] = np.bincount(ep.r_inst, ll, ep.n) / cnt
    return dict(br=br, alc=br @ E.W, pair=ep.inst_pair.copy())


def inner_H(B, b):
    H = Lad.fit_hyper([x for x in E.BENCHES if x not in (B, b)])
    H["_means"] = V1.train_means(H)
    return H


def run():
    t0 = time.perf_counter()
    sha = M.check_hash()
    cfg = M.baseline_cfg()
    out = {"experiment_id": "EXP-0053", "preregistered": True, "reference_sha256": sha,
           "dataset_revision": "bc8204d811", "outer_seeds": M.SEEDS, "inner_seeds": INNER_SEEDS,
           "temperature_convention": "p_T = sigmoid(logit(p)/T), then clip [0.005,0.995]; T>1 softens",
           "grid": [float(TGRID[0]), float(TGRID[-1]), 0.005], "config": cfg, "regimes": {}}
    for R in M.REGIMES:
        res, res_ll, folds = {}, {}, {}
        for B in E.EVAL_FOLDS:
            curves = []
            per_inner = {}
            for b in E.EVAL_FOLDS:
                if b == B:
                    continue
                rp = M.replay(b, inner_H(B, b), cfg, R, INNER_SEEDS[R])
                c = logloss_curve(rp["ep"], logits(rp["P"]), TGRID)
                curves.append(c)
                per_inner[b] = float(TGRID[(c @ E.W).argmin()])
                del rp
            tot = np.mean([c @ E.W for c in curves], 0)
            T = float(TGRID[tot.argmin()])
            T_budget = [float(TGRID[np.mean([c[:, j] for c in curves], 0).argmin()]) for j in range(6)]
            rp = M.replay(B, V1.H_fold(B), cfg, R, M.SEEDS[R])
            L = logits(rp["P"])
            PT = temper(L, T)
            ep = rp["ep"]
            res[B] = {"T1": M.score(ep, rp["P"]), "Tfit": M.score(ep, PT)}
            res_ll[B] = {"T1": ll_score(ep, rp["P"]), "Tfit": ll_score(ep, PT)}
            c_out = logloss_curve(ep, L, TGRID)
            folds[B] = {"T_nested": T, "T_nested_by_budget": T_budget, "T_single_inner_benchmark": per_inner,
                        "inner_logloss_T1": float(tot[np.argmin(np.abs(TGRID - 1.0))]), "inner_logloss_Tfit": float(tot.min()),
                        "descriptive_only_T_on_outer_fold": float(TGRID[(c_out @ E.W).argmin()]),
                        "descriptive_only_T_on_outer_fold_by_budget": [float(TGRID[c_out[:, j].argmin()]) for j in range(6)]}
            M.log(f"[{R} {B}] T={T:.3f} by_budget={T_budget} inner={per_inner} "
                  f"outer_descr={folds[B]['descriptive_only_T_on_outer_fold']:.3f} {time.perf_counter() - t0:.0f}s")
            del rp
        summ = V2.summarise(res, ["T1", "Tfit"], "T1")
        summ_ll = V2.summarise(res_ll, ["T1", "Tfit"], "T1")
        d = summ["Tfit"]["d"]
        Ts = [folds[B]["T_nested"] for B in E.EVAL_FOLDS]
        out["regimes"][R] = {
            "folds": folds, "brier": summ, "logloss_alc": summ_ll,
            "mde_80pct_x1e3": 2.8 * (d["hi"] - d["lo"]) / 3.92,
            "n_folds_T_outside_0.9_1.1": int(sum(t < 0.9 or t > 1.1 for t in Ts)),
            "stable_outside": bool(all(t > 1.1 for t in Ts) or all(t < 0.9 for t in Ts)),
            "baseline_matches_exp0044": abs(summ["T1"]["alc"] - M.EXP0044_BASELINE[R]) < 5e-6,
        }
        M.log(f"[{R}] T={Ts} ALC T1={summ['T1']['alc']:.6f} Tfit d={d['pt']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}]e-3 "
              f"folds={ {k: round(v, 3) for k, v in d['per_fold'].items()} } "
              f"LL d={summ_ll['Tfit']['d']['pt']:+.3f}e-3")
    out["elapsed_seconds"] = time.perf_counter() - t0
    path = M.OUTDIR / "exp_0053_temperature_2026-10-15.json"
    path.write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8")
    M.log(f"wrote {path} in {out['elapsed_seconds']:.1f}s")
    return out


if __name__ == "__main__":
    run()
