"""EXP-0054: deliberate prior mis-specification stress test of the frozen v1_3 predictor (preregistration §5).

Not a tuning grid. Per outer fold B the baseline offset prior is t3(c0, s0) with c0 = G_B/2 and
s0 = max(0.5, rms(training means - c0)) (v1_3's 'half'/'learned' rule). Arms, all on identical episodes:

  centre -1      t3(c0 - 1, s0)
  centre +1      t3(c0 + 1, s0)
  half-life x0.5 t3(c0, s0*sqrt(2))   n_eff = 4(nu+1)/(nu s^2) ∝ 1/s^2, so h x k  <=>  s x 1/sqrt(k)
  half-life x2   t3(c0, s0/sqrt(2))

A numeric-parameter copy of the baseline ('identity') must reproduce the baseline exactly. The realised
model half-life of each arm is re-measured with the EXP-0052 w_mod(n) curve.

Run: python scripts/run_slot.py -- python -m paiec.analysis.exp_0054_stress
"""
from __future__ import annotations

import json
import math
import time

import numpy as np

from paiec.evaluation import episodes as E
from paiec.models import v1_1 as V1
from paiec.models import v1_2 as V2
from paiec.analysis import exp_0052_shrinkage as M

R2 = math.sqrt(2.0)
ARMS = {  # name -> (centre shift, scale multiplier)
    "identity": (0.0, 1.0),
    "centre_minus1": (-1.0, 1.0),
    "centre_plus1": (+1.0, 1.0),
    "halflife_x0.5": (0.0, R2),
    "halflife_x2": (0.0, 1 / R2),
}


def run():
    t0 = time.perf_counter()
    sha = M.check_hash()
    base = M.baseline_cfg()
    rng = np.random.default_rng(20261016)
    out = {"experiment_id": "EXP-0054", "preregistered": True, "reference_sha256": sha,
           "dataset_revision": "bc8204d811", "seeds": M.SEEDS, "baseline_config": base,
           "arms": {k: {"centre_shift_logit": a, "scale_multiplier": b} for k, (a, b) in ARMS.items()},
           "halflife_mapping": "n_eff = 4*(nu+1)/(nu*s^2) is proportional to 1/s^2; half-life x k implemented as prior scale x 1/sqrt(k)",
           "regimes": {}}
    names = ["baseline"] + list(ARMS)
    for R in M.REGIMES:
        res, folds = {}, {}
        for B in E.EVAL_FOLDS:
            H = V1.H_fold(B)
            s0, c0 = M.learned_scale(H)
            D = E.compact(B)
            ep = E.build(D, R, M.SEEDS[R])
            res[B], folds[B] = {}, {"c0": c0, "s0": s0, "arms": {}}
            ref_P = None
            for nm in names:
                if nm == "baseline":
                    cfg = base
                else:
                    dc, ks = ARMS[nm]
                    cfg = M.baseline_cfg(prior=("t", 3, c0 + dc, s0 * ks))
                rp = M.replay(B, H, cfg, R, M.SEEDS[R], ep=ep)
                res[B][nm] = M.score(ep, rp["P"])
                (wm, we), boot = M.curves([M.pair_sums(rp)], rng, 300)
                info = {"h_model": float(M.fit_h(wm)[0]), "h_model_ci": M.ci(M.fit_h(boot[0])),
                        "kappa_model": float(M.fit_kappa(wm)[0]), "w_mod": wm.tolist()}
                if nm == "baseline":
                    ref_P = rp["P"]
                else:
                    info["prior"] = list(cfg["prior"])
                    info["n_eff_mu_prior"] = M.n_eff_mu(cfg["prior"][3])
                if nm == "identity":
                    info["max_abs_diff_vs_baseline"] = float(max(np.abs(rp["P"][b] - ref_P[b]).max() for b in E.BUDGETS))
                folds[B]["arms"][nm] = info
                del rp
            M.log(f"[{R} {B}] c0={c0:.3f} s0={s0:.3f} identity diff={folds[B]['arms']['identity']['max_abs_diff_vs_baseline']:.2e} "
                  f"{time.perf_counter() - t0:.0f}s")
            del ep
        summ = V2.summarise(res, names, "baseline")
        dec = {}
        for nm in ARMS:
            if nm == "identity":
                continue
            d = summ[nm]["d"]
            dec[nm] = {
                "delta_alc_x1e3": d["pt"], "pair_ci_x1e3": [d["lo"], d["hi"]],
                "hier_ci_x1e3": [d["hlo"], d["hhi"]], "t3_fold_ci_x1e3": d["t3"], "new_benchmark_pi_x1e3": d["pi"],
                "fold_deltas_x1e3": d["per_fold"], "n_neg_folds": d["n_neg"], "worst_fold_x1e3": d["worst_fold"],
                "regret_x1e3": max(d["pt"], 0.0),
                "budget_deltas_x1e3": [x["pt"] for x in summ[nm]["d_budget"]],
                "mde_80pct_x1e3": 2.8 * (d["hi"] - d["lo"]) / 3.92,
                "passes_adoption_bar": bool(d["pt"] <= -1.0 and d["hi"] < 0 and d["n_neg"] >= 3 and d["worst_fold"] <= 0.5),
                "significant_harm_pair": bool(d["lo"] > 0),
                "fold_regression_gt_0.5e-3": bool(d["worst_fold"] > 0.5),
            }
            M.log(f"[{R}] {nm}: d={d['pt']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}] t3[{d['t3'][0]:+.2f},{d['t3'][1]:+.2f}] "
                  f"folds={ {k: round(v, 3) for k, v in d['per_fold'].items()} }")
        out["regimes"][R] = {
            "folds": folds, "results": summ, "decisions": dec,
            "worst_case_regret_x1e3": max(v["regret_x1e3"] for v in dec.values()),
            "worst_case_arm": max(dec, key=lambda k: dec[k]["delta_alc_x1e3"]),
            "identity_max_abs_diff": max(f["arms"]["identity"]["max_abs_diff_vs_baseline"] for f in folds.values()),
            "baseline_matches_exp0044": abs(summ["baseline"]["alc"] - M.EXP0044_BASELINE[R]) < 5e-6,
        }
    out["elapsed_seconds"] = time.perf_counter() - t0
    path = M.OUTDIR / "exp_0054_prior_stress_2026-10-15.json"
    path.write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8")
    M.log(f"wrote {path} in {out['elapsed_seconds']:.1f}s")
    return out


if __name__ == "__main__":
    run()
