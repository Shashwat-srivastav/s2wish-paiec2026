"""EXP-0052: empirical shrinkage curve of the frozen v1_3 pair prediction (preregistration §5).

Budgets are nested posterior updates, so no AR(1)/OU stationarity fit is made. For each label budget n
the (pair-weighted) linear data weight of the pair-level update is measured on the probability scale:

    x_i(n) = ybar_i(n) - q0_i            acquired-label mean minus the B0 pair prediction
    w_mod(n) = sum w x (q_n - q0) / sum w x^2          weight the model actually gives the labels
    w_emp(n) = sum w x (z - q0)  / sum w x^2           Bayes-optimal linear weight (z = held-out row mean)

w(n) = 1 - exp(-n/h) is fitted (least squares over n = 1,3,7,15,31) to the training-benchmark empirical
curve of each outer fold (hyperparameters H_fold(B), outer benchmark B never touched), to the held-out
empirical curve and to the model curve; h is bootstrapped by pair. The t3 prior's centre curvature gives
n_eff = 4 * (nu+1)/(nu s^2). Brier evidence: a bounded correction p' = p + (lambda_n - 1)(q_n - q0) with
lambda_n = w_emp_train / w_mod_train from the training benchmarks only, paired against the baseline.

Run: python scripts/run_slot.py -- python -m paiec.analysis.exp_0052_shrinkage
"""
from __future__ import annotations

import hashlib
import json
import math
import time

import numpy as np

from paiec import paths
from paiec.evaluation import episodes as E
from paiec.models import v1_1 as V1
from paiec.models import v1_2 as V2

REFERENCE_ARCHIVE = paths.DIST / "v1_3_named_facets_noguard.zip"
REFERENCE_SHA256 = "A4B0E841D4A344FF207BED2C855342212E95F7D550195B667446EA4E92685B86"
FROZEN = dict(G=-0.5804949802000193, prior_center=-0.29024749010000966, prior_scale=0.7781603399165817,
              tau=1.1298321532152709, sigma_beta=2.701516919156687, rho0=0.05)
OUTDIR = paths.RUNS / "preregistration"
SEEDS = {"R2": list(range(10)), "R1": list(range(100))}
REGIMES = ("R2", "R1")
NB = np.array([1, 3, 7, 15, 31], float)          # budgets with labels
HGRID = np.exp(np.linspace(np.log(0.05), np.log(2000.0), 1500))
NBOOT = 1000
EXP0044_BASELINE = {"R2": 0.208824, "R1": 0.214294}


def baseline_cfg(**kw):
    return V2.cfg_of(facets="all", guard=0, **kw)


def check_hash():
    h = hashlib.sha256(REFERENCE_ARCHIVE.read_bytes()).hexdigest().upper()
    if h != REFERENCE_SHA256:
        raise RuntimeError(f"reference hash mismatch {h}")
    return h


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# One benchmark x hyperparameter x regime replay keeping pair-level quantities
# ---------------------------------------------------------------------------
def replay(X, H, cfg, R, seeds, ep=None):
    D = E.compact(X)
    ep = ep if ep is not None else E.build(D, R, seeds)
    fe = V2.fe_for(X, D, cfg)
    P, Q, YB = {}, {}, {}
    for b in E.BUDGETS:
        li, lk, ly = ep.labels(b)
        P[b] = V2.predict(ep, H, b, cfg, fe)
        Q[b], _ = V1.pooled_pair(ep, H, li, ly, cfg)
        if b:
            YB[b] = np.bincount(li, ly, ep.n) / np.maximum(np.bincount(li, minlength=ep.n), 1)
    cnt = np.bincount(ep.r_inst, minlength=ep.n)
    z = np.bincount(ep.r_inst, ep.r_y, ep.n) / cnt
    return dict(ep=ep, P=P, Q=Q, YB=YB, z=z, pair=ep.inst_pair.copy())


def score(ep, P):
    br, ec = ep.score(P)
    return dict(br=br, ec=ec, alc=br @ E.W, pair=ep.inst_pair.copy())


def pair_sums(rp):
    """Per-pair (instance-averaged) sums: Sxx, Sxm, Sxz each [n_pairs, 5]."""
    q0 = rp["Q"][0]
    x = np.stack([rp["YB"][int(b)] - q0 for b in NB], 1)
    m = np.stack([rp["Q"][int(b)] - q0 for b in NB], 1)
    zt = (rp["z"] - q0)[:, None]
    u, inv = np.unique(rp["pair"], return_inverse=True)
    c = np.bincount(inv, minlength=len(u)).astype(float)[:, None]

    def agg(a):
        return np.stack([np.bincount(inv, a[:, j], len(u)) for j in range(a.shape[1])], 1) / c

    return dict(xx=agg(x * x), xm=agg(x * m), xz=agg(x * zt), n=len(u))


def curves(sums_list, rng=None, nboot=0):
    """Benchmark-equal pooled w_mod, w_emp (point and optional pair-bootstrap draws [nboot,5])."""
    def one(weights):
        xx = xm = xz = 0.0
        for s, w in zip(sums_list, weights):
            ww = w / w.sum()
            xx = xx + ww @ s["xx"]; xm = xm + ww @ s["xm"]; xz = xz + ww @ s["xz"]
        return xm / xx, xz / xx

    pt = one([np.ones(s["n"]) for s in sums_list])
    if not nboot:
        return pt, None
    bm, be = np.empty((nboot, 5)), np.empty((nboot, 5))
    for k in range(nboot):
        wts = [np.bincount(rng.integers(0, s["n"], s["n"]), minlength=s["n"]).astype(float) for s in sums_list]
        bm[k], be[k] = one(wts)
    return pt, (bm, be)


def fit_h(w):
    """Least-squares h of w(n) = 1-exp(-n/h); w is [5] or [k,5]. Grid search on HGRID."""
    w = np.atleast_2d(w)
    f = 1.0 - np.exp(-NB[None, :] / HGRID[:, None])          # [G,5]
    sse = ((w[:, None, :] - f[None, :, :]) ** 2).sum(2)     # [k,G]
    return HGRID[sse.argmin(1)]


def fit_kappa(w):
    """Least-squares kappa of w(n) = n/(n+kappa) (Beta-binomial form), same grid."""
    w = np.atleast_2d(w)
    f = NB[None, :] / (NB[None, :] + HGRID[:, None])
    sse = ((w[:, None, :] - f[None, :, :]) ** 2).sum(2)
    return HGRID[sse.argmin(1)]


def ci(a):
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


# ---------------------------------------------------------------------------
# Prior pseudo-counts
# ---------------------------------------------------------------------------
def n_eff_mu(s, nu=3):
    """4 * [-d^2 log t_nu(mu; c, s) / dmu^2] at the centre = 4 (nu+1)/(nu s^2)."""
    return 4.0 * (nu + 1) / (nu * s * s)


def n_eff_pair(s, tau, nu=3):
    """Same curvature for the R2 single-pair logit prior t_nu(c,s) * N(0,tau^2) (numerical)."""
    mu = np.linspace(-40, 40, 160001)
    dm = mu[1] - mu[0]
    lt = -(nu + 1) / 2 * np.log1p(mu * mu / (nu * s * s))
    t = np.exp(lt); t /= t.sum() * dm
    h = 1e-3

    def f(th):
        return float((t * np.exp(-0.5 * ((th - mu) / tau) ** 2)).sum() * dm)

    d2 = (math.log(f(h)) - 2 * math.log(f(0.0)) + math.log(f(-h))) / (h * h)
    return 4.0 * -d2


def kappa_moment(c, s, tau, nu=3, n=2_000_000, seed=20261015):
    """Report-11 style Beta-binomial pseudo-count: m(1-m)/Var(p) - 1, p = sig(mu + tau x)."""
    rng = np.random.default_rng(seed)
    mu = c + s * (rng.standard_normal(n) if nu is None else rng.standard_t(nu, n))
    p = 1 / (1 + np.exp(-(mu + tau * rng.standard_normal(n))))
    m = p.mean()
    return float(m * (1 - m) / p.var() - 1)


def learned_scale(H):
    c = H["G"] / 2
    return float(max(0.5, math.sqrt(np.mean((H["_means"] - c) ** 2)))), c


# ---------------------------------------------------------------------------
def run():
    t0 = time.perf_counter()
    sha = check_hash()
    cfg = baseline_cfg()
    rng = np.random.default_rng(20261015)
    fz = FROZEN
    prior_counts = {
        "frozen": {
            "n_eff_mu_prior": n_eff_mu(fz["prior_scale"]),
            "n_eff_pair_logit_R2": n_eff_pair(fz["prior_scale"], fz["tau"]),
            "kappa_moment_match_pair_prob": kappa_moment(fz["prior_center"], fz["prior_scale"], fz["tau"]),
            "kappa_moment_match_pair_prob_normal_offset": kappa_moment(fz["prior_center"], fz["prior_scale"], fz["tau"], nu=None),
            "kappa_moment_match_report11_inputs_t3": kappa_moment(-0.58, 0.81, 1.13),
            "kappa_moment_match_report11_inputs_normal": kappa_moment(-0.58, 0.81, 1.13, nu=None),
        }
    }
    out = {"experiment_id": "EXP-0052", "preregistered": True, "reference_sha256": sha,
           "dataset_revision": "bc8204d811", "seeds": SEEDS, "budgets_with_labels": NB.tolist(),
           "config": cfg, "prior_counts": prior_counts, "regimes": {}}
    for R in REGIMES:
        res, folds = {}, {}
        held_sums, skipped = [], {}
        for B in E.EVAL_FOLDS:
            H = V1.H_fold(B)
            s, c = learned_scale(H)
            # training-fold curve: other benchmarks, outer-fold hyperparameters, outer B untouched
            tr_sums = []
            for X in E.BENCHES:
                if X == B:
                    continue
                try:
                    rp = replay(X, H, cfg, R, SEEDS[R])
                except ValueError:  # R1 draw rule selects no pair of X in any seed (swe_rebench)
                    skipped.setdefault(B, []).append(X)
                    log(f"  [{R} {B}] training benchmark {X}: no {R} episodes, skipped")
                    continue
                tr_sums.append(pair_sums(rp))
                del rp
            (wm_tr, we_tr), boot_tr = curves(tr_sums, rng, NBOOT)
            lam = we_tr / wm_tr
            # held-out
            rp = replay(B, H, cfg, R, SEEDS[R])
            hs = pair_sums(rp)
            held_sums.append(hs)
            (wm_ho, we_ho), boot_ho = curves([hs], rng, NBOOT)
            # Brier evidence: bounded correction learned on training benchmarks
            Pc = {0: rp["P"][0]}
            for j, b in enumerate(NB.astype(int)):
                shift = (lam[j] - 1.0) * (rp["Q"][b] - rp["Q"][0])
                Pc[b] = np.clip(rp["P"][b] + shift[rp["ep"].t_inst], 0.005, 0.995)
            res[B] = {"baseline": score(rp["ep"], rp["P"]), "lambda_corrected": score(rp["ep"], Pc)}
            h_tr, h_ho, h_mod = fit_h(we_tr)[0], fit_h(we_ho)[0], fit_h(wm_ho)[0]
            bh_tr, bh_ho, bh_mod = fit_h(boot_tr[1]), fit_h(boot_ho[1]), fit_h(boot_ho[0])
            ne = n_eff_mu(s)
            folds[B] = {
                "prior_centre": c, "prior_scale": s, "tau": H["tau"],
                "n_eff_mu_prior": ne, "n_eff_pair_logit_R2": n_eff_pair(s, H["tau"]),
                "h_equivalent_of_n_eff_mu": float(fit_h(NB / (NB + ne))[0]),
                "w_emp_train": we_tr.tolist(), "w_mod_train": wm_tr.tolist(), "lambda": lam.tolist(),
                "w_emp_heldout": we_ho.tolist(), "w_mod_heldout": wm_ho.tolist(),
                "w_emp_heldout_ci": [ci(boot_ho[1][:, j]) for j in range(5)],
                "h_emp_train": float(h_tr), "h_emp_train_ci": ci(bh_tr),
                "h_emp_heldout": float(h_ho), "h_emp_heldout_ci": ci(bh_ho),
                "h_model_heldout": float(h_mod), "h_model_heldout_ci": ci(bh_mod),
                "h_emp_minus_model_heldout_ci": ci(bh_ho - bh_mod),
                "kappa_emp_heldout": float(fit_kappa(we_ho)[0]), "kappa_emp_heldout_ci": ci(fit_kappa(boot_ho[1])),
                "kappa_model_heldout": float(fit_kappa(wm_ho)[0]),
                "kappa_emp_train": float(fit_kappa(we_tr)[0]),
                "n_pairs_heldout": hs["n"],
            }
            log(f"[{R} {B}] s={s:.3f} n_eff={ne:.2f} h_tr={h_tr:.2f} h_ho={h_ho:.2f} h_mod={h_mod:.2f} "
                f"lam={np.round(lam, 3).tolist()} {time.perf_counter() - t0:.0f}s")
            del rp
        (wm_all, we_all), boot_all = curves(held_sums, rng, NBOOT)
        summ = V2.summarise(res, ["baseline", "lambda_corrected"], "baseline")
        d = summ["lambda_corrected"]["d"]
        se = (d["hi"] - d["lo"]) / 3.92
        out["regimes"][R] = {
            "folds": folds,
            "pooled_heldout": {"w_emp": we_all.tolist(), "w_mod": wm_all.tolist(),
                               "h_emp": float(fit_h(we_all)[0]), "h_emp_ci": ci(fit_h(boot_all[1])),
                               "h_model": float(fit_h(wm_all)[0]), "h_model_ci": ci(fit_h(boot_all[0])),
                               "h_emp_minus_model_ci": ci(fit_h(boot_all[1]) - fit_h(boot_all[0])),
                               "kappa_emp": float(fit_kappa(we_all)[0]), "kappa_emp_ci": ci(fit_kappa(boot_all[1])),
                               "kappa_model": float(fit_kappa(wm_all)[0]), "kappa_model_ci": ci(fit_kappa(boot_all[0]))},
            "brier": summ, "mde_80pct_x1e3": 2.8 * se, "training_benchmarks_skipped": skipped,
            "baseline_matches_exp0044": abs(summ["baseline"]["alc"] - EXP0044_BASELINE[R]) < 5e-6,
        }
        log(f"[{R}] baseline ALC={summ['baseline']['alc']:.6f} corr d={d['pt']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}]e-3 "
            f"folds={ {k: round(v, 3) for k, v in d['per_fold'].items()} }")
    out["elapsed_seconds"] = time.perf_counter() - t0
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / "exp_0052_shrinkage_2026-10-15.json"
    path.write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8")
    log(f"wrote {path} in {out['elapsed_seconds']:.1f}s")
    return out


if __name__ == "__main__":
    run()
