"""EXP-0046 (E3): tinyBenchmarks p-IRT / gp-IRT as an external baseline in the LOBO harness.

Pre-registration: docs/preregistrations/2026-10-14_comprehensive_research_preregistration.md §2 (EXP-0046).
Reference estimator: Maia Polo et al., ICML 2024 (arXiv 2402.14992), Eqs. 4.1, 4.3, 4.4 and the
lambda heuristic of §4.3 (Song 1988, Cor. 2):

  IRT          P(Y_il=1) = sigmoid(alpha_i' theta_l - beta_i)
  theta_l      point estimate from the anchors with item parameters held fixed
  p-IRT        unseen-item prediction p_hat_il = P(Y_il=1 | theta_hat_l, alpha_i, beta_i)
  gp-IRT       lambda * mean(anchor Y) + (1 - lambda) * p-IRT
  lambda       b^2 / (sigma^2 / n + b^2); sigma^2 = mean over training models of the per-model sample
               variance of Y; b^2 = (mean |p-IRT score - true score|)^2 on held-out training models.

Here the "anchors" are the labels revealed at each budget, which come from a random stream.
The scored quantity is the per-item probability for the pair's unseen eval items, so the
per-item gp-IRT prediction is p_i = lambda*ybar + (1-lambda)*p_hat_i.

What transfers to an unseen benchmark: the hidden benchmark's items have no calibrated
(alpha_i, beta_i). Only the *population* of item parameters can come from the training
benchmarks. Consequences, implemented here:
  * d > 1 is unidentified without per-item alpha (rotation invariance), so d = 1 with a common alpha*.
  * New item: beta_k = mu + b_g + e_k, with b_g ~ N(0, tau^2) the unknown benchmark offset (shared
    within a runtime benchmark group) and e_k ~ N(0, sw^2). Unseen items are exchangeable, so
    p-IRT predicts the same f(eta_n) = E_e sigmoid(eta_n - e) for every unseen item of a pair.
  * "cold"  : theta_n (and b_g) MAP from own/group labels, all items marginal  -> per-pair constant.
  * "online": additionally, any target key already labelled in the same runtime group (other
              subjects in R1, or a same-content item of the subject) gets its e_k posterior from those
              labels (in-run calibration of that anchor). In R2 only same-key own labels can do this.
  * "mle"   : the paper's theta MLE. With exchangeable unseen items f is monotone, so the MLE
              plug-in equals the anchor mean ybar exactly, and gp-IRT = p-IRT = ybar.

Run (heavy): python scripts/run_slot.py -- python -m paiec.analysis.exp_0046_gpirt --step check,run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time

import numpy as np
from scipy.optimize import minimize

from paiec import paths
from paiec.evaluation import episodes as E
from paiec.models import v1_1 as V1
from paiec.models import v1_2 as V2

GX, GW = np.polynomial.hermite_e.hermegauss(20)
GW = GW / GW.sum()
LGW = np.log(GW)
OUT_DIR = paths.RUNS / "preregistration"
OUTPUT = OUT_DIR / "exp_0046_gpirt_2026-10-15.json"
CHECK_OUT = OUT_DIR / "exp_0046_gpirt_checks_2026-10-15.json"
REFERENCE_ARCHIVE = paths.DIST / "v1_3_named_facets_noguard.zip"
REFERENCE_SHA256 = "A4B0E841D4A344FF207BED2C855342212E95F7D550195B667446EA4E92685B86"
SEEDS = {"R2": list(range(10)), "R1": list(range(100))}
REGIMES = ("R2", "R1")
CALIB = ("matharena", "multi_swebench", "real_webagents", "researchcodebench")  # swe_rebench: 1 pair
LAMBDA_REPS = 5
NMAX = 31
CLIP = (0.005, 0.995)
ARMS = ("gpirt", "gpirt_cold", "pirt_cold", "gpirt_mle")
PRIMARY = "gpirt"


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def log_sig(x):
    return -np.logaddexp(0.0, -x)


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# Training-benchmark calibration: 1-D IRT, logit = alpha * theta_s - beta_k, theta ~ N(0, 1)
# ---------------------------------------------------------------------------
_FIT = {}


def fit_benchmark(bench):
    """JML (weak N(0,10^2) ridge) on one training benchmark, then theta standardised to mean 0, sd 1. Items are grouped by runtime content key."""
    if bench in _FIT:
        return _FIT[bench]
    D = E.compact(bench)
    S = len(D["pair_sid"])
    keys = D["item_key"][D["row_item"]].astype(np.int64)
    K = int(keys.max()) + 1
    cell = D["row_pair"].astype(np.int64) * K + keys
    uc, inv = np.unique(cell, return_inverse=True)
    t = np.bincount(inv).astype(float)
    k = np.bincount(inv, D["row_y"].astype(float))
    cs, ck = uc // K, uc % K
    uk, ck = np.unique(ck, return_inverse=True)
    nK = len(uk)

    def nlp(x):
        th, be, la = x[:S], x[S:S + nK], x[-1]
        a = math.exp(la)
        z = a * th[cs] - be[ck]
        ll = k * log_sig(z) + (t - k) * log_sig(-z)
        g = k - t * sig(z)
        f = -ll.sum() + 0.5 * (th ** 2).sum() / 100.0 + 0.5 * (be ** 2).sum() / 100.0
        gth = -np.bincount(cs, g * a, S) + th / 100.0
        gbe = np.bincount(ck, g, nK) + be / 100.0
        gla = -(g * a * th[cs]).sum()
        return f, np.concatenate([gth, gbe, [gla]])

    x0 = np.zeros(S + nK + 1)
    r = minimize(nlp, x0, jac=True, method="L-BFGS-B", options=dict(maxiter=3000))
    th, be, a = r.x[:S], r.x[S:S + nK], math.exp(r.x[-1])
    # identification: weak N(0, 10^2) penalties only stabilise; fix the scale by theta ~ (0, 1)
    c, sd = th.mean(), th.std()
    be, a, th = be - a * c, a * sd, (th - c) / sd
    ybar = np.bincount(D["row_pair"], D["row_y"].astype(float), S) / np.bincount(D["row_pair"], minlength=S)
    out = dict(alpha=a, m=float(be.mean()), v=float(be.var()), n_pairs=S, n_keys=nK,
               converged=bool(r.success), theta_sd=float(th.std()),
               sigma2=float(np.mean(ybar * (1 - ybar))))
    _FIT[bench] = out
    return out


def prior_from(benches):
    fits = [fit_benchmark(b) for b in benches]
    ms = np.array([f["m"] for f in fits])
    return dict(alpha=float(np.mean([f["alpha"] for f in fits])), mu=float(ms.mean()),
                tau2=float(ms.var(ddof=1)), sw=float(math.sqrt(np.mean([f["v"] for f in fits]))),
                sigma2=float(np.mean([f["sigma2"] for f in fits])), benches=list(benches))


# ---------------------------------------------------------------------------
# Link and point estimation
# ---------------------------------------------------------------------------
def link(eta, sw):
    """f(eta) = E_{e~N(0,sw^2)} sigmoid(eta - e) and its derivative."""
    s = sig(np.asarray(eta)[..., None] - sw * GX)
    return s @ GW, (s * (1 - s)) @ GW


def map_eta(n1, n0, grp, n_groups, P, iters=200, tol=1e-10):
    """Joint MAP of u_n = alpha*theta_n ~ N(0, alpha^2) and b_g ~ N(0, tau^2); eta = u - mu - b_g.

    Block-coordinate Fisher scoring; the log-likelihood n1 log f + n0 log(1-f) is concave in eta
    because f is the CDF of a logistic + Gaussian sum (log-concave).
    """
    a2, t2 = P["alpha"] ** 2, P["tau2"]
    u = np.zeros(len(n1))
    b = np.zeros(n_groups)
    n = n1 + n0
    for _ in range(iters):
        eta = u - P["mu"] - b[grp]
        f, fp = link(eta, P["sw"])
        f = np.clip(f, 1e-12, 1 - 1e-12)
        sc = n1 * fp / f - n0 * fp / (1 - f)
        info = n * fp ** 2 / (f * (1 - f))
        du = (sc - u / a2) / (info + 1 / a2)
        u = u + du
        eta = u - P["mu"] - b[grp]
        f, fp = link(eta, P["sw"])
        f = np.clip(f, 1e-12, 1 - 1e-12)
        sc = n1 * fp / f - n0 * fp / (1 - f)
        info = n * fp ** 2 / (f * (1 - f))
        db = (-np.bincount(grp, sc, n_groups) - b / t2) / (np.bincount(grp, info, n_groups) + 1 / t2)
        b = b + db
        if max(np.abs(du).max(initial=0), np.abs(db).max(initial=0)) < tol:
            break
    return u - P["mu"] - b[grp]


def online_item(eta_inst, li, lk, ly, t_inst, t_key, grp_of_inst, K, sw):
    """For targets whose (group, key) was labelled: posterior predictive under e_k ~ N(0, sw^2)
    updated by those labels with plug-in eta. Returns (mask, p) over targets."""
    g_l = grp_of_inst[li].astype(np.int64) * K + lk
    g_t = grp_of_inst[t_inst].astype(np.int64) * K + t_key
    mask = np.zeros(len(t_inst), bool)
    p = np.zeros(len(t_inst))
    if not len(li):
        return mask, p
    uk, inv = np.unique(g_l, return_inverse=True)
    z = eta_inst[li][:, None] - sw * GX[None, :]
    contrib = np.where(ly[:, None] > 0.5, log_sig(z), log_sig(-z))
    acc = np.zeros((len(uk), len(GX)))
    np.add.at(acc, inv, contrib)
    pos = np.minimum(np.searchsorted(uk, g_t), len(uk) - 1)
    mask = uk[pos] == g_t
    if mask.any():
        a = acc[pos[mask]] + LGW
        w = np.exp(a - a.max(1, keepdims=True))
        w /= w.sum(1, keepdims=True)
        p[mask] = (w * sig(eta_inst[t_inst[mask]][:, None] - sw * GX[None, :])).sum(1)
    return mask, p


# ---------------------------------------------------------------------------
# lambda heuristic (nested leave-one-training-benchmark-out, as the hidden benchmark is uncalibrated)
# ---------------------------------------------------------------------------
def _pair_tables(bench, reps=LAMBDA_REPS):
    """Per pair x rep: cumulative anchor successes (first response per key in random order) and
    held-out true score after removing the first n keys (n = 1..NMAX)."""
    D = E.compact(bench)
    keys = D["item_key"][D["row_item"]].astype(np.int64)
    order = np.argsort(D["row_pair"], kind="stable")
    rp, rk, ry = D["row_pair"][order], keys[order], D["row_y"][order].astype(float)
    bounds = np.searchsorted(rp, np.arange(len(D["pair_sid"]) + 1))
    cum1, true = [], []
    for p in range(len(D["pair_sid"])):
        k, y = rk[bounds[p]:bounds[p + 1]], ry[bounds[p]:bounds[p + 1]]
        uk, first, inv = np.unique(k, return_index=True, return_inverse=True)
        ks, kc = np.bincount(inv, y), np.bincount(inv).astype(float)
        S, C = ks.sum(), kc.sum()
        h = int(hashlib.sha1(f"exp0046|{bench}|{p}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(h)
        for _ in range(reps):
            perm = rng.permutation(len(uk))[:NMAX]
            fy = y[first[perm]]
            cum1.append(np.cumsum(fy))
            true.append((S - np.cumsum(ks[perm])) / (C - np.cumsum(kc[perm])))
    return np.array(cum1), np.array(true)


def lambda_table(train):
    """b^2(n), sigma^2 and lambda(n) for n = 0..NMAX using only the fold's training benchmarks."""
    P = prior_from(train)
    absdev = np.zeros(NMAX)
    for j in train:
        Pj = prior_from([b for b in train if b != j])
        c1, tr = _pair_tables(j)
        devs = np.zeros(NMAX)
        for n in range(1, NMAX + 1):
            n1 = c1[:, n - 1]
            eta = map_eta(n1, n - n1, np.arange(len(n1)), len(n1), Pj)
            devs[n - 1] = np.mean(np.abs(link(eta, Pj["sw"])[0] - tr[:, n - 1]))
        absdev += devs / len(train)
    b2 = absdev ** 2
    n = np.arange(1, NMAX + 1)
    lam = np.concatenate([[0.0], b2 / (P["sigma2"] / n + b2)])
    return P, dict(b2=b2.tolist(), sigma2=P["sigma2"], lam=lam.tolist())


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_all(ep, P, lam, b):
    li, lk, ly = ep.labels(b)
    n1 = np.bincount(li, ly, ep.n)
    nn = np.bincount(li, minlength=ep.n).astype(float)
    grp = np.unique(ep.inst_group, return_inverse=True)[1]
    G = int(grp.max()) + 1
    eta = map_eta(n1, nn - n1, grp, G, P)
    p_cold = link(eta, P["sw"])[0]
    ybar = np.where(nn > 0, n1 / np.maximum(nn, 1), 0.0)
    lam_n = np.asarray(lam)[np.minimum(nn, NMAX).astype(int)]
    gp = lambda pi: lam_n[ep.t_inst] * ybar[ep.t_inst] + (1 - lam_n[ep.t_inst]) * pi
    pc = p_cold[ep.t_inst]
    K = len(ep.D["key_jev"])
    mask, p_on = online_item(eta, li, lk, ly, ep.t_inst, ep.t_key, grp, K, P["sw"])
    po = np.where(mask, p_on, pc)
    mle = np.where(nn > 0, ybar, link(np.array([-P["mu"]]), P["sw"])[0][0])[ep.t_inst]
    out = {"gpirt": gp(po), "gpirt_cold": gp(pc), "pirt_cold": pc, "gpirt_mle": mle}
    return {k: np.clip(v, *CLIP) for k, v in out.items()}, float(mask.mean())


# ---------------------------------------------------------------------------
# Numerical checks (synthetic, brute force)
# ---------------------------------------------------------------------------
def check():
    rng = np.random.default_rng(0)
    res = {}
    # 1. link vs Monte Carlo
    sw = 1.3
    e = rng.normal(0, sw, 2_000_000)
    res["link_vs_mc_maxabs"] = float(max(abs(link(np.array([x]), sw)[0][0] - sig(x - e).mean())
                                         for x in (-3, -1, 0, 0.7, 2.5)))
    # 2. map_eta vs scipy on a grouped synthetic problem
    P = dict(alpha=1.4, mu=0.3, tau2=1.7, sw=1.1)
    grp = np.repeat(np.arange(4), [1, 3, 5, 2])
    nn = rng.integers(0, 32, len(grp)).astype(float)
    n1 = np.floor(nn * rng.uniform(0, 1, len(grp)))
    eta = map_eta(n1, nn - n1, grp, 4, P)

    def obj(x):
        u, bb = x[:len(grp)], x[len(grp):]
        f = np.clip(link(u - P["mu"] - bb[grp], P["sw"])[0], 1e-15, 1 - 1e-15)
        return -(n1 * np.log(f) + (nn - n1) * np.log1p(-f)).sum() + 0.5 * (u ** 2).sum() / P["alpha"] ** 2 \
            + 0.5 * (bb ** 2).sum() / P["tau2"]
    r = minimize(obj, np.zeros(len(grp) + 4), method="Nelder-Mead",
                 options=dict(maxiter=200000, maxfev=200000, xatol=1e-10, fatol=1e-14))
    r = minimize(obj, r.x, method="BFGS", options=dict(gtol=1e-10))
    ref = r.x[:len(grp)] - P["mu"] - r.x[len(grp):][grp]
    res["map_vs_scipy_maxabs_eta"] = float(np.abs(eta - ref).max())
    # 3. online item posterior vs fine-grid integration
    eta_i = np.array([0.4, -1.0, 1.5])
    li = np.array([0, 1, 2, 1]); lk = np.array([7, 7, 7, 3]); ly = np.array([1., 0., 1., 1.])
    m, p = online_item(eta_i, li, lk, ly, np.array([0, 2]), np.array([7, 3]), np.zeros(3, int), 10, sw)
    xs = np.linspace(-10 * sw, 10 * sw, 200001)
    pr = np.exp(-0.5 * (xs / sw) ** 2)
    lik = sig(eta_i[0] - xs) * (1 - sig(eta_i[1] - xs)) * sig(eta_i[2] - xs)
    w = pr * lik
    ref0 = (w * sig(eta_i[0] - xs)).sum() / w.sum()
    w2 = pr * sig(eta_i[1] - xs)
    ref1 = (w2 * sig(eta_i[2] - xs)).sum() / w2.sum()
    res["online_vs_grid_maxabs"] = float(max(abs(p[0] - ref0), abs(p[1] - ref1)))
    res["online_mask"] = m.tolist()
    # 4. IRT fit recovery on synthetic data (fit routine re-used via a fake compact dict)
    S, Kk, a_true = 60, 300, 1.5
    th, be = rng.normal(0, 1, S), rng.normal(-0.5, 1.2, Kk)
    rp = np.repeat(np.arange(S), Kk); rk = np.tile(np.arange(Kk), S)
    y = (rng.uniform(size=S * Kk) < sig(a_true * th[rp] - be[rk])).astype(np.int8)
    fake = dict(pair_sid=np.arange(S), item_key=np.arange(Kk), row_item=rk, row_pair=rp, row_y=y)
    orig = E.compact
    E.compact = lambda b: fake
    try:
        _FIT.pop("__synthetic__", None)
        fit = fit_benchmark("__synthetic__")
    finally:
        E.compact = orig
        _FIT.pop("__synthetic__", None)
    # identified truth: theta standardised in-sample
    a_id, m_id = a_true * th.std(), float(be.mean() - a_true * th.mean())
    res["synthetic_fit"] = dict(alpha_true_identified=a_id, alpha_hat=fit["alpha"], m_true_identified=m_id,
                                m_hat=fit["m"], sd_true=float(be.std()), sd_hat=math.sqrt(fit["v"]),
                                converged=fit["converged"])
    ok = (res["link_vs_mc_maxabs"] < 2e-3 and res["map_vs_scipy_maxabs_eta"] < 1e-4
          and res["online_vs_grid_maxabs"] < 1e-6 and abs(fit["alpha"] / a_id - 1) < 0.15 and abs(fit["m"] - m_id) < 0.2)
    res["pass"] = bool(ok)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECK_OUT.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    log(json.dumps(res, indent=1))
    assert ok, res
    return res


# ---------------------------------------------------------------------------
# Main replay
# ---------------------------------------------------------------------------
def adoption(d, b1):
    return bool(d["pt"] <= -1.0 and d["hi"] < 0 and d["n_neg"] >= 3 and d["worst_fold"] <= 0.5)


def run():
    actual = hashlib.sha256(REFERENCE_ARCHIVE.read_bytes()).hexdigest().upper()
    if actual != REFERENCE_SHA256:
        raise RuntimeError("reference archive hash mismatch")
    from paiec.analysis.evaluate_exp0044_tempering import _validate_reference
    ref_eq = _validate_reference()
    t0 = time.perf_counter()
    base_cfg = V2.cfg_of(facets="all", guard=0)
    calib, lam_by = {}, {}
    for B in E.EVAL_FOLDS:
        train = [b for b in CALIB if b != B]
        P, lt = lambda_table(train)
        calib[B] = dict(prior=P, lambda_table=lt)
        lam_by[B] = (P, lt["lam"])
        log(f"[calib {B}] alpha={P['alpha']:.3f} mu={P['mu']:.3f} tau2={P['tau2']:.3f} sw={P['sw']:.3f} "
            f"sigma2={P['sigma2']:.4f} lam(1,7,31)={lt['lam'][1]:.3f},{lt['lam'][7]:.3f},{lt['lam'][31]:.3f}")
    calib["fits"] = {b: fit_benchmark(b) for b in CALIB}
    raw, diag = {}, {}
    names = ["v1_3", *ARMS]
    for R in REGIMES:
        res = {}
        for B in E.EVAL_FOLDS:
            D = E.compact(B)
            ep = E.build(D, R, SEEDS[R])
            out = V2.run_cfgs(B, D, ep, V1.H_fold(B), {"v1_3": base_cfg})
            P, lam = lam_by[B]
            probs = {a: {} for a in ARMS}
            online = []
            for b in E.BUDGETS:
                pr, frac = predict_all(ep, P, lam, b)
                online.append(frac)
                for a in ARMS:
                    probs[a][b] = pr[a]
            for a in ARMS:
                br, ec = ep.score(probs[a])
                out[a] = dict(br=br, ec=ec, alc=br @ E.W, pair=ep.inst_pair.copy())
            res[B] = out
            diag[f"{R}|{B}"] = dict(n_inst=int(ep.n), online_target_frac_by_budget=online)
            log(f"  [{R} {B}] n_inst={ep.n} online_frac={online} {time.perf_counter()-t0:.0f}s")
        raw[R] = res
    summ = {R: V2.summarise(raw[R], names, "v1_3") for R in REGIMES}
    decisions = {}
    for R in REGIMES:
        decisions[R] = {}
        for a in ARMS:
            d = summ[R][a]["d"]
            decisions[R][a] = dict(
                mean_delta_alc_x1e3=d["pt"], pair_bootstrap_95ci_x1e3=[d["lo"], d["hi"]],
                hier_bootstrap_95ci_x1e3=[d["hlo"], d["hhi"]], fold_t3_95ci_x1e3=d["t3"],
                fold_mean_delta_alc_x1e3=d["per_fold"], negative_folds=d["n_neg"],
                worst_fold_delta_x1e3=d["worst_fold"],
                budget_delta_x1e3=[x["pt"] for x in summ[R][a]["d_budget"]],
                passes_registered_adoption_bar=adoption(d, None))
            log(f"{R} {a}: ALC={summ[R][a]['alc']:.5f} d={d['pt']:+.2f}e-3 "
                f"[{d['lo']:+.2f},{d['hi']:+.2f}] folds={ {k[:5]: round(v, 2) for k, v in d['per_fold'].items()} }")
        log(f"{R} v1_3 ALC={summ[R]['v1_3']['alc']:.5f}")
    report = dict(
        experiment_id="EXP-0046", preregistered=True, reference_sha256=actual,
        reference_equivalence=ref_eq, dataset_revision="bc8204d811", folds=list(E.EVAL_FOLDS),
        regimes={R: SEEDS[R] for R in REGIMES}, budgets=list(E.BUDGETS), primary_arm=PRIMARY,
        arms={"gpirt": "gp-IRT, 1-D, item population prior, theta/b_g MAP, in-run calibration of labelled keys",
              "gpirt_cold": "gp-IRT with every target item uncalibrated (per-pair constant)",
              "pirt_cold": "p-IRT (lambda=0) with every target item uncalibrated",
              "gpirt_mle": "paper's theta MLE; equals anchor mean ybar for n>0, prior plug-in at n=0"},
        calibration=calib, diagnostics=diag, elapsed_seconds=time.perf_counter() - t0,
        results=summ, decisions=decisions)
    OUTPUT.write_text(json.dumps(report, indent=1, default=float) + "\n", encoding="utf-8")
    log(f"wrote {OUTPUT} elapsed={report['elapsed_seconds']:.0f}s")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="check")
    for s in ap.parse_args().step.split(","):
        globals()[s]()
