"""EXP-0043: runtime item-text level predictor for the budget-0 benchmark level.

Pre-registration: docs/preregistrations/2026-10-14_comprehensive_research_preregistration.md, row EXP-0043.

Runtime computability (the binding constraint: predict() never sees the hidden benchmark's item pool)
------------------------------------------------------------------------------------------------
The arm uses, for each target item, ONLY that item's own runtime dict fields (`content` and
`item_features`). Ten scalar text statistics are computed from that single item, standardised with
constants fitted on training benchmarks, and mapped through a ridge fitted on training benchmarks
(labels of training benchmarks only, no hidden-benchmark labels, no benchmark identity, no pool
aggregation). The resulting per-item level deviation Delta(x) shifts the v1_3 offset prior centre
for that item at budget 0:   centre(x) = (G + Delta(x)) / 2   (v1_3: centre = G / 2, i.e. Delta = 0).
The prior width is v1_3's learned width, unchanged. Budgets >= 1 are v1_3 unchanged. Everything the
deployable predictor needs (feature means/sds, ridge coefficients, G, width, tau) is a constant shipped
with the model; at B0 nothing else is needed, so the arm is computable inside predict().

Ridge strength is selected by inner leave-one-training-benchmark-out CV on the training folds only
(alpha = inf, i.e. intercept only == v1_3, is a candidate). The outer held-out benchmark never chooses.

Evaluation: real data (R2 seeds 0-9, R1 seeds 0-99) + the report-17 A3 base-rate sweep
(Rasch resample of the held-out benchmark, logit shift delta in {-3..+3}, synthetic seeds 0,1,2,
identical across arms; R2 seeds 0-9, R1 seeds 0-49). Regret at each delta = ALC(arm) - min over arms.

Run:  python scripts/run_slot.py -- python -m paiec.analysis.exp_0043_text_level
"""
from __future__ import annotations

import json
import math
import re
import time

import numpy as np

from paiec import paths
from paiec.evaluation import episodes as E
from paiec.models import ladder as Lad
from paiec.models import v1_1 as V1
from paiec.models import v1_2 as V

OUTPUT = paths.RUNS / "preregistration" / "exp_0043_text_level_2026-10-15.json"
BASE_CFG = V.cfg_of(facets="all", guard=0)        # frozen v1_3 in the analysis engine
REAL_SEEDS = {"R2": list(range(10)), "R1": list(range(100))}
SWEEP_SEEDS = {"R2": list(range(10)), "R1": list(range(50))}
DELTAS = [-3, -2, -1, 0, 1, 2, 3]
SYN_SEEDS = [0, 1, 2]
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, math.inf]
FEATS = ["log_chars", "log_lines", "digit_frac", "math_frac", "code_frac", "upper_frac",
         "code_fence", "n_feature_fields", "has_url", "space_frac"]

_CODE = set("{}();=<>[]_")


def text_stats(content: str, feats: dict) -> np.ndarray:
    """Ten statistics of ONE runtime item (content + item_features). No pool information."""
    s = content or ""
    n = max(len(s), 1)
    letters = [c for c in s if c.isalpha()]
    return np.array([
        math.log1p(len(s)),
        math.log1p(s.count("\n")),
        sum(c.isdigit() for c in s) / n,
        (s.count("$") + s.count("\\")) / n,
        sum(c in _CODE for c in s) / n,
        (sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0,
        float("```" in s),
        float(len(feats or {})),
        float(bool(re.search(r"https?://", s))),
        sum(c.isspace() for c in s) / n,
    ])


_XB = {}


def bench_items(B):
    """Per-key text statistics (runtime-visible) and real-label item logits for benchmark B."""
    if B in _XB:
        return _XB[B]
    D = E.compact(B)
    kf = V.key_features(B, D)
    K = len(D["key_content"])
    X = np.stack([text_stats(str(D["key_content"][k]), kf.get(k, {})) for k in range(K)])
    key_row = D["item_key"][D["row_item"]]
    s = np.bincount(key_row, D["row_y"].astype(float), K)
    n = np.bincount(key_row, minlength=K).astype(float)
    q = (s + 0.5) / (n + 1)
    y = np.log(q / (1 - q))
    out = dict(X=X, y=y, n=n)
    _XB[B] = out
    return out


def fit_ridge(train, alpha):
    """Benchmark-balanced ridge (each benchmark total weight 1) on standardised stats."""
    Xs = [bench_items(b)["X"] for b in train]
    ys = [bench_items(b)["y"] for b in train]
    w = np.concatenate([np.full(len(x), 1.0 / len(x)) for x in Xs])
    X = np.concatenate(Xs); y = np.concatenate(ys)
    mu = (w[:, None] * X).sum(0) / w.sum()
    sd = np.sqrt((w[:, None] * (X - mu) ** 2).sum(0) / w.sum()) + 1e-9
    Z = (X - mu) / sd
    b0 = float((w * y).sum() / w.sum())
    if math.isinf(alpha):
        coef = np.zeros(X.shape[1])
    else:
        Zc = Z - (w[:, None] * Z).sum(0) / w.sum()
        A = Zc.T @ (w[:, None] * Zc) + (alpha / 100.0) * np.eye(X.shape[1])   # lam = alpha/100; data weight sums to #benchmarks
        coef = np.linalg.solve(A, Zc.T @ (w * (y - b0)))
        b0 = b0 - float(((w[:, None] * Z).sum(0) / w.sum()) @ coef)
    return dict(mu=mu, sd=sd, b0=b0, coef=coef, alpha=alpha)


def ridge_pred(m, X):
    return m["b0"] + ((X - m["mu"]) / m["sd"]) @ m["coef"]


def select_alpha(train):
    """Inner leave-one-training-benchmark-out; loss = benchmark-mean item-logit MSE."""
    cv = {}
    for a in ALPHAS:
        losses = []
        for h in train:
            m = fit_ridge([b for b in train if b != h], a)
            it = bench_items(h)
            losses.append(float(np.mean((ridge_pred(m, it["X"]) - it["y"]) ** 2)))
        cv[a] = float(np.mean(losses))
    best = min(ALPHAS, key=lambda a: cv[a])
    return best, cv


def level_model(B, finite_only=False):
    """finite_only=False: registered arm (alpha incl. inf chosen by inner CV).
    finite_only=True: non-registered sensitivity forcing a text fit (best finite inner-CV alpha)."""
    H = V1.H_fold(B)
    train = list(H["train"])
    alpha, cv = select_alpha(train)
    if finite_only:
        alpha = min([a for a in ALPHAS if not math.isinf(a)], key=lambda a: cv[a])
    m = fit_ridge(train, alpha)
    # Delta(x) is centred so that its benchmark-balanced training mean is exactly zero.
    m["centre_offset"] = float(np.mean([ridge_pred(m, bench_items(b)["X"]).mean() for b in train]))
    return dict(model=m, alpha=alpha, cv={str(k): v for k, v in cv.items()}, train=train)


def b0_q_for_centres(H, centres):
    """v1_3 B0 pair probability for a vector of prior centres (exact v1_1 pooled_pair with no labels)."""
    MU = V1.MU
    m = H.get("_means")
    G = float(H["G"])
    s = float(max(0.5, math.sqrt(np.mean((m - G / 2) ** 2))))        # v1_3 learned width, unchanged
    tau = H["tau"] * BASE_CFG["tau_scale"]
    eta = MU[:, None] + tau * V1.GH_X[None, :]
    Eq = (Lad.sig(eta) * V1.GH_W[None, :]).sum(1) / V1.GH_W.sum()
    uc, inv = np.unique(np.round(centres, 12), return_inverse=True)
    lp = np.stack([V1._t_logpdf_norm(MU, c, s, 3) for c in uc])
    lp -= lp.max(1, keepdims=True)
    post = np.exp(lp); post /= post.sum(1, keepdims=True)
    return (post @ Eq)[inv]


def b0_text(ep, H, lm, Xkey):
    """Per-target B0 prediction under centre(x) = (G + Delta(x))/2, using only the target item."""
    m = lm["model"]
    dlt = ridge_pred(m, Xkey[ep.t_key]) - m["centre_offset"]
    c = (float(H["G"]) + dlt) / 2
    return np.clip(b0_q_for_centres(H, c), 0.005, 0.995), dlt


ARMS = ("v1_3", "text_level", "text_forced")


def run_fold(B, D, R, seeds, lms, check=False):
    H = V1.H_fold(B)
    ep = E.build(D, R, seeds)
    fe = V.fe_for(B, D, BASE_CFG)
    probs = {b: V.predict(ep, H, b, BASE_CFG, fe) for b in E.BUDGETS}
    Xkey = bench_items(B)["X"]        # text depends on content only (identical for synthetic D)
    out, diag = {}, {}
    br, ec = ep.score(probs)
    out["v1_3"] = dict(br=br, ec=ec, alc=br @ E.W, pair=ep.inst_pair.copy())
    for arm in ARMS[1:]:
        p0, dlt = b0_text(ep, H, lms[arm][B], Xkey)
        br2, ec2 = ep.score({**probs, 0: p0})
        out[arm] = dict(br=br2, ec=ec2, alc=br2 @ E.W, pair=ep.inst_pair.copy())
        diag[arm] = dict(delta_mean=float(dlt.mean()), delta_sd=float(dlt.std()), p0_mean=float(p0.mean()))
    diag.update(p0_mean_v13=float(probs[0].mean()), obs_rate=float(ep.r_y.mean()))
    if check:
        # Equivalence: Delta == 0 must reproduce v1_3's B0 exactly.
        q0 = np.clip(b0_q_for_centres(H, np.full(len(ep.t_key), float(H["G"]) / 2)), 0.005, 0.995)
        diag["equiv_max_abs_diff_delta0"] = float(np.abs(q0 - probs[0]).max())
        # Scoring-path check: substituting Delta=0 B0 through the same {**probs, 0: p} path reproduces v1_3.
        brc, _ = ep.score({**probs, 0: q0})
        diag["equiv_scoring_path_max_abs_alc"] = float(np.abs(brc @ E.W - br @ E.W).max())
        if diag["equiv_max_abs_diff_delta0"] > 1e-9 or diag["equiv_scoring_path_max_abs_alc"] > 1e-12:
            raise RuntimeError(f"Delta=0 equivalence failed on {B} {R}: {diag}")
    return out, diag


def evaluate(D_of, seeds, lms, check=False):
    res, diags = {}, {}
    for R in seeds:
        res[R] = {}
        for B in E.EVAL_FOLDS:
            res[R][B], diags[f"{R}|{B}"] = run_fold(B, D_of(B), R, seeds[R], lms, check)
    return res, diags


def stats(res, arm):
    d = V1.delta(res, arm, "v1_3")
    d.update(V.fold_stats(d))
    d["b0"] = V1.delta(res, arm, "v1_3", budget=0)
    d["b0"].update(V.fold_stats(d["b0"]))
    se = (d["hi"] - d["lo"]) / (2 * 1.96)
    d["mde_x1e3"] = 2.8 * se          # 80% power, two-sided 5%, from the pair-bootstrap half-width
    d["alc"] = V1.mean_alc(res, arm); d["alc_v13"] = V1.mean_alc(res, "v1_3")
    d["curve"] = V1.mean_curve(res, arm).tolist(); d["curve_v13"] = V1.mean_curve(res, "v1_3").tolist()
    return d


def main():
    from paiec.analysis.redteam import synth_D
    t0 = time.time()
    lms = {"text_level": {B: level_model(B) for B in E.EVAL_FOLDS},
           "text_forced": {B: level_model(B, finite_only=True) for B in E.EVAL_FOLDS}}
    lm_report = {}
    for arm in lms:
        lm_report[arm] = {}
        for B, lm in lms[arm].items():
            m = lm["model"]
            held = bench_items(B)
            pred = ridge_pred(m, held["X"]) - m["centre_offset"]
            lm_report[arm][B] = dict(
                train=lm["train"], alpha=str(lm["alpha"]), inner_cv_mse=lm["cv"],
                coef=dict(zip(FEATS, m["coef"].round(5).tolist())),
                heldout_mean_delta=float(pred.mean()), heldout_sd_delta=float(pred.std()),
                heldout_true_level_minus_G=float(Lad.bench_stats(B)["mean"] - V1.H_fold(B)["G"]),
                heldout_item_spearman=(float(_spearman(pred, held["y"])) if pred.std() > 0 else None))
            print(arm, B, lm_report[arm][B], flush=True)

    real, real_diag = evaluate(lambda B: E.compact(B), REAL_SEEDS, lms, check=True)
    real_stats = {R: {arm: stats(real[R], arm) for arm in ARMS[1:]} for R in real}
    print(f"real done {time.time()-t0:.0f}s", flush=True)
    for R in real_stats:
        for arm in ARMS[1:]:
            print(R, arm, V.fmt(real_stats[R][arm]), "| B0", V.fmt(real_stats[R][arm]["b0"]), flush=True)

    sweep = {}
    for d in DELTAS:
        acc = {R: {nm: [] for nm in ARMS} for R in SWEEP_SEEDS}
        b0 = {R: {nm: [] for nm in ARMS} for R in SWEEP_SEEDS}
        for s in SYN_SEEDS:
            res, _ = evaluate(lambda B, s=s, d=d: synth_D(B, d, s), SWEEP_SEEDS, lms)
            for R in res:
                for nm in ARMS:
                    acc[R][nm].append(V1.mean_alc(res[R], nm))
                    b0[R][nm].append(float(V1.mean_curve(res[R], nm)[0]))
        sweep[d] = {R: {nm: dict(alc=float(np.mean(acc[R][nm])), b0=float(np.mean(b0[R][nm])))
                        for nm in ARMS} for R in acc}
        print(f"delta {d:+d} {time.time()-t0:.0f}s {json.dumps(sweep[d])}", flush=True)

    regret = {}
    for R in SWEEP_SEEDS:
        regret[R] = {}
        for metric in ("alc", "b0"):
            r = {nm: [1e3 * (sweep[d][R][nm][metric] - min(sweep[d][R][x][metric] for x in ARMS))
                      for d in DELTAS] for nm in ARMS}
            regret[R][metric] = dict(
                by_delta=r, max_regret={nm: max(v) for nm, v in r.items()},
                mean_regret={nm: float(np.mean(v)) for nm, v in r.items()},
                worst_value_x1e3={nm: 1e3 * max(sweep[d][R][nm][metric] for d in DELTAS) for nm in ARMS})
    kill = {R: {arm: not (regret[R]["alc"]["max_regret"][arm] < regret[R]["alc"]["max_regret"]["v1_3"])
                for arm in ARMS[1:]} for R in SWEEP_SEEDS}
    report = dict(experiment_id="EXP-0043", preregistered=True, dataset_revision="bc8204d811",
                  reference_cfg=str(BASE_CFG), features=FEATS, alphas=[str(a) for a in ALPHAS],
                  ridge_penalty=("lambda = alpha/100 on standardised stats; benchmark-balanced weights "
                                 "(each training benchmark total weight 1)"),
                  level_models=lm_report, real=real_stats, real_diagnostics=real_diag,
                  sweep=sweep, regret=regret, kill_rule_fires=kill,
                  seeds=dict(real=REAL_SEEDS, sweep=SWEEP_SEEDS, synthetic=SYN_SEEDS),
                  elapsed_seconds=time.time() - t0)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=1, default=float) + "\n", encoding="utf-8")
    print("regret", json.dumps(regret, default=float), flush=True)
    print(f"kill={kill} wrote {OUTPUT} {time.time()-t0:.0f}s", flush=True)


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


if __name__ == "__main__":
    main()
