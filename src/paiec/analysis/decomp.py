"""Signal decomposition for PAIEC (orchestrator §7 Components A-E), Level 1/2 evidence.

Outputs:
  docs/research_notes/11_SIGNAL_DECOMPOSITION.md   (written by hand from the printed results)
  artifacts/priors_v1.json             (written here)
  artifacts/decomp_results.json        (raw numbers, written here)

Run:  python src/paiec/analysis/decomp.py   (~ a few minutes, numpy/scipy/pandas only, no network, no caches)

Model family (binomial counts k/m per subject-item cell):
  logit p = mu + sum_g u_g[idx_g(cell)],  u_g ~ N(0, var_g[class])   (Gaussian random effects)
fit by MAP (L-BFGS) with empirical-Bayes variance updates using a diagonal Laplace correction
(var = mean(u^2 + post_var)). This is an approximation (diag Laplace) - adequate for variance
components at these sizes, documented as such in the report.
"""
from __future__ import annotations
import json, os, itertools, time
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit, gammaln

from paiec import paths

ROOT = str(paths.ROOT)
DATA = str(paths.DATA)
ART = str(paths.ARTIFACTS)
BINARY = ["matharena", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench"]
ELIG = ["multi_swebench", "real_webagents", "researchcodebench", "swe_rebench"]
BUDGETS = [0, 1, 3, 7, 15, 31]
W = {0: .1, 1: .2, 3: .2, 7: .2, 15: .2, 31: .1}
RNG = np.random.default_rng(20261001)


# ----------------------------------------------------------------------------- data
def load_cells(b, binary_only=True):
    r = pd.read_parquet(os.path.join(DATA, b, "response.parquet"), columns=["subject_id", "item_id", "response"])
    s = pd.read_parquet(os.path.join(DATA, b, "subjects.parquet"))
    if binary_only:
        r = r[r.response.isin([0.0, 1.0])]
    g = r.groupby(["subject_id", "item_id"]).response.agg(k="sum", m="size").reset_index()
    g = g.merge(s[["subject_id", "normalized_name", "provider"]], on="subject_id", how="left")
    g["provider"] = g.provider.fillna("").replace("", "UNK")
    # runtime sees "" for missing names (52/104 matharena subjects): keep them, but never match across benchmarks
    g["normalized_name"] = g.normalized_name.fillna("").astype(str)
    miss = g.normalized_name == ""
    g.loc[miss, "normalized_name"] = "UNK:" + g.loc[miss, "subject_id"]
    g["benchmark"] = b
    return g


def pair_table(c):
    p = c.groupby(["benchmark", "subject_id", "normalized_name", "provider"]).agg(
        k=("k", "sum"), m=("m", "sum"), n_items=("item_id", "nunique")).reset_index()
    p["acc"] = p.k / p.m
    return p


# ----------------------------------------------------------------------------- GLMM
def codes(x):
    u, inv = np.unique(np.asarray(x).astype(str), return_inverse=True)
    return inv, len(u), u


def fit_glmm(k, m, groups, iters=8, eb=True, mu_init=None, fixed_var=None, mu_sd=100.0, offset=None):
    """groups: list of dict(name, idx[ncell], n, cls[n] (class per level), var[ncls]).
    Returns dict with mu, effects, vars, and per-cell eta."""
    k = np.asarray(k, float); m = np.asarray(m, float)
    sizes = [g["n"] for g in groups]
    offs = np.concatenate([[0], np.cumsum(sizes)])
    P = offs[-1]
    mu0 = logit(np.clip(k.sum() / m.sum(), 1e-3, 1 - 1e-3)) if mu_init is None else mu_init
    x = np.zeros(P + 1); x[0] = mu0
    for g in groups:
        g.setdefault("cls", np.zeros(g["n"], int))
        g["var"] = np.asarray(g.get("var", np.ones(g["cls"].max() + 1)), float)

    def unpack(x):
        return x[0], [x[1 + offs[j]:1 + offs[j + 1]] for j in range(len(groups))]

    def eta_of(x):
        mu, us = unpack(x)
        e = np.full(len(k), mu) + (0.0 if offset is None else offset)
        for g, u in zip(groups, us):
            e += u[g["idx"]]
        return e

    def f(x):
        e = eta_of(x)
        ll = k * e - m * np.logaddexp(0, e)
        p = expit(e); r = k - m * p
        mu, us = unpack(x)
        pm = 1.0 / mu_sd ** 2
        obj = -ll.sum(); grad = np.zeros_like(x); grad[0] = -r.sum() + (mu - mu0) * pm
        obj += 0.5 * pm * (mu - mu0) ** 2
        for j, (g, u) in enumerate(zip(groups, us)):
            prec = 1.0 / g["var"][g["cls"]]
            obj += 0.5 * np.sum(prec * u * u)
            grad[1 + offs[j]:1 + offs[j + 1]] = -np.bincount(g["idx"], r, g["n"]) + prec * u
        return obj, grad

    hist = []
    for it in range(iters if eb else 1):
        res = minimize(f, x, jac=True, method="L-BFGS-B", options=dict(maxiter=500))
        x = res.x
        if not eb:
            break
        e = eta_of(x); p = expit(e); w = m * p * (1 - p)
        mu, us = unpack(x)
        newv = []
        for g, u in zip(groups, us):
            if g.get("fixed", False):
                newv.append(g["var"]); continue
            H = np.bincount(g["idx"], w, g["n"]) + 1.0 / g["var"][g["cls"]]
            pv = 1.0 / H
            v = np.array([np.mean((u * u + pv)[g["cls"] == c]) if np.any(g["cls"] == c) else g["var"][c]
                          for c in range(len(g["var"]))])
            g["var"] = np.maximum(v, 1e-4); newv.append(g["var"].copy())
        hist.append([v.tolist() for v in newv])
    mu, us = unpack(x)
    e = eta_of(x); w = m * expit(e) * (1 - expit(e))
    pvar = {g["name"]: 1.0 / (np.bincount(g["idx"], w, g["n"]) + 1.0 / g["var"][g["cls"]]) for g in groups}
    mu_pvar = 1.0 / (w.sum() + 1.0 / mu_sd ** 2)
    return dict(mu=float(mu), mu_pvar=float(mu_pvar), pvar=pvar, u={g["name"]: u for g, u in zip(groups, us)},
                var={g["name"]: g["var"].tolist() for g in groups}, eta=eta_of(x), hist=hist)


def predict_eta(fit, groups_eval, ncell):
    e = np.full(ncell, fit["mu"])
    for name, idx, valid in groups_eval:
        u = fit["u"][name]
        e += np.where(valid, u[np.clip(idx, 0, len(u) - 1)], 0.0)
    return e


# ----------------------------------------------------------------------------- Brier helpers
def cell_brier(p, k, m):
    """expected per-trial Brier of predicting p for a cell with k/m successes (mean over trials)."""
    yb = k / m
    return p * p - 2 * p * yb + yb


def pair_mean_brier(df, pcol):
    """equal-pair-weighted Brier; within pair, equal weight per trial (official: each trial a target)."""
    b = cell_brier(df[pcol].values, df.k.values, df.m.values) * df.m.values
    t = pd.DataFrame({"pair": df.subject_id.values, "b": b, "m": df.m.values}).groupby("pair").sum()
    return float((t.b / t.m).mean())


# ----------------------------------------------------------------------------- 1. variance decomposition
def per_benchmark_rasch(c, with_cell=False):
    si, ns, _ = codes(c.subject_id); ii, ni, _ = codes(c.item_id)
    groups = [dict(name="subj", idx=si, n=ns, var=[1.0]), dict(name="item", idx=ii, n=ni, var=[1.0])]
    if with_cell:
        groups.append(dict(name="cell", idx=np.arange(len(c)), n=len(c), var=[0.5]))
    return fit_glmm(c.k.values, c.m.values, groups, iters=10), si, ii


def cv_components(c, folds=5):
    """5-fold CV over cells: constant/benchmark-mean/pair-mean/item-mean/additive Rasch."""
    fold = RNG.integers(0, folds, len(c))
    out = {k: [] for k in ["const05", "bench", "pair", "item", "rasch"]}
    preds = {k: np.zeros(len(c)) for k in out}
    si, ns, _ = codes(c.subject_id); ii, ni, _ = codes(c.item_id)
    for f in range(folds):
        tr, te = fold != f, fold == f
        k, m = c.k.values[tr], c.m.values[tr]
        preds["const05"][te] = 0.5
        preds["bench"][te] = k.sum() / m.sum()
        for name, gl in [("pair", [("subj", si, ns)]), ("item", [("item", ii, ni)]),
                         ("rasch", [("subj", si, ns), ("item", ii, ni)])]:
            groups = [dict(name=g, idx=idx[tr], n=n, var=[1.0]) for g, idx, n in gl]
            fit = fit_glmm(k, m, groups, iters=6)
            seen = {g: np.isin(idx[te], np.unique(idx[tr])) for g, idx, n in gl}
            e = predict_eta(fit, [(g, idx[te], seen[g]) for g, idx, n in gl], te.sum())
            preds[name][te] = expit(e)
    d = c.copy()
    res = {}
    for kname, p in preds.items():
        d["p"] = p
        res[kname] = pair_mean_brier(d, "p")
    rep = c[c.m >= 2]
    if len(rep):
        yb = rep.k / rep.m
        floor_cell = (yb * (1 - yb) * rep.m / (rep.m - 1))
        res["noise_floor_repeat_cells"] = float(floor_cell.mean())
        res["share_cells_repeated"] = float(len(rep) / len(c))
    return res


# ----------------------------------------------------------------------------- 2. transfer
def name_dev_table(cells_by_b, fits):
    """within-benchmark ability per (benchmark, name): Rasch theta (avg over subject_ids with that name),
    raw, and z-scored within benchmark; plus logit pair accuracy deviation."""
    rows = []
    for b, c in cells_by_b.items():
        fit, si, ii = fits[b]
        _, _, sl = codes(c.subject_id)
        th = pd.Series(fit["u"]["subj"], index=sl)
        pt = pair_table(c).set_index("subject_id")
        pt["theta"] = th.reindex(pt.index).values
        pt["lacc"] = logit(np.clip((pt.k + .5) / (pt.m + 1), 1e-3, 1 - 1e-3))
        pt = pt[pt.n_items >= 80] if b != "swe_rebench" else pt
        g = pt.groupby("normalized_name").agg(theta=("theta", "mean"), lacc=("lacc", "mean"),
                                              provider=("provider", "first")).reset_index()
        g["benchmark"] = b
        g["lacc_dev"] = g.lacc - g.lacc.mean()
        sd = g.theta.std() if len(g) > 1 else 1.0
        g["z"] = (g.theta - g.theta.mean()) / (sd if sd > 0 else 1.0)
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def transfer_analysis(T, col="z"):
    out = {"pairwise": []}
    bs = sorted(T.benchmark.unique())
    piv = T.pivot_table(index="normalized_name", columns="benchmark", values=col)
    for a, b in itertools.combinations(bs, 2):
        sub = piv[[a, b]].dropna()
        if len(sub) >= 4:
            out["pairwise"].append(dict(a=a, b=b, n=len(sub), pearson=float(sub.corr().iloc[0, 1]),
                                        spearman=float(sub.corr("spearman").iloc[0, 1])))
    multi = piv[piv.notna().sum(1) >= 2]
    # one-way random effects ICC on names with >=2 benchmarks
    vals = [r.dropna().values for _, r in multi.iterrows()]
    grand = np.mean(np.concatenate(vals))
    nk = np.array([len(v) for v in vals]); N = nk.sum(); G = len(vals)
    ssb = sum(len(v) * (v.mean() - grand) ** 2 for v in vals); ssw = sum(((v - v.mean()) ** 2).sum() for v in vals)
    msb, msw = ssb / (G - 1), ssw / (N - G)
    n0 = (N - (nk ** 2).sum() / N) / (G - 1)
    icc = (msb - msw) / (msb + (n0 - 1) * msw)
    out.update(n_names_multi=int(G), n_obs=int(N), icc=float(icc))
    # leave-one-benchmark-out prediction of a name's value from mean of its other benchmarks
    xs, ys = [], []
    for name, r in multi.iterrows():
        r = r.dropna()
        for b in r.index:
            others = r.drop(b)
            xs.append(others.mean()); ys.append(r[b])
    xs, ys = np.array(xs), np.array(ys)
    slope = float(np.sum(xs * ys) / np.sum(xs * xs))
    out.update(loo_r=float(np.corrcoef(xs, ys)[0, 1]), loo_slope_through0=slope, loo_n=len(xs),
               loo_mse_0=float(np.mean(ys ** 2)), loo_mse_slope=float(np.mean((ys - slope * xs) ** 2)))
    return out


def transfer_additive(T, col="lacc", lam=1.0):
    """Two-way additive model lacc[name,b] = d_b + a_name fit on all benchmarks except b (ridge on a);
    then on held-out b: correlation between a_name and lacc_b over shared names (offset-free, i.e. what a
    model gets once the benchmark offset is learned from pooled labels). Pooled across folds after
    within-fold centering; slope = optimal multiplier on a_name (shrinkage)."""
    xs, ys, per = [], [], {}
    for hb in sorted(T.benchmark.unique()):
        tr = T[T.benchmark != hb]; te = T[T.benchmark == hb]
        a = pd.Series(0.0, index=tr.normalized_name.unique()); d = tr.groupby("benchmark")[col].mean()
        for _ in range(50):
            r = tr[col].values - d.reindex(tr.benchmark).values
            g = pd.DataFrame({"n": tr.normalized_name.values, "r": r}).groupby("n").r.agg(["sum", "size"])
            a = g["sum"] / (g["size"] + lam)
            d = pd.DataFrame({"b": tr.benchmark.values, "r": tr[col].values - a.reindex(tr.normalized_name).values}).groupby("b").r.mean()
        sh = te[te.normalized_name.isin(a.index) & (tr.groupby("normalized_name").benchmark.nunique().reindex(te.normalized_name).fillna(0).values >= 1)]
        if len(sh) < 4:
            continue
        x = a.reindex(sh.normalized_name).values; y = sh[col].values
        per[hb] = dict(n=len(sh), r=float(np.corrcoef(x, y)[0, 1]))
        xs.append(x - x.mean()); ys.append(y - y.mean())
    x, y = np.concatenate(xs), np.concatenate(ys)
    slope = float(np.sum(x * y) / np.sum(x * x))
    return dict(per_fold=per, pooled_r=float(np.corrcoef(x, y)[0, 1]), slope=slope, n=len(x),
                var_y=float(np.var(y)), resid_var=float(np.mean((y - slope * x) ** 2)))


def bootstrap_icc(T, col="z", B=500):
    names = T.normalized_name.unique()
    multi = T.groupby("normalized_name").benchmark.nunique()
    multi = multi[multi >= 2].index.values
    vals = []
    for _ in range(B):
        pick = RNG.choice(multi, len(multi))
        TT = pd.concat([T[T.normalized_name == n].assign(normalized_name=f"{n}#{j}") for j, n in enumerate(pick)])
        try:
            vals.append(transfer_analysis(TT, col)["icc"])
        except Exception:
            pass
    return np.percentile(vals, [5, 50, 95]).tolist()


# ----------------------------------------------------------------------------- 3. hierarchical pair-mean predictor
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(40)  # probabilists' Hermite: weight exp(-x^2/2)
GH_W = GH_W / GH_W.sum()
MU_GRID = np.linspace(-6, 4, 201)


def pair_posteriors(kv, nv, G, sB, tau, off=None, pool=True):
    """Logit-normal hierarchy: mu_b ~ N(G, sB^2); logit pi_p = mu_b + off_p + a_p, a_p ~ N(0,tau^2).
    kv,nv: successes/labels per pair (all pairs on the benchmark). Returns posterior mean pi per pair.
    pool=False: each pair treated as if alone on its benchmark (no cross-pair evidence)."""
    kv = np.asarray(kv, float); nv = np.asarray(nv, float)
    off = np.zeros(len(kv)) if off is None else np.asarray(off, float)
    # lik[p, g, q] for mu grid g, quadrature node q
    a = tau * GH_X
    eta = MU_GRID[None, :, None] + off[:, None, None] + a[None, None, :]
    lp = kv[:, None, None] * (-np.logaddexp(0, -eta)) + (nv - kv)[:, None, None] * (-np.logaddexp(0, eta))
    mx = lp.max(axis=2, keepdims=True)
    Lq = np.exp(lp - mx) * GH_W[None, None, :]            # per node
    Lp = Lq.sum(2)                                         # [p, g] (scaled by exp(mx))
    logLp = np.log(Lp) + mx[:, :, 0]
    Ep = (Lq * expit(eta)).sum(2) / Lp                     # E[pi | mu, data_p]
    lprior = -0.5 * ((MU_GRID - G) / sB) ** 2
    if pool:
        lpost = lprior + logLp.sum(0)
        wpost = np.exp(lpost - lpost.max()); wpost /= wpost.sum()
        return (Ep * wpost[None, :]).sum(1)
    lpost = lprior[None, :] + logLp
    wp = np.exp(lpost - lpost.max(1, keepdims=True)); wp /= wp.sum(1, keepdims=True)
    return (Ep * wp).sum(1)


def effective_kappa(G, sB, tau, known_mu=False, n_mc=20000):
    """moment-matched Beta concentration of pair accuracy pi (marginal over items via logit pair mean).
    Bayes-optimal posterior mean ~= (kappa*m + k)/(kappa + n); data weight n/(n+kappa)."""
    mu = np.full(n_mc, G) if known_mu else RNG.normal(G, sB, n_mc)
    pi = expit(mu + RNG.normal(0, tau, n_mc))
    mm, vv = pi.mean(), pi.var()
    return float(mm * (1 - mm) / vv - 1), float(mm)


# ----------------------------------------------------------------------------- episodic evaluation
def episodes(c, pri, name_prior=None, reps=20, with_item=True, seed=0):
    """Real-data episodes on one benchmark. For each rep: per pair (>=80 distinct items) split items 50/50,
    random nested acquisition order; acquired label = one random trial of the cell.
    Predictors: P0 per-pair hierarchical (no pooling), P1 pooled benchmark offset, P2 = P1 + name prior,
    P3 = full EB Rasch on all acquired labels (pooled item difficulty) with mu prior from P1,
    PX oracle known benchmark mean (per pair hierarchy with mu fixed to the true pooled logit-mean)."""
    rng = np.random.default_rng(seed)
    pt = pair_table(c); keep = pt[pt.n_items >= 80].subject_id.values
    c = c[c.subject_id.isin(keep)].reset_index(drop=True)
    if len(keep) == 0:
        return None
    si, ns, sl = codes(c.subject_id); ii, ni, _ = codes(c.item_id)
    off = np.zeros(ns)
    if name_prior is not None:
        nm = c.groupby("subject_id").normalized_name.first().reindex(sl).values
        off = np.array([name_prior.get(n, 0.0) for n in nm])
    true_lacc = logit(np.clip(pt.set_index("subject_id").loc[sl].acc.values, 1e-3, 1 - 1e-3))
    mu_oracle = float(np.mean(true_lacc))
    rows_by_pair = [np.where(si == s)[0] for s in range(ns)]
    out = {nm_: {b: [] for b in BUDGETS} for nm_ in ["P0", "P1", "P2", "P3", "PX"]}
    for rep in range(reps):
        acq_order, ev = [], []
        for s in range(ns):
            r = rng.permutation(rows_by_pair[s]); h = len(r) // 2
            acq_order.append(r[:h]); ev.append(r[h:])
        ev_all = np.concatenate(ev)
        ydraw = (rng.random(len(c)) < c.k.values / c.m.values).astype(float)  # one trial per cell -> nested
        for n in BUDGETS:
            kv = np.zeros(ns); nv = np.zeros(ns); lab_rows, lab_y = [], []
            for s in range(ns):
                rr = acq_order[s][:n]
                y = ydraw[rr]
                kv[s], nv[s] = y.sum(), len(rr); lab_rows.append(rr); lab_y.append(y)
            P = {}
            P["P0"] = pair_posteriors(kv, nv, pri["G"], pri["sB"], pri["tau"], pool=False)
            P["P1"] = pair_posteriors(kv, nv, pri["G"], pri["sB"], pri["tau"], pool=True)
            P["P2"] = pair_posteriors(kv, nv, pri["G"], pri["sB"], pri["tau"], off=off * pri["lam_name"], pool=True)
            P["PX"] = pair_posteriors(kv, nv, mu_oracle, 0.05, pri["tau"], pool=True)
            pcell = {kk: v[si[ev_all]] for kk, v in P.items()}
            if with_item and n > 0:
                rr = np.concatenate(lab_rows); yy = np.concatenate(lab_y)
                mu_hat = float(logit(np.clip(np.mean(P["P2"]), 1e-3, 1 - 1e-3)))
                groups = [dict(name="subj", idx=si[rr], n=ns, var=[pri["tau"] ** 2], fixed=True),
                          dict(name="item", idx=ii[rr], n=ni, var=[pri["sigma"] ** 2], fixed=True)]
                # subject prior mean = name offset: implemented by fitting deviation from offset
                # conditional (item-level) Rasch intercept prior: benchmark prior G is on the marginal
                # pair-mean scale; conditional median-item logit is ~ G*sqrt(1+pi*sigma^2/8) (probit approx)
                infl = np.sqrt(1 + np.pi * pri["sigma"] ** 2 / 8)
                offr = (off * pri["lam_name"])[si[rr]]
                fit = fit_glmm(yy, np.ones_like(yy), groups, eb=False, mu_init=pri["G"] * infl,
                               mu_sd=pri["sB"] * infl, offset=offr)
                e = fit["mu"] + (off * pri["lam_name"])[si[ev_all]] + fit["u"]["subj"][si[ev_all]] + fit["u"]["item"][ii[ev_all]]
                v = fit["mu_pvar"] + fit["pvar"]["subj"][si[ev_all]] + fit["pvar"]["item"][ii[ev_all]]
                pcell["P3"] = expit(e / np.sqrt(1 + np.pi * v / 8))   # posterior-predictive (marginal) prob
            else:
                pcell["P3"] = pcell["P2"]
            d = c.iloc[ev_all]
            for kk, p in pcell.items():
                b = cell_brier(p, d.k.values, d.m.values) * d.m.values
                t = pd.DataFrame({"s": si[ev_all], "b": b, "m": d.m.values}).groupby("s").sum()
                out[kk][n].append(float((t.b / t.m).mean()))
    res = {}
    for kk, v in out.items():
        bb = {n: float(np.mean(v[n])) for n in BUDGETS}
        res[kk] = dict(curve=bb, alc=float(sum(W[n] * bb[n] for n in BUDGETS)))
    return res


def simulate_hierarchy(G, sB, tau, sigma, S=40, N=300, cover=1.0, reps=30, seed=1):
    """Synthetic benchmarks from the fitted hierarchy; same predictors as `episodes`."""
    rng = np.random.default_rng(seed)
    acc = {kk: {n: [] for n in BUDGETS} for kk in ["P0", "P1", "PX", "P3"]}
    for rep in range(reps):
        mub = rng.normal(G, sB); th = rng.normal(0, tau, S); be = rng.normal(0, sigma, N)
        pmat = expit(mub + th[:, None] + be[None, :])
        lacc = logit(np.clip(pmat.mean(1), 1e-3, 1 - 1e-3))
        Y = (rng.random((S, N)) < pmat).astype(float)
        orders = [rng.permutation(N) for _ in range(S)]
        for n in BUDGETS:
            kv = np.array([Y[s, orders[s][:n]].sum() for s in range(S)]); nv = np.full(S, n)
            P = dict(P0=pair_posteriors(kv, nv, G, sB, tau, pool=False),
                     P1=pair_posteriors(kv, nv, G, sB, tau, pool=True),
                     PX=pair_posteriors(kv, nv, lacc.mean(), 0.05, tau, pool=True))
            for kk, v in P.items():
                bs = [np.mean((v[s] - pmat[s, orders[s][N // 2:]]) ** 2 + pmat[s, orders[s][N // 2:]] * (1 - pmat[s, orders[s][N // 2:]])) for s in range(S)]
                acc[kk][n].append(np.mean(bs))
            if n > 0:
                rows_s = np.concatenate([np.full(n, s) for s in range(S)]); rows_i = np.concatenate([orders[s][:n] for s in range(S)])
                yy = Y[rows_s, rows_i]
                groups = [dict(name="subj", idx=rows_s, n=S, var=[tau ** 2], fixed=True),
                          dict(name="item", idx=rows_i, n=N, var=[sigma ** 2], fixed=True)]
                infl = np.sqrt(1 + np.pi * sigma ** 2 / 8)
                fit = fit_glmm(yy, np.ones_like(yy), groups, eb=False, mu_init=G * infl, mu_sd=sB * infl)
                bs = []
                for s in range(S):
                    ev = orders[s][N // 2:]
                    v = fit["mu_pvar"] + fit["pvar"]["subj"][s] + fit["pvar"]["item"][ev]
                    p = expit((fit["mu"] + fit["u"]["subj"][s] + fit["u"]["item"][ev]) / np.sqrt(1 + np.pi * v / 8))
                    bs.append(np.mean((p - pmat[s, ev]) ** 2 + pmat[s, ev] * (1 - pmat[s, ev])))
                acc["P3"][n].append(np.mean(bs))
            else:
                acc["P3"][n].append(acc["P1"][n][-1])
    res = {}
    for kk, v in acc.items():
        bb = {n: float(np.mean(v[n])) for n in BUDGETS}
        res[kk] = dict(curve=bb, alc=float(sum(W[n] * bb[n] for n in BUDGETS)))
    return res


# ----------------------------------------------------------------------------- 4. LOBO budget 0
def hyper_from(pairs):
    """EB hyperparameters from training pairs (equal benchmark weight for G, sB)."""
    pp = pairs[pairs.n_items >= 80].copy()
    pp["lacc"] = logit(np.clip((pp.k + .5) / (pp.m + 1), 1e-3, 1 - 1e-3))
    bm = pp.groupby("benchmark").lacc.mean()
    # within-benchmark sd of logit pair acc, minus binomial noise (delta method)
    taus = []
    for b, g in pp.groupby("benchmark"):
        if len(g) < 3:
            continue
        noise = np.mean(1.0 / (g.m * g.acc.clip(.01, .99) * (1 - g.acc.clip(.01, .99))))
        taus.append(np.sqrt(max(g.lacc.var() - noise, 0.01)))
    return dict(G=float(bm.mean()), sB=float(bm.std(ddof=1)) if len(bm) > 1 else 1.5,
                tau=float(np.sqrt(np.mean(np.square(taus)))), bench_means=bm.to_dict())


def dev_prior(pairs, key, k_shrink):
    """name/provider deviation prior: mean within-benchmark logit-acc deviation, shrunk n/(n+k)."""
    pp = pairs[pairs.n_items >= 80].copy()
    pp["lacc"] = logit(np.clip((pp.k + .5) / (pp.m + 1), 1e-3, 1 - 1e-3))
    pp["dev"] = pp.lacc - pp.groupby("benchmark").lacc.transform("mean")
    g = pp.groupby([key, "benchmark"]).dev.mean().reset_index().groupby(key).dev.agg(["mean", "size"])
    return (g["mean"] * g["size"] / (g["size"] + k_shrink)).to_dict()


def lobo_budget0(cells_by_b, pairs_all, k_name, k_prov):
    res = {}
    for hb in BINARY:
        tr = pairs_all[(pairs_all.benchmark != hb) & pairs_all.benchmark.isin(BINARY)]
        hy = hyper_from(tr)
        c = cells_by_b[hb]
        pt = pair_table(c); keep = pt[pt.n_items >= 80].subject_id.values
        d = c[c.subject_id.isin(keep)].copy()
        tp = tr[tr.n_items >= 80]
        G = hy["G"]
        npri = dev_prior(tr, "normalized_name", k_name); ppri = dev_prior(tr, "provider", k_prov)
        # marginal (over subjects) probability for a benchmark-mean logit G with tau spread
        def pmarg(off):
            # E over unknown benchmark mean and subject deviation: integrate both (cold start)
            sdt = np.sqrt(hy["sB"] ** 2 + hy["tau"] ** 2)
            return float(np.sum(GH_W * expit(G + off + sdt * GH_X)))
        d["const05"] = 0.5
        d["global_pairmean"] = float((tp.k / tp.m).mean())
        d["global_benchmean"] = float(tp.assign(a=tp.k / tp.m).groupby("benchmark").a.mean().mean())
        d["hier_marginal"] = pmarg(0.0)
        d["name_prior"] = [pmarg(npri.get(n, 0.0)) for n in d.normalized_name]
        d["provider_prior"] = [pmarg(ppri.get(p, 0.0)) for p in d.provider]
        d["name_then_provider"] = [pmarg(npri[n]) if n in npri else pmarg(ppri.get(p, 0.0))
                                   for n, p in zip(d.normalized_name, d.provider)]
        true_mean = float(pt[pt.n_items >= 80].acc.mean())
        d["oracle_bench_mean"] = true_mean
        cols = ["const05", "global_pairmean", "global_benchmean", "hier_marginal", "name_prior",
                "provider_prior", "name_then_provider", "oracle_bench_mean"]
        res[hb] = {col: pair_mean_brier(d, col) for col in cols}
        res[hb]["n_pairs"] = int(len(keep))
        res[hb]["names_seen_frac"] = float(np.mean([n in npri for n in d.groupby("subject_id").normalized_name.first()]))
        res[hb]["prov_seen_frac"] = float(np.mean([p in ppri for p in d.groupby("subject_id").provider.first()]))
        res[hb]["hyper"] = {k: v for k, v in hy.items() if k != "bench_means"}
        res[hb]["values"] = {col: float(d[col].mean()) for col in cols}
        res[hb]["true_pair_acc_mean"] = true_mean
    return res


# ----------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    os.makedirs(ART, exist_ok=True)
    cells = {b: load_cells(b) for b in BINARY}
    pairs_all = pd.concat([pair_table(c) for c in cells.values()], ignore_index=True)
    R = {}

    # ---- 1. per-benchmark Rasch + variance components
    fits, vc = {}, {}
    for b, c in cells.items():
        fit, si, ii = per_benchmark_rasch(c)
        fits[b] = (fit, si, ii)
        vc[b] = dict(mu=fit["mu"], var_subject=fit["var"]["subj"][0], var_item=fit["var"]["item"][0],
                     n_subj=int(c.subject_id.nunique()), n_items=int(c.item_id.nunique()), n_cells=len(c))
        if c.m.max() >= 2 and c.subject_id.nunique() > 1 or b == "swe_rebench":
            fc, _, _ = per_benchmark_rasch(c, with_cell=True)
            vc[b]["with_cell"] = dict(var_subject=fc["var"]["subj"][0], var_item=fc["var"]["item"][0],
                                      var_cell=fc["var"]["cell"][0])
        print(b, vc[b], f"{time.time()-t0:.0f}s", flush=True)
    R["variance_components"] = vc

    # trial noise share (cells with repeats)
    tn = {}
    for b in ["matharena", "swe_rebench"]:
        c = cells[b]; rep = c[c.m >= 2]; yb = rep.k / rep.m
        within = float(np.average(yb * (1 - yb) * rep.m / (rep.m - 1), weights=rep.m))
        tot_p = rep.k.sum() / rep.m.sum()
        tn[b] = dict(n_cells=len(rep), mean_trials=float(rep.m.mean()), within_cell_var=within,
                     total_var=float(tot_p * (1 - tot_p)), share_within=within / float(tot_p * (1 - tot_p)),
                     frac_cells_unanimous=float(np.mean((yb == 0) | (yb == 1))))
    R["trial_noise"] = tn; print("trial noise", tn, flush=True)

    # CV Brier by component
    cv = {}
    for b, c in cells.items():
        if c.subject_id.nunique() < 2:
            continue
        cv[b] = cv_components(c); print("cv", b, cv[b], f"{time.time()-t0:.0f}s", flush=True)
    R["cv_brier"] = cv

    # joint model across binary benchmarks: bench fixed + name + pair + item(class per benchmark)
    J = pd.concat(cells.values(), ignore_index=True)
    bi, nb, bl = codes(J.benchmark); ni_, nn, _ = codes(J.normalized_name)
    pi_, npair, _ = codes(J.benchmark + "|" + J.subject_id); pv_, npv, _ = codes(J.provider)
    ii, nitem, il = codes(J.benchmark + "|" + J.item_id)
    item_cls = codes(pd.Series(il).str.split("|").str[0])[0]
    joint = {}
    for label, use_prov in [("name", False), ("provider+name", True)]:
        groups = [dict(name="bench", idx=bi, n=nb, var=[25.0], fixed=True),
                  dict(name="name", idx=ni_, n=nn, var=[0.5]),
                  dict(name="pair", idx=pi_, n=npair, var=[0.5]),
                  dict(name="item", idx=ii, n=nitem, cls=item_cls, var=np.ones(nb))]
        if use_prov:
            groups.insert(1, dict(name="prov", idx=pv_, n=npv, var=[0.3]))
        fj = fit_glmm(J.k.values, J.m.values, groups, iters=10)
        joint[label] = dict(var={k: v for k, v in fj["var"].items() if k != "bench"},
                            bench_offsets=dict(zip(bl, (fj["mu"] + fj["u"]["bench"]).tolist())),
                            item_var_by_bench=dict(zip(bl, fj["var"]["item"])))
        print("joint", label, joint[label], f"{time.time()-t0:.0f}s", flush=True)
    R["joint"] = joint

    # ---- 2. transfer (binary benchmarks + mmdocrag continuous score as sensitivity)
    T = name_dev_table(cells, fits)
    md = pd.read_parquet(os.path.join(DATA, "mmdocrag", "response.parquet"), columns=["subject_id", "item_id", "response"])
    ms = pd.read_parquet(os.path.join(DATA, "mmdocrag", "subjects.parquet"))
    mm = md.groupby("subject_id").agg(score=("response", "mean"), n_items=("item_id", "nunique")).reset_index().merge(ms, on="subject_id")
    mm = mm[mm.n_items >= 80]
    mg = mm.groupby("normalized_name").agg(theta=("score", lambda s: float(logit(np.clip(s.mean(), 1e-3, 1 - 1e-3)))),
                                           provider=("provider", "first")).reset_index()
    mg["benchmark"] = "mmdocrag"; mg["lacc"] = mg.theta; mg["lacc_dev"] = mg.lacc - mg.lacc.mean()
    mg["z"] = (mg.theta - mg.theta.mean()) / mg.theta.std()
    T6 = pd.concat([T, mg], ignore_index=True)
    R["transfer"] = {"binary_z": transfer_analysis(T, "z"), "binary_lacc_dev": transfer_analysis(T, "lacc_dev"),
                     "with_mmdocrag_z": transfer_analysis(T6, "z"),
                     "binary_z_icc_boot90": bootstrap_icc(T, "z", 300)}
    R["transfer"]["additive_lacc"] = transfer_additive(T, "lacc")
    R["transfer"]["additive_lacc_with_mmdocrag"] = transfer_additive(T6, "lacc")
    print("additive", R["transfer"]["additive_lacc"], R["transfer"]["additive_lacc_with_mmdocrag"], flush=True)
    # provider ICC on name-level z (names nested in providers, within benchmark)
    R["transfer"]["provider_share"] = float(T.groupby(["benchmark", "provider"]).z.transform("mean").var() / T.z.var())
    print("transfer", json.dumps(R["transfer"], indent=1)[:3000], flush=True)

    # ---- 3. priors + shrinkage
    hy = hyper_from(pairs_all[pairs_all.benchmark.isin(BINARY)])
    hy_elig = hyper_from(pairs_all[pairs_all.benchmark.isin(ELIG)])
    sig_items = {b: float(np.sqrt(vc[b]["var_item"])) for b in BINARY}
    sigma = float(np.sqrt(np.mean([vc[b]["var_item"] for b in BINARY if b != "swe_rebench"])))
    r_icc = max(R["transfer"]["binary_lacc_dev"]["icc"], 0.0)
    k_name = (1 - r_icc) / max(r_icc, 1e-3)  # n/(n+k) == n r /(1+(n-1) r)
    shrink = {}
    for n in [1, 3, 7, 15, 31]:
        kc, mc = effective_kappa(hy["G"], hy["sB"], hy["tau"], known_mu=False)
        kw, mw = effective_kappa(hy["G"], hy["sB"], hy["tau"], known_mu=True)
        shrink[n] = dict(kappa_cold=kc, w_data_cold=n / (n + kc), kappa_known_bench=kw, w_data_known=n / (n + kw))
    R["shrinkage_analytic"] = shrink
    print("hyper", hy, "elig", hy_elig, "sigma", sigma, sig_items, "shrink", shrink, flush=True)

    sim = {}
    for S in [5, 20, 80]:
        sim[S] = simulate_hierarchy(hy["G"], hy["sB"], hy["tau"], sigma, S=S, N=300, reps=25)
        print("sim", S, {k: round(v["alc"], 4) for k, v in sim[S].items()}, f"{time.time()-t0:.0f}s", flush=True)
    R["simulation"] = sim

    # ---- 4. LOBO budget 0
    R["lobo_b0"] = lobo_budget0(cells, pairs_all, k_name, k_name * 2)
    print("lobo", json.dumps(R["lobo_b0"], indent=1), flush=True)

    # ---- 5. real-data episodic (Level 2): LOBO hyperparameters, held-out benchmark episodes
    ep = {}
    for hb in ELIG + ["matharena"]:
        tr = pairs_all[(pairs_all.benchmark != hb) & pairs_all.benchmark.isin(BINARY)]
        h = hyper_from(tr)
        trb = [b for b in BINARY if b != hb and b != "swe_rebench"]
        h["sigma"] = float(np.sqrt(np.mean([vc[b]["var_item"] for b in trb])))
        h["lam_name"] = 1.0
        npri = dev_prior(tr, "normalized_name", k_name)
        e = episodes(cells[hb], h, name_prior=npri, reps=12 if hb != "multi_swebench" else 6, seed=7)
        ep[hb] = dict(hyper={k: v for k, v in h.items() if k != "bench_means"}, res=e)
        print("episodes", hb, {k: round(v["alc"], 4) for k, v in e.items()},
              {k: [round(v["curve"][n], 4) for n in BUDGETS] for k, v in e.items()}, f"{time.time()-t0:.0f}s", flush=True)
    R["episodes_lobo"] = ep

    priors = dict(
        version="priors_v1", created="2026-10-01", source="src/paiec/analysis/decomp.py",
        scale="logit P(correct); pair-level quantities use logit of pair accuracy (marginal over items)",
        benchmark_mean_logit=dict(mean=hy["G"], sd=hy["sB"], n_benchmarks=len(hy["bench_means"]),
                                  points=hy["bench_means"], eligible_only=dict(mean=hy_elig["G"], sd=hy_elig["sB"]),
                                  note="only 5 (binary) points incl. swe_rebench with 1 subject; treat sd as lower bound; consider Student-t(3)"),
        subject_dev_within_benchmark_sd=dict(tau_pair_logit=hy["tau"],
                                             rasch_theta_sd={b: float(np.sqrt(vc[b]["var_subject"])) for b in BINARY}),
        item_difficulty_sd=dict(pooled=sigma, by_benchmark=sig_items),
        joint_model_variances=joint,
        name_transfer=dict(icc_lacc_dev=R["transfer"]["binary_lacc_dev"]["icc"],
                           icc_z=R["transfer"]["binary_z"]["icc"],
                           k_shrink_name=k_name, rule="dev_name = mean_b(dev_name,b) * n_b/(n_b+k_shrink_name)"),
        name_deviation_prior=dev_prior(pairs_all[pairs_all.benchmark.isin(BINARY)], "normalized_name", k_name),
        provider_deviation_prior=dev_prior(pairs_all[pairs_all.benchmark.isin(BINARY)], "provider", 2 * k_name),
        shrinkage=shrink,
    )
    with open(os.path.join(ART, "priors_v1.json"), "w") as fh:
        json.dump(priors, fh, indent=1, default=float)
    with open(os.path.join(ART, "decomp_results.json"), "w") as fh:
        json.dump(R, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("done", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
