"""E7 — priors from the enlarged training pool (revision 54074453, 281 benchmarks).

Pre-registration: docs/preregistrations/E7_expanded_pool_preregistration.md (not edited by this module).
Run:  python -m paiec.analysis.e7_expanded_pool --step pool        -> artifacts/e7/{pool_pairs,pool_bench}.parquet
      python -m paiec.analysis.e7_expanded_pool --step primary --seeds 10
      python -m paiec.analysis.e7_expanded_pool --step secondary
Outputs: experiments/runs/e7/{e7_pool.log,e7_primary_s<N>.json,e7_primary.log,e7_secondary.json,...}

Data rules: binary rows only (response in {0,1}); item-level; release == public; a pool pair needs
>= MIN_ITEMS distinct items (the competition's eligibility rule, as in episodes.compact); a pool
benchmark needs >= MIN_P such pairs of the fitted population (2025-26 releases, or all years).
Cross-benchmark key = lower-cased ``normalized_name`` only (never subject_id).

Ability prior (arm ``subject``), fitted per held-out fold on the pool minus the fold and relatives:
  r_bs      = y_bs - mean_b(y)                    y = logit((S+.5)/(n+1)), mean over fitted population
  r_kb      = mean of r_bs over pairs of name k in benchmark b
  s_eps^2   = pooled within-name between-benchmark variance of r_kb (names in >= 2 benchmarks)
  s_th^2    = var of name means - s_eps^2 * mean(1/J)       (one-way random effects, moments)
  theta_k   = J s_th^2 / (J s_th^2 + s_eps^2) * mean_b r_kb                     (shrunk mean)
  rel_k     = J s_th^4 / ((J s_th^2 + s_eps^2)(s_th^2 + s_eps^2))   share of a new residual explained
  unseen names: ridge on release (clipped) + provider dummies fitted on the pool's pair residuals,
                rel_fb = leave-benchmark-out R^2 (clipped to [0, .7])
  injection (E1 metadata_prior mechanism): pair effect ~ N(m, tau_H^2 (1 - rel)); m = theta_k or ridge.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time

import numpy as np
import pandas as pd

from paiec import paths
from paiec.evaluation import episodes as E
from paiec.models import ladder as Lad
from paiec.models import v1_1 as Base
from paiec.models import v1_2 as V

NEW = paths.DATA / "_main_54074453"
ART = paths.ARTIFACTS / "e7"
OUT = paths.RUNS / "e7"
REGIME = "R2"
CFG = V.cfg_of(facets="named", guard=0)
SANITY_REF, SANITY_TOL = 0.208722, 5e-4
MIN_ITEMS, MIN_P = 80, 5
MU, GH_X, LOG_GH_W = Base.MU, Base.GH_X, Base.LOG_GH_W
sig, log_sig = Base.sig, Base.log_sig
LOGNAME = "e7.log"

# Relatives of each held-out fold, fixed before any response table was read:
# the fold itself, any benchmark with the same source_url, and the name-stem family.
STEMS = {
    "multi_swebench": r"^swe|swebench",
    "matharena": r"^matharena",
    "real_webagents": r"webagent|webarena",
    "researchcodebench": r"^researchc",
}


def log(*a):
    print(*a, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / LOGNAME, "a", encoding="utf-8") as f:
        print(*a, file=f)


def nkey(v):
    s = E._s(v)
    return s.lower()


# ---------------------------------------------------------------------------
# Stage 1: pool tables
# ---------------------------------------------------------------------------
def _agg(path):
    """Per subject: n binary rows, n correct, n distinct items; plus Rasch inputs (codes)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    sid_ix, iid_ix = {}, {}
    cs, ci, cy = [], [], []
    for g in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(g, columns=["subject_id", "item_id", "response"])
        r = t["response"]
        t = t.filter(pc.fill_null(pc.or_(pc.equal(r, 0.0), pc.equal(r, 1.0)), False))
        if not t.num_rows:
            continue
        for col, ix, dst in (("subject_id", sid_ix, cs), ("item_id", iid_ix, ci)):
            d = pc.dictionary_encode(t[col].combine_chunks())
            if isinstance(d, pa.ChunkedArray):
                d = d.combine_chunks()
            vals = d.dictionary.to_pylist()
            m = np.array([ix.setdefault(v, len(ix)) for v in vals], np.int64)
            dst.append(m[d.indices.to_numpy(zero_copy_only=False)].astype(np.int32))
        cy.append(t["response"].to_numpy().astype(np.int8))
    if not cs:
        return None
    return dict(sid=np.concatenate(cs), iid=np.concatenate(ci), y=np.concatenate(cy),
                sids=np.array(list(sid_ix), dtype=object), n_items=len(iid_ix))


def _rasch_sb(A, keep, rng, max_items=20000):
    """sigma_beta of ladder.rasch on the kept subjects (item_id as key; items subsampled if huge)."""
    m = keep[A["sid"]]
    sid, iid, y = A["sid"][m], A["iid"][m], A["y"][m]
    ui = np.unique(iid)
    if len(ui) > max_items:
        ui = rng.choice(ui, max_items, replace=False)
        mm = np.isin(iid, ui)
        sid, iid, y = sid[mm], iid[mm], y[mm]
    us, sp = np.unique(sid, return_inverse=True)
    ui, ik = np.unique(iid, return_inverse=True)
    D = dict(row_pair=sp, row_item=np.arange(len(ik)), item_key=ik, row_y=y,
             pair_sid=us, key_jev=np.zeros(len(ui)))
    return float(Lad.rasch(D)[2])


def pool():
    global LOGNAME
    LOGNAME = "e7_pool.log"
    ART.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / LOGNAME).write_text("", encoding="utf-8")
    rng = np.random.default_rng(7)
    brow, prow = [], []
    t0 = time.time()
    for d in sorted(p for p in NEW.iterdir() if p.is_dir()):
        ft = d / "formatted_tables"
        if not (ft / "benchmarks.parquet").exists():
            continue
        b = pd.read_parquet(ft / "benchmarks.parquet").iloc[0]
        dom = "|".join(map(str, b.domain)) if b.domain is not None else ""
        rec = dict(bench=d.name, response_type=E._s(b.response_type), granularity=E._s(b.granularity),
                   release=E._s(b.release), domain=dom, source_url=E._s(b.source_url),
                   has_responses=(ft / "responses.parquet").exists(), used=False, n_pairs80=0,
                   sb_2526=np.nan, sb_n=0)
        ok = (rec["has_responses"] and rec["release"] == "public" and rec["granularity"] == "item"
              and rec["response_type"] in ("binary", "mixed"))
        if ok:
            A = _agg(ft / "responses.parquet")
            if A is not None:
                S = len(A["sids"])
                n = np.bincount(A["sid"], minlength=S)
                s = np.bincount(A["sid"], A["y"].astype(float), S)
                pi = np.unique(A["sid"].astype(np.int64) * (A["n_items"] + 1) + A["iid"])
                ni = np.bincount(pi // (A["n_items"] + 1), minlength=S)
                sub = pd.read_parquet(ft / "subjects.parquet").set_index("subject_id")
                yrs = np.array([E.release_years(E._s(sub.release_date.get(x))) if x in sub.index else np.nan
                                for x in A["sids"]])
                for k, x in enumerate(A["sids"]):
                    r = sub.loc[x] if x in sub.index else {}
                    prow.append(dict(bench=d.name, n=int(n[k]), s=float(s[k]), n_items=int(ni[k]),
                                     name=nkey(r.get("normalized_name")), provider=E._s(r.get("provider")),
                                     release=float(yrs[k]), pair_ix=k))
                rec["used"] = True
                rec["n_pairs80"] = int((ni >= MIN_ITEMS).sum())
                keep = (ni >= MIN_ITEMS) & (yrs >= 2025) & (yrs < 2027)
                rec["sb_n"] = int(keep.sum())
                if keep.sum() >= MIN_P:
                    rec["sb_2526"] = _rasch_sb(A, keep, rng)
                del A
        brow.append(rec)
        log(f"{time.time() - t0:6.0f}s {d.name:32s} used={rec['used']} pairs80={rec['n_pairs80']} "
            f"elig2526={rec['sb_n']} sb={rec['sb_2526']:.3f}")
    Bd, Pd = pd.DataFrame(brow), pd.DataFrame(prow)
    Bd.to_parquet(ART / "pool_bench.parquet")
    Pd.to_parquet(ART / "pool_pairs.parquet")
    log(f"benchmarks listed={len(Bd)} used={int(Bd.used.sum())} pairs={len(Pd)} "
        f"pairs>=80 items={int((Pd.n_items >= MIN_ITEMS).sum())}")


# ---------------------------------------------------------------------------
# Stage 2: fits on a pool subset
# ---------------------------------------------------------------------------
_POOL = {}


def load_pool():
    if not _POOL:
        Bd = pd.read_parquet(ART / "pool_bench.parquet")
        Pd = pd.read_parquet(ART / "pool_pairs.parquet")
        Pd = Pd[Pd.n_items >= MIN_ITEMS].reset_index(drop=True)
        q = (Pd.s + 0.5) / (Pd.n + 1)
        Pd["y"] = np.log(q / (1 - q))
        Pd["v"] = 1 / ((Pd.n + 1) * q * (1 - q))
        Pd["is2526"] = (Pd.release >= 2025) & (Pd.release < 2027)
        _POOL.update(B=Bd[Bd.used].reset_index(drop=True), P=Pd)
    return _POOL["B"], _POOL["P"]


def relatives(fold, Bd, keep_family=False):
    """Benchmarks removed from the pool for a held-out fold (sorted list)."""
    ex = {fold}
    if not keep_family:
        url = Bd.loc[Bd.bench == fold, "source_url"]
        url = url.iloc[0] if len(url) else None
        for b, u in zip(Bd.bench, Bd.source_url):
            if (url and u == url) or re.search(STEMS.get(fold, r"^\b$"), b):
                ex.add(b)
    return sorted(ex)


def subset(exclude, all_years=False, perm=None):
    """Pool pairs of the fitted population, with benchmark residuals; benchmarks with >= MIN_P pairs."""
    _, Pd = load_pool()
    T = Pd[~Pd.bench.isin(set(exclude))]
    if not all_years:
        T = T[T.is2526]
    cnt = T.groupby("bench").y.transform("size")
    T = T[cnt >= MIN_P].copy()
    if perm is not None:                         # control: descriptors permuted across pool pairs
        ix = np.random.default_rng(perm).permutation(len(T))
        for c in ("name", "provider", "release"):
            T[c] = T[c].to_numpy()[ix]
    T["mean_b"] = T.groupby("bench").y.transform("mean")
    T["r"] = T.y - T.mean_b
    T["P_b"] = T.groupby("bench").y.transform("size")
    return T


def fit_names(T, twoway=False, iters=20):
    """One-way random-effects fit of name ability on benchmark residuals."""
    T = T.copy()
    th = {}
    for it in range(iters if twoway else 1):
        if twoway and it:
            adj = T.name.map(th).fillna(0.0)
            T["mean_b"] = (T.y - adj).groupby(T.bench).transform("mean")
            T["r"] = T.y - T.mean_b
        N = T[T.name != ""]
        kb = N.groupby(["name", "bench"]).agg(r=("r", "mean"), v=("v", "mean")).reset_index()
        g = kb.groupby("name").r
        J, mk = g.size(), g.mean()
        multi = J[J >= 2].index
        ss = ((kb.r - kb.name.map(mk)) ** 2)[kb.name.isin(multi)].sum()
        df = float((J[multi] - 1).sum())
        s_eps2 = ss / df if df > 0 else float("nan")
        vbar = float(kb.v.mean())
        s_th2 = max(float(mk.var(ddof=1) - s_eps2 * (1 / J).mean()), 0.0) if len(mk) > 2 else 0.0
        w = J * s_th2 / (J * s_th2 + s_eps2)
        th = (w * mk).to_dict()
    tot = s_th2 + s_eps2
    rel = (w * s_th2 / tot) if tot > 0 else w * 0
    return dict(theta=th, rel=rel.to_dict(), J=J.to_dict(), s_th2=float(s_th2), s_eps2=float(s_eps2),
                s_eps2_net=float(max(s_eps2 - vbar, 0.0)), n_names=int(len(J)), n_names2=int(len(multi)),
                n_names3=int((J >= 3).sum()), n_bench=int(T.bench.nunique()), n_pairs=int(len(T)),
                n_named_pairs=int(len(N)), resid_var=float((T.r ** 2).mean()))


def _fb_X(release, provider, M):
    rows = []
    for r, pv in zip(release, provider):
        miss = not np.isfinite(r)
        rc = 0.0 if miss else float(np.clip(r, M["rel_lo"], M["rel_hi"]) - 2025.0)
        rows.append([rc, 1.0 if miss else 0.0] + [1.0 if pv == p else 0.0 for p in M["provs"]])
    return np.array(rows, float).reshape(len(rows), 2 + len(M["provs"]))


def fit_fallback(T, alpha=3.0, folds=10):
    """Ridge of pair residual on release date + provider (benchmark-balanced); grouped-CV R^2."""
    rel = T.release.to_numpy()
    pv = T.provider.to_numpy()
    u, c = np.unique(pv[pv != ""], return_counts=True)
    fin = rel[np.isfinite(rel)]
    M = dict(rel_lo=float(np.percentile(fin, 1)) if len(fin) else 2025.0,
             rel_hi=float(np.percentile(fin, 99)) if len(fin) else 2026.0,
             provs=[p for p, k in zip(u, c) if k >= 10])
    X = _fb_X(rel, pv, M)
    Y = T.r.to_numpy()
    W = (1.0 / T.P_b).to_numpy()
    M["coef"] = Lad._ridge(X, Y, W, alpha).tolist()
    ub, bi = np.unique(T.bench.to_numpy(), return_inverse=True)
    gf = np.random.default_rng(0).permutation(len(ub)) % folds
    pred = np.zeros(len(Y))
    for f in range(folds):
        te = gf[bi] == f
        if te.any() and (~te).any():
            cb = Lad._ridge(X[~te], Y[~te], W[~te], alpha)
            pred[te] = np.hstack([X[te], np.ones((te.sum(), 1))]) @ cb
    sse, sst = float(np.sum(W * (Y - pred) ** 2)), float(np.sum(W * Y ** 2))
    M["cv_r2"] = 1 - sse / sst if sst > 0 else 0.0
    M["rel"] = float(np.clip(M["cv_r2"], 0.0, 0.7))
    return M


def fit_hyper_pool(exclude):
    """Offset centre/scale inputs, tau, sigma_beta from the pool (2025-26 eligible pairs)."""
    Bd, _ = load_pool()
    T = subset(exclude)
    g = T.groupby("bench")
    means = g.y.mean()
    tau2 = g.y.var(ddof=1) - g.v.mean()
    sb = Bd.set_index("bench").sb_2526.reindex(means.index)
    sb = sb[np.isfinite(sb)]
    return dict(G=float(means.mean()), _means=means.to_numpy(), tau=math.sqrt(max(float(tau2.mean()), 0.05)),
                sigma_beta=math.sqrt(float((sb ** 2).mean())), n_bench=int(len(means)), n_sb=int(len(sb)),
                tau_median=math.sqrt(max(float(tau2.median()), 0.0)), mean_sd=float(means.std(ddof=1)),
                sb_median=float(sb.median()))


def prior_for(names, release, provider, F, M, tau):
    """Per-subject (m, t, kind) of the pair-effect prior. kind 2 = name match, 1 = fallback."""
    n = len(names)
    m, t, kind = np.zeros(n), np.full(n, float(tau)), np.zeros(n, int)
    fb = np.hstack([_fb_X(release, provider, M), np.ones((n, 1))]) @ np.array(M["coef"])
    for j, nm in enumerate(names):
        k = nm[2:] if isinstance(nm, str) and nm.startswith("n:") else ""
        if k and k in F["theta"]:
            m[j], t[j], kind[j] = F["theta"][k], tau * math.sqrt(1 - F["rel"][k]), 2
        else:
            m[j], t[j], kind[j] = fb[j], tau * math.sqrt(1 - M["rel"]), 1
    return m, t, kind


# ---------------------------------------------------------------------------
# Model with a pair-effect prior mean / sd (v1_1.pooled_pair generalised; m=0, t=tau is identical)
# ---------------------------------------------------------------------------
def pooled_pair_m(ep, H, li, ly, cfg, m, t):
    n = ep.n
    w = cfg["w_od"]
    S = (np.bincount(li, ly, n) if len(li) else np.zeros(n)) * w
    N = (np.bincount(li, minlength=n).astype(float) if len(li) else np.zeros(n)) * w
    eta = MU[None, :, None] + m[:, None, None] + t[:, None, None] * GH_X[None, None, :]
    ll = S[:, None, None] * log_sig(eta) + (N - S)[:, None, None] * log_sig(-eta) + LOG_GH_W
    mx = ll.max(2, keepdims=True)
    wt = np.exp(ll - mx)
    Ln = wt.sum(2)
    logL = np.log(Ln) + mx[:, :, 0]
    Eq = (wt * sig(eta)).sum(2) / Ln
    groups, ginv = np.unique(ep.inst_group, return_inverse=True)
    lp = np.zeros((len(groups), len(MU)))
    np.add.at(lp, ginv, logL)
    lp += Base.log_prior(H, cfg["prior"])[None, :]
    lp -= lp.max(1, keepdims=True)
    post = np.exp(lp); post /= post.sum(1, keepdims=True)
    return (post[ginv] * Eq).sum(1)


def predict_m(ep, H, b, cfg, fe, m, t):
    """v1_2.predict with the pair level replaced (kappa == 1 in the shipped config)."""
    assert cfg["kappa"] == 1.0
    li, lk, ly = ep.labels(b)
    q = pooled_pair_m(ep, H, li, ly, cfg, m, t)
    p = V.item_level(ep, H, q, li, lk, ly, cfg["rho"], cfg.get("rho_pool", False), cfg.get("guard", 0))
    if cfg.get("facets") and fe is not None:
        p = V.facet_adjust(ep, H, q, p, li, lk, ly, fe, None)
    return np.clip(p, 0.005, 0.995)


ARMS = ["base", "hyper", "subject", "subject+hyper", "subject_allyears", "subject_relkept",
        "control_perm", "SUPP_evalperm", "SUPP_subject_2way", "SUPP_hyper_tau_only", "SUPP_hyper_offset_only",
        "SUPP_hyper_sb_only", "SUPP_name_only"]


def fold_fits(B):
    Bd, _ = load_pool()
    ex = relatives(B, Bd)
    exk = relatives(B, Bd, keep_family=True)
    present = set(Bd.bench)
    fits = {}
    for nm, e, kw in (("subject", ex, {}), ("subject_allyears", ex, dict(all_years=True)),
                      ("subject_relkept", exk, {}), ("control_perm", ex, dict(perm=20261008))):
        T = subset(e, **kw)
        fits[nm] = (fit_names(T), fit_fallback(T))
    T = subset(ex)
    fits["SUPP_subject_2way"] = (fit_names(T, twoway=True), fits["subject"][1])
    return dict(exclude=[b for b in ex if b in present], exclude_not_in_pool=[b for b in ex if b not in present],
                exclude_relkept=exk, fits=fits, hyper=fit_hyper_pool(ex))


def run_fold(B, seeds):
    D = E.compact(B)
    H = Base.H_fold(B)
    ff = fold_fits(B)
    hp = ff["hyper"]
    Hh = dict(H, G=hp["G"], _means=hp["_means"], tau=hp["tau"], sigma_beta=hp["sigma_beta"])
    Hvar = {"SUPP_hyper_tau_only": dict(H, tau=hp["tau"]),
            "SUPP_hyper_offset_only": dict(H, G=hp["G"], _means=hp["_means"]),
            "SUPP_hyper_sb_only": dict(H, sigma_beta=hp["sigma_beta"])}
    ep = E.build(D, REGIME, seeds)
    fe = V.fe_for(B, D, CFG)
    p = ep.inst_pair
    names, rel, prov = D["pair_name"][p], D["pair_release"][p], D["pair_provider"][p]
    pri = {}
    for nm, (F, M) in ff["fits"].items():
        pri[nm] = prior_for(names, rel, prov, F, M, H["tau"])
    Fs, Ms = ff["fits"]["subject"]
    pri["subject+hyper"] = prior_for(names, rel, prov, Fs, Ms, hp["tau"])
    m, t, kind = pri["subject"]
    # supplementary: eval-side mismatch (priors reassigned to other subjects of the fold, same seed block)
    P = len(D["pair_sid"])
    pm = np.random.default_rng(20261008).permutation(P)
    first = {int(pp): j for j, pp in reversed(list(enumerate(p)))}
    src = np.array([first[int(pm[int(pp)])] for pp in p])
    pri["SUPP_evalperm"] = (m[src], t[src], kind[src])
    pri["SUPP_name_only"] = (np.where(kind == 2, m, 0.0), np.where(kind == 2, t, H["tau"]), kind)
    zero, tau0 = np.zeros(ep.n), np.full(ep.n, H["tau"])
    probs = {nm: {} for nm in ARMS}
    ident = 0.0
    for b in E.BUDGETS:
        pb = V.predict(ep, H, b, CFG, fe, None)
        probs["base"][b] = pb
        ident = max(ident, float(np.abs(predict_m(ep, H, b, CFG, fe, zero, tau0) - pb).max()))
        probs["hyper"][b] = V.predict(ep, Hh, b, CFG, fe, None)
        for nm, Hx in Hvar.items():
            probs[nm][b] = V.predict(ep, Hx, b, CFG, fe, None)
        for nm in pri:
            Hx = Hh if nm == "subject+hyper" else H
            probs[nm][b] = predict_m(ep, Hx, b, CFG, fe, pri[nm][0], pri[nm][1])
    res = {}
    for nm in ARMS:
        br, ec = ep.score(probs[nm])
        res[nm] = dict(br=br, ec=ec, alc=br @ E.W, pair=ep.inst_pair.copy(), seed=ep.inst_seed.copy())
    # coverage per evaluation subject (pairs, not instances)
    u, fi = np.unique(p, return_index=True)
    k1 = kind[fi]
    nmk = [x[2:] if isinstance(x, str) and x.startswith("n:") else "" for x in names[fi]]
    Jm = np.array([Fs["J"].get(k, 0) for k in nmk])[k1 == 2]
    relm = np.array([Fs["rel"].get(k, 0) for k in nmk])[k1 == 2]
    st = Lad.bench_stats(B)
    theta_true = (st["y"] - st["mean"])[u]
    cov = dict(n_subjects=int(len(u)), has_normalized_name=float(np.mean([bool(x) for x in nmk])),
               name_match=float((k1 == 2).mean()), n_match=int((k1 == 2).sum()),
               J_mean=float(Jm.mean()) if len(Jm) else 0.0, J_median=float(np.median(Jm)) if len(Jm) else 0.0,
               J_min=int(Jm.min()) if len(Jm) else 0, J_ge3=float((Jm >= 3).mean()) if len(Jm) else 0.0,
               rel_mean=float(relm.mean()) if len(relm) else 0.0,
               corr_m_true_matched=float(np.corrcoef(m[fi][k1 == 2], theta_true[k1 == 2])[0, 1]) if (k1 == 2).sum() > 2 else None,
               corr_m_true_all=float(np.corrcoef(m[fi], theta_true)[0, 1]),
               slope_true_on_m=float(np.polyfit(m[fi], theta_true, 1)[0]), sd_m=float(m[fi].std()),
               sd_true=float(theta_true.std()),
               match_relkept=float((pri["subject_relkept"][2][fi] == 2).mean()),
               match_allyears=float((pri["subject_allyears"][2][fi] == 2).mean()))
    info = dict(exclude=ff["exclude"], exclude_not_in_pool=ff["exclude_not_in_pool"],
                exclude_relkept=ff["exclude_relkept"], coverage=cov, zero_prior_maxdiff=ident,
                H_base=dict(G=H["G"], tau=H["tau"], sigma_beta=H["sigma_beta"], train=H["train"],
                            offset_scale=float(max(0.5, math.sqrt(np.mean((H["_means"] - H["G"] / 2) ** 2))))),
                H_pool={k: v for k, v in hp.items() if k != "_means"} | dict(
                    offset_scale=float(max(0.5, math.sqrt(np.mean((hp["_means"] - hp["G"] / 2) ** 2))))),
                fits={nm: {k: v for k, v in F.items() if k not in ("theta", "rel", "J")}
                      | dict(fb_cv_r2=M["cv_r2"], fb_rel=M["rel"], fb_provs=len(M["provs"]))
                      for nm, (F, M) in ff["fits"].items()},
                n_inst=int(ep.n))
    return res, info


def fold_mean(x, pair):
    return float(Base.pair_mean(x, pair).mean())


def primary(n_seeds=10):
    global LOGNAME
    LOGNAME = f"e7_primary_s{n_seeds}.log"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / LOGNAME).write_text("", encoding="utf-8")
    seeds = list(range(n_seeds))
    log(f"E7 primary regime={REGIME} seeds={seeds} cfg={CFG}")
    t0 = time.time()
    res, info = {}, {}
    for B in E.EVAL_FOLDS:
        res[B], info[B] = run_fold(B, seeds)
        log(f"  [{B}] {time.time() - t0:.0f}s")
        log(f"     excluded from pool: {info[B]['exclude']} (listed but no pool data: {info[B]['exclude_not_in_pool']})")
        log(f"     relatives-kept arm excludes only: {info[B]['exclude_relkept']}")
        log(f"     coverage {json.dumps(info[B]['coverage'])}")
        log(f"     fits {json.dumps(info[B]['fits'])}")
        log(f"     H_base {json.dumps(info[B]['H_base'])}")
        log(f"     H_pool {json.dumps(info[B]['H_pool'])}")
        log("     ALC " + " ".join(f"{nm}={fold_mean(res[B][nm]['alc'], res[B][nm]['pair']):.4f}" for nm in ARMS))
    out = dict(config=dict(regime=REGIME, seeds=seeds, cfg=CFG, min_items=MIN_ITEMS, min_p=MIN_P, stems=STEMS),
               info=info, arms={})
    base = Base.mean_alc(res, "base")
    out["sanity"] = dict(base_alc=base, ref=SANITY_REF, ok=bool(abs(base - SANITY_REF) < SANITY_TOL),
                         zero_prior_maxdiff=max(i["zero_prior_maxdiff"] for i in info.values()))
    log(f"SANITY base ALC={base:.6f} ref={SANITY_REF} (ref is seeds 0-9) ok={out['sanity']['ok']} "
        f"zero-prior identity maxdiff={out['sanity']['zero_prior_maxdiff']:.2e}")
    for nm in ARMS:
        a = dict(alc=Base.mean_alc(res, nm), curve=Base.mean_curve(res, nm).tolist(),
                 per_fold={B: fold_mean(res[B][nm]["alc"], res[B][nm]["pair"]) for B in res})
        if nm != "base":
            a["d"] = Base.delta(res, nm, "base")
            a["d_budget"] = [Base.delta(res, nm, "base", budget=j) for j in range(6)]
        out["arms"][nm] = a
        log(f"{nm:24s} ALC={a['alc']:.5f} curve=" + " ".join(f"{x:.4f}" for x in a["curve"])
            + (f"\n{'':24s} dALC(x1e-3) {Base.fmt(a['d'])}\n{'':24s} d by budget: "
               + " ".join(f"B{b}={d['pt']:+.2f}" for b, d in zip(E.BUDGETS, a["d_budget"])) if "d" in a else ""))
    json.dump(out, open(OUT / f"e7_primary_s{n_seeds}.json", "w", encoding="utf-8"), indent=1, default=str)
    log(f"done {time.time() - t0:.0f}s")
    return out


# ---------------------------------------------------------------------------
# Secondary: pair-level LOBO over the new benchmarks, budgets 0 and 1, no item model
# ---------------------------------------------------------------------------
def pair_q(S, N, m, t, lprior):
    """Single-pair group posterior mean (same grid x GH math as pooled_pair_m)."""
    eta = MU[None, :, None] + m[:, None, None] + t[:, None, None] * GH_X[None, None, :]
    ll = S[:, None, None] * log_sig(eta) + (N - S)[:, None, None] * log_sig(-eta) + LOG_GH_W
    mx = ll.max(2, keepdims=True)
    wt = np.exp(ll - mx)
    Ln = wt.sum(2)
    lp = np.log(Ln) + mx[:, :, 0] + lprior[None, :]
    Eq = (wt * sig(eta)).sum(2) / Ln
    lp -= lp.max(1, keepdims=True)
    post = np.exp(lp); post /= post.sum(1, keepdims=True)
    return (post * Eq).sum(1)


def _brier01(a, m, t, lprior):
    """Expected item Brier of the pair-mean prediction at budget 0 and 1 (exact over the one label)."""
    n = len(a)
    z, o = np.zeros(n), np.ones(n)
    br = lambda q: (q - a) ** 2 + a * (1 - a)
    b0 = br(pair_q(z, z, m, t, lprior))
    b1 = a * br(pair_q(o, o, m, t, lprior)) + (1 - a) * br(pair_q(z, o, m, t, lprior))
    return b0, b1


DOMAIN_ORDER = [("agentic_swe", r"agent|tool_use|software_engineering|ml_engineering|cybersecurity|gui"),
                ("mathematics", r"mathematics"), ("code_other", r"\bcod|programming"),
                ("medicine", r"medic"), ("safety", r"safety"), ("multilingual", r"multilingual|cultural|translation"),
                ("knowledge_reasoning", r"knowledge|reasoning|general|science|education|law|finance")]


def domain_of(dom):
    for nm, pat in DOMAIN_ORDER:
        if re.search(pat, dom):
            return nm
    return "other"


def secondary(perm_control=True):
    global LOGNAME
    LOGNAME = "e7_secondary.log"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / LOGNAME).write_text("", encoding="utf-8")
    Bd, Pd = load_pool()
    cnt = Pd[Pd.is2526].groupby("bench").size()
    targets = sorted(cnt[cnt >= MIN_P].index)
    url = Bd.set_index("bench").source_url.to_dict()
    dom = Bd.set_index("bench").domain.to_dict()
    old = [f for f in STEMS]
    rows = []
    t0 = time.time()
    for B in targets:
        ex = {B} | {b for b in Bd.bench if url.get(b) and url[b] == url.get(B)}
        for f in old:                              # name-stem families are removed together
            if re.search(STEMS[f], B) or B == f:
                ex |= set(relatives(f, Bd))
        hp = fit_hyper_pool(ex)
        H = dict(G=hp["G"], _means=hp["_means"], sB=1.0)
        lprior = Base.log_prior(H, CFG["prior"])
        te = Pd[(Pd.bench == B) & Pd.is2526]
        a = (te.s / te.n).to_numpy()
        names = np.array(["n:" + x if x else "" for x in te.name], dtype=object)
        n = len(te)
        b0, b1 = _brier01(a, np.zeros(n), np.full(n, hp["tau"]), lprior)
        rec = dict(bench=B, domain=domain_of(dom.get(B, "")), domain_raw=dom.get(B, ""), n_pairs=n,
                   base_b0=float(b0.mean()), base_b1=float(b1.mean()), excluded=len(ex))
        for arm, kw in (("subject", {}),) + ((("perm", dict(perm=20261008)),) if perm_control else ()):
            T = subset(ex, **kw)
            F, M = fit_names(T), fit_fallback(T)
            m, t, kind = prior_for(names, te.release.to_numpy(), te.provider.to_numpy(), F, M, hp["tau"])
            s0, s1 = _brier01(a, m, t, lprior)
            rec[f"{arm}_b0"], rec[f"{arm}_b1"] = float(s0.mean()), float(s1.mean())
            if arm == "subject":
                rec["match"] = float((kind == 2).mean())
                mm = kind == 2
                rec["d0_matched"] = float((s0 - b0)[mm].mean()) if mm.any() else np.nan
                rec["d0_unmatched"] = float((s0 - b0)[~mm].mean()) if (~mm).any() else np.nan
                yb = np.log((te.s + .5) / (te.n - te.s + .5)).to_numpy()
                rec["corr"] = float(np.corrcoef(m, yb - yb.mean())[0, 1]) if n > 2 and m.std() > 0 else np.nan
        rows.append(rec)
        log(f"{time.time() - t0:5.0f}s {B:30s} {rec['domain']:20s} P={n:4d} match={rec['match']:.2f} "
            f"B0 {rec['base_b0']:.4f}->{rec['subject_b0']:.4f}  B1 {rec['base_b1']:.4f}->{rec['subject_b1']:.4f}")
    R = pd.DataFrame(rows)
    R.to_parquet(ART / "secondary_per_bench.parquet")
    rng = np.random.default_rng(0)

    def summ(G):
        o = dict(n_bench=int(len(G)), n_pairs=int(G.n_pairs.sum()), match=float(G.match.mean()))
        for arm in ["subject"] + (["perm"] if perm_control else []):
            for b in ("b0", "b1"):
                d = (G[f"{arm}_{b}"] - G[f"base_{b}"]).to_numpy()
                bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
                o[f"{arm}_d{b}"] = [float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
                o[f"{arm}_{b}_neg"] = int((d < 0).sum())
        o["base_b0"], o["base_b1"] = float(G.base_b0.mean()), float(G.base_b1.mean())
        return o

    out = dict(all=summ(R), by_domain={k: summ(G) for k, G in R.groupby("domain")}, n_targets=len(targets))
    for k, o in [("ALL", out["all"])] + sorted(out["by_domain"].items()):
        log(f"{k:20s} nb={o['n_bench']:3d} pairs={o['n_pairs']:5d} match={o['match']:.2f} baseB0={o['base_b0']:.4f} "
            f"dB0={o['subject_db0'][0]*1e3:+.2f} [{o['subject_db0'][1]*1e3:+.2f},{o['subject_db0'][2]*1e3:+.2f}] "
            f"neg={o['subject_b0_neg']} dB1={o['subject_db1'][0]*1e3:+.2f} [{o['subject_db1'][1]*1e3:+.2f},"
            f"{o['subject_db1'][2]*1e3:+.2f}]"
            + (f" | perm dB0={o['perm_db0'][0]*1e3:+.2f} dB1={o['perm_db1'][0]*1e3:+.2f}" if perm_control else ""))
    json.dump(out, open(OUT / "e7_secondary.json", "w", encoding="utf-8"), indent=1, default=str)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="primary")
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    for s in a.step.split(","):
        if s == "pool":
            pool()
        elif s == "primary":
            primary(a.seeds)
        elif s == "secondary":
            secondary()
