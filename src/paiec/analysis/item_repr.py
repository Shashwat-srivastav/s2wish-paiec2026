"""Item-representation study: does item content carry cold-start difficulty signal that
transfers ACROSS benchmarks?  (report: reports/13_ITEM_REPRESENTATION.md)

Design (Representation Scientist + Minimalist: same downstream model, same split, only the
representation changes):
  target   beta_i = per-benchmark Rasch difficulty (L2-regularised JML on trial counts), centred
           within benchmark.  Secondary target: logit of smoothed mean correctness.  Each
           benchmark's target uses only its own responses, so a held-out benchmark's labels never
           enter training (they are used for scoring only).
  features every representation is z-scored / centred WITHIN benchmark (label-free, legal at
           runtime since predict() sees the hidden benchmark's items) -> the model learns
           within-benchmark ranking, which is what survives an unknown benchmark offset.
  model    ridge, benchmark-balanced sample weights, alpha by inner leave-benchmark-out CV
           (within-benchmark CV: inner 5-fold).
  eval     LOBO (primary) and 5-fold within-benchmark CV over items (diagnostic).
           Spearman(pred, beta) + bootstrap CI; Brier on (subject,item) rows with
           logit p = theta_s(oracle) - [mean beta_b(oracle) + f(x_i)], compared against the
           item-free baseline f=0 and the oracle-beta ceiling.  Also 'slope-oracle' Brier, which
           refits only a scalar slope on the held-out benchmark (separates ranking from scale).
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from paiec import paths
ROOT = paths.ROOT
DATA = paths.DATA
EXP = ROOT / "paiec" / "experiments"
BENCHES = ["matharena", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench"]
EMB_BENCHES = ["real_webagents", "researchcodebench", "swe_rebench"]
ALPHAS = [0.1, 1, 10, 100, 1000, 1e4]
RNG = np.random.default_rng(0)

# ---------------------------------------------------------------- targets

def rasch(resp: pd.DataFrame, lam_b: float = 1.0, lam_t: float = 0.1):
    g = resp.groupby(["subject_id", "item_id"])["response"].agg(["sum", "count"]).reset_index()
    s_idx, subs = pd.factorize(g.subject_id)
    i_idx, items = pd.factorize(g.item_id)
    k, n = g["sum"].to_numpy(float), g["count"].to_numpy(float)
    S, I = len(subs), len(items)

    def f(x):
        th, b = x[:S], x[S:]
        eta = th[s_idx] - b[i_idx]
        p = expit(eta)
        ll = k * np.log(p + 1e-12) + (n - k) * np.log(1 - p + 1e-12)
        r = k - n * p  # d ll / d eta
        gth = -np.bincount(s_idx, r, S) + lam_t * th
        gb = np.bincount(i_idx, r, I) + lam_b * b
        return -ll.sum() + 0.5 * lam_t * th @ th + 0.5 * lam_b * b @ b, np.r_[gth, gb]

    x = minimize(f, np.zeros(S + I), jac=True, method="L-BFGS-B").x
    return pd.Series(x[:S], subs), pd.Series(x[S:], items), g


def load():
    items, resp_all = [], {}
    swe_ids = set(pd.read_parquet(EXP / "jev_in" / "swe_rebench.parquet").item_id)
    for b in BENCHES:
        it = pd.read_parquet(DATA / b / "items.parquet")[["item_id", "content", "item_features"]]
        r = pd.read_parquet(DATA / b / "response.parquet")[["subject_id", "item_id", "response"]]
        r = r[r.item_id.isin(it.item_id)].dropna(subset=["response"])
        r["response"] = (r.response > 0.5).astype(float)
        if b == "swe_rebench":  # restrict to the Jev sample so every representation sees the same items
            it = it[it.item_id.isin(swe_ids)]
            r = r[r.item_id.isin(swe_ids)]
        th, be, g = rasch(r)
        pm = r.groupby(["subject_id", "item_id"]).response.mean().groupby("item_id").agg(["sum", "count"])
        it = it[it.item_id.isin(be.index)].copy()
        it["bench"] = b
        it["beta"] = be.reindex(it.item_id).to_numpy()
        it["beta_c"] = it.beta - it.beta.mean()
        pp = pm.reindex(it.item_id)
        it["plogit_c"] = -np.log((pp["sum"] + .5) / (pp["count"] - pp["sum"] + .5)).to_numpy()
        it["plogit_c"] -= it.plogit_c.mean()  # difficulty-signed
        it["n_subj"] = pp["count"].to_numpy()
        resp_all[b] = (r, th, be)
        items.append(it)
    return pd.concat(items, ignore_index=True), resp_all

# ---------------------------------------------------------------- representations

FILE_RE = re.compile(r"[\w/\-]+\.(?:py|js|ts|tsx|jsx|java|go|rs|c|cc|cpp|h|hpp|rb|php|json|ya?ml|toml|md|txt|html|css|sh)\b")

def trivial(texts: pd.Series) -> pd.DataFrame:
    t = texts.fillna("")
    d = pd.DataFrame({
        "log_chars": np.log1p(t.str.len()),
        "log_words": np.log1p(t.str.split().str.len()),
        "log_lines": np.log1p(t.str.count("\n")),
        "code_fence": t.str.count("```").clip(upper=20),
        "code_chars": t.str.count(r"[{};()=<>\[\]]") / (t.str.len() + 1),
        "n_files": np.log1p(t.apply(lambda s: len(set(FILE_RE.findall(s))))),
        "n_urls": np.log1p(t.str.count(r"https?://")),
        "digit_frac": t.str.count(r"\d") / (t.str.len() + 1),
        "latex": np.log1p(t.str.count(r"\$|\\[a-zA-Z]+")),
        "n_numbers": np.log1p(t.str.count(r"\b\d+(?:\.\d+)?\b")),
        "questions": t.str.count(r"\?").clip(upper=20),
        "traceback": t.str.contains(r"Traceback|Error|Exception", regex=True).astype(float),
        "upper_frac": t.str.count(r"[A-Z]") / (t.str.len() + 1),
    })
    return d


def jev(items: pd.DataFrame) -> pd.DataFrame | None:
    parts = [pd.read_parquet(p) for p in EXP.glob("jev_out_*.parquet")]
    if not parts:
        return None
    j = pd.concat(parts).drop_duplicates("item_id").set_index("item_id")
    cols = ["jev_difficulty", "jev_difficulty_conf", "jev_multistep", "jev_tool_use",
            "jev_expertise", "jev_expertise_conf", "jev_verifiable", "jev_ambiguous", "jev_scope",
            "jev_scope_conf", "jev_obscure_knowledge", "jev_refusal_risk"]
    return j.reindex(items.item_id)[cols].reset_index(drop=True)


def emb(items: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for b in EMB_BENCHES:
        e = pd.read_parquet(DATA / "embed" / f"{b}.parquet")
        rows.append(e[["content_sha1", "embedding"]])  # embed item_ids differ from items' ids; join on sha1(content)
    e = pd.concat(rows).drop_duplicates("content_sha1").set_index("content_sha1")
    sha = items.content.fillna("").map(lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest())
    m = sha.map(e.embedding)
    X = np.full((len(items), 1536), np.nan)
    ok = m.notna().to_numpy()
    X[ok] = np.stack(m[ok].to_numpy())
    return pd.DataFrame(X)


def bench_center(X: np.ndarray, bench: np.ndarray, scale: bool) -> np.ndarray:
    X = X.astype(np.float32, copy=True)
    for b in np.unique(bench):
        m = bench == b
        mu = np.nanmean(X[m], 0)
        X[m] -= mu
        if scale:
            sd = np.nanstd(X[m], 0)
            X[m] /= np.where(sd > 1e-9, sd, 1.0)
    return np.nan_to_num(X)

# ---------------------------------------------------------------- model / eval

def tfidf_block(texts_tr, texts_te):
    v = TfidfVectorizer(max_features=2000, min_df=3, sublinear_tf=True, ngram_range=(1, 2),
                        dtype=np.float32)
    clip = lambda s: s.fillna("").str.slice(0, 20000)
    return v.fit_transform(clip(texts_tr)).toarray(), v.transform(clip(texts_te)).toarray()


def fit_ridge(X, y, w, groups):
    """alpha by inner group CV (groups = benchmarks, or folds); returns fitted model."""
    ug = np.unique(groups)
    if len(ug) >= 2:
        best, ba = np.inf, ALPHAS[0]
        for a in ALPHAS:
            err = 0.0
            for g in ug:
                tr, te = groups != g, groups == g
                p = Ridge(alpha=a).fit(X[tr], y[tr], sample_weight=w[tr]).predict(X[te])
                err += np.mean((p - y[te]) ** 2)
            if err < best:
                best, ba = err, a
    else:
        ba = 100.0
    return Ridge(alpha=ba).fit(X, y, sample_weight=w), ba


def brier_eval(b, ids, pred, resp_all):
    """Brier on (subject,item) pairs of held-out benchmark b for item offsets pred (centred)."""
    r, th, be = resp_all[b]
    off = pd.Series(pred - pred.mean(), index=ids)
    pairs = r.groupby(["subject_id", "item_id"]).response.agg(["mean", "count"]).reset_index()
    pairs = pairs[pairs.item_id.isin(off.index)]
    y2 = r.assign(sq=r.response ** 2).groupby(["subject_id", "item_id"]).sq.mean()
    y2 = y2.reindex(pd.MultiIndex.from_frame(pairs[["subject_id", "item_id"]])).to_numpy()
    base = th.reindex(pairs.subject_id).to_numpy() - be.mean()
    ybar = pairs["mean"].to_numpy()
    o = off.reindex(pairs.item_id).to_numpy()
    orc = (be - be.mean()).reindex(pairs.item_id).to_numpy()

    def brier(eta):  # mean over trials of (p - y)^2, averaged per pair
        p = expit(eta)
        return float(np.mean(p * p - 2 * p * ybar + y2))

    res = {"brier_none": brier(base), "brier_pred": brier(base - o), "brier_oracle": brier(base - orc)}
    oz = o / (o.std() + 1e-12)  # slope in logits per SD of prediction
    s = minimize_scalar(lambda c: brier(base - c * oz), bounds=(-4, 4), method="bounded")
    res["brier_slope"] = s.fun
    res["slope"] = s.x
    return res


def boot_ci(y, p, n=500):
    idx = np.arange(len(y))
    vals = [spearmanr(y[i], p[i])[0] for i in (RNG.choice(idx, len(idx)) for _ in range(n))]
    return np.nanpercentile(vals, [2.5, 97.5])


def build_X(rep, items, tr, te, blocks):
    """Return (Xtr, Xte) for representation `rep` (a '+'-joined list of blocks)."""
    parts_tr, parts_te = [], []
    bench = items.bench.to_numpy()
    for name in rep.split("+"):
        if name == "tfidf":
            a, c = tfidf_block(items.content[tr], items.content[te])
            A = np.zeros((len(items), a.shape[1]), np.float32)
            A[tr], A[te] = a, c
            A = bench_center(A, bench, scale=False)
            parts_tr.append(A[tr] * 3.0); parts_te.append(A[te] * 3.0)
        else:
            A = bench_center(blocks[name], bench, scale=(name != "emb"))
            if name == "emb":
                A = A * 10.0  # raw cosine-space units are tiny; fixed rescale, alpha CV handles the rest
            parts_tr.append(A[tr]); parts_te.append(A[te])
    return np.hstack(parts_tr), np.hstack(parts_te)


def run(items, resp_all, blocks, reps, benches, target="beta_c", mode="lobo"):
    out = []
    for rep in reps:
        for hb in benches:
            pool = items.bench.isin(benches).to_numpy()
            hold = (items.bench == hb).to_numpy()
            if mode == "lobo":
                splits = [(pool & ~hold, hold, items.bench.to_numpy())]
            else:
                fold = np.full(len(items), -1)
                # fold by content group: matharena has exact-duplicate problems across competitions
                gid = pd.factorize(items.content[hold])[0]
                fold[hold] = (RNG.permutation(gid.max() + 1) % 5)[gid]
                splits = [(hold & (fold != f), hold & (fold == f), fold) for f in range(5)]
            preds = np.zeros(len(items))
            for tr, te, grp in splits:
                Xtr, Xte = build_X(rep, items, tr, te, blocks)
                y = items[target].to_numpy()
                ytr = y[tr] - np.array([y[tr][items.bench[tr] == b].mean() for b in items.bench[tr]]) \
                    if mode == "lobo" else y[tr] - y[tr].mean()
                w = 1.0 / items[tr].groupby("bench").bench.transform("count").to_numpy()
                m, a = fit_ridge(Xtr, ytr, w, grp[tr])
                preds[te] = m.predict(Xte)
            y = items.loc[hold, target].to_numpy(); p = preds[hold]
            rho = spearmanr(y, p)[0]
            lo, hi = boot_ci(y, p)
            rec = {"mode": mode, "rep": rep, "alpha": a, "held": hb, "n": int(hold.sum()), "spearman": rho,
                   "ci_lo": lo, "ci_hi": hi,
                   "spearman_plogit": spearmanr(items.loc[hold, "plogit_c"], p)[0]}
            rec.update(brier_eval(hb, items.loc[hold, "item_id"].to_numpy(), p, resp_all))
            out.append(rec)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}),
                  flush=True)
    return pd.DataFrame(out)


def zero_param(items, blocks):
    """Univariate, zero-fit baselines: raw feature vs beta within each benchmark."""
    rows = []
    for b in BENCHES:
        m = (items.bench == b).to_numpy()
        for name, v in [("log_chars", blocks["triv"][m, 0]),
                        ("jev_difficulty", blocks["jev"][m, 0] if "jev" in blocks else None),
                        ("jev_scope", blocks["jev"][m, 8] if "jev" in blocks else None),
                        ("jev_expertise", blocks["jev"][m, 4] if "jev" in blocks else None)]:
            if v is None or np.all(np.isnan(v)):
                continue
            rows.append({"bench": b, "feature": name, "spearman": spearmanr(items.beta_c[m], v, nan_policy="omit")[0]})
    return pd.DataFrame(rows).pivot(index="feature", columns="bench", values="spearman").round(3).to_string()


def main():
    items, resp_all = load()
    print(items.groupby("bench").agg(n=("item_id", "size"), beta_sd=("beta", "std"),
                                     beta_mean=("beta", "mean"), nsubj=("n_subj", "median")), flush=True)
    blocks = {"triv": trivial(items.content).to_numpy()}
    J = jev(items)
    if J is not None and J.notna().all(axis=1).mean() > 0.95:
        blocks["jev"] = J.to_numpy()
        blocks["jevd"] = J[["jev_difficulty"]].to_numpy()  # minimalist: one scalar
        print(f"jev coverage {J.notna().all(axis=1).mean():.3f}", flush=True)
    E = emb(items)
    print("emb coverage by bench", pd.Series(~np.isnan(E.to_numpy()[:, 0])).groupby(items.bench).mean().to_dict(), flush=True)
    blocks["emb"] = E.to_numpy()
    print("zero-param within-benchmark Spearman vs beta:\n", zero_param(items, blocks), flush=True)

    reps = ["triv", "tfidf"] + (["jevd", "jev", "triv+jev", "triv+jev+tfidf"] if "jev" in blocks else [])
    res = [run(items, resp_all, blocks, reps, BENCHES, mode="lobo"),
           run(items, resp_all, blocks, reps, BENCHES, mode="within")]
    ereps = ["triv", "tfidf", "emb"] + (["jevd", "jev", "emb+jev", "triv+jev"] if "jev" in blocks else [])
    sub = items[items.bench.isin(EMB_BENCHES)].reset_index(drop=True)
    sblocks = {k: v[items.bench.isin(EMB_BENCHES).to_numpy()] for k, v in blocks.items()}
    r3 = pd.concat([run(sub, resp_all, sblocks, ereps, EMB_BENCHES, mode="lobo"),
                    run(sub, resp_all, sblocks, ereps, EMB_BENCHES, mode="within")])
    r3["mode"] = r3["mode"] + "_emb3"
    res.append(r3)
    R = pd.concat(res, ignore_index=True)
    R["dBrier_pred"] = R.brier_pred - R.brier_none
    R["dBrier_slope"] = R.brier_slope - R.brier_none
    R["dBrier_oracle"] = R.brier_oracle - R.brier_none
    R.to_csv(EXP / "item_repr_results.csv", index=False)
    print(R.round(4).to_string())


if __name__ == "__main__":
    main()
