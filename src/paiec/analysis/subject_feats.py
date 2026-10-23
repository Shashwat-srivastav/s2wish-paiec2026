"""Subject-feature analysis (report 14): what about a subject predicts its ability on an
UNSEEN benchmark at budget 0, and how should the runtime match "the same subject"?

Usage:  python src/paiec/analysis/subject_feats.py [--data E:/NeurIPS/aims/data] [--min-items 80]
Prints all tables used in reports/14_SUBJECT_FEATURES.md. Writes nothing to disk.
Runtime subject = 8 string fields (README): normalized_name, provider, release_date, access_date,
harness, reasoning_effort, harness_version, subject_features_extra. display_name and subject_id are
NOT visible at runtime; we use subject_id only as an analysis-side identity check.
"""
from __future__ import annotations
import argparse, re
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from paiec import paths

FIELDS = ["normalized_name", "provider", "release_date", "access_date", "harness",
          "reasoning_effort", "harness_version", "subject_features_extra"]
BENCH = ["matharena", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench"]
BENCH_DATE = {}  # filled from benchmarks.parquet release_date
EFFORT = {"": 0, "minimal": 1, "low": 1, "medium": 2, "high": 3, "xhigh": 4, "max": 5}
SMALL = {"mini", "nano", "flash", "lite", "haiku", "small", "turbo", "fast", "air", "scout", "tiny"}
BIG = {"pro", "opus", "max", "ultra", "large", "plus", "maverick", "deep", "speciale"}
REASON = {"r1", "o1", "o3", "o4", "think", "thinking", "reasoning", "qwq", "k2-think", "deep"}


def logit(p): return np.log(p / (1 - p))
def sig(x): return 1 / (1 + np.exp(-x))


# ---------------------------------------------------------------- subject parsing
def parse_extra(s: str) -> dict:
    out = {}
    for kv in (s or "").split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1); out[k.strip()] = v.strip()
    return out


def base_name(row) -> str:
    """Runtime-computable name: normalized_name, else model_config from extra (effort suffix stripped)."""
    n = (row["normalized_name"] or "").strip()
    if n:
        return n.lower()
    mc = parse_extra(row["subject_features_extra"]).get("model_config", "")
    mc = re.sub(r"-+(low|medium|high|xhigh|minimal)$", "", mc)
    return ("cfg:" + mc.lower()) if mc else "<anon>"


def family(name: str) -> str:
    """Coarse family: provider/series word + size tier; version numbers and sizes dropped.
    e.g. 'anthropic claude 3.5 sonnet' -> 'anthropic claude sonnet'; 'openai gpt-4o mini' -> 'openai gpt mini'."""
    n = name.replace("cfg:", "")
    n = re.sub(r"\(.*?\)", " ", n)
    n = re.sub(r"[-_/.]", " ", n)
    toks = [t for t in n.split() if not re.fullmatch(r"[\d.]+[a-z]?|v\d+|\d+b|a\d+b|\d+x\d+b|k\d+|\d{2,4}", t)]
    toks = [re.sub(r"\d+", "", t) for t in toks]
    toks = [t for t in toks if t]
    return " ".join(toks[:4]) or "<anon>"


def params_b(name: str) -> float:
    m = re.findall(r"(\d+(?:\.\d+)?)b\b", name.replace("-", " "))
    return float(m[0]) if m else np.nan


def subject_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    rd = pd.to_datetime(df["release_date"].replace("", None), errors="coerce")
    X["rel_years"] = ((rd - pd.Timestamp("2024-01-01")).dt.days / 365.25)
    X["rel_missing"] = X["rel_years"].isna().astype(float)
    X["rel_years"] = X["rel_years"].fillna(X["rel_years"].median())
    eff = df["reasoning_effort"].str.lower().map(EFFORT).fillna(0)
    X["effort"] = eff; X["effort_set"] = (eff > 0).astype(float)
    X["has_harness"] = (df["harness"] != "").astype(float)
    X["has_extra"] = (df["subject_features_extra"] != "").astype(float)
    X["has_name"] = (df["normalized_name"] != "").astype(float)
    nm = df["bname"].str.replace(r"[-_/()]", " ", regex=True)
    tok = nm.str.split()
    X["tier_small"] = tok.apply(lambda t: float(any(x in SMALL for x in t)))
    X["tier_big"] = tok.apply(lambda t: float(any(x in BIG for x in t)))
    X["reason_cue"] = tok.apply(lambda t: float(any(x in REASON for x in t)))
    X["agent_cue"] = (nm.str.contains("agent") | df["provider"].str.contains("agent")).astype(float)
    pb = df["bname"].apply(params_b)
    X["log_params"] = np.log10(pb).fillna(0); X["params_known"] = pb.notna().astype(float)
    for p in ["OpenAI", "Anthropic", "Google", "DeepSeek", "Alibaba", "Meta", "xAI"]:
        X["prov_" + p] = (df["provider"] == p).astype(float)
    X["prov_missing"] = (df["provider"] == "").astype(float)
    return X


FEATSETS = {
    "release": ["rel_years", "rel_missing"],
    "provider": [c for c in ["prov_OpenAI", "prov_Anthropic", "prov_Google", "prov_DeepSeek",
                             "prov_Alibaba", "prov_Meta", "prov_xAI", "prov_missing"]],
    "effort+harness+extra": ["effort", "effort_set", "has_harness", "has_extra", "has_name"],
    "name_tokens": ["tier_small", "tier_big", "reason_cue", "agent_cue", "log_params", "params_known"],
}
FEATSETS["all_meta"] = sum(FEATSETS.values(), [])


# ---------------------------------------------------------------- data
def load(data: str, min_items: int):
    rows, subs = [], []
    for b in BENCH:
        s = pd.read_parquet(f"{data}/{b}/subjects.parquet")
        r = pd.read_parquet(f"{data}/{b}/response.parquet", columns=["subject_id", "item_id", "response"])
        bd = pd.read_parquet(f"{data}/{b}/benchmarks.parquet")["release_date"].iloc[0]
        BENCH_DATE[b] = str(bd)
        r = r.dropna(subset=["response"])
        g = r.groupby("subject_id").agg(m=("response", "mean"), n_resp=("response", "size"),
                                        n_items=("item_id", "nunique")).reset_index()
        g["bench"] = b
        rows.append(g); subs.append(s.assign(bench=b))
    P = pd.concat(rows); S = pd.concat(subs)
    for c in FIELDS + ["display_name"]:
        S[c] = S[c].fillna("").astype(str)
    P = P.merge(S, on=["subject_id", "bench"], how="left")
    P = P[P.n_items >= min_items].reset_index(drop=True)
    P["bname"] = P.apply(base_name, axis=1)
    P["fam"] = P["bname"].apply(family)
    eps = 0.5 / P["n_resp"]
    P["lg"] = logit(P["m"].clip(eps, 1 - eps))
    P["bmean"] = P.groupby("bench")["lg"].transform("mean")
    P["y"] = P["lg"] - P["bmean"]
    return P, S


# ---------------------------------------------------------------- analyses
def acc_table(P):
    print("\n## 1a. pair accuracy (response-level mean, pairs >= min_items)")
    t = P.groupby("bench").agg(pairs=("m", "size"), acc_mean=("m", "mean"), acc_sd=("m", "std"),
                               logit_mean=("lg", "mean"), y_sd=("y", "std"),
                               names=("bname", "nunique"), fams=("fam", "nunique"))
    print(t.round(3).to_string())


def overlap(P, key):
    A = P.groupby([key, "bench"])["y"].mean().unstack()
    n_multi = int((A.notna().sum(1) >= 2).sum())
    res = []
    bs = [b for b in A.columns if A[b].notna().sum() > 2]
    for i, b1 in enumerate(bs):
        for b2 in bs[i + 1:]:
            m = A[[b1, b2]].dropna()
            if len(m) >= 4:
                res.append((b1, b2, len(m), spearmanr(m[b1], m[b2]).statistic))
    return n_multi, res, A


def cross_bench(P):
    print("\n## 1b. cross-benchmark rank correlation of within-benchmark ability y")
    for key in ["subject_id", "bname", "fam"]:
        n_multi, res, _ = overlap(P, key)
        w = [r[3] * r[2] for r in res]; nn = sum(r[2] for r in res)
        print(f"key={key}: keys in >=2 benches={n_multi}; pooled n-weighted rho={sum(w)/max(nn,1):.3f}")
        for r in res:
            print(f"   {r[0]:>17} x {r[1]:<17} n={r[2]:3d} rho={r[3]:+.3f}")


def family_purity(P):
    print("\n## 1c. family normalization: does it corrupt meaning? (within-bench spread of y inside a family)")
    for key in ["bname", "fam"]:
        g = P.groupby(["bench", key])["y"]
        within = g.std().dropna()
        sizes = g.size()
        print(f"key={key}: groups={len(sizes)}, multi-member groups={int((sizes>1).sum())}, "
              f"mean within-group sd(y)={within.mean():.3f} vs overall sd(y)={P['y'].std():.3f}")
    big = P.groupby(["bench", "fam"]).filter(lambda d: d.bname.nunique() > 1)
    worst = big.groupby(["bench", "fam"]).agg(names=("bname", lambda x: sorted(set(x))[:4]),
                                              ysd=("y", "std")).sort_values("ysd", ascending=False).head(8)
    print(worst.to_string())


def identity_checks(P, S):
    print("\n## 4a. identity: is the 8-field dict == subject_id? does it repeat across benchmarks?")
    S2 = S.copy(); S2["k8"] = S2[FIELDS].agg("|".join, axis=1)
    print("8-field keys:", S2.k8.nunique(), " subject_ids:", S2.subject_id.nunique(),
          " max ids per key:", S2.groupby("k8").subject_id.nunique().max(),
          " max keys per id:", S2.groupby("subject_id").k8.nunique().max())
    m = S2.groupby("k8").bench.nunique()
    print("8-field keys appearing in >=2 benchmarks:", int((m >= 2).sum()))
    S2["bname"] = S2.apply(base_name, axis=1)
    m2 = S2.groupby("bname").bench.nunique()
    print("base names in >=2 benchmarks:", int((m2 >= 2).sum()), "; >=3:", int((m2 >= 3).sum()))
    S2["fam"] = S2.bname.apply(family)
    print("families in >=2 benchmarks:", int((S2.groupby('fam').bench.nunique() >= 2).sum()))
    # which fields differ between benchmarks for the same name
    diff = {f: 0 for f in FIELDS}; npair = 0
    for nmv, d in S2.groupby("bname"):
        if d.bench.nunique() < 2 or nmv == "<anon>": continue
        for b1 in d.bench.unique():
            for b2 in d.bench.unique():
                if b1 >= b2: continue
                x, z = d[d.bench == b1].iloc[0], d[d.bench == b2].iloc[0]; npair += 1
                for f in FIELDS: diff[f] += int(x[f] != z[f])
    print("cross-bench same-name pairs:", npair, " fraction differing per field:",
          {k: round(v / max(npair, 1), 2) for k, v in diff.items()})
    print("subjects within one bench sharing a base name (multi-config):",
          int((S2.groupby(['bench', 'bname']).size() > 1).sum()), "groups")
    print("extra keys:", S2.subject_features_extra.apply(lambda s: tuple(sorted(parse_extra(s)))).value_counts().to_dict())
    print("empty normalized_name by bench:", S2[S2.normalized_name == ""].groupby("bench").size().to_dict())


def harness_effort(P):
    print("\n## 2a. natural experiments")
    d = P[P.bench == "multi_swebench"]
    d = d[d.groupby("bname").harness.transform("nunique") >= 3]
    if len(d):
        tot = d.y.var(ddof=0)
        rn = d.y - d.groupby("bname").y.transform("mean")
        rh = d.y - d.groupby("harness").y.transform("mean")
        res2 = rn - (d.assign(r=rn).groupby("harness").r.transform("mean"))
        print(f"multi_swebench (names with >=3 harnesses, n={len(d)}, names={d.bname.nunique()}, harnesses={d.harness.nunique()}):"
              f" R2 name-only={1-rn.var(ddof=0)/tot:.3f}, harness-only={1-rh.var(ddof=0)/tot:.3f}, name+harness={1-res2.var(ddof=0)/tot:.3f}")
        print("  harness mean y:", d.groupby("harness").y.mean().round(2).to_dict())
    m = P[(P.bench == "matharena")]
    m = m.assign(eff=m.reasoning_effort.str.lower().map(EFFORT).fillna(0), core=m.bname.str.replace(r"-(low|high|medium)$", "", regex=True))
    g = m[m.groupby("core").eff.transform("nunique") > 1]
    for c, dd in g.groupby("core"):
        print(f"  effort ladder {c}: ", dd.sort_values("eff")[["reasoning_effort", "m"]].round(3).values.tolist())


def ridge_cv(P, X, cols, groups, alpha=3.0, center=False, extra_pred=None):
    yh = pd.Series(np.nan, index=P.index)
    Xc = X[cols].copy()
    if center:  # within-benchmark centering of features (uses benchmark's subject set)
        Xc = Xc - Xc.groupby(P.bench).transform("mean")
    for g in groups.unique():
        te = groups == g; tr = ~te
        if tr.sum() < 5: continue
        mdl = Ridge(alpha=alpha).fit(Xc[tr], P.y[tr])
        yh[te] = mdl.predict(Xc[te])
    return yh


def name_transfer(P, key, groups):
    """Predict y on held-out group from mean y of the same key on OTHER benchmarks (training folds)."""
    yh = pd.Series(0.0, index=P.index); hit = pd.Series(False, index=P.index)
    for g in groups.unique():
        te = groups == g
        tr = ~te
        for i in P.index[te]:
            k = P.at[i, key]
            src = P[tr & (P[key] == k) & (P.bench != P.at[i, "bench"])]
            if len(src):
                yh[i] = 0.7 * src.y.mean(); hit[i] = True  # 0.7 = shrinkage toward 0 (fixed a priori)
    return yh, hit


def scores(P, yh, mask=None):
    mask = P.bench != "swe_rebench" if mask is None else mask
    y, p = P.y[mask], yh[mask].fillna(0)
    r2 = 1 - ((y - p) ** 2).sum() / (y ** 2).sum()  # vs subject-agnostic prior y=0
    # Brier at budget 0, oracle benchmark offset (isolates subject term). Response-level, equal pair weight.
    m, b = P.m[mask], P.bmean[mask]
    def brier(q): return (m * (1 - q) ** 2 + (1 - m) * q ** 2).mean()
    q0, q1 = sig(b), sig(b + p)
    # unknown offset: global prior = mean of benchmark logit means
    gb = P.groupby("bench").lg.mean().drop("swe_rebench", errors="ignore").mean()
    return r2, brier(q0), brier(q1), brier(sig(gb + 0 * p)), brier(sig(gb + p))


def models(P):
    X = subject_features(P)
    print("\n## 2b. predicting within-benchmark ability y (R2 vs subject-agnostic y=0; Brier at budget 0)")
    print("   B0 columns: oracle-offset agnostic / oracle-offset +subject / unknown-offset agnostic / unknown-offset +subject")
    for split, groups in [("LOBO", P.bench), ("LOFO(family)", P.fam)]:
        print(f"--- {split}")
        for fs, cols in FEATSETS.items():
            for ctr in [False, True]:
                yh = ridge_cv(P, X, cols, groups, center=ctr)
                r2, b0, b1, u0, u1 = scores(P, yh)
                print(f"   {fs:22s} center={int(ctr)}  R2={r2:+.3f}  B0 {b0:.4f}->{b1:.4f} (d={b1-b0:+.4f}) | {u0:.4f}->{u1:.4f}")
        for key in ["bname", "fam"]:
            yh, hit = name_transfer(P, key, groups)
            r2, b0, b1, u0, u1 = scores(P, yh)
            r2h = scores(P, yh, mask=hit & (P.bench != "swe_rebench"))[0] if hit.sum() else np.nan
            print(f"   transfer[{key:5s}] cover={hit[P.bench!='swe_rebench'].mean():.2f}  R2={r2:+.3f} (on covered: {r2h:+.3f})  B0 {b0:.4f}->{b1:.4f} (d={b1-b0:+.4f})")
            yr = ridge_cv(P, X, FEATSETS["all_meta"], groups, center=True)
            comb = yh.where(hit, yr)
            r2, b0, b1, u0, u1 = scores(P, comb)
            print(f"   transfer[{key}]+meta fallback          R2={r2:+.3f}  B0 {b0:.4f}->{b1:.4f} (d={b1-b0:+.4f})")
    print("   per-benchmark LOBO R2 (all_meta, centered):")
    yh = ridge_cv(P, X, FEATSETS["all_meta"], P.bench, center=True)
    for b in BENCH[:-1]:
        mk = P.bench == b
        print(f"     {b:18s} n={mk.sum():3d} R2={scores(P, yh, mk)[0]:+.3f}")
    mdl = Ridge(alpha=3.0).fit(X[FEATSETS['all_meta']] - X[FEATSETS['all_meta']].groupby(P.bench).transform('mean'), P.y)
    print("   full-fit coefs:", dict(zip(FEATSETS["all_meta"], np.round(mdl.coef_, 2))))


def transfer_diag(P):
    print("\n## 2c. name-transfer diagnostics (LOBO; source = mean y of same bname on other benches)")
    yh, hit = name_transfer(P, "bname", P.bench)
    raw = yh / 0.7
    for b in BENCH[:-1]:
        mk = hit & (P.bench == b)
        if mk.sum() >= 3:
            r = spearmanr(raw[mk], P.y[mk]).statistic
            sl = np.polyfit(raw[mk], P.y[mk], 1)[0]
            print(f"   target {b:18s} covered={mk.sum():3d}/{(P.bench==b).sum():3d} rho={r:+.3f} OLS slope={sl:+.2f}")
    # harness-collapsed: average y over harness variants of a name within bench
    Q = P.groupby(["bench", "bname"], as_index=False).agg(y=("y", "mean"))
    A = Q.pivot(index="bname", columns="bench", values="y")
    pr, ys = [], []
    for b in A.columns:
        for nmv in A.index[A[b].notna()]:
            o = A.loc[nmv].drop(b).dropna()
            if len(o): pr.append(o.mean()); ys.append(A.at[nmv, b])
    pr, ys = np.array(pr), np.array(ys)
    print(f"   harness-collapsed name level: n={len(ys)} rho={spearmanr(pr, ys).statistic:+.3f} slope={np.polyfit(pr, ys, 1)[0]:+.2f} "
          f"R2(0.7*src)={1-((ys-0.7*pr)**2).sum()/(ys**2).sum():+.3f}")


def temporal(P):
    print("\n## 3. temporal")
    rd = pd.to_datetime(P.release_date.replace("", None), errors="coerce")
    for b in BENCH[:-1]:
        mk = (P.bench == b) & rd.notna()
        r = spearmanr(rd[mk].astype("int64"), P.y[mk])
        print(f"   {b:18s} n={mk.sum():3d} rho(release,y)={r.statistic:+.3f} p={r.pvalue:.3g}  "
              f"release range {rd[mk].min().date()}..{rd[mk].max().date()} (bench release {BENCH_DATE[b]}); "
              f"access range {P.access_date[mk & (P.access_date!='')].min() or '-'}..{P.access_date[mk & (P.access_date!='')].max() or '-'}")
    allr = rd.dropna()
    print("   release quantiles (all pairs):", allr.quantile([.1, .5, .9]).dt.date.tolist(),
          " frac >= 2026-01-01:", round(float((allr >= "2026-01-01").mean()), 3))
    mk = rd.notna() & (P.bench != "swe_rebench")
    print("   pooled rho(release, y):", round(spearmanr(rd[mk].astype('int64'), P.y[mk]).statistic, 3))
    # frontier-ness: release relative to benchmark's latest subject release
    fr = (rd - rd.groupby(P.bench).transform("max")).dt.days / 365.25
    print("   pooled rho(release - bench max release, y):", round(spearmanr(fr[mk], P.y[mk]).statistic, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(paths.DATA))
    ap.add_argument("--min-items", type=int, default=80)
    a = ap.parse_args()
    P, S = load(a.data, a.min_items)
    acc_table(P); cross_bench(P); family_purity(P); identity_checks(P, S)
    harness_effort(P); models(P); transfer_diag(P); temporal(P)


if __name__ == "__main__":
    main()
