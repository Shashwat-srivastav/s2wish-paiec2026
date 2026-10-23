"""audit_real.py -- Leakage + benchmark-forensics audit on the REAL public pool.

Personas: Statistical Leakage Hunter (#11) + Benchmark Forensicist (#8).
Reads E:/NeurIPS/aims/data/<bench>/{benchmarks,subjects,items,response}.parquet,
prints every result to stdout (report 12 is written from that output).
Writes NO files. No network. Never inspects private data / feedback.

Run:  python src/paiec/analysis/audit_real.py [--section all|runtime|leak|dup|trials|meta|lobo]

Runtime mapping (reconstructed; see report 12 section 1 for provenance):
  subject = {k: str(subjects[k] or "").strip() for k in SUBJECT_FIELDS}   (8 keys; subject_id,
            display_name DROPPED; subject_id appears only as model_id in CSV, not in predict())
  item    = {item_content: items.content or "",
             item_features: (items.item_features or "").strip(),
             interactors: (response.interactors or "").strip(),   # response-level column!
             benchmark_id: opaque alias "benchmark_NNNNNN" (random, not the table value)}
  NOT visible: raw_item_id, grading_criterion, verifier, content_hash, asset_manifest,
               response_id, trial, test_condition, benchmarks.* metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from paiec import paths
DATA = str(paths.DATA)
BENCHES = ["matharena", "mmdocrag", "multi_swebench", "real_webagents",
           "researchcodebench", "swe_rebench"]
SUBJECT_FIELDS = ("normalized_name", "provider", "release_date", "access_date", "harness",
                  "reasoning_effort", "harness_version", "subject_features_extra")
ITEM_FIELDS = ("item_content", "item_features", "interactors", "benchmark_id")


def s(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip()


def alias(name: str) -> str:
    """Stand-in for the platform's random opaque alias (real ones are random, not derivable)."""
    return "benchmark_" + str(int(hashlib.sha256(name.encode()).hexdigest(), 16) % 900000 + 100000)


def load(b):
    t = {k: pd.read_parquet(f"{DATA}/{b}/{k}.parquet")
         for k in ("benchmarks", "subjects", "items", "response")}
    return t


ALL = {b: load(b) for b in BENCHES}


def runtime_subject(row) -> dict:
    return {k: s(row.get(k)) for k in SUBJECT_FIELDS}


def runtime_item(item_row, interactors, bench) -> dict:
    c = item_row.get("content")
    return {"item_content": "" if c is None else str(c),
            "item_features": s(item_row.get("item_features")),
            "interactors": s(interactors),
            "benchmark_id": alias(bench)}


def long_frame(b, binary_only=True):
    t = ALL[b]
    r = t["response"].merge(t["items"][["item_id", "content", "item_features", "content_hash"]],
                            on="item_id", how="left")
    r = r.merge(t["subjects"], on="subject_id", how="left")
    r["bench"] = b
    if binary_only:
        r = r[r.response.isin([0.0, 1.0])].copy()
    return r


# ----------------------------------------------------------------------------- 1 runtime
def sec_runtime():
    print("\n## 1. RUNTIME STRINGS (2 examples per benchmark, content truncated)")
    for b in BENCHES:
        t = ALL[b]
        r = t["response"].drop_duplicates(["subject_id", "item_id"]).sample(2, random_state=0)
        items = t["items"].set_index("item_id")
        subs = t["subjects"].set_index("subject_id")
        print(f"\n### {b}")
        for _, row in r.iterrows():
            subj = runtime_subject(subs.loc[row.subject_id].to_dict())
            it = runtime_item(items.loc[row.item_id].to_dict(), row.interactors, b)
            it["item_content"] = (it["item_content"][:160].replace("\n", " ") +
                                  f" ...[{len(it['item_content'])} chars]")
            print(json.dumps([subj, it], ensure_ascii=False))
            print(f"   (hidden: y={row.response}, trial={row.trial}, test_condition={row.test_condition})")
        # identity collisions at runtime
        sk = t["subjects"].apply(lambda r_: tuple(s(r_[k]) for k in SUBJECT_FIELDS), axis=1)
        coll = sk.value_counts()
        ik = t["items"].apply(lambda r_: (s(r_["content"]), s(r_["item_features"])), axis=1)
        icoll = ik.value_counts()
        print(f"   subject_ids={len(sk)} distinct runtime subject dicts={sk.nunique()} "
              f"(collisions: {int((coll > 1).sum())} dicts covering {int(coll[coll > 1].sum())} ids)")
        print(f"   item_ids={len(ik)} distinct runtime item dicts={ik.nunique()} "
              f"(collisions: {int((icoll > 1).sum())} dicts covering {int(icoll[icoll > 1].sum())} ids)")
        tc = t["response"].test_condition.nunique()
        print(f"   test_condition distinct={tc} (invisible at runtime -> collapsed into one input)")


# ----------------------------------------------------------------------------- 2 leakage scan
NAME_TOKENS = {
    "matharena": ["matharena", "math arena"],
    "mmdocrag": ["mmdocrag", "mmdoc"],
    "multi_swebench": ["multi-swe", "multi_swe", "swe-bench", "swebench"],
    "real_webagents": ["realevals", "real bench", "agi inc", "webagent"],
    "researchcodebench": ["researchcodebench", "research code bench", "research-code-bench"],
    "swe_rebench": ["swe-rebench", "swe_rebench", "rebench", "swe-bench", "swebench"],
}


def _ref_answer(g):
    try:
        return json.loads(g).get("reference_answer")
    except Exception:
        return None


def sec_leak():
    print("\n## 2. LEAKAGE SCAN OF RUNTIME-VISIBLE FIELDS")
    for b in BENCHES:
        t = ALL[b]
        it = t["items"].copy()
        it["c"] = it.content.fillna("")
        it["f"] = it.item_features.fillna("")
        cl = it.c.str.lower()
        fl = it.f.str.lower()
        out = {}
        # benchmark name tokens
        for tok in NAME_TOKENS[b] + [b]:
            n = int(cl.str.contains(re.escape(tok)).sum() + fl.str.contains(re.escape(tok)).sum())
            if n:
                out[f"name_tok[{tok}]"] = n
        # reference answer in content
        ref = it.grading_criterion.map(_ref_answer)
        def contains(row_c, a):
            if a is None:
                return None
            a = str(a).strip()
            if len(a) == 0:
                return None
            if len(a) <= 4:  # short numeric answers: token match only
                return bool(re.search(r"(?<![\d\w])" + re.escape(a) + r"(?![\d\w])", row_c))
            # long answers (patches/code): line-level containment of non-trivial lines
            lines = [ln.strip() for ln in a.splitlines() if len(ln.strip()) >= 25
                     and not ln.startswith(("diff --git", "index ", "---", "+++", "@@"))]
            lines = [ln.lstrip("+-").strip() for ln in lines]
            lines = [ln for ln in lines if len(ln) >= 25]
            if not lines:
                return a in row_c
            return float(np.mean([ln in row_c for ln in lines]))
        cont = [contains(c, a) for c, a in zip(it.c, ref)]
        vals = [v for v in cont if v is not None]
        it["ref_in"] = [np.nan if v is None else float(v) for v in cont]
        if vals:
            v = np.array(vals, dtype=float)
            out["ref_answer_in_content: n_with_ref"] = len(v)
            out["  frac items with any overlap"] = round(float((v > 0).mean()), 3)
            out["  frac items with >=50% answer lines in content"] = round(float((v >= .5).mean()), 3)
        # verifier spec fragments (e.g. FAIL_TO_PASS test names) in content
        def vt(vj):
            try:
                spec = json.loads(vj).get("spec", "")
                sp = json.loads(spec) if isinstance(spec, str) and spec.startswith("{") else {}
                tests = (sp.get("FAIL_TO_PASS") or []) if isinstance(sp, dict) else []
                return [x.split("::")[-1] for x in tests if isinstance(x, str)]
            except Exception:
                return []
        vts = it.verifier.map(vt)
        has = [bool(ts) for ts in vts]
        if any(has):
            hit = [np.mean([x in c for x in ts]) if ts else np.nan for ts, c in zip(vts, it.c)]
            out["FAIL_TO_PASS test names in content (mean frac)"] = round(float(np.nanmean(hit)), 3)
            it["f2p_in"] = hit
        # verifier/grading wording in content
        out["'reference_answer'/'gold' literal in content"] = int(
            cl.str.contains("reference_answer|gold_answer|ground truth answer").sum())
        print(f"\n### {b}: " + json.dumps(out))
        # does ref-in-content correlate with outcome?
        lf = long_frame(b)
        if len(lf) and it["ref_in"].notna().any():
            m = lf.merge(it[["item_id", "ref_in"]], on="item_id")
            m = m[m.ref_in.notna()]
            if m.ref_in.nunique() > 1:
                hi = m[m.ref_in >= .5].response.mean() if (m.ref_in >= .5).any() else float("nan")
                lo = m[m.ref_in < .5].response.mean()
                print(f"   P(y=1 | ref>=50% in content)={hi:.3f}  P(y=1 | <50%)={lo:.3f}  "
                      f"n_hi_items={(it.ref_in >= .5).sum()}")
        if "f2p_in" in it and len(lf):
            m = lf.merge(it[["item_id", "f2p_in"]], on="item_id")
            m = m[m.f2p_in.notna()]
            if len(m):
                print(f"   corr(y, frac FAIL_TO_PASS names in content) = "
                      f"{np.corrcoef(m.response, m.f2p_in)[0, 1]:.3f}")
        # item_features keys
        keys = Counter(k.split("=")[0] for f in it.f for k in f.split(";") if k)
        print(f"   item_features keys: {dict(keys)}")
        sfe = Counter(k.split("=")[0] for f in t["subjects"].subject_features_extra.fillna("")
                      for k in f.split(";") if k)
        print(f"   subject_features_extra keys: {dict(sfe)}")
    # researchcodebench cross-item answer exposure
    print("\n### researchcodebench cross-item exposure (answer of item A visible in content of item B)")
    it = ALL["researchcodebench"]["items"]
    paper = it.item_features.fillna("")
    ref = it.grading_criterion.map(_ref_answer).fillna("")
    exposed = []
    for i in range(len(it)):
        lines = [ln.strip() for ln in str(ref.iloc[i]).splitlines() if len(ln.strip()) >= 25]
        if not lines:
            exposed.append(np.nan); continue
        others = it.content[(paper == paper.iloc[i]) & (it.index != it.index[i])]
        if not len(others):
            exposed.append(0.0); continue
        exposed.append(max(np.mean([ln in c for ln in lines]) for c in others))
    exposed = np.array(exposed, dtype=float)
    print(f"   items={len(it)} mean max-frac of answer lines visible in a sibling item's content="
          f"{np.nanmean(exposed):.3f}; frac items with >=50% exposed={np.nanmean(exposed >= .5):.3f}")


# ----------------------------------------------------------------------------- 3 duplication
def _norm(t):
    t = (t or "").lower()
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def sec_dup():
    from sklearn.feature_extraction.text import TfidfVectorizer
    print("\n## 3. DUPLICATION")
    rows = []
    for b in BENCHES:
        it = ALL[b]["items"]
        for r in it.itertuples():
            rows.append((b, r.item_id, r.content or "", r.content_hash, r.raw_item_id, r.item_features))
    df = pd.DataFrame(rows, columns=["bench", "item_id", "content", "content_hash", "raw", "feat"])
    # exact
    for b, g in df.groupby("bench"):
        vc = g.content_hash.value_counts()
        vn = g.content.map(_norm).value_counts()
        print(f"   {b}: items={len(g)} dup content_hash groups={int((vc > 1).sum())} "
              f"(items in them={int(vc[vc > 1].sum())}); normalized-text dup groups={int((vn > 1).sum())}")
    xh = df.groupby("content_hash").bench.nunique()
    print(f"   cross-benchmark exact content_hash shared: {int((xh > 1).sum())}")
    df["raw_tail"] = df.raw.astype(str).str.replace(r"^.*::", "", regex=True)
    xr = df.groupby("raw").bench.nunique()
    print(f"   cross-benchmark identical raw_item_id: {int((xr > 1).sum())}")
    # repo-level overlap between the two SWE-style benchmarks
    def repo(x):
        m = re.match(r"(.+?)__(.+)-\d+$", str(x))
        return f"{m.group(1)}/{m.group(2)}" if m else None
    rm = set(df[df.bench == "multi_swebench"].raw.map(repo).dropna())
    rs = set(df[df.bench == "swe_rebench"].raw.map(repo).dropna())
    print(f"   repos: multi_swebench={len(rm)} swe_rebench={len(rs)} shared={len(rm & rs)} {sorted(rm & rs)[:10]}")
    # near-dup: char 5-gram tfidf on head 1500 + tail 500
    txt = df.content.map(lambda c: _norm(c[:1500] + " " + c[-500:]))
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
    X = HashingVectorizer(analyzer="char_wb", ngram_range=(5, 5), n_features=2 ** 18,
                          alternate_sign=False, norm=None, dtype=np.float32).transform(txt)
    X = TfidfTransformer(sublinear_tf=True).fit_transform(X).astype(np.float32)
    n = X.shape[0]
    benches = df.bench.values
    pairs_within = Counter(); pairs_cross = Counter(); examples = []
    best_cross = np.zeros(n, dtype=np.float32)
    for st in range(0, n, 1000):
        S = (X[st:st + 1000] @ X.T).toarray()
        for k in range(S.shape[0]):
            i = st + k
            S[k, i] = 0
            cross = benches != benches[i]
            best_cross[i] = S[k, cross].max() if cross.any() else 0
            js = np.where(S[k] >= 0.95)[0]
            for j in js:
                if j <= i:
                    continue
                if benches[j] == benches[i]:
                    pairs_within[benches[i]] += 1
                else:
                    pairs_cross[(benches[i], benches[j])] += 1
                    if len(examples) < 3:
                        examples.append((df.item_id[i], df.item_id[j], float(S[k, j])))
    print(f"   near-dup pairs (char5 tfidf cos>=0.95, head1500+tail500) within: {dict(pairs_within)}")
    print(f"   near-dup pairs cross-benchmark: {dict(pairs_cross)} ex={examples}")
    for b in BENCHES:
        m = benches == b
        print(f"   {b}: max cross-bench cos p50={np.median(best_cross[m]):.3f} p99={np.quantile(best_cross[m], .99):.3f} max={best_cross[m].max():.3f}")
    # outcome agreement on within-benchmark exact dups with distinct item_ids
    for b in BENCHES:
        it = ALL[b]["items"]
        vc = it.content_hash.value_counts()
        dh = set(vc[vc > 1].index)
        if not dh:
            continue
        lf = long_frame(b)
        lf = lf[lf.content_hash.isin(dh)]
        if not len(lf):
            continue
        pm = lf.groupby(["subject_id", "content_hash", "item_id"]).response.mean().reset_index()
        g = pm.groupby(["subject_id", "content_hash"]).response.agg(["count", "min", "max"])
        g = g[g["count"] > 1]
        if len(g):
            agree = float((g["min"] == g["max"]).mean())
            print(f"   {b}: dup-content item groups, same-subject outcome agreement across item_ids = {agree:.3f} (n={len(g)})")
        diff = it[it.content_hash.isin(dh)].groupby("content_hash").item_features.nunique()
        print(f"   {b}: dup-content groups whose item_features differ: {int((diff > 1).sum())}/{len(diff)}")
    # subjects
    print("\n### duplicated subjects (same normalized_name, multiple subject_ids)")
    allsub = pd.concat([ALL[b]["subjects"].assign(bench=b) for b in BENCHES])
    for b in BENCHES:
        sb = allsub[allsub.bench == b]
        g = sb.groupby(sb.normalized_name.fillna("<NULL>"))
        multi = [(k, v) for k, v in g if len(v) > 1]
        print(f"   {b}: subjects={len(sb)} normalized_names={sb.normalized_name.nunique()} null_name={sb.normalized_name.isna().sum()} "
              f"names_with_>1_id={len(multi)}")
        reasons = Counter()
        for k, v in multi:
            diffcols = [c for c in SUBJECT_FIELDS if v[c].fillna("").nunique() > 1]
            reasons[tuple(diffcols) or ("IDENTICAL_8_FIELDS",)] += 1
        print(f"      varying fields among duplicates: {dict(reasons)}")
        for k, v in multi[:2]:
            print("      e.g.", k, v[["display_name", "harness", "reasoning_effort", "access_date",
                                      "subject_features_extra"]].fillna("").values.tolist()[:4])
    xs = allsub.groupby("subject_id").bench.nunique()
    print(f"   subject_id shared across benchmarks: {int((xs > 1).sum())} ids "
          f"(max benches={int(xs.max())}); normalized_name in >=2 benches: "
          f"{int((allsub.groupby('normalized_name').bench.nunique() > 1).sum())}")


# ----------------------------------------------------------------------------- 4 trials
def sec_trials():
    print("\n## 4. REPEATED TRIALS / test_condition")
    for b in BENCHES:
        r = ALL[b]["response"].copy()
        binary = r.response.isin([0.0, 1.0])
        r["tc"] = r.test_condition.fillna("")
        key = ["subject_id", "item_id", "tc"]
        g = r[binary].groupby(key).response.agg(["count", "mean"])
        multi = g[g["count"] > 1]
        info = {"rows": len(r), "binary_rows": int(binary.sum()),
                "pairs(subj,item,tc)": len(g), "pairs_with_>1_trial": len(multi)}
        if len(multi):
            info["agreement(all trials equal)"] = round(float(((multi["mean"] == 0) | (multi["mean"] == 1)).mean()), 3)
            # pairwise agreement between two random trials
            p = multi["mean"]; c = multi["count"]
            pw = ((p * (p * c - 1) + (1 - p) * ((1 - p) * c - 1)) / (c - 1)).mean()
            info["pairwise trial agreement"] = round(float(pw), 3)
            # irreducible Brier floor for a predictor constant across trials of a runtime input
            w = multi["count"]
            info["Brier floor on multi-trial pairs"] = round(float((w * p * (1 - p)).sum() / w.sum()), 4)
            info["Brier floor all rows (single-trial=0)"] = round(float((w * p * (1 - p)).sum() / g["count"].sum()), 4)
        # collapse over test_condition (runtime cannot see it)
        if r.tc.nunique() > 1:
            g2 = r.groupby(["subject_id", "item_id"]).response.agg(["count", "mean", "var"])
            info["test_conditions"] = r.tc.value_counts().to_dict()
            info["within (subj,item) variance across conditions (fractional y)"] = round(float(g2["var"].mean()), 4)
            info["frac rows binary"] = round(float(binary.mean()), 3)
        print(f"   {b}: {json.dumps(info)}")
    # matharena fractional rows
    r = ALL["matharena"]["response"]
    fr = r[~r.response.isin([0.0, 1.0])]
    print(f"   matharena non-binary rows: {len(fr)} values={sorted(fr.response.round(3).unique())[:10]}")


# ----------------------------------------------------------------------------- 5 metadata
DATE_PATTERNS = {"YYYY-MM-DD": r"^\d{4}-\d{2}-\d{2}$", "YYYY-MM": r"^\d{4}-\d{2}$", "YYYY": r"^\d{4}$"}


def sec_meta():
    print("\n## 5. SUBJECT METADATA QUALITY")
    allsub = pd.concat([ALL[b]["subjects"].assign(bench=b) for b in BENCHES])
    miss = allsub.groupby("bench")[list(SUBJECT_FIELDS)].agg(lambda x: round(float((x.fillna("").astype(str).str.strip() == "").mean()), 2))
    print(miss.to_string())
    for col in ("release_date", "access_date"):
        v = allsub[col].dropna().astype(str)
        fmt = Counter(next((k for k, p in DATE_PATTERNS.items() if re.match(p, x)), "OTHER:" + x[:20]) for x in v)
        print(f"   {col} formats: {dict(fmt)}  range {v.min()}..{v.max()}")
    # release after access?
    both = allsub.dropna(subset=["release_date", "access_date"])
    print(f"   release_date > access_date: {int((both.release_date > both.access_date).sum())}/{len(both)}")
    # name normalization inconsistencies
    canon = allsub.normalized_name.dropna().map(lambda x: re.sub(r"[^a-z0-9]", "", x.lower()))
    grp = allsub.dropna(subset=["normalized_name"]).assign(canon=canon).groupby("canon").normalized_name.unique()
    bad = grp[grp.map(len) > 1]
    print(f"   canonical-key collisions (spelling variants of same name): {len(bad)} {bad.head(8).map(list).to_dict()}")
    nn = allsub.dropna(subset=["normalized_name"]).groupby("normalized_name")
    inc_rel = nn.release_date.nunique(); inc_prov = nn.provider.nunique()
    print(f"   same normalized_name with >1 release_date: {int((inc_rel > 1).sum())} {inc_rel[inc_rel > 1].index.tolist()[:8]}")
    print(f"   same normalized_name with >1 provider: {int((inc_prov > 1).sum())} {inc_prov[inc_prov > 1].index.tolist()[:8]}")
    # display_name -> normalized_name collapse (e.g. date-suffixed versions)
    coll = allsub.dropna(subset=["normalized_name"]).groupby("normalized_name").display_name.nunique()
    print(f"   normalized_names absorbing >1 display_name: {int((coll > 1).sum())}; e.g. "
          + str({k: allsub[allsub.normalized_name == k].display_name.unique().tolist()[:4]
                 for k in coll[coll > 1].index[:5]}))
    odd = allsub[allsub.normalized_name.fillna("").str.contains(r"^Agent:|\(|scaffold", regex=True)
                 | allsub.provider.fillna("").str.contains(r"\(")]
    print(f"   non-model 'subjects' (agent products / scaffold provider): {len(odd)} "
          f"{odd[['bench', 'normalized_name', 'provider']].head(6).values.tolist()}")
    print(f"   null normalized_name rows: {allsub.normalized_name.isna().groupby(allsub.bench).sum().to_dict()}")


# ----------------------------------------------------------------------------- 6 LOBO per-field
def _brier(y, p, w):
    return float(np.sum(w * (y - p) ** 2) / np.sum(w))


def sec_lobo():
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from scipy import sparse
    print("\n## 6. PER-FIELD MODELS (leave-benchmark-out + within-benchmark item-grouped CV)")
    use = ["matharena", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench"]
    lf = pd.concat([long_frame(b) for b in use], ignore_index=True)
    # aggregate to runtime inputs: (subject, item) with weight = n trials, y = mean
    agg = lf.groupby(["bench", "subject_id", "item_id"]).agg(
        y=("response", "mean"), w=("response", "size")).reset_index()
    items = pd.concat([ALL[b]["items"] for b in use]).drop_duplicates("item_id").set_index("item_id")
    subs = pd.concat([ALL[b]["subjects"] for b in use]).drop_duplicates("subject_id").set_index("subject_id")
    agg = agg.join(items[["content", "item_features"]], on="item_id").join(subs[list(SUBJECT_FIELDS)], on="subject_id")
    for c in SUBJECT_FIELDS + ("content", "item_features"):
        agg[c] = agg[c].map(s)
    agg["content_head"] = agg.content.map(lambda c: c[:3000] + " " + c[-1000:])

    def date_num(x):
        m = re.match(r"(\d{4})-(\d{2})", x)
        return (int(m.group(1)) + (int(m.group(2)) - 1) / 12) if m else np.nan

    def feats(name, tr, te):
        if name == "item_features":
            hv = HashingVectorizer(n_features=2 ** 14, token_pattern=r"[^;]+", alternate_sign=False)
            return hv.transform(tr.item_features), hv.transform(te.item_features)
        if name == "item_content_tfidf":
            tv = TfidfVectorizer(max_features=20000, min_df=3, sublinear_tf=True)
            ut = tr.drop_duplicates("item_id")
            tv.fit(ut.content_head)
            return tv.transform(tr.content_head), tv.transform(te.content_head)
        if name == "item_content_length":
            f = lambda d: sparse.csr_matrix(np.c_[np.log1p(d.content.str.len())])
            return f(tr), f(te)
        if name in ("release_date", "access_date"):
            def f(d):
                v = d[name].map(date_num).values
                mu = np.nanmean(tr[name].map(date_num).values) if np.isfinite(tr[name].map(date_num)).any() else 0
                miss = np.isnan(v)
                v = np.where(miss, mu, v) - 2025
                return sparse.csr_matrix(np.c_[v, miss.astype(float)])
            return f(tr), f(te)
        hv = HashingVectorizer(n_features=2 ** 14, token_pattern=r"[^;]+", alternate_sign=False)
        col = name
        return hv.transform(tr[col].replace("", "<EMPTY>")), hv.transform(te[col].replace("", "<EMPTY>"))

    fields = ["item_features", "item_content_tfidf", "item_content_length", "normalized_name",
              "provider", "release_date", "access_date", "harness", "reasoning_effort",
              "harness_version", "subject_features_extra"]
    print("   LOBO: train on 4 benchmarks, test on 5th. Brier(const=train mean), Brier(oracle=test mean, "
          "unattainable at B0), Brier(field), AUC(field). Weighted by #trials.")
    res = []
    for hold in use:
        tr = agg[agg.bench != hold]; te = agg[agg.bench == hold]
        ytr = tr.y.values; yte = te.y.values; wte = te.w.values
        b_const = _brier(yte, np.average(ytr, weights=tr.w), wte)
        b_orac = _brier(yte, np.average(yte, weights=wte), wte)
        for f in fields:
            Xtr, Xte = feats(f, tr, te)
            # fractional y -> duplicate rows with weights
            Xd = sparse.vstack([Xtr, Xtr]); yd = np.r_[np.ones(len(ytr)), np.zeros(len(ytr))]
            wd = np.r_[tr.w.values * ytr, tr.w.values * (1 - ytr)]
            clf = LogisticRegression(C=1.0, max_iter=300).fit(Xd, yd, sample_weight=wd)
            p = clf.predict_proba(Xte)[:, 1]
            yb = (yte >= .5).astype(int)
            auc = roc_auc_score(yb, p, sample_weight=wte) if 0 < yb.mean() < 1 and np.ptp(p) > 0 else np.nan
            res.append((hold, f, round(b_const, 4), round(b_orac, 4), round(_brier(yte, p, wte), 4), round(float(auc), 3)))
    R = pd.DataFrame(res, columns=["holdout", "field", "B_const", "B_oracle", "B_field", "AUC"])
    R["beats_oracle"] = R.B_field < R.B_oracle - 0.005
    print(R.to_string(index=False))
    print("   FLAG rows (field beats the unattainable oracle test-mean by >0.005):")
    print(R[R.beats_oracle].to_string(index=False) if R.beats_oracle.any() else "   none")

    # within-benchmark, item-grouped 5-fold: does item-side text trivially predict y beyond the benchmark mean?
    print("\n   Within-benchmark item-grouped 5-fold CV (item fields only) — vs benchmark mean")
    rng = np.random.RandomState(0)
    for b in use:
        d = agg[agg.bench == b].reset_index(drop=True)
        uit = d.item_id.unique(); fold = dict(zip(uit, rng.randint(0, 5, len(uit))))
        fo = d.item_id.map(fold).values
        for f in ["item_features", "item_content_tfidf", "item_content_length", "ITEM_ID_ORACLE"]:
            P = np.zeros(len(d))
            for k in range(5):
                tr = d[fo != k]; te = d[fo == k]
                if f == "ITEM_ID_ORACLE":
                    # upper bound: leave-one-subject-out item mean on test items (uses other subjects' labels)
                    gs = te.groupby("item_id")
                    tot = gs.apply(lambda g: (g.y * g.w).sum()); cnt = gs.w.sum()
                    P[fo == k] = ((te.item_id.map(tot) - te.y * te.w) /
                                  (te.item_id.map(cnt) - te.w).clip(lower=1)).values
                    continue
                if f == "item_features" and d.item_features.nunique() < 2:
                    P[fo == k] = np.average(tr.y, weights=tr.w); continue
                Xtr, Xte = feats(f, tr, te)
                Xd = sparse.vstack([Xtr, Xtr]); yd = np.r_[np.ones(len(tr)), np.zeros(len(tr))]
                wd = np.r_[tr.w * tr.y, tr.w * (1 - tr.y)]
                P[fo == k] = LogisticRegression(C=1.0, max_iter=300).fit(Xd, yd, sample_weight=wd).predict_proba(Xte)[:, 1]
            bm = _brier(d.y.values, np.average(d.y, weights=d.w), d.w.values)
            yb = (d.y.values >= .5).astype(int)
            auc = roc_auc_score(yb, P, sample_weight=d.w) if 0 < yb.mean() < 1 and np.ptp(P) > 0 else np.nan
            print(f"   {b:18s} {f:22s} B_mean={bm:.4f} B_field={_brier(d.y.values, P, d.w.values):.4f} AUC={auc:.3f}")


def sec_groupcv():
    """Does item-grouped CV leak through duplicate content? Compare grouping by item_id vs by
    runtime item dict (content+features) vs content_hash."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from scipy import sparse
    print("\n## 6b. CV GROUPING LEAK (item_content tfidf, within benchmark, 5-fold)")
    for b in ["matharena", "multi_swebench", "swe_rebench"]:
        lf = long_frame(b)
        d = lf.groupby(["subject_id", "item_id"]).agg(y=("response", "mean"), w=("response", "size")).reset_index()
        it = ALL[b]["items"].set_index("item_id")
        d["content"] = d.item_id.map(it.content).fillna("")
        d["ch"] = d.item_id.map(it.content_hash)
        d["ch_f"] = d.ch + "|" + d.item_id.map(it.item_features).fillna("")
        d["txt"] = d.content.map(lambda c: c[:3000] + " " + c[-1000:])
        for gcol in ["item_id", "ch_f", "ch"]:
            rng = np.random.RandomState(0)
            ug = d[gcol].unique(); fold = dict(zip(ug, rng.randint(0, 5, len(ug))))
            fo = d[gcol].map(fold).values
            P = np.zeros(len(d))
            for k in range(5):
                tr = d[fo != k]; te = d[fo == k]
                tv = TfidfVectorizer(max_features=20000, min_df=3, sublinear_tf=True).fit(tr.drop_duplicates("item_id").txt)
                Xtr = tv.transform(tr.txt); Xte = tv.transform(te.txt)
                Xd = sparse.vstack([Xtr, Xtr]); yd = np.r_[np.ones(len(tr)), np.zeros(len(tr))]
                wd = np.r_[tr.w * tr.y, tr.w * (1 - tr.y)]
                P[fo == k] = LogisticRegression(max_iter=300).fit(Xd, yd, sample_weight=wd).predict_proba(Xte)[:, 1]
            yb = (d.y >= .5).astype(int)
            print(f"   {b:15s} group_by={gcol:8s} Brier={_brier(d.y.values, P, d.w.values):.4f} "
                  f"AUC={roc_auc_score(yb, P, sample_weight=d.w):.3f}")


SECTIONS = {"runtime": sec_runtime, "leak": sec_leak, "dup": sec_dup, "trials": sec_trials,
            "meta": sec_meta, "lobo": sec_lobo, "groupcv": sec_groupcv}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", default="all")
    a = ap.parse_args()
    todo = SECTIONS if a.section == "all" else {k: SECTIONS[k] for k in a.section.split(",")}
    for k, fn in todo.items():
        fn()
        sys.stdout.flush()
