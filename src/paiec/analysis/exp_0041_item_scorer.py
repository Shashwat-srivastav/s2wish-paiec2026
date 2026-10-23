"""EXP-0041 -- can an offline item-demand scorer provide transferable item ranking?

Locked procedure: docs/preregistrations/2026-10-14_comprehensive_research_preregistration.md §2 (EXP-0041 row);
scope: §9 addendum (runtime-legal structural-feature arm only; ADeLe annotator arm and LLM-estimated human
minutes are blocked; Jev is a separately labelled, non-deployable offline upper bound).

Plan written BEFORE running (operationalisation of the locked row; no procedure change)
-------------------------------------------------------------------------------------
Why: reports 13/20/25 suggest content does not transfer across benchmarks (LOBO triv rho ~ -0.01 against
in-sample beta, with an illegal within-hidden-benchmark z-scoring). EXP-0041 re-tests the structural arm under
runtime-legal preprocessing, content-grouped folds, inner-fold scorer choice and a leave-subject-out target.

Procedure
  items     every item of the five public benchmarks (rev bc8204d811); content seen by predict() only
            (item `content`; `item_features` is excluded because its keys/values identify the benchmark).
  features  structural: length (chars/words/lines), code-block count, inline-code count, step count
            (numbered lines, bullet lines, 'step' words), headings, table rows, multiple-choice options,
            URLs, file paths, LaTeX, numbers, digit/upper/code-char fractions, question marks, traceback flag,
            format flags (has code block / LaTeX / URL / table / options / ends with '?').
            Standardised with TRAINING-pool statistics only (never the hidden benchmark's pool).
  folds     outer LOBO over multi_swebench, real_webagents, researchcodebench, matharena; swe_rebench is
            training-only.  Content groups = sha1(whitespace-normalised content); any training item whose
            content group occurs in the held-out benchmark is dropped (same rule in inner folds).
  training target  per-benchmark Rasch beta (item_repr.rasch, all subjects), centred within benchmark.
  scorer choice    inside training folds only: inner leave-one-training-benchmark-out over the 4 training
            benchmarks; mean inner Spearman picks among {ridge alpha in 0.1..1e4 on training-benchmark-centred
            features, HistGradientBoosting (depth 3, 200 it, lr .05, leaf 50), zero-fit log-length}.
            Benchmark-balanced sample weights.  Chosen scorer refit on all training benchmarks.
  outer target (leave-subject-out item difficulty)  subjects of the held-out benchmark are split into K=5
            groups by sha1(subject_id); beta^(-k) = Rasch beta fitted without group k.
            PRIMARY rho_fold = mean_k Spearman(score, beta^(-k)).
            Secondary: mean_k Spearman(score, d_k) where d_k = empirical logit difficulty among the left-out
            subjects of group k (fully out-of-sample subjects); ceiling = mean_k Spearman(beta^(-k), d_k).
  uncertainty  cluster bootstrap over held-out content groups (1000 reps) for the primary rho.
  control   permutation null: training targets permuted within each training benchmark, whole pipeline
            (selection + fit) rerun, N_PERM=20; per-fold null mean/sd and empirical p.
  upper bound  zero-fit Jev `jev_difficulty` (offline cache, no API) against the same targets; labelled
            NON-DEPLOYABLE.
Outcome -> action (kill rule from the registration / brief)
  primary rho < 0.30 on >= 3/4 folds  -> close the item-content (structural) route; EXP-0055 gate stays shut.
  primary rho >= 0.30 on >= 3/4 folds -> scorer continues to an ALC test; EXP-0055 gate opens.
  otherwise (2/4)                      -> inconclusive; route not closed, EXP-0055 gate stays shut.

Run:  python scripts/run_slot.py -- python -m paiec.analysis.exp_0041_item_scorer
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from paiec import paths
from paiec.analysis.item_repr import rasch

BENCHES = ["matharena", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench"]
OUTER = ["multi_swebench", "real_webagents", "researchcodebench", "matharena"]
ALPHAS = [0.1, 1, 10, 100, 1000, 1e4]
K_SUBJ = 5
N_BOOT = 1000
N_PERM = 20
RHO_BAR = 0.30
OUT = paths.RUNS / "preregistration" / "exp_0041_item_scorer_2026-10-15.json"
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ------------------------------------------------------------------ features (runtime-legal, content only)
FILE_RE = re.compile(r"[\w/\-]+\.(?:py|js|ts|tsx|jsx|java|go|rs|c|cc|cpp|h|hpp|rb|php|json|ya?ml|toml|md|txt|html|css|sh)\b")


def structural(texts: pd.Series) -> pd.DataFrame:
    t = texts.fillna("").astype(str)
    n = t.str.len() + 1
    fences = t.str.count("```")
    numbered = t.str.count(r"(?m)^\s*\d+[.)]\s")
    bullets = t.str.count(r"(?m)^\s*[-*•]\s")
    table = t.str.count(r"(?m)^\s*\|.*\|\s*$")
    options = t.str.count(r"(?m)^\s*\(?[A-E][).:]\s")
    latex = t.str.count(r"\$|\\[a-zA-Z]+")
    urls = t.str.count(r"https?://")
    return pd.DataFrame({
        "log_chars": np.log1p(t.str.len()),
        "log_words": np.log1p(t.str.split().str.len()),
        "log_lines": np.log1p(t.str.count("\n")),
        "code_blocks": np.log1p(fences // 2),
        "inline_code": np.log1p(t.str.count(r"`[^`\n]+`")),
        "step_numbered": np.log1p(numbered),
        "step_bullets": np.log1p(bullets),
        "step_words": np.log1p(t.str.count(r"(?i)\bstep\b")),
        "headings": np.log1p(t.str.count(r"(?m)^#{1,6}\s")),
        "table_rows": np.log1p(table),
        "mc_options": np.log1p(options),
        "n_urls": np.log1p(urls),
        "n_files": np.log1p(t.apply(lambda s: len(set(FILE_RE.findall(s))))),
        "latex": np.log1p(latex),
        "n_numbers": np.log1p(t.str.count(r"\b\d+(?:\.\d+)?\b")),
        "digit_frac": t.str.count(r"\d") / n,
        "upper_frac": t.str.count(r"[A-Z]") / n,
        "code_chars": t.str.count(r"[{};()=<>\[\]]") / n,
        "questions": np.log1p(t.str.count(r"\?")),
        "traceback": t.str.contains(r"Traceback|Error|Exception", regex=True).astype(float),
        "f_code_block": (fences >= 2).astype(float),
        "f_latex": (latex > 0).astype(float),
        "f_url": (urls > 0).astype(float),
        "f_table": (table >= 2).astype(float),
        "f_options": (options >= 2).astype(float),
        "f_ends_q": t.str.rstrip().str.endswith("?").astype(float),
    })


def cgroup(s: str) -> str:
    return hashlib.sha1(" ".join(str(s or "").split()).encode("utf-8", "replace")).hexdigest()


def subj_group(sid: str) -> int:
    return int(hashlib.sha1(str(sid).encode()).hexdigest(), 16) % K_SUBJ


# ------------------------------------------------------------------ data
def load():
    items, targets = [], {}
    jev = {}
    for b in BENCHES:
        it = pd.read_parquet(paths.DATA / b / "items.parquet", columns=["item_id", "content"])
        r = pd.read_parquet(paths.DATA / b / "response.parquet", columns=["subject_id", "item_id", "response"])
        r = r[r.item_id.isin(it.item_id)].dropna(subset=["response"])
        r["response"] = (r.response > 0.5).astype(float)
        _, be, _ = rasch(r)
        it = it[it.item_id.isin(be.index)].copy()
        it["bench"] = b
        it["beta"] = be.reindex(it.item_id).to_numpy()
        it["beta"] -= it.beta.mean()
        it["grp"] = it.content.map(cgroup)
        if b in OUTER:
            sg = r.subject_id.map(subj_group)
            tk = {"beta_loso": [], "d_out": [], "n_subj": [int((r.subject_id.nunique()))]}
            for k in range(K_SUBJ):
                rin, rout = r[sg != k], r[sg == k]
                if rout.subject_id.nunique() == 0:
                    continue
                _, bk, _ = rasch(rin)
                tk["beta_loso"].append(bk.reindex(it.item_id).to_numpy())
                agg = rout.groupby("item_id").response.agg(["sum", "count"]).reindex(it.item_id)
                d = -np.log((agg["sum"] + .5) / (agg["count"] - agg["sum"] + .5)).to_numpy()
                tk["d_out"].append(d)
            targets[b] = tk
            jo = paths.RUNS / "jev" / f"jev_out_{b}.parquet"
            if jo.exists():
                j = pd.read_parquet(jo, columns=["item_id", "jev_difficulty"]).drop_duplicates("item_id")
                jev[b] = j.set_index("item_id").jev_difficulty.reindex(it.item_id).to_numpy()
        log(b, len(it), "items", r.subject_id.nunique(), "subjects")
        del r
        items.append(it)
    items = pd.concat(items, ignore_index=True)
    X = structural(items.content)
    return items, X, targets, jev


# ------------------------------------------------------------------ scorers
def bal_w(bench: np.ndarray) -> np.ndarray:
    _, inv, cnt = np.unique(bench, return_inverse=True, return_counts=True)
    return (1.0 / cnt)[inv] * len(bench) / len(cnt)


def fit_scorer(name, Xtr, ytr, btr):
    """Return a function X -> score (higher = harder). Training statistics only."""
    if name == "loglen":
        return lambda X: X[:, 0]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Z = (Xtr - mu) / sd
    w = bal_w(btr)
    if name.startswith("ridge"):
        Zc = Z.copy()
        for b in np.unique(btr):  # centre within TRAINING benchmarks (offsets are not the target)
            m = btr == b
            Zc[m] -= Zc[m].mean(0)
        m = Ridge(alpha=float(name.split("_")[1])).fit(Zc, ytr, sample_weight=w)
        return lambda X: ((X - mu) / sd) @ m.coef_
    if name == "hgb":
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=50,
                                          random_state=0).fit(Z, ytr, sample_weight=w)
        return lambda X: m.predict((X - mu) / sd)
    raise ValueError(name)


CANDS = [f"ridge_{a:g}" for a in ALPHAS] + ["hgb", "loglen"]


def sp(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5 or np.std(a[m]) == 0:
        return 0.0
    return float(spearmanr(a[m], b[m]).correlation)


def select_and_fit(items, X, y, train_benches, held_grps):
    """Inner leave-one-training-benchmark-out choice, then refit. Content groups of the outer benchmark are
    excluded from training."""
    keep = items.bench.isin(train_benches).to_numpy() & ~items.grp.isin(held_grps).to_numpy()
    inner = {c: [] for c in CANDS}
    for ib in train_benches:
        te = keep & (items.bench.to_numpy() == ib)
        tr = keep & (items.bench.to_numpy() != ib) & ~items.grp.isin(set(items.grp[te])).to_numpy()
        for c in CANDS:
            f = fit_scorer(c, X[tr], y[tr], items.bench.to_numpy()[tr])
            inner[c].append(sp(f(X[te]), y[te]))
    inner = {c: float(np.mean(v)) for c, v in inner.items()}
    best = max(CANDS, key=lambda c: inner[c])
    fits = {c: fit_scorer(c, X[keep], y[keep], items.bench.to_numpy()[keep]) for c in CANDS}
    return best, inner, fits, int(keep.sum())


def rho_targets(score, tk):
    return (float(np.mean([sp(score, t) for t in tk["beta_loso"]])),
            float(np.mean([sp(score, t) for t in tk["d_out"]])))


def boot_ci(score, tk, grp, rng):
    codes, inv = np.unique(grp, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(codes))]
    out = []
    for _ in range(N_BOOT):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(codes), len(codes))])
        out.append(np.mean([sp(score[idx], t[idx]) for t in tk["beta_loso"]]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def main():
    items, Xdf, targets, jev = load()
    X = Xdf.to_numpy(float)
    y = items.beta.to_numpy()
    rng = np.random.default_rng(0)
    res = {"experiment": "EXP-0041", "data_revision": (paths.DATA / "REVISION").read_text().strip()[:10],
           "features": list(Xdf.columns), "candidates": CANDS, "K_subject_groups": K_SUBJ,
           "n_boot": N_BOOT, "n_perm": N_PERM, "folds": {}}
    for ob in OUTER:
        te = (items.bench == ob).to_numpy()
        held_grps = set(items.grp[te])
        train_b = [b for b in BENCHES if b != ob]
        n_dup = int((items.bench.isin(train_b) & items.grp.isin(held_grps)).sum())
        best, inner, fits, ntr = select_and_fit(items, X, y, train_b, held_grps)
        tk = targets[ob]
        score = fits[best](X[te])
        prim, sec = rho_targets(score, tk)
        ci = boot_ci(score, tk, items.grp[te].to_numpy(), rng)
        full = sp(score, y[te])
        ceiling = float(np.mean([sp(b_, d) for b_, d in zip(tk["beta_loso"], tk["d_out"])]))
        diag = {c: dict(zip(["rho_loso", "rho_out_subj"], rho_targets(fits[c](X[te]), tk))) for c in CANDS}
        # permutation null: whole pipeline on targets permuted within each training benchmark
        null = []
        for p in range(N_PERM):
            yp = y.copy()
            for b in train_b:
                m = np.flatnonzero((items.bench == b).to_numpy())
                yp[m] = yp[rng.permutation(m)]
            bp, _, fp, _ = select_and_fit(items, X, yp, train_b, held_grps)
            null.append(rho_targets(fp[bp](X[te]), tk)[0])
        null = np.array(null)
        jv = None
        if ob in jev:
            jv = dict(zip(["rho_loso", "rho_out_subj"], rho_targets(jev[ob], tk)))
            jv["ci_loso"] = boot_ci(jev[ob], tk, items.grp[te].to_numpy(), rng)
            jv["n_scored"] = int(np.isfinite(jev[ob]).sum())
        res["folds"][ob] = {
            "n_items": int(te.sum()), "n_content_groups": len(held_grps), "n_subjects": tk["n_subj"][0],
            "n_train_items": ntr, "n_train_dropped_dup_content": n_dup,
            "selected": best, "inner_mean_rho": inner,
            "rho_loso_primary": prim, "rho_loso_ci95": ci, "rho_out_subject": sec, "rho_full_beta": full,
            "target_ceiling_loso_vs_out": ceiling,
            "perm_null_mean": float(null.mean()), "perm_null_sd": float(null.std(ddof=1)),
            "perm_p_one_sided": float((1 + (null >= prim).sum()) / (1 + len(null))),
            "all_candidates_outer_diag": diag, "jev_upper_bound_NON_DEPLOYABLE": jv,
        }
        log(ob, best, f"rho={prim:+.3f} CI={ci} out={sec:+.3f} null={null.mean():+.3f}+-{null.std():.3f}",
            "jev", None if jv is None else round(jv["rho_loso"], 3))
    rhos = [res["folds"][b]["rho_loso_primary"] for b in OUTER]
    n_below = sum(r < RHO_BAR for r in rhos)
    n_above = sum(r >= RHO_BAR for r in rhos)
    t_mean, t_sd = float(np.mean(rhos)), float(np.std(rhos, ddof=1))
    res["summary"] = {
        "rho_per_fold": dict(zip(OUTER, rhos)), "mean_rho": t_mean,
        "fold_t95": [t_mean - 3.182 * t_sd / 2, t_mean + 3.182 * t_sd / 2],
        "n_folds_below_0.30": n_below, "n_folds_at_or_above_0.30": n_above,
        "kill_rule_fires": n_below >= 3, "exp0055_gate_open": n_above >= 3,
        "decision": ("CLOSE structural item-content route" if n_below >= 3 else
                     "CONTINUE to ALC test" if n_above >= 3 else "INCONCLUSIVE"),
    }
    try:
        import psutil
        res["peak_rss_mb"] = psutil.Process().memory_info().peak_wset / 2**20
    except Exception:
        res["peak_rss_mb"] = None
    res["wall_s"] = time.time() - T0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    log("summary", json.dumps(res["summary"]))
    log("wrote", OUT)


if __name__ == "__main__":
    sys.exit(main())
