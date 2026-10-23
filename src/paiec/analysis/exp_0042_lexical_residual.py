"""EXP-0042 (A2): online lexical residual smoothing over already-revealed items, vs frozen v1_3.

Locked procedure: docs/preregistrations/2026-10-14_comprehensive_research_preregistration.md §2 (EXP-0042 row).

Method (frozen before any evaluation; nothing below was changed after seeing a score)
------------------------------------------------------------------------------------
* Base = frozen v1_3 reproduced as ``v1_2.cfg_of(facets="all", guard=0)`` (contract §2); p_t = base target
  probability, q_n = pooled pair probability (``v1_1.pooled_pair``) at the same budget.
* Representation: TF-IDF fitted on the *training* benchmarks of the outer fold only (four non-held-out benchmarks
  incl. swe_rebench): lowercase, token ``[a-z0-9_]{2,}``, uni+bi-grams, min_df=2, max_features=20000,
  sublinear tf, l2 rows. The held-out benchmark's items are only *transformed* with that frozen vocabulary/idf.
* At budget b, target t of pair n sees only the first b revealed items j of the *same pair* (never the hidden pool,
  never other pairs, even in R1). Exact same-key neighbours are excluded (the v1_2 item posterior owns them).
  s_tj = cosine(t, j) >= 0;  w_tj = s_tj^gamma, optionally top-k by weight;  r_j = y_j - q_n;
  rhat_t = sum_j w_tj r_j / (m0 + sum_j w_tj);   logit p'_t = logit p_t + a * rhat_t;  clip [0.005, 0.995].
* Frozen hyperparameter grid (selected *inside training folds only*, per outer fold):
  gamma in {1, 4, 16}, m0 in {0.25, 1, 4}, a in {0.5, 1, 2, 4}, k in {3, all}  (72 cells).
  Selection: run R2 seeds 0-9 on each of the four training benchmarks of the outer fold with the outer fold's
  hyperparameters H_fold(outer) (so the outer benchmark never enters), minimise the unweighted mean over training
  benchmarks of the pair-weighted mean ALC delta vs base. a=0 is not in the grid (the test is of the method).
* Arms on the outer fold: baseline; lexical (frozen); shuffled-neighbour control (same frozen hyperparameters, TF-IDF
  rows permuted across keys of the held-out benchmark, rng seed 0); descriptive: lexical without near-duplicates
  (s>0.95 dropped); descriptive (R2 only): the experimental v1_4 kernel (token-overlap, strength 0.2) on the same base.
* Signal quality for the EXP-0055 gate: Spearman rho between rhat_t and leave-pair-out empirical item easiness of the
  target key (logit of smoothed mean correctness over all *other* pairs of the held-out benchmark), pooled over
  targets with sum_j w_tj > 0, per fold, at budgets 3/7/15/31 (R2). Also rho vs realised residual y - q.

Run:  python scripts/run_slot.py -- python -m paiec.analysis.exp_0042_lexical_residual
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy import stats as SS
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from paiec import paths
from paiec.evaluation import episodes as E
from paiec.models import v1_1 as Base
from paiec.models import v1_2 as V

OUTPUT = paths.RUNS / "preregistration" / "exp_0042_lexical_residual_2026-10-15.json"
SEEDS = {"R2": list(range(10)), "R1": list(range(100))}
TRAIN_ONLY = ["swe_rebench"]
GRID = [dict(gamma=g, m0=m, a=a, k=k)
        for g, m, a, k in itertools.product((1, 4, 16), (0.25, 1.0, 4.0), (0.5, 1.0, 2.0, 4.0), (3, None))]
TFIDF = dict(lowercase=True, token_pattern=r"(?u)\b[a-z0-9_]{2,}\b", ngram_range=(1, 2), min_df=2,
             max_features=20000, sublinear_tf=True, dtype=np.float32)
NEAR_DUP = 0.95
MAXL = 31


def log(*a):
    print(*a, flush=True)


def base_cfg():
    return V.cfg_of(facets="all", guard=0)


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------
def fit_vectorizer(outer):
    texts = []
    for T in [b for b in E.BENCHES if b != outer]:
        texts += [str(c) for c in E.compact(T)["key_content"]]
    vec = TfidfVectorizer(**TFIDF)
    vec.fit(texts)
    return vec


def key_matrix(vec, D):
    return vec.transform([str(c) for c in D["key_content"]]).tocsr()


# ---------------------------------------------------------------------------
# Per-(benchmark, regime) preparation: base predictions + similarity tensor
# ---------------------------------------------------------------------------
class Prep:
    def __init__(self, B, D, regime, seeds, H, X, v14=False):
        self.B = B
        cfg = base_cfg()
        ep = E.build(D, regime, seeds)
        self.ep = ep
        fe = V.fe_for(B, D, cfg)
        self.p, self.q = {}, {}
        for b in E.BUDGETS:
            self.p[b] = V.predict(ep, H, b, cfg, fe)
            li, lk, ly = ep.labels(b)
            self.q[b] = Base.pooled_pair(ep, H, li, ly, cfg)[0]
        T = len(ep.t_key)
        self.lab_k = np.full((ep.n, MAXL), -1, np.int64)
        self.lab_y = np.zeros((ep.n, MAXL), np.float32)
        for n, a in enumerate(ep.acq):
            m = min(MAXL, len(a))
            self.lab_k[n, :m] = a[:m, 0]
            self.lab_y[n, :m] = a[:m, 1]
        self.m = (self.lab_k >= 0).sum(1)
        self.X = X
        self.S = self._sims(X)
        self.exact = self.lab_k[ep.t_inst] == ep.t_key[:, None]
        self.S[self.exact] = 0.0
        self.base_alc = self._score(self.p)
        self.v14_S = None
        if v14:
            cv = CountVectorizer(lowercase=True, token_pattern=r"[a-z0-9]{2,}", binary=True, dtype=np.float32)
            Bm = cv.fit_transform([str(c) for c in D["key_content"]]).tocsr()  # runtime: token sets of seen items
            inter = self._sims(Bm)
            sz = np.asarray(Bm.sum(1)).ravel()
            szt = sz[ep.t_key][:, None]
            szl = np.where(self.lab_k[ep.t_inst] >= 0, sz[np.maximum(self.lab_k[ep.t_inst], 0)], 0)
            s = np.where((inter >= 2) & (szt > 0) & (szl > 0), inter / np.sqrt(np.maximum(szt * szl, 1)), 0.0)
            s[self.exact.any(1)] = 0.0  # original skips a target whose key is labelled
            self.v14_S = np.where(s >= 0.20, s, 0.0).astype(np.float32)
        assert T == self.S.shape[0]

    def _sims(self, X):
        ep = self.ep
        out = np.zeros((len(ep.t_key), MAXL), np.float32)
        bounds = np.searchsorted(ep.t_inst, np.arange(ep.n + 1))  # targets of an instance are contiguous
        for n in range(ep.n):
            a, z, m = bounds[n], bounds[n + 1], int(self.m[n])
            if z > a and m:
                out[a:z, :m] = (X[ep.t_key[a:z]] @ X[self.lab_k[n, :m]].T).toarray()
        return out

    def _score(self, probs):
        br, _ = self.ep.score(probs)
        return br @ E.W, br

    def residuals(self, b):
        ep = self.ep
        r = self.lab_y - self.q[b][:, None].astype(np.float32)
        return r[ep.t_inst]

    def rhat(self, b, hp, S=None, R=None):
        ep = self.ep
        S = self.S if S is None else S
        cols = np.arange(MAXL)[None, :] < np.minimum(b, self.m)[ep.t_inst][:, None]
        Wt = np.where(cols & (S > 0), np.power(np.maximum(S, 0), hp["gamma"]), 0.0).astype(np.float32)
        k = hp["k"]
        if k is not None and b > k:
            thr = -np.partition(-Wt, k - 1, axis=1)[:, k - 1:k]
            Wt = np.where(Wt >= thr, Wt, 0.0)
        R = self.residuals(b) if R is None else R
        ws = Wt.sum(1)
        return (Wt * R).sum(1) / (hp["m0"] + ws), ws

    def arm_probs(self, hp, S=None):
        out = {}
        for b in E.BUDGETS:
            if b == 0:
                out[b] = self.p[b]
                continue
            rh, _ = self.rhat(b, hp, S)
            lp = np.log(self.p[b] / (1 - self.p[b]))
            out[b] = np.clip(sig(lp + hp["a"] * rh), 0.005, 0.995)
        return out

    def v14_probs(self, strength=0.20):
        ep = self.ep
        out = {}
        for b in E.BUDGETS:
            if b == 0:
                out[b] = self.p[b]
                continue
            cols = np.arange(MAXL)[None, :] < np.minimum(b, self.m)[ep.t_inst][:, None]
            Wt = np.where(cols, self.v14_S ** 2, 0.0)
            ws = Wt.sum(1)
            r = np.where(ws > 0, (Wt * self.residuals(b)).sum(1) / np.maximum(ws, 1e-30), 0.0)
            gain = strength * (1.0 - np.exp(-ws))
            out[b] = np.where(ws > 0, np.clip(self.p[b] + gain * r, 0.005, 0.995), self.p[b])
        return out

    def result(self, probs):
        alc, br = self._score(probs)
        return dict(br=br, ec=np.zeros_like(br), alc=alc, pair=self.ep.inst_pair.copy())

    def delta_alc(self, probs):
        alc, _ = self._score(probs)
        return float(Base.pair_mean(alc - self.base_alc[0], self.ep.inst_pair).mean())


def _peak_mb():
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t)] +                        [(f"f{i}", ctypes.c_size_t) for i in range(6)]
        c = PMC(); c.cb = ctypes.sizeof(PMC)
        ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.PeakWorkingSetSize / 2**20
    except Exception:
        return float("nan")


def shuffled_matrix(X, seed=0):
    perm = np.random.default_rng(seed).permutation(X.shape[0])
    return X[perm]


def easiness_lpo(D, ep):
    """Leave-pair-out logit easiness of each (instance, target) key, from all other pairs of the benchmark."""
    K = len(D["key_jev"])
    key = D["item_key"][D["row_item"]].astype(np.int64)
    y = D["row_y"].astype(float)
    Sk = np.bincount(key, y, K); Nk = np.bincount(key, minlength=K).astype(float)
    pk = D["row_pair"].astype(np.int64) * K + key
    u, inv = np.unique(pk, return_inverse=True)
    Spk = np.bincount(inv, y); Npk = np.bincount(inv).astype(float)
    tp = ep.inst_pair[ep.t_inst].astype(np.int64) * K + ep.t_key
    pos = np.minimum(np.searchsorted(u, tp), len(u) - 1)
    hit = u[pos] == tp
    s = Sk[ep.t_key] - np.where(hit, Spk[pos], 0); n = Nk[ep.t_key] - np.where(hit, Npk[pos], 0)
    e = np.log((s + 0.5) / (n - s + 0.5))
    return e, n > 0


def rho_diag(prep, D, hp, S=None):
    ep = prep.ep
    e, ok = easiness_lpo(D, ep)
    out = {}
    for b in (3, 7, 15, 31):
        rh, ws = prep.rhat(b, hp, S)
        m = ok & (ws > 0)
        r_e = SS.spearmanr(rh[m], e[m]).statistic if m.sum() > 10 else float("nan")
        rr = ep.r_y - prep.q[b][ep.r_inst]
        mr = (ws > 0)[ep.r_t]
        r_y = SS.spearmanr(rh[ep.r_t][mr], rr[mr]).statistic if mr.sum() > 10 else float("nan")
        out[b] = dict(rho_easiness=float(r_e), rho_realised_residual=float(r_y), n_targets=int(m.sum()),
                      frac_targets_with_neighbour=float((ws > 0).mean()))
    return out


# ---------------------------------------------------------------------------
# Reference + harness
# ---------------------------------------------------------------------------
def check_reference():
    from paiec.analysis import evaluate_exp0044_tempering as T
    h = hashlib.sha256(T.REFERENCE_ARCHIVE.read_bytes()).hexdigest().upper()
    if h != T.REFERENCE_SHA256:
        raise RuntimeError("reference archive hash mismatch")
    return dict(sha256=h, equivalence=T._validate_reference())


def check_v14(B="real_webagents"):
    """Vectorised v1_4 kernel == original python kernel (R2, seed 0)."""
    from paiec.models import v1_3_kernel as K14
    D = E.compact(B); H = Base.H_fold(B)
    X = csr_matrix((len(D["key_content"]), 1), dtype=np.float32)
    pr = Prep(B, D, "R2", [0], H, X, v14=True)
    worst = 0.0
    ours = pr.v14_probs()
    for b in (1, 7, 31):
        li, lk, ly = pr.ep.labels(b)
        ref = K14._kernel(pr.ep, H, pr.q[b], pr.p[b], li, lk, ly, 0.20)
        worst = max(worst, float(np.abs(ref - ours[b]).max()))
    return worst


def select(outer, vec):
    H = Base.H_fold(outer)
    inner = [b for b in E.BENCHES if b != outer]
    table = np.zeros((len(inner), len(GRID)))
    for i, T in enumerate(inner):
        D = E.compact(T)
        pr = Prep(T, D, "R2", SEEDS["R2"], H, key_matrix(vec, D))
        for j, hp in enumerate(GRID):
            table[i, j] = pr.delta_alc(pr.arm_probs(hp))
        log(f"   inner {outer}<-{T}: best cell {GRID[int(table[i].argmin())]} d={table[i].min()*1e3:+.3f}e-3")
        del pr
    obj = table.mean(0)
    j = int(obj.argmin())
    return GRID[j], dict(inner_benchmarks=inner, inner_delta_x1e3_at_selected={T: float(table[i, j] * 1e3)
                         for i, T in enumerate(inner)}, inner_objective_x1e3=float(obj[j] * 1e3),
                         inner_objective_range_x1e3=[float(obj.min() * 1e3), float(obj.max() * 1e3)])


def run(smoke=False):
    t0 = time.perf_counter()
    ref = check_reference() if not smoke else {"skipped": "smoke"}
    v14_check = check_v14()
    log(f"v1_4 vectorised vs original max|dp| = {v14_check:.2e}")
    assert v14_check < 1e-6, v14_check  # float32 similarities
    seeds = SEEDS if not smoke else {"R2": [0], "R1": [0, 1]}
    frozen, sel_info, res, rho = {}, {}, {"R2": {}, "R1": {}}, {}
    names = ["baseline", "lexical", "shuffled", "lexical_no_neardup", "v1_4_kernel"]
    for B in E.EVAL_FOLDS:
        vec = fit_vectorizer(B)
        log(f"[{B}] vocab={len(vec.vocabulary_)} {time.perf_counter()-t0:.0f}s")
        if smoke:
            hp, info = GRID[0], {"smoke": True}
        else:
            hp, info = select(B, vec)
        frozen[B], sel_info[B] = hp, info
        log(f"[{B}] frozen {hp}  {info.get('inner_objective_x1e3')}")
        D = E.compact(B)
        H = Base.H_fold(B)
        X = key_matrix(vec, D)
        Xs = shuffled_matrix(X, 0)
        for R in ("R2", "R1"):
            pr = Prep(B, D, R, seeds[R], H, X, v14=(R == "R2"))
            S_sh = pr._sims(Xs); S_sh[pr.exact] = 0.0
            S_nd = np.where(pr.S > NEAR_DUP, 0.0, pr.S).astype(np.float32)
            r = {"baseline": pr.result(pr.p),
                 "lexical": pr.result(pr.arm_probs(hp)),
                 "shuffled": pr.result(pr.arm_probs(hp, S_sh)),
                 "lexical_no_neardup": pr.result(pr.arm_probs(hp, S_nd))}
            if R == "R2":
                r["v1_4_kernel"] = pr.result(pr.v14_probs())
                rho[B] = dict(lexical=rho_diag(pr, D, hp), shuffled=rho_diag(pr, D, hp, S_sh),
                              frac_neardup_neighbour=float(((pr.S > NEAR_DUP).any(1)).mean()),
                              mean_max_sim=float(pr.S.max(1).mean()))
            res[R][B] = r
            log(f"  [{R} {B}] n_inst={pr.ep.n} " + " ".join(
                f"{k}={Base.pair_mean(v['alc'], v['pair']).mean():.5f}" for k, v in r.items())
                + f" {time.perf_counter()-t0:.0f}s")
            del pr, S_sh, S_nd
    summ = {}
    for R in ("R2", "R1"):
        nm = [n for n in names if all(n in res[R][B] for B in E.EVAL_FOLDS)]
        summ[R] = V.summarise(res[R], nm, "baseline")
    peak = _peak_mb()
    report = dict(experiment_id="EXP-0042", preregistered=True, dataset_revision="bc8204d811",
                  reference=ref, v14_vectorisation_check_max_abs=v14_check, folds=list(E.EVAL_FOLDS),
                  regimes={k: v for k, v in seeds.items()}, budgets=list(E.BUDGETS), acquisition_policy="A1 (accept-all)",
                  base_config=base_cfg(), tfidf=dict(TFIDF, dtype="float32"), grid=GRID, near_dup_threshold=NEAR_DUP,
                  frozen_hyperparameters=frozen, selection=sel_info, results=summ, rho=rho,
                  elapsed_seconds=time.perf_counter() - t0, peak_working_set_mb=peak, smoke=smoke)
    out = OUTPUT if not smoke else paths.ROOT / ".tmp" / "exp_0042_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    for R in ("R2", "R1"):
        for n, x in summ[R].items():
            if "d" in x:
                d = x["d"]
                log(f"{R} {n:20s} ALC={x['alc']:.5f} d={d['pt']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}] "
                    f"t3[{d['t3'][0]:+.3f},{d['t3'][1]:+.3f}] neg={d['n_neg']}/4 worst={d['worst_fold']:+.3f} "
                    + " ".join(f"B{b}={db['pt']:+.3f}" for b, db in zip(E.BUDGETS, x["d_budget"])))
    log(f"wrote {out}  elapsed={report['elapsed_seconds']:.0f}s peak_ws={report['peak_working_set_mb']:.0f}MB")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    run(ap.parse_args().smoke)
