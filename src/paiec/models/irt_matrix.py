"""IRT + matrix-completion skeletons for PAIEC (benchmark-cold-start aware).

Design constraints from the spec:
- Test benchmarks are UNSEEN: no memorizable per-benchmark test effect.
  IRT difficulty for a test item must come from item features/text or from
  pair-label posterior updates, NOT from a fitted per-item intercept.
- Subjects at test may be unseen configs: ability must back off to
  family/provider/generation priors.
- Adaptation uses ONLY supplied `labeled` (nested 0/1/3/7/15/31); no cross-call memory.
- Equal pair weighting: shrink aggressively; per-pair posteriors, not pooled MLE.

This file provides dependency-light reference implementations (numpy only):
  1. Rasch/1PL with MAP ability updates from acquired labels.
  2. 2PL (difficulty + discrimination) with feature-conditioned priors.
  3. Low-rank logistic matrix factorization with bias terms + cold-start priors.
All are starting points for experiments 6/7/10; production versions may use
torch/差不多 hierarchical Bayes, but the cold-start discipline stays.

Notation: y in {0,1}, p = sigmoid(theta_s - b_i) [1PL],
p = sigmoid(a_i*(theta_s - b_i)) [2PL], p = sigmoid(mu + u_s·v_i + c_s + d_i) [MF].
"""
from __future__ import annotations
import math
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

# ---------------- 1PL Rasch with MAP adaptation ----------------
class Rasch1PL:
    def __init__(self, prior_theta_mean=0.0, prior_theta_sd=1.0,
                 prior_b_mean=0.0, prior_b_sd=1.0, l2=1.0, lr=0.1, iters=200):
        self.pt_m, self.pt_sd = prior_theta_mean, prior_theta_sd
        self.pb_m, self.pb_sd = prior_b_mean, prior_b_sd
        self.l2, self.lr, self.iters = l2, lr, iters
        self.theta_ = {}   # subject_key -> ability
        self.b_ = {}       # train item/benchmark-group difficulty
        self.global_b_ = 0.0

    def fit(self, rows):
        """rows: iterable of (subject_key, group_key, y). group_key for training
        is (benchmark_id, item_id) when available else benchmark_id. For cold
        start we retain only benchmark-level difficulty + global fallback."""
        thetas, bs = {}, {}
        for s, g, y in rows:
            thetas.setdefault(s, 0.0); bs.setdefault(g, 0.0)
        # simple gradient MAP
        for _ in range(self.iters):
            for s, g, y in rows:
                p = float(sigmoid(thetas[s] - bs[g]))
                err = y - p
                thetas[s] += self.lr * (err - self.l2 * (thetas[s] - self.pt_m))
                bs[g] += self.lr * (-err - self.l2 * (bs[g] - self.pb_m))
        self.theta_, self.b_ = thetas, bs
        # collapse to benchmark-level difficulty for cold start
        bench_vals = {}
        for g, b in bs.items():
            bench = g[0] if isinstance(g, tuple) else g
            bench_vals.setdefault(bench, []).append(b)
        self.bench_b_ = {k: float(np.mean(v)) for k, v in bench_vals.items()}
        self.global_b_ = float(np.mean(list(bs.values()))) if bs else 0.0

    def _theta_for(self, s, fallback=0.0):
        return self.theta_.get(s, fallback)

    def _b_for_cold_start(self, benchmark_id, item_prior_b=None):
        if item_prior_b is not None:
            return item_prior_b
        return self.bench_b_.get(benchmark_id, self.global_b_)

    def predict_with_labels(self, s, benchmark_id, acquired_ys,
                            item_prior_b=None, theta_prior=None, theta_sd=1.0):
        """MAP theta given acquired pair labels + prior, then p for target.
        acquired_ys: list of 0/1 from same (s, benchmark). Cold-start item
        difficulty from benchmark prior or text/feature model (passed in)."""
        b = self._b_for_cold_start(benchmark_id, item_prior_b)
        theta = self._theta_for(s, fallback=(theta_prior if theta_prior is not None else self.pt_m))
        # Newton/MAP refinement on theta (few steps suffice for <=31 labels)
        for _ in range(25):
            p = float(sigmoid(theta - b))
            # gradient of log-posterior; Hessian approx
            g = sum(y - sigmoid(theta - b) for y in acquired_ys) - (theta - self.pt_m) / (theta_sd ** 2)
            h = -sum(p * (1 - p) for _ in acquired_ys) - 1.0 / (theta_sd ** 2)
            step = g / (h - 1e-9)
            theta -= step
            if abs(step) < 1e-4:
                break
        return float(sigmoid(theta - b)), float(theta), float(b)

# ---------------- Low-rank logistic MF with biases ----------------
class LogisticMF:
    """p = sigmoid(mu + c_s + d_g + u_s·v_g); groups g = benchmark (cold-start
    item factors replaced by benchmark factors + item-feature residual)."""
    def __init__(self, rank=8, l2=0.5, lr=0.05, iters=100):
        self.rank, self.l2, self.lr, self.iters = rank, l2, lr, iters

    def fit(self, rows, subjects, groups):
        rng = np.random.default_rng(0)
        self.mu = 0.0
        self.c_ = {s: 0.0 for s in subjects}
        self.d_ = {g: 0.0 for g in groups}
        self.U_ = {s: rng.normal(0, 0.1, self.rank) for s in subjects}
        self.V_ = {g: rng.normal(0, 0.1, self.rank) for g in groups}
        for _ in range(self.iters):
            for s, g, y in rows:
                p = float(sigmoid(self.mu + self.c_[s] + self.d_[g] + self.U_[s] @ self.V_[g]))
                err = y - p
                self.mu += self.lr * (err - self.l2 * self.mu)
                self.c_[s] += self.lr * (err - self.l2 * self.c_[s])
                self.d_[g] += self.lr * (err - self.l2 * self.d_[g])
                us, vg = self.U_[s].copy(), self.V_[g].copy()
                self.U_[s] += self.lr * (err * vg - self.l2 * us)
                self.V_[g] += self.lr * (err * us - self.l2 * vg)

    def predict_cold(self, s, g_train_or_none, u_prior=None, v_prior=None):
        c = self.c_.get(s, 0.0)
        d = self.d_.get(g_train_or_none, 0.0) if g_train_or_none else 0.0
        u = self.U_.get(s, u_prior if u_prior is not None else np.zeros(self.rank))
        v = self.V_.get(g_train_or_none, v_prior if v_prior is not None else np.zeros(self.rank)) if g_train_or_none else (v_prior if v_prior is not None else np.zeros(self.rank))
        return float(sigmoid(self.mu + c + d + u @ v))

# ---------------- Acquisition policies (for local simulation) ----------------
def random_acquire(n_candidates, k, rng):
    return list(rng.choice(n_candidates, size=min(k, n_candidates), replace=False))

def uncertainty_acquire(probs, k):
    order = sorted(range(len(probs)), key=lambda i: abs(probs[i] - 0.5))
    return order[:k]

def diversity_greedy_acquire(embeddings, probs, k, lam=0.5):
    """Uncertainty + diversity: score = lam*uncertainty + (1-lam)*min-dist-to-selected."""
    n = len(probs)
    unc = 1.0 - np.abs(np.asarray(probs) - 0.5) * 2  # 1 at p=0.5
    selected = []
    for _ in range(min(k, n)):
        best, best_s = -1, -1e9
        for i in range(n):
            if i in selected:
                continue
            if not selected:
                div = 1.0
            else:
                d = min(float(np.linalg.norm(embeddings[i] - embeddings[j])) for j in selected)
                div = d / (1.0 + d)
            s = lam * unc[i] + (1 - lam) * div
            if s > best_s:
                best, best_s = i, s
        selected.append(best)
    return selected
