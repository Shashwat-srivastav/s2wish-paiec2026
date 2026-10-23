"""Naive baselines for PAIEC — pure-python reference implementations.

Each baseline implements predict(input, labeled) -> float with the official
8-field subject + 4-field item dicts. Fitting is explicit (no hidden state):
fit_* functions precompute tables from a training frame; predict_* closures
use only those tables + supplied `labeled` evidence (nested per-budget).

Training frame columns (long form):
  subject_key (tuple of 8 subject strings), benchmark_id,
  normalized_name, provider, harness, reasoning_effort, release_date,
  item_id (optional), y (0/1)

All priors are Laplace/Beta-smoothed: (succ + a) / (n + a + b).
Default a=b=5 (~ weak prior centered at 0.5 with 10 pseudo-obs); per-baseline
tuning via internal benchmark-holdout, NOT via Codabench.

Do NOT call these "generalizing" until benchmark-holdout is measured.
"""
from __future__ import annotations
from collections import defaultdict

BUDGETS = [0, 1, 3, 7, 15, 31]

def _smoothed(succ: int, n: int, a: float = 5.0, b: float = 5.0) -> float:
    return (succ + a) / (n + a + b)

def subject_key_of(subject: dict) -> tuple:
    return tuple(subject.get(k, "") or "" for k in (
        "normalized_name", "provider", "release_date", "access_date",
        "harness", "harness_version", "reasoning_effort", "subject_features_extra"))

class FittedPriors:
    """Precomputed smoothed success-rate tables."""
    def __init__(self, a: float = 5.0, b: float = 5.0):
        self.a, self.b = a, b
        self.global_rate = 0.5
        self.by_benchmark = {}
        self.by_subject = {}
        self.by_provider = {}
        self.by_family = {}   # normalized_name prefix before first '-'/'_'/' ' as proxy
        self.by_harness = {}
        self.by_reasoning = {}
        self.by_release_bucket = {}
        self.by_pair = {}     # (subject_key, benchmark_id) historical mean

    @staticmethod
    def _family(normalized_name: str) -> str:
        n = (normalized_name or "").strip()
        for sep in ("-", "_", " ", "/"):
            if sep in n:
                n = n.split(sep)[0]
        return n or "(unknown)"

    @staticmethod
    def _release_bucket(release_date: str) -> str:
        d = (release_date or "").strip()
        return d[:7] if len(d) >= 7 else "(unknown)"

def fit_priors(rows: list[dict], a: float = 5.0, b: float = 5.0) -> FittedPriors:
    """rows: list of dicts with keys subject dict fields + benchmark_id + y."""
    fp = FittedPriors(a, b)
    acc = defaultdict(lambda: [0, 0])  # key -> [succ, n]
    g_succ = g_n = 0
    for r in rows:
        y = int(r["y"])
        g_succ += y; g_n += 1
        subj = {k: r.get(k, "") for k in (
            "normalized_name", "provider", "release_date", "access_date",
            "harness", "harness_version", "reasoning_effort", "subject_features_extra")}
        sk = subject_key_of(subj)
        bch = r.get("benchmark_id", "")
        acc[("bench", bch)][0] += y; acc[("bench", bch)][1] += 1
        acc[("subj", sk)][0] += y; acc[("subj", sk)][1] += 1
        acc[("prov", r.get("provider", ""))][0] += y; acc[("prov", r.get("provider", ""))][1] += 1
        acc[("fam", FittedPriors._family(r.get("normalized_name", "")))][0] += y
        acc[("fam", FittedPriors._family(r.get("normalized_name", "")))][1] += 1
        acc[("harn", r.get("harness", ""))][0] += y; acc[("harn", r.get("harness", ""))][1] += 1
        acc[("reas", r.get("reasoning_effort", ""))][0] += y; acc[("reas", r.get("reasoning_effort", ""))][1] += 1
        acc[("rel", FittedPriors._release_bucket(r.get("release_date", "")))][0] += y
        acc[("rel", FittedPriors._release_bucket(r.get("release_date", "")))][1] += 1
        acc[("pair", (sk, bch))][0] += y; acc[("pair", (sk, bch))][1] += 1
    fp.global_rate = _smoothed(g_succ, g_n, a, b)
    for (kind, key), (s, n) in acc.items():
        v = _smoothed(s, n, a, b)
        {"bench": fp.by_benchmark, "subj": fp.by_subject, "prov": fp.by_provider,
         "fam": fp.by_family, "harn": fp.by_harness, "reas": fp.by_reasoning,
         "rel": fp.by_release_bucket, "pair": fp.by_pair}[kind][key] = v
    return fp

def _pair_labels(subject: dict, benchmark: str, labeled) -> list[int]:
    out = []
    for (osub, oitem), y in (labeled or []):
        if osub != subject:
            continue
        if (oitem.get("benchmark_id") or "") != (benchmark or ""):
            continue
        out.append(int(y))
    return out

def make_predictor(name: str, fp: FittedPriors, adapt_a: float = 5.0, adapt_b: float = 5.0):
    """Return predict(input, labeled) for the named naive baseline.

    Names: global | benchmark | subject | provider | family | harness |
           reasoning | release_bucket | pair_history | hierarchical
    Adaptation: posterior mean of prior + pair-matched acquired labels:
      p = (prior*(a+b) + succ) / (a+b+n). For 'global', prior=global_rate etc.
    'hierarchical': precision-weighted blend of global/benchmark/subject/providers
      as prior, then updated with pair labels. 'pair_history': historical pair
      mean as prior (falls back down the hierarchy when unseen).
    """
    def prior_for(subject, benchmark):
        sk = subject_key_of(subject)
        chain = [
            fp.by_pair.get((sk, benchmark)),
            fp.by_subject.get(sk),
            fp.by_benchmark.get(benchmark),
            fp.by_family.get(FittedPriors._family(subject.get("normalized_name", ""))),
            fp.by_provider.get(subject.get("provider", "")),
            fp.by_harness.get(subject.get("harness", "")),
            fp.by_reasoning.get(subject.get("reasoning_effort", "")),
            fp.by_release_bucket.get(FittedPriors._release_bucket(subject.get("release_date", ""))),
            fp.global_rate,
        ]
        if name == "global":
            return fp.global_rate
        if name == "benchmark":
            return fp.by_benchmark.get(benchmark, fp.global_rate)
        if name == "subject":
            return fp.by_subject.get(sk, fp.global_rate)
        if name == "provider":
            return fp.by_provider.get(subject.get("provider", ""), fp.global_rate)
        if name == "family":
            return fp.by_family.get(FittedPriors._family(subject.get("normalized_name", "")), fp.global_rate)
        if name == "harness":
            return fp.by_harness.get(subject.get("harness", ""), fp.global_rate)
        if name == "reasoning":
            return fp.by_reasoning.get(subject.get("reasoning_effort", ""), fp.global_rate)
        if name == "release_bucket":
            return fp.by_release_bucket.get(FittedPriors._release_bucket(subject.get("release_date", "")), fp.global_rate)
        if name == "pair_history":
            for c in chain:
                if c is not None:
                    return c
            return fp.global_rate
        if name == "hierarchical":
            vals = [v for v in [fp.by_subject.get(sk), fp.by_benchmark.get(benchmark),
                    fp.by_provider.get(subject.get("provider", "")), fp.global_rate] if v is not None]
            return sum(vals) / len(vals) if vals else 0.5
        raise ValueError(f"unknown baseline {name}")

    def predict(input, labeled=None) -> float:
        subject, item = input
        benchmark = item.get("benchmark_id", "")
        prior = prior_for(subject, benchmark)
        ys = _pair_labels(subject, benchmark, labeled)
        pseudo = adapt_a + adapt_b
        p = (prior * pseudo + sum(ys)) / (pseudo + len(ys))
        return float(min(max(p, 0.0), 1.0))

    predict.__name__ = f"predict_{name}"
    return predict

# Empirical-mean (official, unsmoothed) re-stated for side-by-side comparison.
def predict_empirical_mean(input, labeled=None) -> float:
    subject, item = input
    if not labeled:
        return 0.5
    benchmark = item.get("benchmark_id")
    if not isinstance(benchmark, str) or not benchmark:
        raise ValueError("The mean predictor requires an anonymous item benchmark_id.")
    ys = _pair_labels(subject, benchmark, labeled)
    # replicate official strictness: error if acquired items lack benchmark_id
    for (osub, oitem), y in labeled:
        if osub != subject:
            continue
        ob = oitem.get("benchmark_id")
        if not isinstance(ob, str) or not ob:
            raise ValueError("Acquired items must include an anonymous benchmark_id.")
        if y not in (0, 1):
            raise ValueError("Acquired responses must be binary (0 or 1).")
    return float(sum(ys) / len(ys)) if ys else 0.5
