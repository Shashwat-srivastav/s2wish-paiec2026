"""v0: hierarchical Beta-Binomial shrinkage (optional logistic-IRT) predictor.

Prediction for target ``[subject, item]``:

    p = (k * m + S) / (k + n)

where ``S`` / ``n`` are successes / count among acquired labels whose subject
dict equals the target subject dict *exactly* (all 8 fields) and whose
``benchmark_id`` equals the target's. ``(m, k)`` is the prior mean / strength,
read from ``prior.json`` (see README.md for the schema and resolution order).

Optional IRT variant (``prior.json: irt.enabled``) uses per-item difficulties
``b_i`` from an artifact file: ``P(y=1) = sigmoid(theta - b_i)``; theta is fitted
by MAP (Newton) under a N(logit(m), sd^2) prior, and the prediction integrates
over the Laplace posterior (probit approximation). If the artifact or the
target item's difficulty is missing, it falls back to Beta-Binomial.

Contract: ``predict`` never raises, is stateless across calls (only read-only
config is cached at import), and always returns a finite float in [clip_lo, clip_hi].
"""

from __future__ import annotations

import hashlib
import json
import math
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

SUBJECT_FIELDS = (
    "normalized_name", "provider", "release_date", "access_date", "harness",
    "reasoning_effort", "harness_version", "subject_features_extra",
)

# Defaults used when prior.json is absent or malformed. m=0.5, k=2 is the
# uniform Beta(1,1) prior (Laplace rule of succession): with no data we assume
# pair accuracies are spread roughly uniformly over [0,1], for which the
# Bayes-optimal shrinkage strength is k = a + b = 2 (see README / analysis.py).
DEFAULT_CONFIG = {
    "default": {"m": 0.5, "k": 2.0},
    "clip": [0.02, 0.98],
    "by_benchmark": {},
    "by_subject_name": {},
    "by_subject_field": {},
    "field_order": ["harness", "reasoning_effort", "provider"],
    "irt": {"enabled": False, "difficulties_file": "", "prior_sd": 1.5,
            "default_difficulty": 0.0, "newton_iters": 25},
}


def _load_json(name):
    try:
        if not name:
            return None
        path = os.path.join(_HERE, os.path.basename(str(name)))
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _merge(base, override):
    out = dict(base)
    if isinstance(override, dict):
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(out.get(key), dict):
                out[key] = _merge(out[key], value)
            else:
                out[key] = value
    return out


def _load_config():
    cfg = _merge(DEFAULT_CONFIG, _load_json("prior.json") or {})
    try:
        lo, hi = float(cfg["clip"][0]), float(cfg["clip"][1])
        if not (0.0 <= lo < hi <= 1.0 and math.isfinite(lo) and math.isfinite(hi)):
            raise ValueError
    except Exception:
        lo, hi = 0.02, 0.98
    cfg["clip"] = [lo, hi]
    return cfg


CONFIG = _load_config()
_IRT = CONFIG.get("irt") or {}
DIFFICULTIES = None
if _IRT.get("enabled"):
    _raw = _load_json(_IRT.get("difficulties_file"))
    if isinstance(_raw, dict):
        DIFFICULTIES = {}
        for _key, _val in _raw.items():
            try:
                _f = float(_val)
                if math.isfinite(_f):
                    DIFFICULTIES[str(_key)] = _f
            except Exception:
                pass


def item_key(item):
    """Stable key for an item: sha1(benchmark_id + US + item_content)[:16]."""
    text = f"{item.get('benchmark_id', '')}\x1f{item.get('item_content', '')}"
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _valid_prior(entry):
    try:
        m, k = float(entry["m"]), float(entry["k"])
        if math.isfinite(m) and math.isfinite(k) and 0.0 < m < 1.0 and k >= 0.0:
            return m, k
    except Exception:
        pass
    return None


def resolve_prior(subject, item):
    """Most specific prior wins: benchmark > subject name > subject field > default."""
    cfg = CONFIG
    try:
        bench = str(item.get("benchmark_id", ""))
        hit = _valid_prior((cfg.get("by_benchmark") or {}).get(bench, {}))
        if hit:
            return hit
        name = str(subject.get("normalized_name", ""))
        hit = _valid_prior((cfg.get("by_subject_name") or {}).get(name, {}))
        if hit:
            return hit
        by_field = cfg.get("by_subject_field") or {}
        for field in cfg.get("field_order") or []:
            table = by_field.get(field) or {}
            hit = _valid_prior(table.get(str(subject.get(field, "")), {}))
            if hit:
                return hit
    except Exception:
        pass
    return _valid_prior(cfg.get("default", {})) or (0.5, 2.0)


def _subject_sig(subject):
    return tuple(str(subject.get(f, "")) for f in SUBJECT_FIELDS)


def matching_labels(subject, item, labeled):
    """[(item_dict, y)] for labels on the same full subject and benchmark."""
    out = []
    if not labeled:
        return out
    try:
        target_sig = _subject_sig(subject)
        bench = item.get("benchmark_id", "")
    except Exception:
        return out
    for entry in labeled:
        try:
            (obs_subject, obs_item), label = entry
            if not isinstance(obs_subject, dict) or not isinstance(obs_item, dict):
                continue
            if obs_item.get("benchmark_id", "") != bench:
                continue
            if _subject_sig(obs_subject) != target_sig:
                continue
            y = float(label)
            if y not in (0.0, 1.0):
                continue
            out.append((obs_item, int(y)))
        except Exception:
            continue
    return out


def _sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def beta_binomial(m, k, obs):
    n = len(obs)
    s = sum(y for _, y in obs)
    if k + n <= 0:
        return m
    return (k * m + s) / (k + n)


def irt_predict(m, item, obs):
    """MAP theta under N(logit m, sd^2); Laplace + probit-approx predictive.

    Returns None when the target difficulty is unavailable (-> fallback).
    """
    if DIFFICULTIES is None:
        return None
    b_target = DIFFICULTIES.get(item_key(item))
    if b_target is None:
        return None
    sd = float(_IRT.get("prior_sd", 1.5) or 1.5)
    b_default = float(_IRT.get("default_difficulty", 0.0) or 0.0)
    mu0, prec0 = _logit(m), 1.0 / (sd * sd)
    data = [(DIFFICULTIES.get(item_key(it), b_default), y) for it, y in obs]
    theta = mu0
    hess = prec0
    for _ in range(int(_IRT.get("newton_iters", 25) or 25)):
        grad = -prec0 * (theta - mu0)
        hess = prec0
        for b, y in data:
            p = _sigmoid(theta - b)
            grad += y - p
            hess += p * (1 - p)
        step = grad / hess
        theta += max(-2.0, min(2.0, step))
        if abs(step) < 1e-8:
            break
    var = 1.0 / hess
    return _sigmoid((theta - b_target) / math.sqrt(1.0 + math.pi * var / 8.0))


def _finish(p):
    lo, hi = CONFIG["clip"]
    try:
        p = float(p)
        if not math.isfinite(p):
            p = 0.5
    except Exception:
        p = 0.5
    return float(min(max(p, lo), hi))


def predict(input: list, labeled: list | None = None) -> float:
    try:
        subject, item = input[0], input[1]
        if not isinstance(subject, dict) or not isinstance(item, dict):
            return _finish(CONFIG["default"].get("m", 0.5))
    except Exception:
        return _finish(0.5)
    try:
        m, k = resolve_prior(subject, item)
    except Exception:
        m, k = 0.5, 2.0
    try:
        obs = matching_labels(subject, item, labeled)
        p = None
        if _IRT.get("enabled"):
            try:
                p = irt_predict(m, item, obs)
            except Exception:
                p = None
        if p is None:
            p = beta_binomial(m, k, obs)
        return _finish(p)
    except Exception:
        return _finish(m)
