"""v1: pooled hierarchical logistic predictor with exact posterior integration.

    logit q_pair = mu_b + theta_pair            (pair-mean / marginal scale)
        mu_b     ~ Student-t3(G, s_B)            pooled over every pair with labels on
                                                 the same benchmark_id in `labeled`
        theta    ~ N(m_subject, tau^2)           m_subject: ridge prior on public-pool
                                                 name ability (shrunk), release date
                                                 (clipped), provider
    logit p_{pair,i} = c_pair + beta_i          (conditional scale)
        beta_i   ~ N(0, sigma_beta^2)            updated by labels other subjects
                                                 received on the SAME item dict
        c_pair   solves E_beta[sig(c + beta)] = q_pair

mu_b is integrated on a 121-point grid, theta and beta by 20-node Gauss-Hermite;
predictions are posterior predictive means (no plug-in). Optional temperature T.
Subjects are matched on the exact 8-field dict; items on (content, features,
interactors) within the benchmark_id. All parameters come from v1_params.json,
fit only on the public training pool (src/paiec/models/ladder.py --build-v1).

Contract: never raises, stateless across calls, returns a finite float in [0.005, 0.995].
"""
from __future__ import annotations

import json
import math
import os
import re

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
SUBJECT_FIELDS = ("normalized_name", "provider", "release_date", "access_date", "harness",
                  "reasoning_effort", "harness_version", "subject_features_extra")
_DEFAULT = {"G": -0.6, "sB": 0.8, "tau": 1.1, "sigma_beta": 2.7, "lam": 0.0, "tau_name": 1.1,
            "name_table": {}, "provs": [], "rel_lo": 2023.7, "rel_hi": 2026.3,
            "meta_coef": None, "tau_meta": 1.1, "T": 1.0, "prior_mode": "meta",
            "item_pool": True, "clip": [0.005, 0.995]}


def _load():
    cfg = dict(_DEFAULT)
    try:
        with open(os.path.join(_HERE, "v1_params.json"), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            cfg.update(data)
    except Exception:
        pass
    return cfg


P = _load()
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(20)
GH_W = GH_W / GH_W.sum()
LOG_GH_W = np.log(GH_W)
MU = np.linspace(-7.0, 5.0, 121)
LOG_T3 = -2.0 * np.log1p(((MU - float(P["G"])) / float(P["sB"])) ** 2 / 3.0)
_FALLBACK = 1.0 / (1.0 + math.exp(-float(P["G"])))


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _log_sig(x):
    return -np.logaddexp(0.0, -x)


def _str(v):
    return "" if v is None else str(v)


def _subject_key(s):
    return tuple(_str(s.get(f, "")) for f in SUBJECT_FIELDS)


def _item_key(it):
    return (_str(it.get("item_content", "")), _str(it.get("item_features", "")),
            _str(it.get("interactors", "")))


def _name_key(s):
    n = _str(s.get("normalized_name", "")).strip()
    if n:
        return "n:" + n.lower()
    m = re.search(r"model_config=([^;]+)", _str(s.get("subject_features_extra", "")))
    if m:
        return "c:" + m.group(1).split("--")[0].strip().lower()
    return ""


def _release(s):
    m = re.match(r"(\d{4})-(\d{2})(?:-(\d{2}))?", _str(s.get("release_date", "")).strip())
    if not m:
        return None
    return int(m.group(1)) + (int(m.group(2)) - 1) / 12 + (int(m.group(3) or 15) - 1) / 365


def _prior(subject):
    """(mean, sd) of theta for this subject on an unseen benchmark."""
    tau = float(P["tau"])
    mode = P.get("prior_mode", "meta")
    if mode == "none":
        return 0.0, tau
    a = P["name_table"].get(_name_key(subject))
    if mode == "name":
        return (float(P["lam"]) * float(a), float(P["tau_name"])) if a is not None else (0.0, tau)
    coef = P.get("meta_coef")
    if not coef:
        return 0.0, tau
    r = _release(subject)
    rc = 0.0 if r is None else min(max(r, float(P["rel_lo"])), float(P["rel_hi"])) - 2025.0
    x = [float(a) if a is not None else 0.0, 1.0 if a is not None else 0.0, rc,
         1.0 if r is None else 0.0]
    prov = _str(subject.get("provider", "")).strip()
    x += [1.0 if prov == p else 0.0 for p in P["provs"]]
    x.append(1.0)
    if len(x) != len(coef):
        return 0.0, tau
    m = float(np.dot(np.asarray(x), np.asarray(coef, float)))
    if not math.isfinite(m):
        return 0.0, tau
    return max(min(m, 4.0), -4.0), float(P["tau_meta"])


def _pair_posterior(S, N, m, t):
    """Posterior predictive pair-mean for every pair of one benchmark group."""
    eta = MU[None, :, None] + m[:, None, None] + t[:, None, None] * GH_X[None, None, :]
    ll = S[:, None, None] * _log_sig(eta) + (N - S)[:, None, None] * _log_sig(-eta) + LOG_GH_W
    mx = ll.max(2, keepdims=True)
    w = np.exp(ll - mx)
    Ln = w.sum(2)
    logL = np.log(Ln) + mx[:, :, 0]
    Eq = (w * _sig(eta)).sum(2) / Ln
    lp = logL.sum(0) + LOG_T3
    post = np.exp(lp - lp.max())
    post /= post.sum()
    return Eq @ post


def _invert_c(q, s):
    q = np.clip(q, 1e-4, 1 - 1e-4)
    c = np.log(q / (1 - q)) * math.sqrt(1 + math.pi * s * s / 8)
    for _ in range(12):
        pr = _sig(c[:, None] + s * GH_X[None, :])
        f = pr @ GH_W - q
        d = (pr * (1 - pr)) @ GH_W
        c = c - f / np.maximum(d, 1e-9)
    return c


def _core(subject, item, labeled):
    bench = _str(item.get("benchmark_id", ""))
    tkey = _subject_key(subject)
    ikey = _item_key(item)
    pairs = {tkey: 0}
    subj_of = [subject]
    S, N = [0.0], [0.0]
    item_obs = []  # (pair index, y)
    for entry in labeled or ():
        try:
            (os_, oi), y = entry
            if not isinstance(os_, dict) or not isinstance(oi, dict):
                continue
            if _str(oi.get("benchmark_id", "")) != bench:
                continue
            y = float(y)
            if y not in (0.0, 1.0):
                continue
            k = _subject_key(os_)
            j = pairs.get(k)
            if j is None:
                j = pairs[k] = len(S)
                S.append(0.0); N.append(0.0); subj_of.append(os_)
            S[j] += y; N[j] += 1.0
            if P.get("item_pool", True) and _item_key(oi) == ikey:
                item_obs.append((j, y))
        except Exception:
            continue
    pri = [_prior(s) for s in subj_of]
    m = np.array([a for a, _ in pri]); t = np.array([b for _, b in pri])
    q = _pair_posterior(np.array(S), np.array(N), m, t)
    if not item_obs:
        return float(q[0])
    sb = float(P["sigma_beta"])
    idx = sorted({j for j, _ in item_obs} | {0})
    c = dict(zip(idx, _invert_c(q[idx], sb)))
    beta = sb * GH_X
    ll = LOG_GH_W.copy()
    for j, y in item_obs:
        e = c[j] + beta
        ll = ll + (_log_sig(e) if y > 0.5 else _log_sig(-e))
    w = np.exp(ll - ll.max())
    w /= w.sum()
    return float(np.dot(w, _sig(c[0] + beta)))


def _finish(p):
    try:
        p = float(p)
        if not math.isfinite(p):
            p = _FALLBACK
        T = float(P.get("T", 1.0))
        if T != 1.0 and math.isfinite(T):
            p = min(max(p, 1e-6), 1 - 1e-6)
            p = 1.0 / (1.0 + math.exp(-T * math.log(p / (1 - p))))
        lo, hi = P.get("clip", [0.005, 0.995])
        p = float(min(max(p, float(lo)), float(hi)))
        return p if math.isfinite(p) else 0.5
    except Exception:
        return 0.5


def predict(input: list, labeled: list | None = None) -> float:
    try:
        subject, item = input[0], input[1]
        if not isinstance(subject, dict) or not isinstance(item, dict):
            return _finish(_FALLBACK)
        with np.errstate(all="ignore"):
            return _finish(_core(subject, item, labeled))
    except Exception:
        return _finish(_FALLBACK)
