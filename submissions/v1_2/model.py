"""v1_2 = v1_1 + item-group facet effect + degenerate-key guard (E4, report 23).

Engine: src/paiec/models/v1_2.py, config cfg_of(facets="all", guard=1). Additions to v1_1:

Facets. Every `key=value` field of item_features is a facet. For a target whose item key
has no label from any subject, the subject's OWN revealed labels on the same (field, value)
cells move the prediction:
    logit p = c_pair + sum_f (gamma_f[v] - s_f) + beta_res,
    gamma ~ N(0, pi sigma_beta^2), beta_res ~ N(0, (1 - pi) sigma_beta^2),
s_f = label-weighted mean posterior gamma of field f (centring). pi in {1e-3, 0.1, 0.3}
with prior weights 0.4/0.4/0.2, averaged by the pair's marginal likelihood over its cells.
Targets with a same-key label, no features or no matching cell (and every B0 target)
get v1_1 exactly.
Guard. rho_k >= 0.5 when the target key's labels all come from the target subject.

v1_1 (unchanged below): v1 (pooled offset + cross-subject item pooling) with two repairs.

    logit q_pair = mu_b + theta_pair            (pair-mean scale)
        mu_b     ~ Student-t3(C, s)              C = G/2 (pulled halfway to logit 0),
                                                 s = max(.5, rms(public means - C))
        theta    ~ N(0, tau^2)
    logit p_{pair,i} = c_pair + beta_i          (conditional scale)
        beta_i   ~ N(0, sigma_beta^2), shared by every label on the SAME item dict
        c_pair   solves E_beta[sig(c + beta)] = q_pair

False-merge guard: an item key may merge unrelated items. With prior merge rate rho_b
the prediction is P(shared|y) * item-level + (1 - P(shared|y)) * q_pair, where
P(shared|y) uses the exact marginal-likelihood ratio (shared beta vs independent).
rho_b is a per-benchmark moment estimate: same-subject label pairs on one key must be
distinct items; their disagreement rate (relative to 2q(1-q)) versus the cross-subject
same-key rate estimates the share of key matches that are false merges (floor rho0).

Exact integration: mu_b on a 121-point grid, theta and beta by 20-node Gauss-Hermite.
Parameters: v1_2_params.json (built by src/paiec/build/v1_2_params.py; public pool only).
Contract: never raises, stateless across calls (no cache), finite float in [0.005, 0.995].
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
SUBJECT_FIELDS = ("normalized_name", "provider", "release_date", "access_date", "harness",
                  "reasoning_effort", "harness_version", "subject_features_extra")
_DEFAULT = {"G": -0.58, "prior_center": -0.29, "prior_scale": 0.78, "tau": 1.13,
            "sigma_beta": 2.70, "rho0": 0.05, "rho_k0": 2.0, "clip": [0.005, 0.995],
            "facets": True, "facet_pis": [1e-3, 0.1, 0.3], "facet_prior_w": [0.4, 0.4, 0.2],
            "guard": 1, "guard_rho": 0.5}


def _load():
    cfg = dict(_DEFAULT)
    try:
        with open(os.path.join(_HERE, "v1_2_params.json"), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            for k in _DEFAULT:
                if k in data:
                    cfg[k] = data[k]
    except Exception:
        pass
    return cfg


P = _load()
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(20)
GH_W = GH_W / GH_W.sum()
LOG_GH_W = np.log(GH_W)
MU = np.linspace(-7.0, 5.0, 121)
_Z = (MU - float(P["prior_center"])) / float(P["prior_scale"])
LOG_PRIOR = -2.0 * np.log1p(_Z * _Z / 3.0)
_FALLBACK = 1.0 / (1.0 + math.exp(-float(P["prior_center"])))


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _log_sig(x):
    return -np.logaddexp(0.0, -x)


def _str(v):
    return v if type(v) is str else ("" if v is None else str(v))


def _subject_key(s):
    return tuple(_str(s.get(f, "")) for f in SUBJECT_FIELDS)


def _item_key(it):
    return (_str(it.get("item_content", "")), _str(it.get("item_features", "")),
            _str(it.get("interactors", "")))


def _pair_posterior(S, N, t):
    eta = MU[None, :, None] + t * GH_X[None, None, :]
    ll = S[:, None, None] * _log_sig(eta) + (N - S)[:, None, None] * _log_sig(-eta) + LOG_GH_W
    mx = ll.max(2, keepdims=True)
    w = np.exp(ll - mx)
    Ln = w.sum(2)
    logL = np.log(Ln) + mx[:, :, 0]
    Eq = (w * _sig(eta)).sum(2) / Ln
    lp = logL.sum(0) + LOG_PRIOR
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


def _merge_rate(cells, q):
    """cells: {(pair j, item key): [n, successes]} on this benchmark."""
    rho0, k0 = float(P["rho0"]), float(P["rho_k0"])
    n_s, t_k = {}, {}
    n_wpairs = n_w = 0.0
    for (j, k), (n, s) in cells.items():
        n_s[j] = n_s.get(j, 0.0) + n
        t_k[k] = t_k.get(k, 0.0) + n
        if n >= 2:
            n_wpairs += n * (n - 1) / 2
            qc = min(max(float(q[j]), 0.02), 0.98)
            n_w += s * (n - s) / (2 * qc * (1 - qc))
    n_w = min(n_w, n_wpairs)
    n_x = sum(t * (t - 1) / 2 for t in t_k.values()) - n_wpairs
    ns = list(n_s.values())
    P_w = sum(n * (n - 1) / 2 for n in ns)
    P_x = (sum(ns) ** 2 - sum(n * n for n in ns)) / 2
    est = n_w * P_x / P_w if P_w > 0 else 0.0
    return min(max((est + k0 * rho0) / (n_x + k0), rho0), 1 - 1e-4)


def _core(subject, item, labeled):
    bench = _str(item.get("benchmark_id", ""))
    tkey = _subject_key(subject)
    ikey = _item_key(item)
    pairs = {tkey: 0}
    S, N = [0.0], [0.0]
    item_obs = []                     # (pair index, y) on the target's item dict
    own = []                          # (item_features, y) of the target subject's labels
    cells = {}                        # (pair, key) -> [n, s] for the merge-rate estimate
    for entry in labeled or ():
        try:
            (os_, oi), y = entry
            if not isinstance(oi, dict) or not isinstance(os_, dict):
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
                S.append(0.0); N.append(0.0)
            S[j] += y; N[j] += 1.0
            ik = _item_key(oi)
            c = cells.get((j, ik))
            if c is None:
                cells[(j, ik)] = [1.0, y]
            else:
                c[0] += 1.0; c[1] += y
            if ik == ikey:
                item_obs.append((j, y))
            if j == 0:
                own.append((oi.get("item_features", ""), y))
        except Exception:
            continue
    q = _pair_posterior(np.array(S), np.array(N), float(P["tau"]))
    if not item_obs:
        if P.get("facets") and own:
            try:
                return _facet(float(q[0]), own, item.get("item_features", ""))
            except Exception:
                pass
        return float(q[0])
    sb = float(P["sigma_beta"])
    idx = sorted({j for j, _ in item_obs} | {0})
    c = dict(zip(idx, _invert_c(q[idx], sb)))
    beta = sb * GH_X
    ll = LOG_GH_W.copy()
    lind = 0.0
    for j, y in item_obs:
        e = c[j] + beta
        qj = min(max(float(q[j]), 1e-12), 1 - 1e-12)
        if y > 0.5:
            ll = ll + _log_sig(e); lind += math.log(qj)
        else:
            ll = ll + _log_sig(-e); lind += math.log1p(-qj)
    mx = ll.max()
    w = np.exp(ll - mx)
    Ls = w.sum()
    w /= Ls
    p_same = float(np.dot(w, _sig(c[0] + beta)))
    rho = _merge_rate(cells, q)
    if int(P.get("guard", 0)) == 1 and not any(j != 0 for j, _ in item_obs):
        rho = max(rho, float(P["guard_rho"]))     # no cross-subject label on this key
    lo = math.log1p(-rho) - math.log(rho) + math.log(Ls) + mx - lind
    ps = 1.0 / (1.0 + math.exp(-lo)) if lo > -700 else 0.0
    return ps * p_same + (1.0 - ps) * float(q[0])


def _features(v):
    """{field: value} from a 'a=x;b=y' item_features string (later duplicates win)."""
    d = {}
    for kv in _str(v).split(";"):
        if "=" in kv:
            a, b = kv.split("=", 1)
            d[a.strip()] = b.strip()
    return d


def _facet(q0, own, tfeat):
    """Facet-adjusted prediction for a target with no same-key label (else q0)."""
    tf = _features(tfeat)
    if not tf:
        return q0
    cells = {}                                    # (field, value) -> [n, s]
    for feat, y in own:
        for a, v in _features(feat).items():
            c = cells.get((a, v))
            if c is None:
                cells[(a, v)] = [1.0, y]
            else:
                c[0] += 1.0; c[1] += y
    keys = list(cells)
    hit = [i for i, (a, v) in enumerate(keys) if a in tf and tf[a] == v]
    if not hit:
        return q0
    n = np.array([cells[k][0] for k in keys])
    s = np.array([cells[k][1] for k in keys])
    fields = {}
    fidx = np.array([fields.setdefault(a, len(fields)) for a, _ in keys])
    sb = float(P["sigma_beta"])
    c0 = float(_invert_c(np.array([q0]), sb)[0])
    lq0 = math.log(q0 / (1 - q0))
    hit = np.array(hit)
    outs, lmls = [], []
    for pi, pw in zip(P["facet_pis"], P["facet_prior_w"]):
        pi = float(pi)
        sg, sr = math.sqrt(pi) * sb, math.sqrt(1 - pi) * sb
        m = _sig(c0 + sg * GH_X[:, None] + sr * GH_X[None, :]) @ GH_W
        m = np.clip(m, 1e-12, 1 - 1e-12)
        lw = s[:, None] * np.log(m)[None, :] + (n - s)[:, None] * np.log1p(-m)[None, :] + LOG_GH_W
        mx = lw.max(1, keepdims=True)
        lml = np.log(np.exp(lw - mx).sum(1)) + mx[:, 0]
        w = np.exp(lw - lml[:, None])
        gm = w @ (sg * GH_X)
        shift = (np.bincount(fidx, n * gm) / np.bincount(fidx, n))[fidx]
        e = (c0 - shift)[:, None, None] + sg * GH_X[None, :, None] + sr * GH_X[None, None, :]
        pt = np.einsum("cj,cjx,x->c", w, _sig(e), GH_W)
        pt = np.clip(pt, 1e-12, 1 - 1e-12)
        off = np.log(pt / (1 - pt)) - lq0
        outs.append(1.0 / (1.0 + math.exp(-(lq0 + float(off[hit].sum())))))
        lmls.append(float(lml.sum()) + math.log(float(pw)))
    L = np.array(lmls)
    wt = np.exp(L - L.max())
    wt /= wt.sum()
    p = float(wt @ np.array(outs))
    return p if math.isfinite(p) else q0


def _finish(p):
    try:
        p = float(p)
        if not math.isfinite(p):
            p = _FALLBACK
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
