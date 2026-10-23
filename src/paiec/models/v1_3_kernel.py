"""v1_3 candidate: v1_2 named facets plus conservative within-benchmark kernel.

The kernel uses only labels already revealed for the current hidden benchmark.
It shares residual difficulty from lexically similar *different* items and is
therefore inactive at budget 0 and on exact-key matches handled by v1_2.
"""
from __future__ import annotations

import math
import re

import numpy as np

from paiec.models import v1_2 as V
from paiec.models import v1_1 as Base

WORD_RE = re.compile(r"[a-z0-9]{2,}")
_TOKENS: dict[str, frozenset[str]] = {}


def _tokens(text):
    text = "" if text is None else str(text).lower()
    out = _TOKENS.get(text)
    if out is None:
        out = frozenset(WORD_RE.findall(text))
        _TOKENS[text] = out
    return out


def _sim(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter < 2:
        return 0.0
    return inter / math.sqrt(len(a) * len(b))


def _kernel(ep, H, q, p, li, lk, ly, strength):
    if not len(li) or not hasattr(ep.D, "__getitem__") or "key_content" not in ep.D:
        return p
    contents = ep.D["key_content"]
    label_tokens = [_tokens(contents[int(k)]) for k in lk]
    qres = ly - q[li]
    out = p.copy()
    K = len(ep.D["key_jev"])
    # Exact-key evidence is already handled by v1_2's item posterior.
    exact = set((int(i), int(k)) for i, k in zip(li, lk))
    for t, (inst, key) in enumerate(zip(ep.t_inst, ep.t_key)):
        if any((int(inst), int(key)) == x for x in exact):
            continue
        tt = _tokens(contents[int(key)])
        weights = []
        vals = []
        for j, (ji, jk) in enumerate(zip(li, lk)):
            if int(ji) != int(inst) and int(ep.inst_group[int(ji)]) != int(ep.inst_group[int(inst)]):
                continue
            if int(jk) == int(key):
                continue
            s = _sim(tt, label_tokens[j])
            if s >= 0.20:
                weights.append(s * s)
                vals.append(float(qres[j]))
        if not weights:
            continue
        w = np.asarray(weights, float)
        r = float(np.dot(w, np.asarray(vals, float)) / w.sum())
        gain = float(strength) * (1.0 - math.exp(-float(w.sum())))
        out[t] = min(max(out[t] + gain * r, 0.005), 0.995)
    return out


def cfg_of(**kw):
    cfg = V.cfg_of(facets="named", guard=0)
    cfg.update(kw)
    return cfg


def predict(ep, H, b, cfg, fe=None, stats=None):
    li, lk, ly = ep.labels(b)
    # v1_2.predict computes the validated base candidate; q is used only for
    # residuals and is otherwise not allowed to change base behavior.
    p = V.predict(ep, H, b, cfg, fe, stats)
    if not len(li):
        return p
    q, _ = Base.pooled_pair(ep, H, li, ly, cfg)
    return _kernel(ep, H, q, p, li, lk, ly, cfg.get("kernel_strength", 0.20))


def run_cfgs(B, D, ep, H, cfgs, stats=None):
    out = {}
    for name, cfg in cfgs.items():
        fe = V.fe_for(B, D, cfg)
        probs = {b: predict(ep, H, b, cfg, fe, None) for b in (0, 1, 3, 7, 15, 31)}
        br, ec = ep.score(probs)
        out[name] = dict(br=br, ec=ec, alc=br @ np.array([0.1, .2, .2, .2, .2, .1]), pair=ep.inst_pair.copy())
    return out
