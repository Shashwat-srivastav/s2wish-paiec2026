"""Realistic LOBO episode generator on the real public pool (Phase 2).

Design (see reports/15_PHASE2_MODEL.md):
* Data are compacted ONE benchmark at a time into small numpy arrays (content is
  hashed into a runtime item key and discarded) and cached under
  artifacts/episode_cache/ (E:, never C:).
* Split / stream order reuse the simulator's hash seeds exactly
  (``_hseed("split", sid, bid, seed)``, ``_hseed("order", ...)``), so an episode
  built here reproduces ``simulator.prepare_pairs`` (verified in ladder.py
  ``--check`` against the packaged v1 model run through simulator.run_direct).
* Acquisition = accept-all over the fixed random stream (== random policy in
  distribution); revealed trial = first data-order row (simulator reveal="first").
* Draw regimes for a held-out benchmark B:
    R1 ~1000-item draw: B's eligible groups + (K-1) phantom hidden benchmarks
       (K ~ U{1..4}) with group sizes resampled from public pair sizes; the
       simulator's ``draw_groups`` preference rule (fewest groups per benchmark,
       then per subject, random tie-break, skip groups that no longer fit) picks
       the groups; only B's chosen pairs are scored. Draws with no B pair are
       re-drawn (we condition on B being evaluated).
    R2 one pair per episode (no cross-pair evidence at all).
    R3 all eligible pairs of B in one episode.
* Pooling groups: every pair instance carries a ``group`` id = (episode, bench);
  predictors may pool evidence only within a group (== same benchmark_id at
  runtime in the same run).
"""
from __future__ import annotations

import hashlib
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from paiec import paths
from paiec.evaluation.simulator import _hseed, draw_groups

DATA = paths.DATA
CACHE = paths.EPISODE_CACHE
BENCHES = ["matharena", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench"]
EVAL_FOLDS = ["multi_swebench", "real_webagents", "researchcodebench", "matharena"]
SUBJECT_FIELDS = ("normalized_name", "provider", "release_date", "access_date", "harness",
                  "reasoning_effort", "harness_version", "subject_features_extra")
MIN_ITEMS = 80
BUDGETS = (0, 1, 3, 7, 15, 31)
W = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])


def _s(v):
    if v is None or (isinstance(v, float) and v != v):
        return ""
    return str(v).strip()


def name_key(subj: dict) -> str:
    """Hierarchical cross-benchmark name link: normalized_name, else model_config
    with any '--effort' suffix stripped (report 14 rec. 2). '' if neither."""
    n = subj.get("normalized_name", "")
    if n:
        return "n:" + n.lower()
    m = re.search(r"model_config=([^;]+)", subj.get("subject_features_extra", ""))
    if m:
        return "c:" + m.group(1).split("--")[0].strip().lower()
    return ""


def release_years(s: str) -> float:
    m = re.match(r"(\d{4})-(\d{2})(?:-(\d{2}))?", s or "")
    if not m:
        return float("nan")
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3) or 15)
    return y + (mo - 1) / 12 + (d - 1) / 365


def _item_key(content, feats, inter) -> str:
    return hashlib.sha1(f"{content}\x1f{feats}\x1f{inter}".encode("utf-8", "replace")).hexdigest()


def compact(bench: str, force=False) -> dict:
    """Binary rows only. Returns dict of numpy arrays + per-pair metadata."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{bench}.npz"
    if path.exists() and not force:
        z = np.load(path, allow_pickle=True)
        if "key_content" in z.files:
            return {k: z[k] for k in z.files}
    import pandas as pd
    d = DATA / bench
    resp = pd.read_parquet(d / "response.parquet", columns=["subject_id", "item_id", "response", "interactors"])
    resp = resp[resp.response.isin([0.0, 1.0])].reset_index(drop=True)
    items = pd.read_parquet(d / "items.parquet", columns=["item_id", "content", "item_features"])
    items["content"] = items.content.map(_s)
    items["item_features"] = items.item_features.map(_s)
    item_content = dict(zip(items.item_id, items.content))
    # interactors are response-level and empty in all public data; key on item fields + ''
    ikey_str = {r.item_id: _item_key(r.content, r.item_features, "") for r in items.itertuples()}
    del items
    subs = pd.read_parquet(d / "subjects.parquet")
    jev = {}
    jp = paths.RUNS / "jev" / f"jev_out_{bench}.parquet"
    if jp.exists():
        j = pd.read_parquet(jp, columns=["item_id", "jev_difficulty"])
        jev = dict(zip(j.item_id, j.jev_difficulty))
    # eligible pairs (>= 80 distinct item_ids)
    n_items = resp.groupby("subject_id").item_id.nunique()
    elig = sorted(n_items[n_items >= MIN_ITEMS].index)
    resp = resp[resp.subject_id.isin(elig)].reset_index(drop=True)
    sid_ix = {s: k for k, s in enumerate(elig)}
    item_ids = sorted(resp.item_id.unique())
    iid_ix = {s: k for k, s in enumerate(item_ids)}
    keys = sorted({ikey_str[i] for i in item_ids})
    k_ix = {s: k for k, s in enumerate(keys)}
    key_content = {}
    for iid in item_ids:
        key_content.setdefault(k_ix[ikey_str[iid]], item_content.get(iid, ""))
    subs = subs.set_index("subject_id")
    meta = []
    for s in elig:
        row = subs.loc[s]
        subj = {f: _s(row.get(f)) for f in SUBJECT_FIELDS}
        meta.append([s, "\x1f".join(subj[f] for f in SUBJECT_FIELDS), name_key(subj),
                     release_years(subj["release_date"]), subj["provider"]])
    out = {
        "row_pair": resp.subject_id.map(sid_ix).to_numpy(np.int32),
        "row_item": resp.item_id.map(iid_ix).to_numpy(np.int32),
        "row_y": resp.response.to_numpy(np.int8),
        "item_ids": np.array(item_ids, dtype=object),
        "item_key": np.array([k_ix[ikey_str[i]] for i in item_ids], np.int32),
        "key_content": np.array([key_content.get(k, "") for k in range(len(keys))], dtype=object),
        "key_jev": np.full(len(keys), np.nan),
        "pair_sid": np.array([m[0] for m in meta], dtype=object),
        "pair_exact": np.array([m[1] for m in meta], dtype=object),
        "pair_name": np.array([m[2] for m in meta], dtype=object),
        "pair_release": np.array([m[3] for m in meta], float),
        "pair_provider": np.array([m[4] for m in meta], dtype=object),
        "bench": np.array(bench, dtype=object),
    }
    for iid, k in zip(item_ids, out["item_key"]):
        v = jev.get(iid)
        if v is not None and np.isfinite(v):
            out["key_jev"][k] = v
    np.savez(path, **out)
    return out


# ---------------------------------------------------------------------------
# Per-pair split / stream (simulator-identical hashing)
# ---------------------------------------------------------------------------
def pair_streams(D: dict, seed: int):
    """For every pair: (acq list of (key, y) in stream order, eval rows (key, y))."""
    bench = str(D["bench"])
    order = np.argsort(D["row_pair"], kind="stable")
    rp, ri, ry = D["row_pair"][order], D["row_item"][order], D["row_y"][order]
    bounds = np.searchsorted(rp, np.arange(len(D["pair_sid"]) + 1))
    res = []
    for p, sid in enumerate(D["pair_sid"]):
        a, b = bounds[p], bounds[p + 1]
        its, ys = ri[a:b], ry[a:b]
        ids = sorted({D["item_ids"][i] for i in its})
        sh = ids[:]
        random.Random(_hseed("split", sid, bench, seed)).shuffle(sh)
        cut = (len(sh) + 1) // 2
        acq_ids = set(sh[:cut])
        first = {}
        ev_k, ev_y = [], []
        for i, y in zip(its, ys):
            iid = D["item_ids"][i]
            if iid in acq_ids:
                first.setdefault(iid, (D["item_key"][i], int(y)))
            else:
                ev_k.append(D["item_key"][i]); ev_y.append(int(y))
        ordl = sorted(acq_ids)
        random.Random(_hseed("order", sid, bench, seed)).shuffle(ordl)
        acq = np.array([first[i] for i in ordl], np.int64).reshape(-1, 2)
        res.append((acq, np.array(ev_k, np.int64), np.array(ev_y, np.int8)))
    return res


def distinct_sizes(D):
    pi = np.unique(D["row_pair"].astype(np.int64) * 10**7 + D["row_item"])
    return np.bincount(pi // 10**7, minlength=len(D["pair_sid"]))


class Episodes:
    """A batch of pair instances, each with its own split seed and pooling group."""

    def __init__(self, D, inst_pair, inst_seed, inst_group):
        self.D = D
        self.inst_pair = np.asarray(inst_pair, np.int64)
        self.inst_seed = np.asarray(inst_seed, np.int64)
        self.inst_group = np.asarray(inst_group, np.int64)
        streams = {}
        for s in np.unique(self.inst_seed):
            streams[int(s)] = pair_streams(D, int(s))
        self.acq = [streams[int(s)][int(p)][0] for p, s in zip(self.inst_pair, self.inst_seed)]
        ev = [streams[int(s)][int(p)][1:] for p, s in zip(self.inst_pair, self.inst_seed)]
        # unique targets per instance
        tk, ti, rows_t, rows_y, rows_i = [], [], [], [], []
        off = 0
        for n, (k, y) in enumerate(ev):
            u, inv = np.unique(k, return_inverse=True)
            tk.append(u); ti.append(np.full(len(u), n))
            rows_t.append(inv + off); rows_y.append(y); rows_i.append(np.full(len(y), n))
            off += len(u)
        self.t_key, self.t_inst = np.concatenate(tk), np.concatenate(ti)
        self.r_t, self.r_y, self.r_inst = np.concatenate(rows_t), np.concatenate(rows_y).astype(float), np.concatenate(rows_i)
        self.n = len(self.inst_pair)

    def labels(self, b):
        """Concatenated labels at cap b: (inst, key, y)."""
        L_i, L_k, L_y = [], [], []
        for n, a in enumerate(self.acq):
            m = min(b, len(a))
            if m:
                L_i.append(np.full(m, n)); L_k.append(a[:m, 0]); L_y.append(a[:m, 1])
        if not L_i:
            z = np.zeros(0, np.int64)
            return z, z, z.astype(float)
        return np.concatenate(L_i), np.concatenate(L_k), np.concatenate(L_y).astype(float)

    def score(self, probs_by_budget):
        """probs_by_budget: {b: array over targets} -> per-instance brier[n,6], ece[n,6]."""
        br = np.zeros((self.n, 6)); ec = np.zeros((self.n, 6))
        cnt = np.bincount(self.r_inst, minlength=self.n)
        for j, b in enumerate(BUDGETS):
            p = probs_by_budget[b][self.r_t]
            br[:, j] = np.bincount(self.r_inst, (p - self.r_y) ** 2, minlength=self.n) / cnt
            bins = np.minimum(np.floor(p * 10).astype(int), 9)
            g = self.r_inst * 10 + bins
            c = np.bincount(g, minlength=self.n * 10).reshape(self.n, 10)
            sp = np.bincount(g, p, minlength=self.n * 10).reshape(self.n, 10)
            sy = np.bincount(g, self.r_y, minlength=self.n * 10).reshape(self.n, 10)
            ec[:, j] = np.abs(sp - sy).sum(1) / cnt
        return br, ec


# ---------------------------------------------------------------------------
# Draw regimes
# ---------------------------------------------------------------------------
_PUBLIC_SIZES = None


def public_pair_sizes():
    global _PUBLIC_SIZES
    if _PUBLIC_SIZES is None:
        sz = []
        for b in BENCHES:
            sz.extend(distinct_sizes(compact(b)).tolist())
        _PUBLIC_SIZES = np.array(sz)
    return _PUBLIC_SIZES


def r1_draw(D, seed, limit=1000, max_tries=50):
    """Pairs of D chosen by the README draw rule in a mixed hidden pool."""
    sizes_B = distinct_sizes(D)
    bench = str(D["bench"])
    pub = public_pair_sizes()
    for t in range(max_tries):
        rng = np.random.default_rng(_hseed("r1", bench, seed, t) % 2**32)
        K = int(rng.integers(1, 5))
        groups = {}
        for p, sid in enumerate(D["pair_sid"]):
            groups[(str(sid), bench)] = int(sizes_B[p])
        for k in range(K - 1):
            ns = int(rng.integers(3, 41))
            for j in range(ns):
                groups[(f"ph{k}_{j}", f"phantom{k}")] = int(rng.choice(pub))
        fake = {g: [{"item_id": str(i)} for i in range(n)] for g, n in groups.items()}
        chosen = draw_groups(fake, limit, seed=_hseed("r1d", bench, seed, t) % 2**31)
        picked = [g for g in chosen if g[1] == bench]
        if picked:
            ix = {str(s): p for p, s in enumerate(D["pair_sid"])}
            return sorted(ix[g[0]] for g in picked), K
    return [], 0


def build(D, regime, seeds):
    """Episodes object for one held-out benchmark under a regime."""
    P = len(D["pair_sid"])
    ip, isd, ig = [], [], []
    info = []
    g = 0
    for s in seeds:
        if regime == "R3":
            ip += list(range(P)); isd += [s] * P; ig += [g] * P; g += 1
        elif regime == "R2":
            for p in range(P):
                ip.append(p); isd.append(s); ig.append(g); g += 1
        elif regime == "R1":
            pk, K = r1_draw(D, s)
            info.append((len(pk), K))
            ip += pk; isd += [s] * len(pk); ig += [g] * len(pk); g += 1
        else:
            raise ValueError(regime)
    ep = Episodes(D, ip, isd, ig)
    ep.draw_info = info
    return ep
