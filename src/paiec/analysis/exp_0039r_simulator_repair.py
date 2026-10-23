"""EXP-0039R: simulator repair with enlarged-pool held-out folds.

Pre-registration: docs/preregistrations/EXP-0039R_simulator_repair_preregistration.md (locked).
Re-runs the EXP-0039 statistics (src/paiec/analysis/exp_0039_simulator_fidelity.py) against the
same 34 official pair records / 11 runs, with the predictive built from leave-one-benchmark-out R1
episodes over the enlarged public release data/_main_54074453 (rev 54074453da).

Inclusion (public rules only): response_type == binary, granularity == item, release == public,
>= 1 eligible pair (>= 80 distinct items) that the existing R1 builder (episodes.r1_draw) can draw.
Equal benchmark weights. Hyperparameters frozen from the shipped archives (fit on bc8204d811);
for the benchmarks that are also bc8204d811 folds the LOBO fit (ladder.hyper_for) is used so that
every benchmark's predictions are blind to its own responses.

The enlarged release ships no items.parquet, so items are keyed by item_id and item_features facets
are unavailable (the v1_2/v1_3 engines fall back exactly to their no-facet path).
Response trials are aggregated per (subject, item) as (n, sum, first-trial y): Brier over the
evaluation rows is computed exactly from these sufficient statistics; the revealed label is the
first data-order trial (simulator reveal='first').

Steps (each through scripts/run_slot.py):
  --step compact   stream responses.parquet -> .tmp/exp0039r_cache/<bench>.npz
  --step check     package-equivalence assert (v1_3 zip) + builder-vs-Episodes exactness check
  --step score     R1 seeds 0-19 per benchmark, 4 version engines -> per-instance ALC
  --step analyse   EXP-0039 statistics (primary composition, independent slots, 7-benchmark stratified)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np

from paiec import paths
from paiec.analysis import exp_0039_simulator_fidelity as F
from paiec.evaluation import episodes as E
from paiec.evaluation.simulator import _hseed
from paiec.models import ladder as Lad, v1_1 as V11, v1_2 as V2

NEW = paths.DATA / "_main_54074453"
CACHE = paths.ROOT / ".tmp" / "exp0039r_cache"
OUT_DIR = paths.RUNS / "preregistration"
SCORES = OUT_DIR / "exp_0039r_scores_2026-10-15.json"
OUT = OUT_DIR / "exp_0039r_simulator_repair_2026-10-15.json"
SEEDS = list(range(20))
ORIG = set(E.BENCHES)
N_SIM = 20000
RNG_SEED = 3939


def log(*a):
    print(*a, flush=True)


def peak_mb():
    try:
        import psutil
        return psutil.Process().memory_info().peak_wset / 2**20
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Inclusion
# ---------------------------------------------------------------------------
def candidates():
    """Public-rule candidates; metadata re-read from benchmarks.parquet (not from E7 summaries)."""
    import pandas as pd
    out = []
    for d in sorted(p for p in NEW.iterdir() if p.is_dir()):
        ft = d / "formatted_tables"
        if not (ft / "benchmarks.parquet").exists() or not (ft / "responses.parquet").exists():
            continue
        b = pd.read_parquet(ft / "benchmarks.parquet").iloc[0]
        if E._s(b.response_type) == "binary" and E._s(b.granularity) == "item" and E._s(b.release) == "public":
            out.append(d.name)
    return out


# ---------------------------------------------------------------------------
# Compaction (streaming, exact sufficient statistics)
# ---------------------------------------------------------------------------
def compact_one(bench, path=None):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    path = path or NEW / bench / "formatted_tables" / "responses.parquet"
    pf = pq.ParquetFile(path)
    sid_ix, iid_ix = {}, {}
    parts = []
    offset = 0
    for g in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(g, columns=["subject_id", "item_id", "response"])
        nrow = t.num_rows
        r = t["response"]
        keep = pc.fill_null(pc.or_(pc.equal(r, 0.0), pc.equal(r, 1.0)), False)
        order = np.flatnonzero(keep.to_numpy(zero_copy_only=False)) + offset
        offset += nrow
        t = t.filter(keep)
        if not t.num_rows:
            continue
        codes = []
        for col, ix in (("subject_id", sid_ix), ("item_id", iid_ix)):
            dct = pc.dictionary_encode(t[col].combine_chunks())
            if isinstance(dct, pa.ChunkedArray):
                dct = dct.combine_chunks()
            m = np.array([ix.setdefault(v, len(ix)) for v in dct.dictionary.to_pylist()], np.int64)
            codes.append(m[dct.indices.to_numpy(zero_copy_only=False)])
        y = t["response"].to_numpy().astype(np.int64)
        cell = (codes[0] << 32) | codes[1]
        u, first, inv = np.unique(cell, return_index=True, return_inverse=True)
        parts.append((u, np.bincount(inv, minlength=len(u)), np.bincount(inv, y, len(u)).astype(np.int64),
                      y[first], order[first]))
    if not parts:
        return None
    cell = np.concatenate([p[0] for p in parts])
    n = np.concatenate([p[1] for p in parts]); s = np.concatenate([p[2] for p in parts])
    fy = np.concatenate([p[3] for p in parts]); fo = np.concatenate([p[4] for p in parts])
    u, inv = np.unique(cell, return_inverse=True)
    N = np.bincount(inv, n, len(u)).astype(np.int64)
    S_ = np.bincount(inv, s, len(u)).astype(np.int64)
    srt = np.lexsort((fo, inv))                      # earliest data-order row per cell first
    first_pos = srt[np.r_[0, np.flatnonzero(np.diff(inv[srt])) + 1]]
    FY = fy[first_pos]
    sid = (u >> 32).astype(np.int64)
    iid = (u & 0xFFFFFFFF).astype(np.int64)
    sids = np.array(list(sid_ix), dtype=object)
    iids = np.array(list(iid_ix), dtype=object)
    nitems = np.bincount(sid, minlength=len(sids))
    elig = np.flatnonzero(nitems >= E.MIN_ITEMS)
    elig = elig[np.argsort(sids[elig].astype(str))]    # pairs sorted by subject id string
    pmap = -np.ones(len(sids), np.int64); pmap[elig] = np.arange(len(elig))
    m = pmap[sid] >= 0
    used_items, item_key = np.unique(iid[m], return_inverse=True)
    return dict(cell_pair=pmap[sid[m]].astype(np.int32), cell_key=item_key.astype(np.int32),
                cell_n=N[m], cell_s=S_[m], cell_fy=FY[m].astype(np.int8),
                item_ids=iids[used_items].astype(str), pair_sid=sids[elig].astype(str), bench=np.array(bench),
                n_rows=int(N.sum()), n_rows_eligible=int(N[m].sum()))


def load_cache(bench):
    z = np.load(CACHE / f"{bench}.npz", allow_pickle=False)
    D = {k: z[k] for k in z.files}
    D["bench"] = str(D["bench"])
    return D


def _draw_groups_fast(sizes, limit, seed):
    """Exact equivalent of simulator.draw_groups(ineligible='skip') when every group has a distinct
    subject (true for r1_draw: one group per subject in B, unique phantom subjects): the subject
    count of every unchosen group is 0, so the key reduces to (groups chosen in its benchmark, tiebreak)."""
    sizes = {k: n for k, n in sizes.items() if n >= E.MIN_ITEMS}
    rng = random.Random(_hseed("draw", seed))
    tiebreak = {k: rng.random() for k in sorted(sizes)}
    by_b = {}
    for k in sizes:
        by_b.setdefault(k[1], []).append(k)
    for b in by_b:
        by_b[b].sort(key=lambda g: tiebreak[g])
    ptr = {b: 0 for b in by_b}
    bench_n = {b: 0 for b in by_b}
    remaining, chosen = limit, []
    while True:
        live = [b for b in by_b if ptr[b] < len(by_b[b])]
        if not live:
            return chosen
        b = min(live, key=lambda x: (bench_n[x], tiebreak[by_b[x][ptr[x]]]))
        k = by_b[b][ptr[b]]
        ptr[b] += 1
        if sizes[k] > remaining:
            continue
        chosen.append(k)
        remaining -= sizes[k]
        bench_n[b] += 1


def r1_draw_fast(sizes_B, pair_sid, bench, seed, limit=1000, max_tries=50):
    """episodes.r1_draw with identical RNG calls, without materialising item lists."""
    pub = E.public_pair_sizes()
    for t in range(max_tries):
        rng = np.random.default_rng(_hseed("r1", bench, seed, t) % 2**32)
        K = int(rng.integers(1, 5))
        groups = {}
        for p, sid in enumerate(pair_sid):
            groups[(str(sid), bench)] = int(sizes_B[p])
        for k in range(K - 1):
            ns = int(rng.integers(3, 41))
            for j in range(ns):
                groups[(f"ph{k}_{j}", f"phantom{k}")] = int(rng.choice(pub))
        if min(groups[(str(s), bench)] for s in pair_sid) > limit:
            continue                               # no B group can ever fit (RNG already consumed)
        chosen = _draw_groups_fast(groups, limit, _hseed("r1d", bench, seed, t) % 2**31)
        picked = [g for g in chosen if g[1] == bench]
        if picked:
            ix = {str(s): p for p, s in enumerate(pair_sid)}
            return sorted(ix[g[0]] for g in picked), K
    return [], 0


def drawable(D, seeds=SEEDS):
    sizes = np.bincount(D["cell_pair"], minlength=len(D["pair_sid"]))   # one cell per distinct item
    return {s: r1_draw_fast(sizes, D["pair_sid"], D["bench"], s) for s in seeds}


def check_fast_draw():
    """r1_draw_fast == episodes.r1_draw on the bc8204d811 folds, seeds 0-19."""
    n = 0
    for B in E.EVAL_FOLDS:
        D = E.compact(B)
        sizes = E.distinct_sizes(D)
        for s in SEEDS:
            assert r1_draw_fast(sizes, D["pair_sid"], B, s) == E.r1_draw(D, s), (B, s)
            n += 1
    return n


def step_compact(names=None):
    CACHE.mkdir(parents=True, exist_ok=True)
    names = names or candidates()
    log(f"candidates (binary/item/public): {len(names)}")
    t0 = time.perf_counter()
    meta = {}
    for k, b in enumerate(names):
        f = CACHE / f"{b}.npz"
        t1 = time.perf_counter()
        if not f.exists():
            D = compact_one(b)
            if D is None or len(D["pair_sid"]) == 0:
                meta[b] = dict(eligible_pairs=0)
                log(f"{k:3d} {b:34s} no eligible pair ({time.perf_counter()-t1:.1f}s)")
                continue
            np.savez(f, **D)
        D = load_cache(b)
        meta[b] = dict(eligible_pairs=int(len(D["pair_sid"])), rows_eligible=int(D["cell_n"].sum()),
                       accuracy=float(D["cell_s"].sum() / D["cell_n"].sum()))
        log(f"{k:3d} {b:34s} pairs={meta[b]['eligible_pairs']:5d} {time.perf_counter()-t1:6.1f}s "
            f"total {time.perf_counter()-t0:6.0f}s rss_peak {peak_mb():.0f}MB")
    (CACHE / "compact_meta.json").write_text(json.dumps(dict(meta=meta, seconds=time.perf_counter() - t0,
                                                              peak_mb=peak_mb()), indent=1))


# ---------------------------------------------------------------------------
# Episodes over aggregated cells (same hash seeds as simulator/episodes)
# ---------------------------------------------------------------------------
class CellEpisodes:
    def __init__(self, D, inst_pair, inst_seed, inst_group):
        self.D = {"key_jev": np.zeros(len(D["item_ids"]))}
        self.inst_pair = np.asarray(inst_pair, np.int64)
        self.inst_seed = np.asarray(inst_seed, np.int64)
        self.inst_group = np.asarray(inst_group, np.int64)
        self.n = len(self.inst_pair)
        bench = D["bench"]
        order = np.argsort(D["cell_pair"], kind="stable")
        cp, ck = D["cell_pair"][order], D["cell_key"][order]
        cn, cs, cf = D["cell_n"][order], D["cell_s"][order], D["cell_fy"][order]
        bounds = np.searchsorted(cp, np.arange(len(D["pair_sid"]) + 1))
        iids = D["item_ids"]
        self.acq, tk, ti, tn, ts = [], [], [], [], []
        for j, (p, sd) in enumerate(zip(self.inst_pair, self.inst_seed)):
            a, b = bounds[p], bounds[p + 1]
            keys = ck[a:b]
            ids = iids[keys]
            srt = sorted(ids.tolist())
            sid = str(D["pair_sid"][p])
            random.Random(_hseed("split", sid, bench, int(sd))).shuffle(srt)
            cut = (len(srt) + 1) // 2
            acq_ids = set(srt[:cut])
            is_acq = np.array([i in acq_ids for i in ids.tolist()])
            pos = {i: q for q, i in enumerate(ids.tolist())}
            ordl = sorted(acq_ids)
            random.Random(_hseed("order", sid, bench, int(sd))).shuffle(ordl)
            q = np.array([pos[i] for i in ordl], np.int64)
            self.acq.append(np.stack([keys[q].astype(np.int64), cf[a:b][q].astype(np.int64)], 1).reshape(-1, 2))
            ev = ~is_acq
            tk.append(keys[ev].astype(np.int64)); ti.append(np.full(int(ev.sum()), j))
            tn.append(cn[a:b][ev].astype(float)); ts.append(cs[a:b][ev].astype(float))
        self.t_key, self.t_inst = np.concatenate(tk), np.concatenate(ti)
        self.t_n, self.t_s = np.concatenate(tn), np.concatenate(ts)
        self.r_t = np.arange(len(self.t_key))

    labels = E.Episodes.labels

    def score(self, probs):
        br = np.zeros((self.n, 6))
        den = np.bincount(self.t_inst, self.t_n, self.n)
        for j, b in enumerate(E.BUDGETS):
            p = probs[b]
            se = self.t_n * p * p - 2 * p * self.t_s + self.t_s
            br[:, j] = np.bincount(self.t_inst, se, self.n) / den
        return br


def build(D, seeds):
    draws = drawable(D, seeds)
    ip, isd, ig = [], [], []
    for g, s in enumerate(seeds):
        pk, _ = draws[s]
        ip += pk; isd += [s] * len(pk); ig += [g] * len(pk)
    return CellEpisodes(D, ip, isd, ig) if ip else None


# ---------------------------------------------------------------------------
# Hyperparameters (frozen archive params; LOBO for bc8204d811 benchmarks)
# ---------------------------------------------------------------------------
def archive_params():
    import zipfile
    out = {}
    for name, v in F.VERSIONS.items():
        with zipfile.ZipFile(paths.DIST / v["zip"]) as z:
            out[name] = json.loads(z.read(v["params"]))
    return out


def H_for(name, bench, params):
    if bench in ORIG:
        return V11.H_fold(bench)
    return F._engine_H(params[name])


def engine_alc(name, ep, H):
    cfg = F.VERSIONS[name]["cfg"]
    if F.VERSIONS[name]["engine"] == "v1_1":
        probs = {b: V11.predict(ep, H, b, cfg) for b in E.BUDGETS}
    else:
        probs = {b: V2.predict(ep, H, b, cfg, None) for b in E.BUDGETS}   # no item_features available
    return ep.score(probs) @ E.W


def step_check():
    """(a) v1_3 package equivalence (EXP-0039 check); (b) CellEpisodes == Episodes on bc8204d811
    real_webagents/researchcodebench with item_id keys (no content merge), R1 seeds 0-4, all engines."""
    out = {"package": F.check_packages()}
    params = archive_params()
    worst = 0.0
    for B in ("real_webagents", "researchcodebench"):
        D_old = dict(E.compact(B))
        D_old["item_key"] = np.arange(len(D_old["item_ids"]), dtype=np.int32)
        D_old["key_jev"] = np.zeros(len(D_old["item_ids"]))
        ref = E.build(D_old, "R1", list(range(5)))
        D_new = compact_one(B, E.DATA / B / "response.parquet")
        D_new["bench"] = B
        ep = build(D_new, list(range(5)))
        assert list(map(str, D_new["pair_sid"][ep.inst_pair])) == list(map(str, D_old["pair_sid"][ref.inst_pair]))
        for name in F.VERSIONS:
            H = F._engine_H(params[name])
            cfg = F.VERSIONS[name]["cfg"]
            if F.VERSIONS[name]["engine"] == "v1_1":
                pr = {b: V11.predict(ref, H, b, cfg) for b in E.BUDGETS}
            else:
                pr = {b: V2.predict(ref, H, b, cfg, None) for b in E.BUDGETS}
            a_ref = ref.score(pr)[0] @ E.W
            a_new = engine_alc(name, ep, H)
            worst = max(worst, float(np.abs(a_ref - a_new).max()))
        log(f"  [check builder] {B}: n_inst={ep.n} worst |dALC|={worst:.2e}")
    out["builder_max_abs_alc_delta"] = worst
    assert worst < 1e-9, worst
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "exp_0039r_checks_2026-10-15.json").write_text(json.dumps(out, indent=1))
    return out


def step_score():
    params = archive_params()
    meta = json.loads((CACHE / "compact_meta.json").read_text())["meta"]
    names = [b for b, m in meta.items() if m["eligible_pairs"] > 0]
    per, excluded = {}, {}
    t0 = time.perf_counter()
    n_eq = check_fast_draw()
    log(f"fast R1 draw == episodes.r1_draw on {n_eq} fold-seed draws ({time.perf_counter()-t0:.1f}s)")
    for k, B in enumerate(names):
        D = load_cache(B)
        ep = build(D, SEEDS)
        if ep is None:
            excluded[B] = "no R1-drawable pair (all eligible pairs exceed the 1000-item draw limit)"
            log(f"{k:3d} {B:34s} excluded: not R1-drawable")
            continue
        alc = {name: engine_alc(name, ep, H_for(name, B, params)).tolist() for name in F.VERSIONS}
        acc = (np.bincount(ep.t_inst, ep.t_s, ep.n) / np.bincount(ep.t_inst, ep.t_n, ep.n)).tolist()
        n_eval = np.bincount(ep.t_inst, minlength=ep.n).tolist()
        per[B] = dict(seed=ep.inst_seed.tolist(), pair=ep.inst_pair.tolist(), alc=alc, eval_acc=acc,
                      n_eval_items=n_eval, hypers="LOBO bc8204d811" if B in ORIG else "frozen archive")
        log(f"{k:3d} {B:34s} n_inst={ep.n:4d} v1_3={np.mean(alc['v1_3']):.4f} acc={np.mean(acc):.3f} "
            f"{time.perf_counter()-t0:6.0f}s rss_peak {peak_mb():.0f}MB")
        del ep, D
    SCORES.write_text(json.dumps(dict(per=per, excluded=excluded, fast_draw_equivalence_checks=n_eq, seconds=time.perf_counter() - t0,
                                      peak_mb=peak_mb()), indent=0))


# ---------------------------------------------------------------------------
# Analysis (EXP-0039 statistics, generalised to many benchmarks)
# ---------------------------------------------------------------------------
class Pred:
    def __init__(self, per, version):
        self.benches = sorted(per)
        self.units = []           # per bench: list of arrays of pair ALCs per seed draw
        for B in self.benches:
            seed = np.array(per[B]["seed"]); v = np.array(per[B]["alc"][version])
            self.units.append([v[seed == s] for s in SEEDS if (seed == s).any()])

    def run(self, n, rng, mode):
        nb = len(self.benches)
        if mode == "stratified7":
            bs = rng.choice(nb, 7, replace=False)
            got = []
            draws = [self.units[b][rng.integers(len(self.units[b]))] for b in bs]
            for d in draws:
                got.append(d[rng.integers(len(d))])
            for _ in range(n - 7):
                j = rng.integers(7)
                d = self.units[bs[j]][rng.integers(len(self.units[bs[j]]))]
                got.append(d[rng.integers(len(d))])
            return got
        got = []
        while len(got) < n:
            b = rng.integers(nb)
            d = self.units[b][rng.integers(len(self.units[b]))]
            if mode == "episode":
                got.extend(d[rng.permutation(len(d))].tolist())
            else:
                got.append(float(d[rng.integers(len(d))]))
        return got[:n]


def analyse(per, pairs_obs, runs, mode, rng):
    preds = {v: Pred(per, v) for v in F.VERSIONS}
    n_of = {"v1": 8, "v1_1": 8, "v1_2": 9, "v1_3": 9}
    sims = {(v, n): np.array([preds[v].run(n, rng, mode) for _ in range(N_SIM)])
            for v in F.VERSIONS for n in (8, 9)}
    res = {"composition": mode}
    rl = []
    for r in runs:
        n = r["n"] or n_of[r["version"]]
        m = sims[(r["version"], n)].mean(1)
        q05, q95 = np.quantile(m, [0.05, 0.95])
        alt = sims[(r["version"], 17 - n)].mean(1)
        a05, a95 = np.quantile(alt, [0.05, 0.95])
        rl.append(dict(r, n_used=n, n_assumed=r["n"] is None, pred_mean=float(m.mean()),
                       pi90=[float(q05), float(q95)], pit=F.pit(m, r["alc"]),
                       outside=bool(r["alc"] < q05 or r["alc"] > q95),
                       outside_other_n=bool(r["alc"] < a05 or r["alc"] > a95)))
    res["runs"] = rl
    res["n_runs_outside"] = int(sum(x["outside"] for x in rl))
    marg, pp = {}, []
    for v in F.VERSIONS:
        s = sims[(v, n_of[v])].ravel()
        q05, q95 = np.quantile(s, [0.05, 0.95])
        marg[v] = dict(q05=float(q05), q50=float(np.median(s)), q95=float(q95), mean=float(s.mean()))
        for x in pairs_obs[v]:
            pp.append(dict(version=v, alc=x, pit=F.pit(s, x),
                           tail="low" if x < q05 else "high" if x > q95 else ""))
    T_lo = sum((sims[(v, n_of[v])] < marg[v]["q05"]).sum(1) for v in F.VERSIONS)
    T_hi = sum((sims[(v, n_of[v])] > marg[v]["q95"]).sum(1) for v in F.VERSIONS)
    T = T_lo + T_hi
    lo, hi = np.quantile(T, [0.05, 0.95], method="inverted_cdf")
    obs = sum(1 for p in pp if p["tail"])
    res["pair_marginals"], res["pairs"] = marg, pp
    res["tail_count"] = dict(observed=int(obs), observed_low=sum(p["tail"] == "low" for p in pp),
                             observed_high=sum(p["tail"] == "high" for p in pp), expected=float(T.mean()),
                             range90=[int(lo), int(hi)], achieved_coverage=float(((T >= lo) & (T <= hi)).mean()),
                             p_upper=float((T >= obs).mean()), p_lower=float((T <= obs).mean()),
                             outside=bool(obs < lo or obs > hi))
    sim_sorted = np.hstack([np.sort(sims[(v, n_of[v])], 1) for v in F.VERSIONS])
    obs_sorted = np.concatenate([np.sort(pairs_obs[v]) for v in F.VERSIONS])
    res["envelope"] = F.extreme_rank_envelope(sim_sorted, obs_sorted)
    allsim = np.concatenate([sims[(v, n_of[v])].ravel() for v in F.VERSIONS])
    allobs = np.concatenate([pairs_obs[v] for v in F.VERSIONS])
    res["pooled"] = dict(obs_mean=float(allobs.mean()), pred_mean=float(allsim.mean()),
                         obs_sd=float(allobs.std(ddof=1)), pred_sd=float(allsim.std()),
                         obs_quantiles=np.quantile(allobs, [0.05, 0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
                         pred_quantiles=np.quantile(allsim, [0.05, 0.1, 0.25, 0.5, 0.75, 0.9]).tolist())
    rep = {}
    for v in ("v1_1", "v1_2"):
        o = np.array([r["alc"] for r in runs if r["version"] == v])
        m = sims[(v, n_of[v])].mean(1)
        k = len(o)
        sds = np.array([m[i:i + k].std(ddof=1) for i in range(0, N_SIM - k + 1, k)])
        rep[v] = dict(n_runs=k, obs_sd=float(o.std(ddof=1)), pred_run_sd=float(m.std(ddof=1)),
                      pit_of_obs_sd=F.pit(sds, o.std(ddof=1)), sd_range90=np.quantile(sds, [0.05, 0.95]).tolist())
    res["repeat_sd"] = rep
    env_ok, tail_ok, runs_ok = res["envelope"]["observed_inside"], not res["tail_count"]["outside"], res["n_runs_outside"] == 0
    res["criteria"] = dict(runs_inside_pi90=runs_ok, tail_count_inside=tail_ok, envelope_inside=env_ok)
    return res


def outcome(c):
    if all(c.values()):
        return "All three EXP-0039 criteria pass"
    if not c["tail_count_inside"] and not c["envelope_inside"]:
        return "Both fail"
    if c["tail_count_inside"] != c["envelope_inside"]:
        return "Envelope passes but tail count fails, or vice versa"
    return "Tail count and envelope pass; run-level criterion fails (not a row of the locked table)"


def step_analyse():
    t0 = time.perf_counter()
    S = json.loads(SCORES.read_text())
    per = S["per"]
    pairs_obs, _, runs = F.official_records()
    rng = np.random.default_rng(RNG_SEED)
    out = {m: analyse(per, pairs_obs, runs, m, rng) for m in ("episode", "independent", "stratified7")}
    acc = np.concatenate([per[B]["eval_acc"] for B in per])
    w = np.concatenate([np.full(len(per[B]["eval_acc"]), 1 / len(per[B]["eval_acc"])) for B in per])
    w /= w.sum()                                       # equal benchmark weights
    o = np.argsort(acc); cw = np.cumsum(w[o])
    wq = lambda q: float(acc[o][np.searchsorted(cw, q)])
    nev = np.concatenate([per[B]["n_eval_items"] for B in per])
    meta = json.loads((CACHE / "compact_meta.json").read_text())
    report = {
        "experiment_id": "EXP-0039R", "preregistered": True, "date": "2026-10-15",
        "data": "data/_main_54074453 (rev 54074453da)", "n_candidates_binary_item_public": len(meta["meta"]),
        "n_with_eligible_pair": sum(1 for m in meta["meta"].values() if m["eligible_pairs"] > 0),
        "n_included_r1_drawable": len(per), "excluded_not_drawable": S["excluded"],
        "orig_benchmarks_included": sorted(set(per) & ORIG), "seeds": SEEDS, "n_sim": N_SIM, "rng_seed": RNG_SEED,
        "n_instances": int(sum(len(per[B]["seed"]) for B in per)),
        "local_mean_alc_equal_bench": {v: float(np.mean([np.mean(per[B]["alc"][v]) for B in per])) for v in F.VERSIONS},
        "base_rate": dict(equal_bench_weighted_median_acc=wq(0.5), frac_near_perfect=float(w[acc >= 0.95].sum()),
                          frac_near_zero=float(w[acc <= 0.05].sum()), quantiles={q: wq(q) for q in (0.1, 0.25, 0.75, 0.9)},
                          unweighted_pairs=dict(median=float(np.median(acc)), frac_near_zero=float((acc <= 0.05).mean()),
                                                frac_near_perfect=float((acc >= 0.95).mean())),
                          bench_share_any_near_zero_pair=float(np.mean([np.mean(np.array(per[B]["eval_acc"]) <= 0.05) > 0 for B in per])),
                          platform_reference="corrected q4_estimate_reproduction: platform median 0.27, 14.7% pairs acc<=0.05, 0% near-perfect; old local 0.41"),
        "eval_pool_items_quantiles": np.quantile(nev, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist(),
        "primary": out["episode"], "sensitivity_independent_slots": out["independent"],
        "secondary_stratified7": out["stratified7"],
        "decision_row_primary": outcome(out["episode"]["criteria"]),
        "decision_row_stratified7": outcome(out["stratified7"]["criteria"]),
        "cost": dict(compact_seconds=meta["seconds"], compact_peak_mb=meta["peak_mb"], score_seconds=S["seconds"],
                     score_peak_mb=S["peak_mb"], analyse_seconds=time.perf_counter() - t0, analyse_peak_mb=peak_mb()),
    }
    OUT.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for k in ("primary", "sensitivity_independent_slots", "secondary_stratified7"):
        p = report[k]
        log(f"[{k}] runs outside {p['n_runs_outside']}/11; tail {p['tail_count']['observed']} "
            f"range {p['tail_count']['range90']}; envelope inside={p['envelope']['observed_inside']} "
            f"p={p['envelope']['p_global']:.3f}; pooled obs {p['pooled']['obs_mean']:.3f} pred {p['pooled']['pred_mean']:.3f}")
    log(f"decision row (primary): {report['decision_row_primary']}")
    log(f"base rate {report['base_rate']}")
    return report


def composition(per, thr=0.09):
    """Equal-benchmark-weighted decomposition of the low-ALC (v1_3 < thr) tail by pair accuracy."""
    A, C, W, Bn = [], [], [], []
    for B, v in per.items():
        a = np.array(v["alc"]["v1_3"]); A += list(a); C += list(v["eval_acc"])
        W += [1 / len(a)] * len(a); Bn += [B] * len(a)
    A, C, W, Bn = np.array(A), np.array(C), np.array(W), np.array(Bn)
    W = W / W.sum()
    low = A < thr
    cnt = {B: int((low & (Bn == B)).sum()) for B in per}
    return dict(threshold=thr, n_low_instances=int(low.sum()), weighted_p_low=float(W[low].sum()),
                share_low_from_acc_le_0_05=float(W[low & (C <= .05)].sum() / W[low].sum()),
                share_low_from_acc_le_0_15=float(W[low & (C <= .15)].sum() / W[low].sum()),
                share_low_from_acc_ge_0_95=float(W[low & (C >= .95)].sum() / W[low].sum()),
                mean_alc_acc_le_0_05=float(np.average(A[C <= .05], weights=W[C <= .05])),
                mean_alc_acc_ge_0_95=float(np.average(A[C >= .95], weights=W[C >= .95])),
                low_instances_by_bench=dict(sorted(cnt.items(), key=lambda kv: -kv[1])))


def step_loo(top=6):
    """Red-team sensitivity: drop each top low-ALC contributor (and crow+monkey jointly), primary composition."""
    t0 = time.perf_counter()
    per = json.loads(SCORES.read_text())["per"]
    pairs_obs, _, runs = F.official_records()
    comp = composition(per)
    tops = [b for b, n in comp["low_instances_by_bench"].items() if n > 0][:top]
    sets = [[b] for b in tops] + [["crow", "monkey_power_laws"], tops]
    res = []
    for drop in sets:
        sub = {B: v for B, v in per.items() if B not in drop}
        a = analyse(sub, pairs_obs, runs, "episode", np.random.default_rng(RNG_SEED))
        v13 = [r for r in a["runs"] if r["version"] == "v1_3"][0]
        res.append(dict(dropped=drop, n_bench=len(sub), criteria=a["criteria"], decision_row=outcome(a["criteria"]),
                        n_runs_outside=a["n_runs_outside"], tail=a["tail_count"]["observed"],
                        tail_range90=a["tail_count"]["range90"], envelope_p=a["envelope"]["p_global"],
                        v1_3_rank=v13["pit"], pooled_pred_mean=a["pooled"]["pred_mean"]))
        log(f"drop {drop}: {res[-1]['decision_row']} runs_out={a['n_runs_outside']} tail={res[-1]['tail']} "
            f"{res[-1]['tail_range90']} env_p={res[-1]['envelope_p']:.3f} v1_3 rank={v13['pit']:.3f}")
    out = dict(experiment_id="EXP-0039R", step="leave-one-benchmark-out sensitivity (red-team)", rng_seed=RNG_SEED,
               composition=comp, drops=res, seconds=time.perf_counter() - t0, peak_mb=peak_mb())
    (OUT_DIR / "exp_0039r_loo_sensitivity_2026-10-16.json").write_text(json.dumps(out, indent=1))
    R = json.loads(OUT.read_text()); R["composition_low_tail"] = comp
    OUT.write_text(json.dumps(R, indent=1) + chr(10), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["compact", "check", "score", "analyse", "loo"])
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    {"compact": lambda: step_compact(a.only), "check": step_check, "score": step_score,
     "analyse": step_analyse, "loo": step_loo}[a.step]()
