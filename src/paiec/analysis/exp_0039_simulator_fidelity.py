"""EXP-0039 (F2): does the local simulator reproduce the distribution of observed platform scores?

Preregistered in docs/preregistrations/2026-10-14_comprehensive_research_preregistration.md section 2.
Blind, version-matched R1 predictive distributions (LOBO hyperparameters fit on public training
folds only, seeds 0-99, accept-all acquisition) are compared with every saved official run-level
ALC and every saved official per-pair ALC. No fitting to official scores; anonymous subject or
benchmark identifiers are never read (only the per-run pair count and pair ALC values are used).

Stages
  1. Verify archive SHA-256s and package-vs-engine equivalence for v1, v1_1, v1_2, v1_3.
  2. Score identical R1 episodes (4 folds x 100 seeds) with each version's engine.
  3. Monte Carlo predictive: a simulated official run of n pairs is composed of whole R1 draws
     (fold uniform, seed uniform, pairs of that draw in random order) until n pairs are filled
     (primary); sensitivity: every pair slot from an independent draw.
  4. Run-level 90% predictive intervals and ranks; per-pair marginal 90% intervals and tail count
     with its 90% predictive range from joint simulation of all saved detailed runs; a
     simultaneous (extreme-rank) 90% envelope for the 34 sorted pair scores.

Run: python scripts/run_slot.py -- python -m paiec.analysis.exp_0039_simulator_fidelity
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

from paiec import paths
from paiec.evaluation import episodes as E, simulator as S
from paiec.models import v1_1 as V11, v1_2 as V2

OUT = paths.RUNS / "preregistration" / "exp_0039_simulator_fidelity_2026-10-15.json"
TMP = paths.ROOT / ".tmp"
CB = paths.ROOT / "experiments" / "platform_submissions"
SEEDS = list(range(100))
N_SIM = 20000
RNG_SEED = 39

VERSIONS = {
    "v1": dict(zip="v1.zip", params="v1_params.json", engine="v1_1", cfg=V11.cfg_of(),
               sha="f5e989391336470ffcd98e424f48741dba49fa49f48b884bbc675770c43eed3e"),
    "v1_1": dict(zip="v1_1.zip", params="v1_1_params.json", engine="v1_1", cfg=V11.BASE2,
                 sha="971d4bceb64bae1ed83983119c7e752b10857e80a748afc7e52e9ae00cddba83"),
    "v1_2": dict(zip="v1_2.zip", params="v1_2_params.json", engine="v1_2",
                 cfg=V2.cfg_of(facets="all", guard=1),
                 sha="010f017992521af66e5adfc7e056062d561ca6e51cc50640083a12059936c8b4"),
    "v1_3": dict(zip="v1_3_named_facets_noguard.zip", params="v1_2_params.json", engine="v1_2",
                 cfg=V2.cfg_of(facets="all", guard=0),
                 sha="a4b0e841d4a344ff207bed2c855342212e95f7d550195b667446ea4e92685b86"),
}


def peak_rss_mb():
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("a", ctypes.c_size_t), ("b", ctypes.c_size_t), ("c", ctypes.c_size_t),
                        ("d", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = PMC(); pmc.cb = ctypes.sizeof(PMC)
        k = ctypes.windll.kernel32
        k.GetCurrentProcess.restype = wintypes.HANDLE
        ctypes.windll.psapi.GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return pmc.PeakWorkingSetSize / 2**20
    except Exception:
        return float("nan")


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# Official records (counted from what is saved)
# ---------------------------------------------------------------------------
def official_records():
    j = lambda f: json.loads((CB / f).read_text(encoding="utf-8"))
    v1, v11, v11r = j("2026-10-01_v1.json"), j("2026-10-01_v1_1.json"), j("2026-10-02_v1_1_repeats.json")
    v12, v13 = j("2026-10-02_v1_2.json"), j("2026-10-07_v1_3_named_facets_noguard.json")
    pairs = {"v1": [p["alc"] for p in v1["pairs"]], "v1_1": [p["alc"] for p in v11["pairs"]],
             "v1_2": [p["alc"] for p in v12["pairs"]], "v1_3": [p["alc"] for p in v13["pairs"]]}
    n_eval = {k: [p["n_unique_pairs"] for p in d["pairs"]]
              for k, d in (("v1", v1), ("v1_1", v11), ("v1_2", v12), ("v1_3", v13))}
    runs = [dict(version="v1", run="v1 2026-10-01", alc=float(np.mean(pairs["v1"])), n=8, detailed=True)]
    detailed_v11 = round(float(np.mean(pairs["v1_1"])), 6)
    for r in v11r["runs"]:
        det = abs(r["alc"] - detailed_v11) < 1e-6
        runs.append(dict(version="v1_1", run=f"v1_1 {r['id']}", alc=r["alc"], n=8 if det else None, detailed=det))
    runs.append(dict(version="v1_2", run=f"v1_2 {v12['codabench_id']}", alc=v12["alc"], n=9, detailed=True))
    for r in v12.get("other_runs", []):
        runs.append(dict(version="v1_2", run=f"v1_2 {r['id']}", alc=r["alc"], n=None, detailed=False,
                         note=r.get("note", "")))
    runs.append(dict(version="v1_3", run=f"v1_3 {v13['runs'][0]['id']}", alc=v13["runs"][0]["alc"], n=9,
                     detailed=True))
    return pairs, n_eval, runs


# ---------------------------------------------------------------------------
# Stage 1: archive hashes + package-vs-engine equivalence
# ---------------------------------------------------------------------------
def _engine_H(params):
    H = {k: params[k] for k in ("G", "tau", "sigma_beta") if k in params}
    if "sB" in params:
        H["sB"] = params["sB"]
    if "prior_center" in params:  # ('t',3,'half','learned'): scale = max(.5, rms(_means - G/2))
        assert abs(params["prior_center"] - params["G"] / 2) < 1e-12
        H["_means"] = np.array([params["prior_center"] + params["prior_scale"]])
    return H


def _engine_probs(name, ep, H, B, D):
    v = VERSIONS[name]
    if v["engine"] == "v1_1":
        return {b: V11.predict(ep, H, b, v["cfg"]) for b in E.BUDGETS}
    fe = V2.fe_for(B, D, v["cfg"])
    return {b: V2.predict(ep, H, b, v["cfg"], fe) for b in E.BUDGETS}


def check_packages(bench="real_webagents", seed=0, n=4):
    out = {}
    D = E.compact(bench)
    sids = [str(s) for s in D["pair_sid"][:n]]
    rows = [r for r in S.load_benchmark_dir(str(E.DATA / bench))
            if str(r["subject_id"]) in set(sids) and r["y"] in (0, 1)]
    pairs = S.prepare_pairs(rows, seed=seed)
    assert [p.subject_id for p in pairs] == sids
    ep = E.Episodes(D, list(range(n)), [seed] * n, [0] * n)
    assert np.array_equal(np.concatenate([[r["y"] for r in p.eval_rows] for p in pairs]), ep.r_y)
    TMP.mkdir(exist_ok=True)
    for name, v in VERSIONS.items():
        zp = paths.DIST / v["zip"]
        sha = hashlib.sha256(zp.read_bytes()).hexdigest()
        if sha != v["sha"]:
            raise RuntimeError(f"{name}: archive hash mismatch {sha}")
        tmp = tempfile.mkdtemp(prefix=f"exp0039_{name}_", dir=TMP)
        try:
            with zipfile.ZipFile(zp) as z:
                z.extractall(tmp)
                params = json.loads(z.read(v["params"]))
            model = S.load_model_module(Path(tmp) / "model.py")
            label = S.load_model_module(Path(tmp) / "labeling.py")
            t0 = time.perf_counter()
            pkg = S.run_direct(pairs, model.predict, label.acquisition_function, isolation="call")
            wall = time.perf_counter() - t0
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        probs = _engine_probs(name, ep, _engine_H(params), bench, D)
        dmax = max(float(np.abs(np.asarray(pkg["probabilities"][b], float) - probs[b][ep.r_t]).max())
                   for b in E.BUDGETS)
        out[name] = dict(sha256=sha, max_abs_prob_delta=dmax, package_wall_s=round(wall, 2),
                         bench=bench, seed=seed, pairs=n)
        log(f"  [check] {name}: max|dp|={dmax:.2e} ({wall:.1f}s)")
        if dmax > 1e-6:
            raise RuntimeError(f"{name}: package differs from engine by {dmax}")
    return out


# ---------------------------------------------------------------------------
# Stage 2: R1 episodes scored by each version
# ---------------------------------------------------------------------------
def score_r1():
    per = {}
    for B in E.EVAL_FOLDS:
        t0 = time.perf_counter()
        D = E.compact(B)
        ep = E.build(D, "R1", SEEDS)
        H = V11.H_fold(B)
        alc = {}
        for name in VERSIONS:
            br, _ = ep.score(_engine_probs(name, ep, H, B, D))
            alc[name] = (br @ E.W).tolist()
        n_targets = np.bincount(ep.t_inst, minlength=ep.n).tolist()
        per[B] = dict(seed=ep.inst_seed.tolist(), pair=ep.inst_pair.tolist(), n_eval_keys=n_targets,
                      K=[k for _, k in ep.draw_info], alc=alc)
        log(f"  [R1 {B}] n_inst={ep.n} " + " ".join(f"{k}={np.mean(v):.4f}" for k, v in alc.items())
            + f" {time.perf_counter()-t0:.0f}s")
        del ep, D
    return per


# ---------------------------------------------------------------------------
# Stage 3-4: predictive simulation
# ---------------------------------------------------------------------------
def _draw_units(per):
    """List of R1 draws: (fold index, array of instance indices into the fold arrays)."""
    units = []
    for fi, B in enumerate(E.EVAL_FOLDS):
        seed = np.array(per[B]["seed"])
        units.append([np.flatnonzero(seed == s) for s in SEEDS])
    return units


def simulate_runs(per, version, n, n_sim, rng, mode="episode"):
    """(n_sim, n) matrix of simulated pair ALCs for an official run of n pairs."""
    units = _draw_units(per)
    vals = [np.array(per[B]["alc"][version]) for B in E.EVAL_FOLDS]
    out = np.empty((n_sim, n))
    for r in range(n_sim):
        got = []
        while len(got) < n:
            f = rng.integers(4)
            s = rng.integers(len(SEEDS))
            ix = units[f][s]
            if len(ix) == 0:
                continue
            if mode == "episode":
                got.extend(vals[f][rng.permutation(ix)].tolist())
            else:
                got.append(float(vals[f][rng.choice(ix)]))
        out[r] = got[:n]
    return out


def pit(sim, x):
    return float(((sim < x).sum() + 0.5 * (sim == x).sum()) / len(sim))


def extreme_rank_envelope(sim_sorted, obs_sorted, alpha=0.10):
    """Two-sided global extreme-rank envelope (Myllymaki et al. 2017) over all columns."""
    R = sim_sorted.shape[0]
    allm = np.vstack([sim_sorted, obs_sorted[None, :]])
    ranks = allm.argsort(0).argsort(0) + 1          # 1..R+1 per column
    ext = np.minimum(ranks, R + 2 - ranks).min(1)   # extreme rank per replicate
    obs_e, sim_e = ext[-1], ext[:-1]
    p_global = float((sim_e <= obs_e).mean())
    # largest k with P(sim extreme rank < k) <= alpha
    k = 1
    while (sim_e < k + 1).mean() <= alpha:
        k += 1
    srt = np.sort(sim_sorted, 0)
    lo, hi = srt[k - 1], srt[R - k]
    inside = bool(((obs_sorted >= lo) & (obs_sorted <= hi)).all())
    coverage = float((sim_e >= k).mean())
    return dict(k=int(k), achieved_coverage=coverage, obs_extreme_rank=int(obs_e), p_global=p_global,
                observed_inside=inside, lo=lo.tolist(), hi=hi.tolist(),
                n_points_outside=int(((obs_sorted < lo) | (obs_sorted > hi)).sum()))


def analyse(per, pairs_obs, n_eval_obs, runs, mode, rng):
    res = {"composition": mode}
    # version-matched predictive for detailed runs (n known) and repeats (n assumed from version)
    n_of = {"v1": 8, "v1_1": 8, "v1_2": 9, "v1_3": 9}
    sims = {(v, n): simulate_runs(per, v, n, N_SIM, rng, mode) for v in VERSIONS for n in (8, 9)}
    rl = []
    for r in runs:
        n = r["n"] or n_of[r["version"]]
        m = sims[(r["version"], n)].mean(1)
        alt = sims[(r["version"], 17 - n)].mean(1)
        q05, q95 = np.quantile(m, [0.05, 0.95])
        rl.append(dict(r, n_used=n, n_assumed=r["n"] is None, pred_mean=float(m.mean()), pi90=[float(q05), float(q95)],
                       pit=pit(m, r["alc"]), outside=bool(r["alc"] < q05 or r["alc"] > q95),
                       pi90_other_n=[float(x) for x in np.quantile(alt, [0.05, 0.95])],
                       outside_other_n=bool(r["alc"] < np.quantile(alt, 0.05) or r["alc"] > np.quantile(alt, 0.95))))
    res["runs"] = rl
    res["n_runs_outside"] = int(sum(x["outside"] for x in rl))
    # per-pair marginals, tail count and its joint predictive range
    marg, tail_obs, pp = {}, 0, []
    for v in VERSIONS:
        s = sims[(v, n_of[v])].ravel()
        q05, q95 = np.quantile(s, [0.05, 0.95])
        marg[v] = dict(q05=float(q05), q50=float(np.median(s)), q95=float(q95), mean=float(s.mean()))
        for x in pairs_obs[v]:
            o = x < q05 or x > q95
            tail_obs += o
            pp.append(dict(version=v, alc=x, pit=pit(s, x), tail="low" if x < q05 else "high" if x > q95 else ""))
    T = np.zeros(N_SIM, int)
    T_lo = np.zeros(N_SIM, int)
    T_hi = np.zeros(N_SIM, int)
    for v in VERSIONS:
        m = sims[(v, n_of[v])]
        T_lo += (m < marg[v]["q05"]).sum(1)
        T_hi += (m > marg[v]["q95"]).sum(1)
    T = T_lo + T_hi
    lo, hi = np.quantile(T, [0.05, 0.95], method="inverted_cdf")
    cov = float(((T >= lo) & (T <= hi)).mean())
    lo_obs = sum(p["tail"] == "low" for p in pp)
    hi_obs = sum(p["tail"] == "high" for p in pp)
    res["pair_marginals"] = marg
    res["pairs"] = pp
    res["tail_count"] = dict(observed=int(tail_obs), observed_low=int(lo_obs), observed_high=int(hi_obs),
                             expected=float(T.mean()), range90=[int(lo), int(hi)], achieved_coverage=cov,
                             p_upper=float((T >= tail_obs).mean()), p_lower=float((T <= tail_obs).mean()),
                             low_range90=[int(x) for x in np.quantile(T_lo, [0.05, 0.95], method="inverted_cdf")],
                             high_range90=[int(x) for x in np.quantile(T_hi, [0.05, 0.95], method="inverted_cdf")],
                             outside=bool(tail_obs < lo or tail_obs > hi))
    # simultaneous envelope over 34 sorted pair scores (sorted within version, versions concatenated)
    sim_sorted = np.hstack([np.sort(sims[(v, n_of[v])], 1) for v in VERSIONS])
    obs_sorted = np.concatenate([np.sort(pairs_obs[v]) for v in VERSIONS])
    res["envelope"] = extreme_rank_envelope(sim_sorted, obs_sorted)
    # descriptive: pooled observed vs predictive pair distributions
    allsim = np.concatenate([sims[(v, n_of[v])].ravel() for v in VERSIONS])
    allobs = np.concatenate([pairs_obs[v] for v in VERSIONS])
    res["pooled"] = dict(obs_mean=float(allobs.mean()), obs_sd=float(allobs.std(ddof=1)),
                         pred_mean=float(allsim.mean()), pred_sd=float(allsim.std()),
                         obs_quantiles=np.quantile(allobs, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
                         pred_quantiles=np.quantile(allsim, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist())
    # run-level sd of same-archive repeats vs predictive sd (v1_1 five runs, v1_2 four runs)
    rep = {}
    for v in ("v1_1", "v1_2"):
        obs = np.array([r["alc"] for r in runs if r["version"] == v])
        m = sims[(v, n_of[v])].mean(1)
        k = len(obs)
        sds = np.array([m[i:i + k].std(ddof=1) for i in range(0, N_SIM - k + 1, k)])
        rep[v] = dict(n_runs=k, obs_sd=float(obs.std(ddof=1)), pred_run_sd=float(m.std(ddof=1)),
                      pit_of_obs_sd=pit(sds, obs.std(ddof=1)),
                      sd_range90=[float(x) for x in np.quantile(sds, [0.05, 0.95])])
    res["repeat_sd"] = rep
    return res


def run():
    t0 = time.perf_counter()
    pairs_obs, n_eval_obs, runs = official_records()
    n_pairs = sum(len(v) for v in pairs_obs.values())
    log(f"saved official pair records: {n_pairs}; runs with run-level ALC: {len(runs)}")
    checks = check_packages()
    per = score_r1()
    rng = np.random.default_rng(RNG_SEED)
    primary = analyse(per, pairs_obs, n_eval_obs, runs, "episode", rng)
    sens = analyse(per, pairs_obs, n_eval_obs, runs, "independent", rng)
    # descriptive: evaluation-pool sizes (distinct targets) local R1 vs official n_unique_pairs
    loc_n = np.concatenate([per[B]["n_eval_keys"] for B in E.EVAL_FOLDS])
    obs_n = np.concatenate([n_eval_obs[v] for v in VERSIONS])
    report = {
        "experiment_id": "EXP-0039", "preregistered": True, "date": "2026-10-15",
        "dataset_revision": "bc8204d811", "folds": list(E.EVAL_FOLDS), "regime": "R1", "seeds": SEEDS,
        "budgets": list(E.BUDGETS), "acquisition_policy": "A1 accept-all (all four archives)",
        "versions": {k: dict(zip=v["zip"], sha256=v["sha"], engine=v["engine"],
                             cfg={a: (list(b) if isinstance(b, tuple) else b) for a, b in v["cfg"].items()})
                     for k, v in VERSIONS.items()},
        "n_official_pair_records": n_pairs, "n_official_runs": len(runs),
        "package_checks": checks, "n_sim": N_SIM, "rng_seed": RNG_SEED,
        "r1_local_mean_alc": {v: float(np.mean([np.mean(per[B]["alc"][v]) for B in E.EVAL_FOLDS]))
                              for v in VERSIONS},
        "r1_fold_mean_alc": {v: {B: float(np.mean(per[B]["alc"][v])) for B in E.EVAL_FOLDS} for v in VERSIONS},
        "r1_instances_per_fold": {B: len(per[B]["seed"]) for B in E.EVAL_FOLDS},
        "eval_pool_size": dict(local_r1_quantiles=np.quantile(loc_n, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist(),
                               official_quantiles=np.quantile(obs_n, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist(),
                               note="local = distinct content keys in eval pool; official = n_unique_pairs"),
        "primary": primary, "sensitivity_independent_slots": sens,
        "elapsed_seconds": time.perf_counter() - t0, "peak_rss_mb": peak_rss_mb(),
        "r1_per_instance": per,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    p = primary
    log(f"runs outside 90% PI: {p['n_runs_outside']}/{len(runs)}")
    for r in p["runs"]:
        log(f"  {r['run']:16s} obs={r['alc']:.4f} PI=[{r['pi90'][0]:.4f},{r['pi90'][1]:.4f}] pit={r['pit']:.3f}"
            f" {'OUT' if r['outside'] else ''}")
    log(f"tail count {p['tail_count']}")
    log(f"envelope k={p['envelope']['k']} inside={p['envelope']['observed_inside']} p={p['envelope']['p_global']:.3f}")
    log(f"sens runs outside {sens['n_runs_outside']}; tail {sens['tail_count']['observed']} {sens['tail_count']['range90']}")
    log(f"elapsed {report['elapsed_seconds']:.1f}s peak RSS {report['peak_rss_mb']:.0f} MB -> {OUT}")
    return report


if __name__ == "__main__":
    run()
