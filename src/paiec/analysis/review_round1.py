"""Review round 1: analyses requested by the mock review (items #4, #7, #1, #11).

    python -m paiec.analysis.review_round1 platform   # #4 baseline / v0 / final replay on exact singleton pairs (s)
    python -m paiec.analysis.review_round1 floor      # #7 bootstrap CIs for floor and ceiling + local R2 floor
    python -m paiec.analysis.review_round1 ladder     # #1 fold-level intervals for every ladder step (minutes)
    python -m paiec.analysis.review_round1 counts     # #11 singleton / exact pair counts
    python -m paiec.analysis.review_round1 all

Heavy parts (floor's local R2 episodes, ladder) should be run through scripts/run_slot.py.
Outputs: experiments/runs/review_round1/<part>.json. All platform numbers are IN-SAMPLE on our own formative draws:
they describe those draws and use only our own feedback and archives. Anonymous IDs are opaque labels.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
import zipfile
from pathlib import Path

import numpy as np

from paiec import paths
from paiec.feedback import (BUDGETS, alc, benchmark_multiplicity, brier_constant, invert_pair, load_feedback,
                            pair_level_bounds, recover_k_path, replay_curve)

OUT = paths.RUNS / "review_round1"
TMP = paths.ROOT / ".tmp" / "review_round1"
FB = paths.EXPERIMENTS / "platform_submissions"
RUNS = {  # version: (feedback file, archive)
    "v1": ("2026-10-01_v1.json", "v1.zip"),
    "v1_1": ("2026-10-01_v1_1.json", "v1_1.zip"),
    "v1_2": ("2026-10-02_v1_2.json", "v1_2.zip"),
    "v1_3": ("2026-10-07_v1_3_named_facets_noguard.json", "v1_3_named_facets_noguard.zip"),
}
try:  # private locator of the explored variant's saved files (not released)
    from paiec import _variant as _V
    VARIANT_DIR, VARIANT_RUN_ID, VARIANT_RUN_KEY = _V.DIR, _V.RUN_ID, _V.RUN_KEY
except ImportError:
    VARIANT_DIR, VARIANT_RUN_ID, VARIANT_RUN_KEY = paths.ROOT / "explored_variant_feedback", None, "run_variant"
PROTO_FB = next(iter(sorted(VARIANT_DIR.glob("*_platform_feedback.json"))), VARIANT_DIR / "platform_feedback.json")
PROTO_PY = next(iter(sorted(VARIANT_DIR.glob("*/model.py"))), VARIANT_DIR / "model.py")  # its unpacked archive
SUBJ = {"normalized_name": "X", "provider": "Y", "release_date": "2026-01-01", "access_date": "2026-02-01",
        "harness": "h", "reasoning_effort": "low", "harness_version": "1", "subject_features_extra": ""}
W = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
NBOOT = 10000
T3 = 3.182446305284263  # t_{0.975, 3}
FACET_RUNS = {"v1_2", "v1_3", "proto"}  # runs whose model shares the final model's item-group (facet) code


# --------------------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------------------
def _module(name: str, py: Path):
    spec = importlib.util.spec_from_file_location(f"rr1_{name}", py)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _models() -> dict:
    out = {}
    for v, (_, z) in RUNS.items():
        d = TMP / v
        zipfile.ZipFile(paths.DIST / z).extractall(d)
        out[v] = _module(v, d / "model.py")
    zipfile.ZipFile(paths.DIST / "v0.zip").extractall(TMP / "v0")
    out["v0"] = _module("v0", TMP / "v0" / "model.py")
    out["proto"] = _module("proto", PROTO_PY)
    out["baseline"] = _module("baseline", paths.BASELINE / "empirical_mean" / "model.py")
    return out


def _item(bench: str, c: str = "q") -> dict:
    return {"item_content": c, "item_features": "", "interactors": "", "benchmark_id": bench}


def rule_of(m, bench: str):
    """Model as a count-only rule p(k, b): k successes among b synthetic, feature-less labels of one pair."""
    def p(k: int, b: int) -> float:
        lab = [((SUBJ, _item(bench, f"z{i}")), 1 if i < k else 0) for i in range(b)]
        return float(m.predict([SUBJ, _item(bench)], lab))
    return p


def _boot_mean(x: np.ndarray, rng, groups: np.ndarray | None = None, nboot: int = NBOOT) -> list[float]:
    """Percentile 95% CI of the mean; resamples rows, or whole groups (clusters) if ``groups`` is given."""
    x = np.asarray(x, float)
    if groups is None:
        bs = x[rng.integers(0, len(x), (nboot, len(x)))].mean(1)
    else:
        ug = np.unique(groups)
        idx = [np.flatnonzero(groups == g) for g in ug]
        bs = np.empty(nboot)
        for i in range(nboot):
            pick = rng.integers(0, len(ug), len(ug))
            ii = np.concatenate([idx[j] for j in pick])
            bs[i] = x[ii].mean()
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def _rows(models: dict) -> list[dict]:
    """Every saved pair-slot (4 runs + the explored-variant run) with its inversion."""
    rows = []
    for v, (f, _) in RUNS.items():
        pairs = load_feedback(FB / f)
        mult = benchmark_multiplicity(pairs)
        for p in pairs:
            p0 = rule_of(models[v], p.benchmark_id)(0, 0)
            rows.append(dict(ver=v, pair=p, rep=mult[p.benchmark_id], p0=p0, inv=invert_pair(p.brier[0], p.ece[0], p0)))
    pairs = load_feedback(PROTO_FB)
    mult = benchmark_multiplicity(pairs)
    for p in pairs:
        p0 = rule_of(models["proto"], p.benchmark_id)(0, 0)
        rows.append(dict(ver="proto", pair=p, rep=mult[p.benchmark_id], p0=p0,
                         inv=invert_pair(p.brier[0], p.ece[0], p0)))
    return rows


def _exact_singletons(rows: list[dict], models: dict) -> dict:
    """(subject, bench) -> {a, ks, runs, rows}; k paths recovered with the run's own model at tolerance k_tol."""
    out = {}
    for r in rows:
        if r["rep"] != 1:
            continue
        key = (r["pair"].subject_id, r["pair"].benchmark_id)
        path = recover_k_path(r["inv"].a, r["pair"].ece, rule_of(models[r["ver"]], key[1]), tol=r["inv"].k_tol)
        r["path"] = path
        if path.exact and path.nested:
            e = out.setdefault(key, dict(a=r["inv"].a, ks=path.ks, runs=[], rows=[]))
            assert e["ks"] == path.ks, f"k path differs across runs for {key}"
            assert abs(e["a"] - r["inv"].a) < 2e-5, f"accuracy differs across runs for {key}"
            e["runs"].append(r["ver"]); e["rows"].append(r)
    return out


def _dump(name: str, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1), encoding="utf-8")
    print(f"written {(OUT / f'{name}.json').relative_to(paths.ROOT)}")


# --------------------------------------------------------------------------------------------------------------
# #4 platform replay of the official baseline and v0 against the final model
# --------------------------------------------------------------------------------------------------------------
def platform() -> dict:
    models = _models()
    rows = _rows(models)
    ex = _exact_singletons(rows, models)
    # rule checks: the baseline is k/b (0.5 at b=0); v0 is (1+k)/(2+b) clipped to [0.02, 0.98]
    for b in BUDGETS:
        for k in range(b + 1):
            assert abs(rule_of(models["baseline"], "x")(k, b) - (k / b if b else 0.5)) < 1e-12
            assert abs(rule_of(models["v0"], "x")(k, b) - min(max((1 + k) / (2 + b), 0.02), 0.98)) < 1e-12
    recs = []
    for (s, bench), e in sorted(ex.items()):
        a, ks = e["a"], e["ks"]
        c = {nm: replay_curve(a, ks, rule_of(models[nm], bench)) for nm in ("baseline", "v0", "v1_3")}
        obs_final = next((r["pair"].brier for r in e["rows"] if r["ver"] == "v1_3"), None)
        strict = bool(set(e["runs"]) & FACET_RUNS)
        recs.append(dict(subject=s, bench=bench, a=a, ks={str(b): k for b, k in ks.items()}, runs=e["runs"],
                         final_exact=strict, alc={nm: alc(v) for nm, v in c.items()},
                         final_observed_alc=None if obs_final is None else alc(obs_final),
                         self_replay_dev=max(abs(c["v1_3"][b] - obs_final[b]) for b in BUDGETS) if obs_final else None,
                         floor=a * (1 - a)))
    rng = np.random.default_rng(0)

    def summ(sel):
        A = {nm: np.array([r["alc"][nm] for r in sel]) for nm in ("baseline", "v0", "v1_3")}
        fl = np.array([r["floor"] for r in sel])
        bench = np.array([r["bench"] for r in sel])
        o = dict(n_pairs=len(sel), n_benchmarks=int(len(set(bench))), alc={k: float(v.mean()) for k, v in A.items()},
                 floor=float(fl.mean()), diffs={})
        for nm in ("baseline", "v0"):
            d = A[nm] - A["v1_3"]
            o["diffs"][f"{nm}_minus_final"] = dict(mean=float(d.mean()), ci95_pair=_boot_mean(d, rng),
                                                   ci95_benchmark_cluster=_boot_mean(d, rng, bench),
                                                   n_final_better=int((d > 0).sum()))
        d = A["baseline"] - A["v0"]
        o["diffs"]["baseline_minus_v0"] = dict(mean=float(d.mean()), ci95_pair=_boot_mean(d, rng),
                                               ci95_benchmark_cluster=_boot_mean(d, rng, bench))
        return o

    res = dict(
        note=("In-sample exact replay on our own formative draws. a and k_b recovered from our feedback "
              "(paiec.feedback); every model is scored on the same revealed labels (our accept-all acquisition order). "
              "Baseline and v0 are count-only by construction, so their replay is exact on every exact singleton pair. "
              "The final model's replay is exact (A4 verified) only where a run with its item-group code was exact "
              "('strict'); elsewhere it uses feature-less labels and is an approximation."),
        rules=dict(baseline="k/b, 0.5 at b=0 (external/official_baselines/empirical_mean, unclipped)",
                   v0="(1+k)/(2+b) clipped [0.02,0.98] (submissions/dist/v0.zip)",
                   final="v1_3_named_facets_noguard.zip predict() on feature-less labels"),
        bootstrap=f"{NBOOT} resamples, percentile, seed 0; pair = resample pairs, cluster = resample benchmark IDs",
        strict=summ([r for r in recs if r["final_exact"]]),
        all_exact_singletons=summ(recs),
        max_final_self_replay_dev=max((r["self_replay_dev"] for r in recs if r["self_replay_dev"] is not None),
                                      default=None),
        pairs=recs,
    )
    _dump("platform_baseline_replay", res)
    for k in ("strict", "all_exact_singletons"):
        print(k, json.dumps({kk: vv for kk, vv in res[k].items()}, indent=0))
    return res


# --------------------------------------------------------------------------------------------------------------
# #7 uncertainty of floor and ceiling; local floor on the R2 LOBO folds
# --------------------------------------------------------------------------------------------------------------
def _local_floor(regime: str = "R2", seeds=range(10)) -> dict:
    from paiec.evaluation import episodes as E
    from paiec.models import v1_1 as V
    from paiec.models import v1_2 as V2
    per_fold, fold_final = {}, {}
    for B in E.EVAL_FOLDS:
        D = E.compact(B)
        ep = E.build(D, regime, list(seeds))
        cnt = np.bincount(ep.r_inst, minlength=ep.n)
        a = np.bincount(ep.r_inst, ep.r_y, minlength=ep.n) / cnt
        fl_inst = np.bincount(ep.r_inst, (ep.r_y - a[ep.r_inst]) ** 2, minlength=ep.n) / cnt  # = a(1-a) for 0/1
        per_fold[B] = float(V.pair_mean(fl_inst, ep.inst_pair).mean())
        r = V2.run_cfgs(B, D, ep, V.H_fold(B), {"final": V2.cfg_of(facets="all")})["final"]
        fold_final[B] = dict(alc=float(V.pair_mean(r["alc"], r["pair"]).mean()),
                             excess=[float(V.pair_mean(r["br"][:, j] - fl_inst, r["pair"]).mean()) for j in range(6)])
    f = np.array(list(per_fold.values()))
    return dict(regime=regime, seeds=list(seeds), per_fold=per_fold, mean=float(f.mean()),
                t3=[float(f.mean() - T3 * f.std(ddof=1) / 2), float(f.mean() + T3 * f.std(ddof=1) / 2)],
                final_model=dict(per_fold=fold_final, alc=float(np.mean([v["alc"] for v in fold_final.values()])),
                                 ceiling=float(f.mean() - np.mean([v["alc"] for v in fold_final.values()]))))


def floor(local: bool = True) -> dict:
    models = _models()
    rows = _rows(models)
    rng = np.random.default_rng(0)
    out = dict(bootstrap=f"{NBOOT} resamples, percentile, seed 0")
    for name, sel in (("earlier_draws_v1_1_to_v1_3", [r for r in rows if r["ver"] in ("v1_1", "v1_2", "v1_3")]),
                      (VARIANT_RUN_KEY, [r for r in rows if r["ver"] == "proto"])):
        a = np.array([r["inv"].a for r in sel])
        B = np.array([[r["pair"].brier[b] for b in BUDGETS] for r in sel])
        bd = pair_level_bounds(a, B)
        fl = a * (1 - a)
        ceil_i = fl - B @ W  # per-slot (floor - ALC); its mean is the ceiling
        bench = np.array([r["pair"].benchmark_id for r in sel])
        pair_id = np.array([r["pair"].subject_id + "|" + r["pair"].benchmark_id for r in sel])
        out[name] = dict(n_slots=len(sel), n_distinct_pairs=int(len(set(pair_id))), n_benchmarks=int(len(set(bench))),
                         floor=bd.floor, floor_ci95_slot=_boot_mean(fl, rng),
                         floor_ci95_distinct_pair_cluster=_boot_mean(fl, rng, pair_id),
                         floor_ci95_benchmark_cluster=_boot_mean(fl, rng, bench),
                         observed_alc=bd.observed_alc, ceiling=bd.ceiling_gain,
                         ceiling_ci95_slot=_boot_mean(ceil_i, rng),
                         ceiling_ci95_distinct_pair_cluster=_boot_mean(ceil_i, rng, pair_id),
                         ceiling_ci95_benchmark_cluster=_boot_mean(ceil_i, rng, bench),
                         excess=bd.excess)
    p0s = {r["ver"]: round(r["p0"], 6) for r in rows if r["ver"] in ("v1_1", "v1_2", "v1_3")}
    out["earlier_draws_v1_1_to_v1_3"]["budget0_predictions"] = p0s
    if local:
        out["local_R2"] = _local_floor("R2", range(10))
    _dump("floor_ceiling", out)
    print(json.dumps(out, indent=1)[:3000])
    return out


# --------------------------------------------------------------------------------------------------------------
# #1 fold-level intervals for the ladder (recomputed on common episodes)
# --------------------------------------------------------------------------------------------------------------
LADDER = [("baseline", "official baseline (empirical mean)"), ("v0", "Beta-binomial shrinkage"),
          ("v1", "+ benchmark offset, item pooling"), ("v1_1", "+ hedged centre, false-merge model"),
          ("v1_2", "+ item-group effects"), ("final", "final model: key guard off")]


def ladder(regimes=("R2", "R1")) -> dict:
    from paiec.evaluation import episodes as E
    from paiec.models import ladder as Lad
    from paiec.models import v1_1 as V
    from paiec.models import v1_2 as V2
    seeds = {"R2": list(range(10)), "R1": list(range(100))}
    cfg_v = {"v1": V.V1_CFG, "v1_1": V.BASE2}
    cfg_2 = {"v1_2": V2.cfg_of(facets="all", guard=1), "final": V2.cfg_of(facets="all")}
    out = {}
    t0 = time.time()
    for R in regimes:
        res = {}
        for B in E.EVAL_FOLDS:
            D = E.compact(B)
            ep = E.build(D, R, seeds[R])
            H = V.H_fold(B)
            r = {}
            for nm, rung in (("baseline", "L0"), ("v0", "L1")):
                probs = {b: Lad.predict_budget(ep, H, rung, b)[0] for b in E.BUDGETS}
                br, ec = ep.score(probs)
                r[nm] = dict(br=br, ec=ec, alc=br @ E.W, pair=ep.inst_pair.copy())
            r.update(V.run_cfgs(ep, H, cfg_v))
            r.update(V2.run_cfgs(B, D, ep, H, cfg_2))
            res[B] = r
            print(f"[ladder {R} {B}] n_inst={ep.n} {time.time() - t0:.0f}s", flush=True)
        o = dict(seeds=len(seeds[R]), alc={nm: V.mean_alc(res, nm) for nm, _ in LADDER},
                 curve={nm: V.mean_curve(res, nm).tolist() for nm, _ in LADDER}, steps={})
        names = [n for n, _ in LADDER]
        pairs = list(zip(names, names[1:])) + [("v0", "v1_1"), ("v1_1", "final"), ("v1_1", "v1_2"),
                                                ("v0", "final"), ("baseline", "final")]
        for a, b in pairs:
            d = V.delta(res, b, a)  # later minus earlier, x1e3; pair and hierarchical bootstrap (2000, seed 0)
            pf = np.array(list(d["per_fold"].values()))
            m, s = float(pf.mean()), float(pf.std(ddof=1))
            d["t3"] = [m - T3 * s / 2, m + T3 * s / 2]
            d["n_folds_improved"] = int((pf < 0).sum())
            o["steps"][f"{a}->{b}"] = d
        out[R] = o
        print(f"[{R}] ALC " + " ".join(f"{k}={v:.4f}" for k, v in o["alc"].items()))
        for k, d in o["steps"].items():
            print(f"  {k:18s} {d['pt']:+.2f} pair[{d['lo']:+.2f},{d['hi']:+.2f}] hier[{d['hlo']:+.2f},{d['hhi']:+.2f}] "
                  f"t3[{d['t3'][0]:+.2f},{d['t3'][1]:+.2f}] folds " +
                  " ".join(f"{v:+.2f}" for v in d["per_fold"].values()))
    out["units"] = "Delta ALC x1e3, later minus earlier (negative = improvement); mean of 4 held-out folds"
    out["intervals"] = ("pair: bootstrap pairs within fixed folds; hier: resample folds then pairs (2000, seed 0); "
                        "t3: mean +/- t_{0.975,3} sd/2 over the 4 fold means")
    _dump("ladder_intervals", out)
    return out


# --------------------------------------------------------------------------------------------------------------
# #11 pair counts
# --------------------------------------------------------------------------------------------------------------
def counts() -> dict:
    models = _models()
    rows = _rows(models)
    ex = _exact_singletons(rows, models)

    def distinct(sel):
        return sorted({(r["pair"].subject_id, r["pair"].benchmark_id) for r in sel})

    later = [r for r in rows if r["ver"] in ("v1_1", "v1_2", "v1_3")]
    single = [r for r in rows if r["rep"] == 1]
    res = {}
    for name, vers in (("later_three", {"v1_1", "v1_2", "v1_3"}),
                       ("later_saved_runs", {"v1_1", "v1_2", "v1_3", "proto"}),
                       ("all_five_runs", {"v1", "v1_1", "v1_2", "v1_3", "proto"}),
                       ("four_saved_runs", {"v1", "v1_1", "v1_2", "v1_3"})):
        s = [r for r in single if r["ver"] in vers]
        ex_rows = [r for r in s if r["path"].exact and r["path"].nested]
        loose = [r for r in s if r["path"].max_residual < 0.005]
        res[name] = dict(singleton_rows=len(s), singleton_distinct=len(distinct(s)),
                         exact_rows=len(ex_rows), exact_distinct=len(distinct(ex_rows)),
                         loose_0p005_distinct=len(distinct(loose)))
    res["pair_slots_saved_four_runs"] = sum(1 for r in rows if r["ver"] != "proto")
    res["pair_slots_later_three"] = len(later)
    res["max_exact_residual"] = max(r["path"].max_residual for r in single if r["path"].exact)
    res["min_inexact_residual"] = min(r["path"].max_residual for r in single if not r["path"].exact)
    res["exact_distinct_total"] = len(ex)
    res["note"] = ("exact = every budget's k residual <= k_tol (2*(5e-7/|1-2p0| + 5e-7)) and unique, path nested; "
                   "loose = the 0.005 threshold used by experiments/pattern_mining/meta_2026-10-16/m3.")
    _dump("pair_counts", res)
    print(json.dumps(res, indent=1))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["platform", "floor", "ladder", "counts", "all"])
    ap.add_argument("--no-local", action="store_true")
    a = ap.parse_args()
    if a.part in ("counts", "all"):
        counts()
    if a.part in ("platform", "all"):
        platform()
    if a.part in ("floor", "all"):
        floor(local=not a.no_local)
    if a.part in ("ladder", "all"):
        ladder()


if __name__ == "__main__":
    main()
