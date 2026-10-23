"""Review round 2: per-budget platform table for the final model, and local-vs-platform table per version.

    python -m paiec.analysis.review_round2          # seconds; writes experiments/runs/review_round2/*.json

Sources:
- per-pair formative feedback: experiments/platform_submissions/{2026-10-01_v1, 2026-10-01_v1_1, 2026-10-02_v1_2,
  2026-10-07_v1_3_named_facets_noguard}.json and the explored variant's saved feedback (not released).
  Only ONE run per version has per-pair (hence per-budget) feedback; all other runs have ALC only;
- run-level ALC: experiments/platform_submissions/2026-10-15_submission_history.json, 2026-10-02_v1_1_repeats.json,
  repeats_*.json, 2026-10-01_v1.json (pair mean);
- local R2: experiments/runs/review_round1/ladder_intervals.json (10 paired seeds; common episodes) and
  experiments/runs/preregistration/exp_0044_tempering_replay_2026-10-14.json (final-model archive replay).
Platform statistics are in-sample descriptions of our own formative draws; each run is a fresh draw.
"""
from __future__ import annotations

import json
import math

import numpy as np
from scipy import stats

from paiec import paths
from paiec.analysis import review_round1 as R1
from paiec.feedback import BUDGETS, alc, load_feedback, replay_curve

OUT = paths.RUNS / "review_round2"
PS = paths.EXPERIMENTS / "platform_submissions"
LADDER = paths.RUNS / "review_round1" / "ladder_intervals.json"
EXP44 = paths.RUNS / "preregistration" / "exp_0044_tempering_replay_2026-10-14.json"
NBOOT = 10000

PER_PAIR = {"v1": "2026-10-01_v1.json", "v1_1": "2026-10-01_v1_1.json", "v1_2": "2026-10-02_v1_2.json",
            "v1_3": "2026-10-07_v1_3_named_facets_noguard.json"}
try:  # private locator of the explored variant's saved files (not released)
    from paiec import _variant as _V
    VARIANT_DIR, VARIANT_RUN_ID, VARIANT_RUN_KEY = _V.DIR, _V.RUN_ID, _V.RUN_KEY
except ImportError:
    VARIANT_DIR, VARIANT_RUN_ID, VARIANT_RUN_KEY = paths.ROOT / "explored_variant_feedback", None, "run_variant"
PER_PAIR_RUN_ID = {"v1": None, "v1_1": 955381, "v1_2": 955660, "v1_3": 967620, "proto": VARIANT_RUN_ID}
LABEL = {"v0": "Beta-binomial shrinkage (v0)", "v1": "first model (v1)", "v1_1": "hedged-centre model (v1_1)",
         "v1_2": "item-group model (v1_2)", "v1_3": "final model (v1_3)",
         "proto": "explored variant (not selected)"}
LOCAL_KEY = {"v0": "v0", "v1": "v1", "v1_1": "v1_1", "v1_2": "v1_2", "v1_3": "final"}


def sign_test(n_above: int, n: int) -> float:
    """Two-sided exact binomial sign test, H0: P(above) = 1/2."""
    if n == 0:
        return float("nan")
    return float(min(1.0, stats.binomtest(n_above, n, 0.5).pvalue))


def t_ci(x) -> list[float] | None:
    x = np.asarray(x, float)
    if len(x) < 2:
        return None
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
    return [float(x.mean() - h), float(x.mean() + h)]


def platform_runs() -> dict:
    """Every logged platform ALC per version (run id -> ALC)."""
    hist = json.loads((PS / "2026-10-15_submission_history.json").read_text(encoding="utf-8"))["runs"]
    by_file: dict[str, dict[int, dict]] = {}
    for r in hist:
        by_file.setdefault(r["file"], {})[r["id"]] = dict(alc=r["alc"], status=r["status"])
    v11 = json.loads((PS / "2026-10-02_v1_1_repeats.json").read_text(encoding="utf-8"))["runs"]
    proto_file = next(iter(sorted(PS.glob("repeats_q4_*.json"))), PS / "repeats_explored_variant.json")
    proto = json.loads(proto_file.read_text(encoding="utf-8"))["runs"]
    v1 = float(np.mean([alc(p.brier) for p in load_feedback(PS / PER_PAIR["v1"])]))
    runs = {
        "v0": {},
        "v1": {"(id not saved)": dict(alc=round(v1, 6), status="Finished")},
        "v1_1": {r["id"]: dict(alc=r["alc"], status="Finished") for r in v11},
        "v1_2": by_file["v1_2.zip"],
        "v1_3": by_file["v1_3_named_facets_noguard.zip"],
        "proto": {r["id"]: dict(alc=r["alc"], status="Finished") for r in proto},
    }
    return dict(runs=runs, v1_2_repackage={967648: by_file["v1_3_named_facets.zip"][967648]})


def local_r2() -> dict:
    lad = json.loads(LADDER.read_text(encoding="utf-8"))["R2"]
    e44 = json.loads(EXP44.read_text(encoding="utf-8"))["results"]["R2"]["baseline"]
    out = {v: dict(alc=lad["alc"][k], curve=lad["curve"][k]) for v, k in LOCAL_KEY.items()}
    out["v1_3_exp0044"] = dict(alc=e44["alc"], curve=e44["curve"])
    # The Q4 table is keyed by anonymous platform benchmark IDs, none of which occurs in the public folds, so the
    # variant falls back to the final model exactly on every local pair (identical by construction, not re-run).
    out["proto"] = dict(alc=e44["alc"], curve=e44["curve"], note="identical to the final model locally by construction")
    return out


# --------------------------------------------------------------------------------------------------------------
# 1. per-budget table
# --------------------------------------------------------------------------------------------------------------
def per_budget(local: dict, runs: dict) -> dict:
    cols = [f"B{b}" for b in BUDGETS] + ["ALC"]

    def curves(fname):
        P = load_feedback(PS / fname)
        return np.array([[p.brier[b] for b in BUDGETS] + [alc(p.brier)] for p in P])

    fin = curves(PER_PAIR["v1_3"])
    loc = local["v1_3_exp0044"]["curve"] + [local["v1_3_exp0044"]["alc"]]
    alc_runs = np.array([r["alc"] for r in runs["runs"]["v1_3"].values()])
    final = {}
    for j, c in enumerate(cols):
        x = fin[:, j]
        row = dict(local_R2=loc[j], per_pair_run=PER_PAIR_RUN_ID["v1_3"], n_pairs=len(x),
                   mean_run_967620=float(x.mean()), within_draw_pair_sd=float(x.std(ddof=1)),
                   within_draw_se=float(x.std(ddof=1) / math.sqrt(len(x))))
        if c == "ALC":
            row.update(n_runs=len(alc_runs), mean_over_runs=float(alc_runs.mean()),
                       one_draw_sd=float(alc_runs.std(ddof=1)), se_mean=float(alc_runs.std(ddof=1) / math.sqrt(len(alc_runs))),
                       ci95_t=t_ci(alc_runs), runs_above_local=int((alc_runs > loc[j]).sum()),
                       sign_test_p=sign_test(int((alc_runs > loc[j]).sum()), len(alc_runs)))
        else:
            row.update(n_runs=1, mean_over_runs=float(x.mean()), one_draw_sd=None, se_mean=None,
                       runs_above_local=int(x.mean() > loc[j]), sign_test_p=sign_test(int(x.mean() > loc[j]), 1),
                       note="only run 967620 has per-budget feedback; between-draw SD not estimable")
        final[c] = row
    # pooled v1_1..v1_3: one per-pair run per version, identical budget-0 prediction p0 = 0.448722
    pooled = {}
    run_means = {v: curves(PER_PAIR[v]) for v in ("v1_1", "v1_2", "v1_3")}
    for j, c in enumerate(cols):
        m = np.array([run_means[v][:, j].mean() for v in run_means])
        lv = np.array([(local[v]["curve"] + [local[v]["alc"]])[j] for v in run_means])
        slots = np.concatenate([run_means[v][:, j] for v in run_means])
        lslot = np.concatenate([np.full(len(run_means[v]), (local[v]["curve"] + [local[v]["alc"]])[j]) for v in run_means])
        pooled[c] = dict(n_runs=3, run_means=dict(zip(run_means, m.round(6).tolist())), mean=float(m.mean()),
                         one_draw_sd=float(m.std(ddof=1)), se_mean=float(m.std(ddof=1) / math.sqrt(3)),
                         local_R2_own_version_mean=float(lv.mean()), runs_above_own_local=int((m > lv).sum()),
                         sign_test_p=sign_test(int((m > lv).sum()), 3),
                         n_slots=len(slots), slots_above_own_local=int((slots > lslot).sum()),
                         slot_sign_test_p=sign_test(int((slots > lslot).sum()), len(slots)))
    return dict(final_model_v1_3=final, pooled_v1_1_to_v1_3=pooled,
                caveats=["Per-budget platform values exist for ONE run per version (v1_3: 967620 only); the other "
                         "three v1_3 runs report ALC only.",
                         "Pooled rows mix three versions (one draw each). They share the budget-0 prediction "
                         "(0.448722), so B0 is the same model; budgets 1-31 differ slightly between versions. Each "
                         "run is compared with its own version's local value.",
                         "The slot-level sign test treats pairs within a draw as independent; it is descriptive.",
                         "In-sample: platform values describe our own formative draws."])


# --------------------------------------------------------------------------------------------------------------
# 2. local vs platform per version
# --------------------------------------------------------------------------------------------------------------
def per_version(local: dict, runs: dict) -> dict:
    models = R1._models()
    rows = R1._rows(models)
    ex = R1._exact_singletons(rows, models)
    rng = np.random.default_rng(0)
    vers = ["v0", "v1", "v1_1", "v1_2", "v1_3", "proto"]
    count_only = {"v0", "v1", "v1_1"}  # no item-level terms on a singleton pair: replay exact on all 18
    rep, strict = {v: [] for v in vers}, []
    for (s, bench), e in sorted(ex.items()):
        strict.append(bool(set(e["runs"]) & R1.FACET_RUNS))
        for v in vers:
            rep[v].append(alc(replay_curve(e["a"], e["ks"], R1.rule_of(models[v], bench))))
    rep = {v: np.array(x) for v, x in rep.items()}
    strict = np.array(strict)

    def boot(d):
        b = d[rng.integers(0, len(d), (NBOOT, len(d)))].mean(1)
        return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]

    table = {}
    for v in vers:
        alcs = [r["alc"] for r in runs["runs"][v].values()]
        table[v] = dict(
            label=LABEL[v], local_R2_alc=local[v]["alc"] if v in local else None,
            platform_runs={str(k): r for k, r in runs["runs"][v].items()}, platform_n=len(alcs),
            platform_mean=float(np.mean(alcs)) if alcs else None, platform_ci95_t=t_ci(alcs),
            platform_note=("not submitted (no logged run)" if not alcs else "n=1: no interval" if len(alcs) == 1 else None),
            replay_alc_18=float(rep[v].mean()), replay_alc_strict8=float(rep[v][strict].mean()),
            replay_exact=("all 18 (count-only)" if v in count_only else "strict 8 only; feature-less approximation on 10"))
    table["v1_3"]["local_R2_alc_exp0044"] = local["v1_3_exp0044"]["alc"]
    alt = [r["alc"] for r in runs["runs"]["v1_2"].values()] + [r["alc"] for r in runs["v1_2_repackage"].values()]
    table["v1_2"]["platform_with_repackage"] = dict(n=len(alt), mean=float(np.mean(alt)), ci95_t=t_ci(alt))
    steps = [("v0", "v1"), ("v1", "v1_1"), ("v1_1", "v1_2"), ("v1_2", "v1_3"), ("v1_3", "proto"), ("v0", "v1_3")]
    paired = {}
    for a, b in steps:
        d18, d8 = rep[b] - rep[a], (rep[b] - rep[a])[strict]
        paired[f"{a}->{b}"] = dict(mean_18=float(d18.mean()), ci95_18=boot(d18), mean_strict8=float(d8.mean()),
                                   ci95_strict8=boot(d8))
    sub = [v for v in ("v1", "v1_1", "v1_2", "v1_3") if table[v]["platform_n"]]
    loc_order = sorted(sub, key=lambda v: table[v]["local_R2_alc"])
    plat_order = sorted(sub, key=lambda v: table[v]["platform_mean"])
    rho = stats.spearmanr([table[v]["local_R2_alc"] for v in sub], [table[v]["platform_mean"] for v in sub])
    overlaps = {f"{a}|{b}": _overlap(table[a]["platform_ci95_t"], table[b]["platform_ci95_t"])
                for a, b in zip(sub, sub[1:])}
    return dict(table=table, paired_replay=paired, n_pairs=int(len(strict)), n_strict=int(strict.sum()),
                ranking=dict(versions=sub, local_order=loc_order, platform_mean_order=plat_order,
                             spearman_local_vs_platform=float(rho.statistic), spearman_p=float(rho.pvalue),
                             adjacent_ci_overlap=overlaps,
                             local_spread=float(max(table[v]["local_R2_alc"] for v in sub) - min(table[v]["local_R2_alc"] for v in sub))),
                caveats=["Platform means are over fresh draws; differences between versions are unpaired across draws.",
                         "Replay ALCs are paired, exact (as marked) and in-sample on 18 pairs over 7 anonymous IDs.",
                         "The explored variant's replay uses its table, built from feedback that includes some of "
                         "these pairs, so its replay advantage is partly in-sample by construction.",
                         "v0 was never submitted; its platform column is empty."])


def _overlap(a, b):
    if a is None or b is None:
        return None
    return bool(a[0] <= b[1] and b[0] <= a[1])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = platform_runs()
    local = local_r2()
    pb = per_budget(local, runs)
    (OUT / "per_budget_final.json").write_text(json.dumps(pb, indent=1), encoding="utf-8")
    pv = per_version(local, runs)
    (OUT / "local_vs_platform.json").write_text(json.dumps(pv, indent=1, default=str), encoding="utf-8")
    for c, r in pb["final_model_v1_3"].items():
        print(f"{c:4s} run967620 {r['mean_run_967620']:.4f} se_pairs {r['within_draw_se']:.4f} local {r['local_R2']:.4f} "
              f"| pooled {pb['pooled_v1_1_to_v1_3'][c]['mean']:.4f} sd {pb['pooled_v1_1_to_v1_3'][c]['one_draw_sd']:.4f} "
              f"above {pb['pooled_v1_1_to_v1_3'][c]['runs_above_own_local']}/3 "
              f"slots {pb['pooled_v1_1_to_v1_3'][c]['slots_above_own_local']}/26 p={pb['pooled_v1_1_to_v1_3'][c]['slot_sign_test_p']:.3g}")
    a = pb["final_model_v1_3"]["ALC"]
    print("ALC 4 runs", a["mean_over_runs"], a["one_draw_sd"], a["se_mean"], a["ci95_t"], a["runs_above_local"], a["sign_test_p"])
    for v, r in pv["table"].items():
        print(v, r["local_R2_alc"], r["platform_n"], r["platform_mean"], r["platform_ci95_t"], round(r["replay_alc_18"], 4),
              round(r["replay_alc_strict8"], 4))
    print(json.dumps(pv["paired_replay"], indent=0))
    print(json.dumps(pv["ranking"], indent=0))


if __name__ == "__main__":
    main()
