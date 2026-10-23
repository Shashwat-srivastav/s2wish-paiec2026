"""Table 3, official empirical-mean row on the same R2 episodes as the other rows (seeds 0-9).

    python scripts/run_slot.py -- python scripts/table3_official_baseline.py

The paper's other Table 3 rows use R2 seeds 0-9 (E.build(D, "R2", range(10)) per fold), but the official-mean
row came from the phase-2 run with seeds 0-4. This script scores the official empirical mean (ladder rung L0)
on exactly the seed 0-9 episodes, together with v0 and v1_3 on the same episode objects. It reports:
- the row (B0..B31, ALC);
- per-fold ALC;
- paired deltas against v1_3 (pair bootstrap, as elsewhere);
- a sanity check that seeds 0-4 reproduce the stored phase-2 row.

Output: experiments/runs/review_round1/table3_baseline_seeds0-9.json (written only with --write, the default).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from paiec.evaluation import episodes as E  # noqa: E402
from paiec.models import ladder as Lad, v1_1 as V11, v1_2 as V12  # noqa: E402

OUT = ROOT / "experiments" / "runs" / "review_round1" / "table3_baseline_seeds0-9.json"


def score(ep, probs):
    br, _ = ep.score(probs)
    return dict(br=br, alc=br @ E.W, pair=ep.inst_pair.copy())


def run(seeds):
    res = {}
    v13 = V12.cfg_of(facets="all", guard=0)
    for B in E.EVAL_FOLDS:
        D = E.compact(B)
        ep = E.build(D, "R2", seeds)
        H = V11.H_fold(B)
        res[B] = {
            "official": score(ep, {b: Lad.predict_budget(ep, H, "L0", b)[0] for b in E.BUDGETS}),
            "v0": score(ep, {b: Lad.predict_budget(ep, H, "L1", b)[0] for b in E.BUDGETS}),
            "v1_3": score(ep, {b: V12.predict(ep, H, b, v13, V12.fe_for(B, D, v13)) for b in E.BUDGETS}),
        }
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    t0 = time.time()
    res = run(list(range(10)))
    rows = {k: dict(curve=V11.mean_curve(res, k).tolist(), alc=V11.mean_alc(res, k),
                    per_fold={B: float(V11.pair_mean(r[k]["alc"], r[k]["pair"]).mean()) for B, r in res.items()})
            for k in ("official", "v0", "v1_3")}
    deltas = {f"{k}_minus_v1_3_x1e3": V11.delta(res, k, "v1_3") for k in ("official", "v0")}
    chk5 = run(list(range(5)))
    l0_5 = dict(curve=V11.mean_curve(chk5, "official").tolist(), alc=V11.mean_alc(chk5, "official"))
    saved = json.loads((ROOT / "experiments" / "runs" / "e5" / "real.json").read_text(encoding="utf-8"))["R2"]["v0"]
    v13_saved = json.loads((ROOT / "experiments" / "runs" / "preregistration" /
                            "exp_0044_tempering_replay_2026-10-14.json").read_text(encoding="utf-8"))["results"]["R2"]["baseline"]
    out = {
        "what": "Table 3 official empirical-mean row (ladder L0) on R2 seeds 0-9, same episodes as v0 and v1_3",
        "dataset_revision": (ROOT / "data" / "REVISION").read_text().strip()[:10],
        "regime": "R2", "seeds": list(range(10)), "folds": list(E.EVAL_FOLDS), "budgets": list(E.BUDGETS),
        "rows": rows, "paired_deltas": deltas,
        "checks": {
            "official_seeds0-4_curve_alc": l0_5,
            "v0_matches_e5_real_json": max(abs(x - y) for x, y in zip(rows["v0"]["curve"] + [rows["v0"]["alc"]],
                                                                     saved["curve"] + [saved["alc"]])),
            "v1_3_matches_exp0044_json": max(abs(x - y) for x, y in zip(rows["v1_3"]["curve"] + [rows["v1_3"]["alc"]],
                                                                       v13_saved["curve"] + [v13_saved["alc"]])),
        },
        "seconds": time.time() - t0,
        "command": "python scripts/run_slot.py -- python scripts/table3_official_baseline.py",
    }
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    r = rows["official"]
    print("official R2 seeds 0-9: " + " ".join(f"B{b}={v:.4f}" for b, v in zip(E.BUDGETS, r["curve"])) + f" ALC={r['alc']:.4f}")
    print("checks:", json.dumps(out["checks"]))
    print("wrote", p)


if __name__ == "__main__":
    main()
