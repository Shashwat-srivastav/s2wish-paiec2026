"""EXP-0057: v1_8 (v1_3 + E5 X1-ext10 offset prior) — package equivalence and local safety check.

Pre-registration: docs/preregistrations/EXP-0057_v1_8_ext10_preregistration.md. Steps 1 (equivalence) and 2 (safety on
the four hard folds, R2 seeds 0-9, paired vs v1_3). Step 3 (easy-benchmark evidence) needs EXP-0039R's harness.

    python scripts/run_slot.py -- python -m paiec.analysis.exp_0057_v1_8_ext10
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

from paiec import paths
from paiec.evaluation import episodes as E, simulator as S
from paiec.models import v1_2 as V, x1_prior as X

ARCHIVE = paths.DIST / "v1_8_ext10.zip"
OUTPUT = paths.RUNS / "preregistration" / "exp_0057_v1_8_ext10_2026-10-15.json"
SEEDS = {"R2": list(range(10))}
EXT10 = X.ext(X.CORE_H, 0.10)  # 0.90 t3(G/2, learned) + 0.05 t3(+3,1) + 0.05 t3(-3,1)


def _cfg(prior=None) -> dict:
    # The shipped model treats any truthy "facets" value as all-feature facets (EXP-0044 addendum).
    return V.cfg_of(facets="all", guard=0, **({"prior": prior} if prior else {}))


def validate_package() -> dict:
    checks = {}
    with zipfile.ZipFile(ARCHIVE) as archive, tempfile.TemporaryDirectory(
            prefix="exp0057_", dir=paths.ROOT / ".tmp") as tmp:
        params = json.loads(archive.read("v1_2_params.json"))
        archive.extractall(tmp)
        model = S.load_model_module(Path(tmp) / "model.py")
        label = S.load_model_module(Path(tmp) / "labeling.py")
        comps = tuple((float(w), float(c), float(s), 3) for w, c, s in params["prior_mix"])
        h = {k: params[k] for k in ("G", "tau", "sigma_beta")}
        cfg = V.cfg_of(facets="all", guard=params["guard"], rho=params["rho0"], prior=("x1", comps))
        for benchmark in E.EVAL_FOLDS:
            for seed in (0, 3, 7):
                data = E.compact(benchmark)
                n = min(5, len(data["pair_sid"]))
                sids = {str(s) for s in data["pair_sid"][:n]}
                rows = [r for r in S.load_benchmark_dir(str(E.DATA / benchmark))
                        if str(r["subject_id"]) in sids and r["y"] in (0, 1)]
                pairs = S.prepare_pairs(rows, seed=seed)
                if [p.subject_id for p in pairs] != [str(s) for s in data["pair_sid"][:n]]:
                    raise RuntimeError(f"episode order mismatch {benchmark} seed {seed}")
                package = S.run_direct(pairs, model.predict, label.acquisition_function, isolation="call")
                ep = E.Episodes(data, list(range(n)), [seed] * n, [0] * n)
                feats = V.fe_for(benchmark, data, cfg)
                worst = max(float(np.abs(np.asarray(package["probabilities"][b], float)
                                         - V.predict(ep, h, b, cfg, feats)[ep.r_t]).max())
                            for b in E.BUDGETS)
                checks[f"{benchmark}|seed{seed}"] = {"pairs": n, "max_abs_delta": worst}
                if worst >= 1e-9:
                    raise RuntimeError(f"package differs from engine on {benchmark} seed {seed}: {worst}")
    return checks


def run() -> dict:
    X.install()
    sha = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    started = time.perf_counter()
    equivalence = validate_package()
    configs = {"v1_3": _cfg(), "v1_8_ext10": _cfg(EXT10)}
    raw, diagnostics = V.evaluate(configs, regimes=("R2",), seeds=SEEDS, tag="EXP-0057")
    summary = V.summarise(raw["R2"], list(configs), "v1_3")["v1_8_ext10"]
    d, b1 = summary["d"], summary["d_budget"][1]["pt"]
    safety_abort = d["pt"] > 1.5 or b1 > 1.0
    report = {
        "experiment_id": "EXP-0057", "preregistered": True, "step": "1-2 (equivalence + hard-fold safety)",
        "archive": str(ARCHIVE.relative_to(paths.ROOT)), "archive_sha256": sha,
        "dataset_revision": "bc8204d811", "folds": list(E.EVAL_FOLDS), "regime": "R2", "seeds": SEEDS["R2"],
        "prior": EXT10, "package_vs_engine": equivalence,
        "delta_alc_x1e3": d["pt"], "pair_bootstrap_95ci_x1e3": [d["lo"], d["hi"]],
        "fold_delta_x1e3": d["per_fold"], "negative_folds": d["n_neg"], "worst_fold_x1e3": d["worst_fold"],
        "delta_by_budget_x1e3": {b: x["pt"] for b, x in zip(E.BUDGETS, summary["d_budget"])},
        "safety_abort": safety_abort,
        "safety_rule": "abort if dALC > +1.5e-3 or dB1 > +1.0e-3 (hard folds, R2)",
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUTPUT.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    return report


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: r[k] for k in ("delta_alc_x1e3", "pair_bootstrap_95ci_x1e3", "fold_delta_x1e3",
                                         "delta_by_budget_x1e3", "safety_abort")}, indent=1, default=str))
