"""Run the preregistered v1_3 likelihood-tempering replay (EXP-0044)."""
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
from paiec.models import v1_2 as V


REFERENCE_ARCHIVE = paths.DIST / "v1_3_named_facets_noguard.zip"
REFERENCE_SHA256 = "A4B0E841D4A344FF207BED2C855342212E95F7D550195B667446EA4E92685B86"
OUTPUT = paths.RUNS / "preregistration" / "exp_0044_tempering_replay_2026-10-14.json"
SEEDS = {"R2": list(range(10)), "R1": list(range(100))}
REGIMES = ("R2", "R1")


def _config(**overrides: float) -> dict:
    # The shipped model treats any truthy "facets" value as all-feature facets.
    return V.cfg_of(facets="all", guard=0, **overrides)


def _validate_reference() -> dict:
    checks = {}
    with zipfile.ZipFile(REFERENCE_ARCHIVE) as archive:
        params = json.loads(archive.read("v1_2_params.json"))
        if params["facets"] != "named" or params["guard"] != 0:
            raise RuntimeError("Reference archive parameters no longer match v1_3")
        with tempfile.TemporaryDirectory(prefix="paiec_exp0044_reference_") as temp_dir:
            archive.extractall(temp_dir)
            model = S.load_model_module(Path(temp_dir) / "model.py")
            label = S.load_model_module(Path(temp_dir) / "labeling.py")
            model.P.update(params)
            for benchmark, seed, pair_count in (
                ("real_webagents", 0, 6),
                ("matharena", 3, 4),
            ):
                data = E.compact(benchmark)
                n = min(pair_count, len(data["pair_sid"]))
                subject_ids = {str(subject) for subject in data["pair_sid"][:n]}
                rows = [
                    row
                    for row in S.load_benchmark_dir(str(E.DATA / benchmark))
                    if str(row["subject_id"]) in subject_ids and row["y"] in (0, 1)
                ]
                pairs = S.prepare_pairs(rows, seed=seed)
                expected_subjects = [str(subject) for subject in data["pair_sid"][:n]]
                if [pair.subject_id for pair in pairs] != expected_subjects:
                    raise RuntimeError(f"Reference episode order mismatch for {benchmark}")
                package = S.run_direct(
                    pairs, model.predict, label.acquisition_function, isolation="call"
                )
                episode = E.Episodes(data, list(range(n)), [seed] * n, [0] * n)
                simulated_labels = np.concatenate(
                    [[row["y"] for row in pair.eval_rows] for pair in pairs]
                )
                if not np.array_equal(simulated_labels, episode.r_y):
                    raise RuntimeError(f"Reference row alignment mismatch for {benchmark}")
                h = {key: params[key] for key in ("G", "tau", "sigma_beta")}
                common = {
                    "guard": params["guard"],
                    "prior": (
                        "t",
                        3,
                        params["prior_center"],
                        params["prior_scale"],
                    ),
                    "rho": params["rho0"],
                }
                max_delta = {}
                named_control_max_delta = {}
                for mode, target in (("effective_all", "all"), ("named_control", "named")):
                    cfg = V.cfg_of(facets=target, **common)
                    features = V.fe_for(benchmark, data, cfg)
                    differences = []
                    for budget in E.BUDGETS:
                        prediction = V.predict(episode, h, budget, cfg, features)[episode.r_t]
                        differences.append(
                            float(
                                np.abs(
                                    np.asarray(package["probabilities"][budget], float)
                                    - prediction
                                ).max()
                            )
                        )
                    if mode == "effective_all":
                        max_delta[benchmark] = max(differences)
                    else:
                        named_control_max_delta[benchmark] = max(differences)
                if max_delta[benchmark] >= 1e-9:
                    raise RuntimeError(
                        f"Reference package differs from all-facet engine on {benchmark}: "
                        f"{max_delta[benchmark]}"
                    )
                checks[benchmark] = {
                    "pairs": n,
                    "max_abs_delta_effective_all": max_delta[benchmark],
                    "max_abs_delta_named_control": named_control_max_delta[benchmark],
                }
    return {
        "archive_parameters_facets_value": params["facets"],
        "archive_parameters_guard": params["guard"],
        "effective_facets": "all item_features fields (the packaged model treats 'named' as truthy)",
        "package_vs_engine": checks,
    }


def run() -> dict:
    actual_hash = hashlib.sha256(REFERENCE_ARCHIVE.read_bytes()).hexdigest().upper()
    if actual_hash != REFERENCE_SHA256:
        raise RuntimeError(
            f"Reference archive hash mismatch: expected {REFERENCE_SHA256}, got {actual_hash}"
        )
    reference_equivalence = _validate_reference()

    configs = {
        "baseline": _config(),
        "w_od_0.8": _config(w_od=0.8),
        "tau_scale_0.8": _config(tau_scale=0.8),
        "both_0.8": _config(w_od=0.8, tau_scale=0.8),
    }
    started = time.perf_counter()
    raw, diagnostics = V.evaluate(
        configs, regimes=REGIMES, seeds=SEEDS, tag="EXP-0044"
    )
    summaries = {
        regime: V.summarise(raw[regime], list(configs), "baseline")
        for regime in REGIMES
    }

    decisions = {}
    for regime in REGIMES:
        decisions[regime] = {}
        for name in configs:
            if name == "baseline":
                continue
            result = summaries[regime][name]
            delta = result["d"]
            b1_delta = result["d_budget"][1]["pt"]
            passes_primary_adoption = (
                delta["pt"] <= -1.0
                and delta["hi"] < 0.0
                and delta["n_neg"] >= 3
                and delta["worst_fold"] <= 0.5
                and b1_delta <= 0.5
            )
            decisions[regime][name] = {
                "mean_delta_alc_x1e3": delta["pt"],
                "pair_bootstrap_95ci_x1e3": [delta["lo"], delta["hi"]],
                "fold_mean_delta_alc_x1e3": delta["per_fold"],
                "negative_folds": delta["n_neg"],
                "worst_fold_delta_x1e3": delta["worst_fold"],
                "b1_delta_x1e3": b1_delta,
                "passes_registered_adoption_bar": passes_primary_adoption,
            }

    report = {
        "experiment_id": "EXP-0044",
        "preregistered": True,
        "reference_archive": str(REFERENCE_ARCHIVE.relative_to(paths.ROOT)),
        "reference_sha256": actual_hash,
        "dataset_revision": "bc8204d811",
        "folds": list(E.EVAL_FOLDS),
        "regimes": {regime: SEEDS[regime] for regime in REGIMES},
        "budgets": list(E.BUDGETS),
        "acquisition_policy": "A1 (accept-all)",
        "configurations": configs,
        "reference_equivalence": reference_equivalence,
        "elapsed_seconds": time.perf_counter() - started,
        "runtime_diagnostics": {
            f"{regime}|{fold}": diagnostics[(regime, fold)]
            for regime in REGIMES
            for fold in E.EVAL_FOLDS
        },
        "results": summaries,
        "decisions": decisions,
        "build_candidate": any(
            decisions["R2"][name]["passes_registered_adoption_bar"]
            and decisions["R1"][name]["mean_delta_alc_x1e3"] < 0.0
            and decisions["R1"][name]["b1_delta_x1e3"] <= 0.5
            for name in decisions["R2"]
        ),
        "limitations": [
            "R2 is the primary adoption regime; R1 is a stress/stability regime.",
            "Pair-bootstrap intervals are conditional on the four LOBO folds; fold-level intervals are reported separately.",
            "A candidate archive is built only if it passes the registered R2 adoption bar and has non-positive R1 mean delta with no B1 regression above 0.5e-3.",
            "This local replay does not authorize a Codabench submission.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"EXP-0044 results: {OUTPUT}")
    print(f"elapsed_seconds={report['elapsed_seconds']:.2f}")
    for regime, candidates in decisions.items():
        for name, result in candidates.items():
            print(
                f"{regime} {name}: delta={result['mean_delta_alc_x1e3']:+.3f}e-3 "
                f"CI=[{result['pair_bootstrap_95ci_x1e3'][0]:+.3f},"
                f"{result['pair_bootstrap_95ci_x1e3'][1]:+.3f}] "
                f"folds={result['negative_folds']}/4 "
                f"B1={result['b1_delta_x1e3']:+.3f}e-3 "
                f"adopt={result['passes_registered_adoption_bar']}"
            )
    print(f"build_candidate={report['build_candidate']}")
    return report


if __name__ == "__main__":
    run()
