"""LOBO evaluation for the research-only correlated hierarchical model."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from paiec import paths
from paiec.evaluation import episodes as E
from paiec.models import ladder as Lad
from paiec.models import v1_2 as V12
from paiec.models.correlated_hierarchical import (
    CorrelatedHierarchicalLogistic,
    EffectScales,
)

OUT = paths.RUNS / "correlated_hierarchical"


def _blend_predictions(
    baseline: dict[int, np.ndarray],
    candidate: dict[int, np.ndarray],
    alpha: float,
) -> dict[int, np.ndarray]:
    """Shrink candidate probabilities toward the established baseline."""
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("blend alpha must be finite and between 0 and 1.")
    if baseline.keys() != candidate.keys():
        raise ValueError("baseline and candidate budgets must match.")
    return {
        budget: (1.0 - alpha) * baseline[budget] + alpha * candidate[budget]
        for budget in baseline
    }


def _system_names(data: dict, pair_indices: np.ndarray, benchmark: str) -> list[str]:
    names = data["pair_name"]
    sids = data["pair_sid"]
    result = []
    for pair_index in pair_indices:
        name = str(names[int(pair_index)]).strip()
        result.append(name if name else f"anonymous:{benchmark}:{sids[int(pair_index)]}")
    return result


def _aggregate_data(data: dict, benchmark: str) -> dict[str, np.ndarray | list]:
    """Aggregate raw binary rows to system-item binomial cells."""
    pair = np.asarray(data["row_pair"], dtype=np.int64)
    item_indices = np.asarray(data["row_item"], dtype=np.int64)
    item_key = np.asarray(data["item_key"], dtype=np.int64)[item_indices]
    cells, inverse = np.unique(
        np.column_stack((pair, item_key)), axis=0, return_inverse=True
    )
    trials = np.bincount(inverse).astype(np.int64)
    successes = np.bincount(
        inverse, weights=np.asarray(data["row_y"], dtype=float)
    ).astype(np.int64)
    system = _system_names(data, cells[:, 0], benchmark)
    return {
        "benchmark": [benchmark] * len(cells),
        "system": system,
        "item": [(benchmark, int(key)) for key in cells[:, 1]],
        "successes": successes,
        "trials": trials,
    }


def _training_rows(held_out: str) -> dict[str, list | np.ndarray]:
    chunks = [
        _aggregate_data(E.compact(benchmark), benchmark)
        for benchmark in E.BENCHES
        if benchmark != held_out
    ]
    return {
        key: sum((chunk[key] for chunk in chunks), [])
        if key in ("benchmark", "system", "item")
        else np.concatenate([chunk[key] for chunk in chunks])
        for key in chunks[0]
    }


def _append_acquired(
    train: dict[str, list | np.ndarray],
    data: dict,
    benchmark: str,
    ep,
    group: int,
    budget: int,
) -> dict[str, list | np.ndarray]:
    li, lk, ly = ep.labels(budget)
    selected = ep.inst_group[li] == group if len(li) else np.zeros(0, dtype=bool)
    pair_indices = ep.inst_pair[li[selected]]
    return {
        "benchmark": list(train["benchmark"]) + [benchmark] * int(selected.sum()),
        "system": list(train["system"]) + _system_names(data, pair_indices, benchmark),
        "item": list(train["item"]) + [
            (benchmark, int(key)) for key in lk[selected]
        ],
        "successes": np.concatenate((
            np.asarray(train["successes"]), np.asarray(ly[selected], dtype=np.int64)
        )),
        "trials": np.concatenate((
            np.asarray(train["trials"]),
            np.ones(int(selected.sum()), dtype=np.int64),
        )),
    }


def _fit_predict(data: dict, train: dict, benchmark: str, ep, group: int,
                 budget: int, scales: EffectScales) -> np.ndarray:
    rows = _append_acquired(train, data, benchmark, ep, group, budget)
    model = CorrelatedHierarchicalLogistic(scales=scales, max_iter=2_000).fit(
        rows["benchmark"], rows["system"], rows["item"],
        rows["successes"], trials=rows["trials"],
    )
    target_mask = ep.inst_group[ep.t_inst] == group
    target_instances = ep.t_inst[target_mask]
    system_ids = _system_names(data, ep.inst_pair[target_instances], benchmark)
    target_keys = ep.t_key[target_mask]
    return model.predict_marginal_mean(
        [benchmark] * int(target_mask.sum()),
        system_ids,
        [(benchmark, int(key)) for key in target_keys],
    )


def evaluate_fold(
    benchmark: str,
    seeds: list[int],
    scales: EffectScales,
    blend_alphas: list[float],
) -> dict:
    data = E.compact(benchmark)
    train = _training_rows(benchmark)
    ep = E.build(data, "R3", seeds)
    cfg = V12.cfg_of(facets="all", guard=1)
    hyper = Lad.hyper_for(benchmark)
    facet_encoding = V12.fe_for(benchmark, data, cfg)
    candidate_probs = {budget: np.empty(len(ep.t_key)) for budget in E.BUDGETS}
    baseline_probs = {budget: np.empty(len(ep.t_key)) for budget in E.BUDGETS}

    for group in np.unique(ep.inst_group):
        mask = ep.inst_group[ep.t_inst] == group
        for budget in E.BUDGETS:
            candidate_probs[budget][mask] = _fit_predict(
                data, train, benchmark, ep, int(group), budget, scales
            )
        for budget in E.BUDGETS:
            baseline_probs[budget][mask] = V12.predict(
                ep, hyper, budget, cfg, facet_encoding
            )[mask]

    candidate_brier, _ = ep.score(candidate_probs)
    baseline_brier, _ = ep.score(baseline_probs)
    blend_pair_alc = {}
    blend_brier_by_budget = {}
    for alpha in blend_alphas:
        blended_brier, _ = ep.score(
            _blend_predictions(baseline_probs, candidate_probs, alpha)
        )
        label = f"{alpha:g}"
        blend_pair_alc[label] = (blended_brier @ E.W).tolist()
        blend_brier_by_budget[label] = blended_brier.mean(0).tolist()
    return {
        "n_pairs": ep.n,
        "n_target_rows": len(ep.r_y),
        "candidate_pair_alc": (candidate_brier @ E.W).tolist(),
        "baseline_pair_alc": (baseline_brier @ E.W).tolist(),
        "candidate_brier_by_budget": candidate_brier.mean(0).tolist(),
        "baseline_brier_by_budget": baseline_brier.mean(0).tolist(),
        "blend_pair_alc": blend_pair_alc,
        "blend_brier_by_budget": blend_brier_by_budget,
    }


def run(
    seeds: list[int],
    scales: EffectScales,
    blend_alphas: list[float],
) -> dict:
    results = {}
    started = time.time()
    for benchmark in E.EVAL_FOLDS:
        results[benchmark] = evaluate_fold(
            benchmark, seeds, scales, blend_alphas
        )
        cand = np.mean(results[benchmark]["candidate_pair_alc"])
        base = np.mean(results[benchmark]["baseline_pair_alc"])
        print(f"{benchmark}: candidate={cand:.5f} baseline={base:.5f} "
              f"delta={cand - base:+.5f}", flush=True)
    candidate_fold_means = [
        np.mean(results[b]["candidate_pair_alc"]) for b in E.EVAL_FOLDS
    ]
    baseline_fold_means = [
        np.mean(results[b]["baseline_pair_alc"]) for b in E.EVAL_FOLDS
    ]
    deltas = np.asarray(candidate_fold_means) - np.asarray(baseline_fold_means)
    rng = np.random.default_rng(17)
    blend_summaries = {}
    for alpha in blend_alphas:
        label = f"{alpha:g}"
        fold_means = [
            np.mean(results[b]["blend_pair_alc"][label])
            for b in E.EVAL_FOLDS
        ]
        blend_deltas = np.asarray(fold_means) - np.asarray(baseline_fold_means)
        fold_bootstraps = []
        for benchmark in E.EVAL_FOLDS:
            blended = np.asarray(results[benchmark]["blend_pair_alc"][label])
            baseline = np.asarray(results[benchmark]["baseline_pair_alc"])
            difference = blended - baseline
            fold_bootstraps.append(np.mean(
                difference[
                    rng.integers(0, len(difference), size=(5_000, len(difference)))
                ],
                axis=1,
            ))
        overall_bootstrap = np.mean(np.stack(fold_bootstraps), axis=0)
        blend_summaries[label] = {
            "fold_balanced_alc": float(np.mean(fold_means)),
            "fold_balanced_delta_vs_baseline": float(np.mean(blend_deltas)),
            "benchmark_fold_deltas": blend_deltas.tolist(),
            "paired_pair_bootstrap_95pct_ci": np.percentile(
                overall_bootstrap, [2.5, 97.5]
            ).tolist(),
        }
        print(
            f"blend alpha={label}: ALC={np.mean(fold_means):.5f} "
            f"baseline_delta={np.mean(blend_deltas):+.5f}",
            flush=True,
        )
    summary = {
        "evaluation": "LOBO; R3 all eligible pairs grouped by benchmark per seed",
        "seeds": seeds,
        "scales": {
            "benchmark": scales.benchmark,
            "system": scales.system,
            "system_benchmark": scales.system_benchmark,
            "item": scales.item,
        },
        "fold_means": {
            b: {
                "candidate_alc": candidate_fold_means[k],
                "baseline_alc": baseline_fold_means[k],
                "delta": float(deltas[k]),
            }
            for k, b in enumerate(E.EVAL_FOLDS)
        },
        "fold_balanced_candidate_alc": float(np.mean(candidate_fold_means)),
        "fold_balanced_baseline_alc": float(np.mean(baseline_fold_means)),
        "fold_balanced_delta": float(np.mean(deltas)),
        "benchmark_fold_deltas": deltas.tolist(),
        "blend_alphas": blend_alphas,
        "blends": blend_summaries,
        "uncertainty_note": (
            "Pair bootstrap is conditional on these four held-out benchmarks; "
            "it does not quantify uncertainty for a new benchmark. Alpha "
            "results are exploratory because the same folds are used to compare "
            "blend strengths; they are not an unbiased tuning estimate."
        ),
        "elapsed_seconds": time.time() - started,
        "folds": results,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    scale_label = (
        f"b{scales.benchmark:g}_s{scales.system:g}_"
        f"x{scales.system_benchmark:g}_i{scales.item:g}"
    )
    seed_label = "_".join(str(seed) for seed in seeds)
    output = OUT / f"lobo_r3_{scale_label}_seed{seed_label}.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--benchmark-scale", type=float, default=0.8)
    parser.add_argument("--system-scale", type=float, default=1.1)
    parser.add_argument("--interaction-scale", type=float, default=0.6)
    parser.add_argument("--item-scale", type=float, default=1.5)
    parser.add_argument("--blend-alpha", type=float, action="append", default=None)
    args = parser.parse_args()
    scales = EffectScales(
        benchmark=args.benchmark_scale,
        system=args.system_scale,
        system_benchmark=args.interaction_scale,
        item=args.item_scale,
    )
    run(args.seed or [0], scales, args.blend_alpha or [0.0, 0.25, 0.5, 0.75, 1.0])


if __name__ == "__main__":
    main()
