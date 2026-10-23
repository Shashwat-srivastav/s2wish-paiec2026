"""Synthetic 2PL-IRT response data in the simulator row schema and in the
starting_kit sample_data CSV format.

P(correct) = sigmoid(a_j * (theta_s + offset_b - d_j)).
Coverage includes: eligible benchmarks of 80-300 items (80 exact boundary, odd
sizes), pairs below 80 distinct items (79, 70, whole 60-item benchmark),
repeated trials (1-3 runs/item) for some pairs, and two subjects that share a
base model name but differ in reasoning effort.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

SUBJECT_FIELDS = ("normalized_name", "provider", "release_date", "access_date",
                  "harness", "reasoning_effort", "harness_version", "subject_features_extra")

# benchmark size, per-subject item-count overrides, subjects with repeated trials
DEFAULT_BENCHMARKS = (
    (80, {}, ()),
    (95, {}, (1, 3)),
    (150, {0: 70, 1: 79}, ()),
    (300, {2: 81, 4: 120}, (0, 5)),
    (60, {}, (2,)),
    (201, {3: 0}, (4,)),
)


def make_synthetic(seed: int = 0, n_subjects: int = 6, benchmarks=DEFAULT_BENCHMARKS,
                   max_trials: int = 3) -> list[dict]:
    rng = np.random.default_rng(seed)
    used = set()

    def alias(prefix):
        while True:
            v = f"{prefix}_{int(rng.integers(100000, 999999))}"
            if v not in used:
                used.add(v)
                return v

    subjects = []
    for s in range(n_subjects):
        base = s if s != 1 else 0  # subjects 0 and 1 share a base model name
        subjects.append({
            "subject_id": alias("subject"), "theta": float(rng.normal(0, 1)),
            "normalized_name": f"SynthModel-{base}", "provider": f"Provider{base % 3}",
            "release_date": f"2026-0{1 + base % 9}-01", "access_date": "2026-06-01",
            "harness": "harness_a" if s % 2 else "harness_b",
            "reasoning_effort": ("low", "high", "medium")[s % 3],
            "harness_version": f"1.{s}", "subject_features_extra": "",
        })
    rows = []
    for b_idx, (size, overrides, repeated) in enumerate(benchmarks):
        bid = alias("benchmark")
        offset = float(rng.normal(0, 0.7))
        disc = rng.lognormal(0.0, 0.35, size)
        diff = rng.normal(0, 1.2, size)
        for s_idx, subj in enumerate(subjects):
            n_items = overrides.get(s_idx, size)
            if n_items <= 0:
                continue
            chosen = np.sort(rng.choice(size, size=n_items, replace=False))
            for j in chosen:
                p = 1.0 / (1.0 + math.exp(-disc[j] * (subj["theta"] + offset - diff[j])))
                trials = int(rng.integers(1, max_trials + 1)) if s_idx in repeated else 1
                for _ in range(trials):
                    rows.append({
                        **{k: subj[k] for k in SUBJECT_FIELDS},
                        "item_content": f"Synthetic task {j} of benchmark {b_idx}: solve it.",
                        "item_features": f"tier={'hard' if diff[j] > 0 else 'easy'};idx={j}",
                        "interactors": "", "benchmark_id": bid,
                        "item_id": f"item_{b_idx}_{j}", "subject_id": subj["subject_id"],
                        "y": int(rng.random() < p), "p_true": p,
                    })
    return rows


def write_sample_data(rows: list[dict], root) -> Path:
    """Write rows as test/{models,test_items,test_pairs}.csv + ref/ground_truth.csv."""
    root = Path(root)
    (root / "test").mkdir(parents=True, exist_ok=True)
    (root / "ref").mkdir(parents=True, exist_ok=True)
    models, items = {}, {}
    for r in rows:
        models.setdefault(r["subject_id"], {"model_id": r["subject_id"], **{k: r[k] for k in SUBJECT_FIELDS}})
        items.setdefault(r["item_id"], {k: r[k] for k in ("item_id", "item_content", "item_features",
                                                           "benchmark_id", "interactors")})

    def dump(path, fields, data):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(data)

    dump(root / "test" / "models.csv", ["model_id", *SUBJECT_FIELDS], models.values())
    dump(root / "test" / "test_items.csv",
         ["item_id", "item_content", "item_features", "benchmark_id", "interactors"], items.values())
    dump(root / "test" / "test_pairs.csv", ["model_id", "item_id"],
         [{"model_id": r["subject_id"], "item_id": r["item_id"]} for r in rows])
    dump(root / "ref" / "ground_truth.csv", ["model_id", "item_id", "label"],
         [{"model_id": r["subject_id"], "item_id": r["item_id"], "label": r["y"]} for r in rows])
    return root
