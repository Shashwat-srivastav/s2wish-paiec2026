"""Official PAIEC metric: Brier score + Brier ALC + ECE (diagnostic).

Authoritative source: https://aimslab.stanford.edu/competition
- Brier score = mean((p - y)^2), y in {0,1}
- Brier ALC = 0.1*B0 + 0.2*B1 + 0.2*B3 + 0.2*B7 + 0.2*B15 + 0.1*B31
  computed PER subject-benchmark pair, then averaged equally across pairs.
- ECE = mean absolute gap between mean predicted prob and observed success
  rate across 10 equal-width bins, weighted by response count. Summary ECE
  uses the same ALC budget weights. Diagnostic only.
- Lower is better for all. Constant 0.5 -> Brier 0.25 at every budget.

This module is a faithful local re-implementation for development.
The hidden evaluation is streaming (see external/official_baselines/tools/streaming_ingestion.py)
with 50/50 acquisition/eval split per pair (>=80 distinct items), nested label
sets, fixed split across submissions/budgets.
"""
from __future__ import annotations
import math
import numpy as np

BUDGETS = [0, 1, 3, 7, 15, 31]
ALC_WEIGHTS = {0: 0.1, 1: 0.2, 3: 0.2, 7: 0.2, 15: 0.2, 31: 0.1}
# Derivation (trapezoid on x=log2(n+1), budgets at 0..5, width 5):
# endpoints weight 0.1, interior 0.2. Sum = 1.0.

def brier_score(y_true, y_prob) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    assert y_true.shape == y_prob.shape, "shape mismatch"
    assert np.all((y_true == 0) | (y_true == 1)), "labels must be binary"
    assert np.all(np.isfinite(y_prob)), "non-finite probability"
    assert np.all((y_prob >= 0) & (y_prob <= 1)), "probability out of [0,1]"
    return float(np.mean((y_prob - y_true) ** 2))

def brier_alc(brier_by_budget: dict) -> float:
    """brier_by_budget: {0:..,1:..,3:..,7:..,15:..,31:..} -> weighted ALC."""
    missing = [b for b in BUDGETS if b not in brier_by_budget]
    assert not missing, f"missing budgets {missing}"
    return float(sum(ALC_WEIGHTS[b] * brier_by_budget[b] for b in BUDGETS))

def ece(y_true, y_prob, n_bins: int = 10) -> float:
    """Equal-width ECE, weighted by bin response count (official diagnostic)."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    if n == 0:
        return 0.0
    # Fixed bins [0,0.1), ..., [0.9,1]. Index by floor(10p) (last bin closed):
    # np.linspace edges are inexact (0.30000000000000004, 0.6000000000000001,
    # 0.7000000000000001) and would drop p=0.3/0.6/0.7 into the lower bin.
    bins = np.minimum(np.floor(y_prob * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for i in range(n_bins):
        mask = bins == i
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        acc = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        total += (cnt / n) * abs(conf - acc)
    return float(total)

def alc_from_curves(pair_curves: dict) -> tuple[float, dict]:
    """Average per-pair ALC equally across subject-benchmark pairs.

    pair_curves: {(subject_key, benchmark_id): {budget: brier}}
    Returns (overall_alc, {pair: alc}).
    """
    pair_alc = {k: brier_alc(v) for k, v in pair_curves.items()}
    overall = float(np.mean(list(pair_alc.values()))) if pair_alc else float("nan")
    return overall, pair_alc

def alc_summary_table(pair_briers: dict, pair_eces: dict | None = None) -> list[dict]:
    """Build a formative-feedback-style ALC summary (illustrative format)."""
    rows = []
    for pair, curve in pair_briers.items():
        row = {"pair": pair, "alc": brier_alc(curve)}
        if pair_eces and pair in pair_eces:
            row["ece_alc"] = float(sum(ALC_WEIGHTS[b] * pair_eces[pair][b] for b in BUDGETS))
        rows.append(row)
    return rows

if __name__ == "__main__":
    # Self-checks against official documentation examples.
    # 1. Constant 0.5 -> 0.25 regardless of labels.
    y = [0, 1, 1, 0, 1]
    assert abs(brier_score(y, [0.5]*5) - 0.25) < 1e-12
    # 2. Perfect -> 0.0.
    assert abs(brier_score(y, y) - 0.0) < 1e-12
    # 3. Illustrative curve on competition page:
    # B = [0.22, 0.18, 0.145, 0.12, 0.1, 0.09] -> ALC 0.140
    ill = dict(zip(BUDGETS, [0.22, 0.18, 0.145, 0.12, 0.1, 0.09]))
    assert abs(brier_alc(ill) - 0.140) < 1e-12, brier_alc(ill)
    # 4. Flat curve -> ALC == Brier.
    flat = {b: 0.2 for b in BUDGETS}
    assert abs(brier_alc(flat) - 0.2) < 1e-12
    # 5. Weights sum to 1.
    assert abs(sum(ALC_WEIGHTS.values()) - 1.0) < 1e-12
    # 6. ECE sanity: perfect confident predictions -> ~0.
    assert ece([0,0,1,1],[0.01,0.02,0.98,0.99]) < 0.05
    # 7. Pair averaging equally regardless of response count.
    overall, _ = alc_from_curves({"a": flat, "b": ill})
    assert abs(overall - (0.2 + 0.140)/2) < 1e-12
    print("metric self-checks OK: Brier, ALC=0.140 illustrative, ECE, pair-averaging verified.")
