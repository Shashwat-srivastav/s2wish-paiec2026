"""Floor and ceiling bounds on what any pair-level method can gain.

Definitions. A *pair-level* method predicts, at each budget, one value per pair (constant over the pair's scored
rows); it may depend on anything else (benchmark ID, revealed labels, priors). For such a method,

    Brier_b = a (1 - a) + (q_b - a)^2 >= a (1 - a),

with equality iff ``q_b = a``. Hence:

* pair-constant floor: ``F = mean_pairs a (1 - a)``; no pair-level method has Brier ALC below ``F`` (ALC weights
  sum to 1), and the bound is attained only by the oracle ``q_b = a``;
* label-removable excess at budget b: ``X_b = mean_pairs (B_b - a (1 - a))`` for the observed curves ``B_b``;
* ceiling on the gain of any pair-level change: ``Delta ALC >= -sum_b w_b X_b``.

Item-varying predictions (e.g. item pooling across subjects of a repeated benchmark) are NOT pair-level and can go
below ``a (1 - a)``; on such pairs ``B_b - a (1 - a)`` can be negative. The floor needs only ``a`` (A1-A3 at budget 0),
or only (B0, ECE0) via ``B0 - ECE0^2``, since it is invariant to the reflection ``a -> 1 - a``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from .scoring import ALC_WEIGHTS, BUDGETS


def pair_constant_floor(a: ArrayLike) -> np.ndarray:
    """``a (1 - a)``: the smallest Brier any prediction constant on the pair can achieve."""
    a = np.asarray(a, dtype=float)
    return a * (1.0 - a)


@dataclass(frozen=True)
class PairLevelBounds:
    """Floor, per-budget excess and ceiling over a set of pairs (equal weight per pair, as on the leaderboard)."""

    n_pairs: int
    floor: float
    observed_alc: float
    excess: dict[int, float]
    ceiling_gain: float  # most negative ALC change any pair-level method can achieve (= floor - observed_alc)


def pair_level_bounds(a: ArrayLike, curves: ArrayLike, budgets: Sequence[int] = BUDGETS,
                      weights: Mapping[int, float] = ALC_WEIGHTS) -> PairLevelBounds:
    """Bounds from success rates ``a`` (n,) and observed Brier curves ``curves`` (n, len(budgets))."""
    a = np.asarray(a, dtype=float)
    B = np.asarray(curves, dtype=float)
    if B.shape != (a.size, len(budgets)):
        raise ValueError("curves must have shape (n_pairs, n_budgets)")
    fl = pair_constant_floor(a)
    w = np.array([weights[b] for b in budgets])
    excess = {b: float(np.mean(B[:, i] - fl)) for i, b in enumerate(budgets)}
    ceiling = -float(sum(w[i] * excess[b] for i, b in enumerate(budgets)))
    return PairLevelBounds(n_pairs=int(a.size), floor=float(fl.mean()), observed_alc=float((B @ w).mean()),
                           excess=excess, ceiling_gain=ceiling)
