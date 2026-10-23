"""Exact, zero-upload counterfactual replay of pair-level model changes.

Setting. For a pair with recovered success rate ``a`` (:mod:`.inversion`) and recovered success counts ``k_b``
(:mod:`.krecovery`), any *pair-level* rule, i.e. a rule ``q(k, b)`` that predicts one value for every scored row of
the pair from the revealed-label count alone, scores exactly

    Brier_b(q) = a (1 - a) + (q(k_b, b) - a)^2.

No outcome other than the platform's own scored rows is involved, nothing is fitted, and nothing is uploaded.

Validity. The replay is exact only where the inputs are exact: ``a`` requires A1-A3 at budget 0, the k path
requires A4-A5 (singleton pairs, count-only predictions) for the run whose feedback it was recovered from, and the
alternative rule must itself be pair-level (A1 and A4 for the alternative). At budget 0 only ``a`` is needed, so
budget-0 counterfactuals are exact on every pair where the observed and the alternative budget-0 predictions are
constant per pair. Replays are in-sample: they score a change on the same draw that informed it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from .scoring import ALC_WEIGHTS, BUDGETS, alc, brier_constant

Rule = Callable[[int, int], float]


@dataclass(frozen=True)
class Replay:
    """Replayed Brier curve of one pair under ``rule``, its ALC and, if supplied, the change from the observed curve."""

    curve: dict[int, float]
    alc: float
    observed_alc: float | None
    delta_alc: float | None


def replay_curve(a: float, ks: Mapping[int, int], rule: Rule, budgets: Sequence[int] = BUDGETS) -> dict[int, float]:
    """Exact Brier at each budget for a pair-level ``rule(k, b)``; ``ks`` maps budget -> recovered ``k_b``."""
    return {b: float(brier_constant(a, rule(int(ks[b]), b))) for b in budgets}


def replay(a: float, ks: Mapping[int, int], rule: Rule, observed: Mapping[int, float] | None = None,
           weights: Mapping[int, float] = ALC_WEIGHTS) -> Replay:
    """Replay ``rule`` on one pair; if ``observed`` (budget -> reported Brier) is given, also return the ALC change."""
    curve = replay_curve(a, ks, rule, tuple(weights))
    r_alc = alc(curve, weights)
    o_alc = alc(observed, weights) if observed is not None else None
    return Replay(curve=curve, alc=r_alc, observed_alc=o_alc, delta_alc=None if o_alc is None else r_alc - o_alc)


def budget0_delta(a: ArrayLike, q_new: ArrayLike, q_old: ArrayLike) -> np.ndarray:
    """Per-pair exact change in B0 when the budget-0 constant prediction moves from ``q_old`` to ``q_new``.

    Equals ``(q_new - a)^2 - (q_old - a)^2``. Its mean times the budget-0 ALC weight (0.1) is the ALC contribution.
    """
    return brier_constant(a, q_new) - brier_constant(a, q_old)
