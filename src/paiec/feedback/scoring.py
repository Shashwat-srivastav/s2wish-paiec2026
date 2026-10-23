"""Platform scoring rules, re-implemented exactly as published, plus the constant-prediction identities.

Definitions (PAIEC competition page / starting-kit README, "Scoring"):

* Outcomes are binary, y in {0, 1}; each stored response row of a pair's fixed evaluation pool is scored at
  every budget. The pool does not change across budgets, so a pair's success rate
  ``a = mean(y)`` over the scored rows is one number shared by all six budgets.
* Brier score at budget b: ``mean((p - y)^2)`` over the scored rows.
* ECE at budget b: ten fixed equal-width bins ``[0, 0.1), ..., [0.9, 1]`` (last bin closed); within each
  occupied bin the absolute gap between mean prediction and observed success rate, weighted by the bin's
  response count.
* Brier ALC: ``0.1 B0 + 0.2 B1 + 0.2 B3 + 0.2 B7 + 0.2 B15 + 0.1 B31`` per pair, then the unweighted mean over pairs.

Constant-prediction identities (the basis of every inversion in this package). If a model predicts one value
``q`` for every scored row of a pair at budget b, then, because ``y^2 = y``,

    Brier_b = a (1 - 2 q) + q^2 = a (1 - a) + (q - a)^2,          ECE_b = |q - a|,

and ECE_b does not depend on the bin edges (all rows fall in one bin).
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

BUDGETS: tuple[int, ...] = (0, 1, 3, 7, 15, 31)
ALC_WEIGHTS: Mapping[int, float] = {0: 0.1, 1: 0.2, 3: 0.2, 7: 0.2, 15: 0.2, 31: 0.1}
N_BINS = 10


def brier(y: ArrayLike, p: ArrayLike) -> float:
    """Brier score ``mean((p - y)^2)`` of binary outcomes ``y`` under predictions ``p``."""
    y = np.asarray(y, dtype=float)
    p = np.broadcast_to(np.asarray(p, dtype=float), y.shape)
    return float(np.mean((p - y) ** 2))


def ece(y: ArrayLike, p: ArrayLike, n_bins: int = N_BINS) -> float:
    """Count-weighted ECE with ``n_bins`` fixed equal-width bins ``[0, 1/n), ..., [(n-1)/n, 1]``.

    Bins are indexed by ``floor(n p)`` with the last bin closed, which avoids the inexact edges of
    ``np.linspace`` (0.30000000000000004 etc.).
    """
    y = np.asarray(y, dtype=float)
    p = np.broadcast_to(np.asarray(p, dtype=float), y.shape)
    if y.size == 0:
        return 0.0
    idx = np.minimum(np.floor(p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for i in np.unique(idx):
        m = idx == i
        total += m.sum() / y.size * abs(p[m].mean() - y[m].mean())
    return float(total)


def alc(curve: Mapping[int, float] | Sequence[float], weights: Mapping[int, float] = ALC_WEIGHTS) -> float:
    """Brier ALC of one pair. ``curve`` is ``{budget: Brier}`` or a sequence aligned with :data:`BUDGETS`."""
    if not isinstance(curve, Mapping):
        curve = dict(zip(BUDGETS, curve))
    return float(sum(w * float(curve[b]) for b, w in weights.items()))


def brier_constant(a: ArrayLike, q: ArrayLike) -> np.ndarray:
    """Exact Brier of a pair with success rate ``a`` under a prediction ``q`` constant on the pair.

    ``a (1 - a) + (q - a)^2``; vectorised over matching shapes.
    """
    a = np.asarray(a, dtype=float)
    q = np.asarray(q, dtype=float)
    return a * (1.0 - a) + (q - a) ** 2


def ece_constant(a: ArrayLike, q: ArrayLike) -> np.ndarray:
    """Exact ECE of a pair with success rate ``a`` under a constant prediction ``q``: ``|q - a|``."""
    return np.abs(np.asarray(q, dtype=float) - np.asarray(a, dtype=float))
