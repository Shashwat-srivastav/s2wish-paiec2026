"""Recover k_b, the number of successes among the first b revealed labels of a pair, from per-budget ECE.

Assumptions, in addition to A1-A3 of :mod:`.inversion` (which give the pair's success rate ``a``):

A4. Count-only, item-independent prediction. At budget b the submitted model predicts one value ``p(k_b, b)`` for
    every scored row of the pair, where ``p(k, b)`` depends on the revealed labels only through their success count
    ``k``. For a hierarchical pair-offset model this holds when the pair's benchmark appears ONCE in the draw
    (a "singleton pair": no other subject's labels on the same items can enter through item pooling) and the model
    uses no item features the scored rows differ on. Pairs whose benchmark repeats in the draw, or whose items
    trigger item-level terms, violate A4; the residual below then exceeds the tolerance, so violations are
    detected rather than silently mis-recovered.
A5. Nested prefixes. With accept-all acquisition the platform reveals labels in a fixed candidate order, so
    ``0 <= k_b' - k_b <= b' - b`` for ``b < b'`` and the same pair yields the same k path in every run.

Method. Under A1-A4 the scoring identities give ``ECE_b = |p(k_b, b) - a|``. Given ``a`` and the model's table
``p(k, b), k = 0..b`` (computed by calling the model on synthetic labels), ``k_b`` is the ``k`` minimising the
residual ``r(k) = ||p(k, b) - a| - ECE_b|``. The recovery is accepted as exact when the minimum residual is within
the rounding tolerance (``PairInversion.k_tol``) and unique when every other ``k`` has residual above it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence, Union

import numpy as np
from numpy.typing import ArrayLike

from .scoring import BUDGETS

PTable = Union[Callable[[int, int], float], Mapping[int, ArrayLike]]


@dataclass(frozen=True)
class KRecovery:
    """One budget: best ``k``, its residual, the second-best residual, and the acceptance flags."""

    budget: int
    k: int
    residual: float
    runner_up: float
    exact: bool
    unique: bool
    p: float


@dataclass(frozen=True)
class KPath:
    """All budgets of one pair. ``ks`` maps budget -> k and includes ``0 -> 0``."""

    ks: dict[int, int]
    steps: dict[int, KRecovery] = field(repr=False)
    nested: bool
    exact: bool
    max_residual: float


def _row(p_table: PTable, b: int) -> np.ndarray:
    if callable(p_table):
        return np.array([float(p_table(k, b)) for k in range(b + 1)])
    row = np.asarray(p_table[b], dtype=float)
    if row.shape != (b + 1,):
        raise ValueError(f"p_table[{b}] must have length {b + 1}")
    return row


def recover_k(a: float, ece_b: float, p_row: ArrayLike, *, tol: float, budget: int | None = None) -> KRecovery:
    """Recover ``k_b`` from ``ECE_b`` given ``a`` and the model's predictions ``p_row[k] = p(k, b)``, ``k = 0..b``."""
    p_row = np.asarray(p_row, dtype=float)
    r = np.abs(np.abs(p_row - a) - ece_b)
    order = np.argsort(r, kind="stable")
    k = int(order[0])
    second = float(r[order[1]]) if r.size > 1 else float("inf")
    return KRecovery(budget=len(p_row) - 1 if budget is None else budget, k=k, residual=float(r[k]),
                     runner_up=second, exact=bool(r[k] <= tol), unique=bool(second > tol), p=float(p_row[k]))


def is_nested(ks: Mapping[int, int]) -> bool:
    """A5: ``0 <= k_b' - k_b <= b' - b`` for consecutive budgets, and ``0 <= k_b <= b``."""
    bs = sorted(ks)
    if any(not 0 <= ks[b] <= b for b in bs):
        return False
    return all(0 <= ks[c] - ks[b] <= c - b for b, c in zip(bs, bs[1:]))


def recover_k_path(a: float, eces: Mapping[int, float], p_table: PTable, *, tol: float,
                   budgets: Sequence[int] = BUDGETS) -> KPath:
    """Recover ``k_b`` at every positive budget of ``budgets``.

    ``eces`` maps budget -> reported ECE. ``p_table`` is either a callable ``p(k, b)`` (the model's own prediction
    after ``k`` successes in ``b`` labels) or a mapping ``b -> array of length b + 1``.
    """
    steps = {b: recover_k(a, float(eces[b]), _row(p_table, b), tol=tol, budget=b) for b in budgets if b > 0}
    ks = {0: 0, **{b: s.k for b, s in steps.items()}}
    return KPath(ks=ks, steps=steps, nested=is_nested(ks), exact=all(s.exact and s.unique for s in steps.values()),
                 max_residual=max((s.residual for s in steps.values()), default=0.0))
