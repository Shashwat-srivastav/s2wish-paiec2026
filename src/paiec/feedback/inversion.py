"""Exact recovery of a pair's success rate from budget-0 formative feedback.

Assumptions (each is checkable and is checked by :func:`invert_pair`):

A1. Constant prediction. At budget 0 the submitted model predicts one value ``p0`` for every scored row of the
    pair. This holds for any model whose budget-0 prediction does not depend on the item (e.g. one global
    constant, or one constant per anonymous benchmark ID). ``p0`` is obtained by calling the model's own
    ``predict()`` with no labels; it is NOT read from the feedback.
A2. Exact scoring rule. The reported Brier and ECE are the published rules (see :mod:`.scoring`) applied to
    binary outcomes of a fixed evaluation pool.
A3. Reported precision. Feedback values are rounded to ``decimals`` places (6 on the PAIEC logs), so each carries
    an absolute error of at most ``half_ulp = 0.5 * 10**-decimals``.

Identity. Under A1-A2, ``B0 = a (1 - 2 p0) + p0^2``, hence for ``p0 != 1/2``

    a = (B0 - p0^2) / (1 - 2 p0),                                                       (1)

and independently ``ECE0 = |a - p0|`` gives ``a in {p0 - ECE0, p0 + ECE0}``.  Under A3 the error of (1) is at most
``half_ulp / |1 - 2 p0|``, and the disagreement between (1) and the nearer ECE root is at most
``half_ulp / |1 - 2 p0| + half_ulp``.

Identifiability and the sign pitfall. Without ``p0`` the feedback (B0, ECE0) identifies only the unordered pair
{a, 1 - a}: the four points ``(a, a +/- ECE0)`` and ``(1 - a, 1 - a +/- ECE0)`` all reproduce B0 and ECE0
(:func:`feedback_only_solutions`). In particular the reflection ``(a, p0) -> (1 - a, 1 - p0)`` leaves B0 and ECE0
unchanged, so substituting ``1 - p0`` for ``p0`` in (1) returns ``1 - a`` and still passes the ECE0 cross-check.
The orientation must therefore come from the archive's own ``p0``. At ``p0 = 1/2`` the Brier score equals 1/4 for
every ``a`` and only ``|a - 1/2|`` is identified. The pair-constant floor ``a (1 - a) = B0 - ECE0^2`` is invariant
under the reflection and is identified from the feedback alone.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_DECIMALS = 6


class NotIdentifiable(ValueError):
    """Raised when ``p0`` is (numerically) 1/2, where B0 carries no information about ``a``."""


def half_ulp(decimals: int = DEFAULT_DECIMALS) -> float:
    """Maximum absolute rounding error of a value reported to ``decimals`` places."""
    return 0.5 * 10.0 ** (-decimals)


def accuracy_from_brier(b0: float, p0: float, *, min_gap: float = 1e-9) -> float:
    """Identity (1): ``a = (B0 - p0^2) / (1 - 2 p0)``. Raises :class:`NotIdentifiable` if ``|1 - 2 p0| < min_gap``.

    The result is not clipped; values just outside [0, 1] indicate rounding (A3) and are clipped by
    :func:`invert_pair`.
    """
    g = 1.0 - 2.0 * p0
    if abs(g) < min_gap:
        raise NotIdentifiable(f"p0={p0!r} is 1/2: Brier = 1/4 for every accuracy")
    return (b0 - p0 * p0) / g


def accuracy_tolerance(p0: float, decimals: int = DEFAULT_DECIMALS) -> float:
    """Worst-case error of identity (1) from rounding B0 (A3): ``half_ulp / |1 - 2 p0|``."""
    return half_ulp(decimals) / abs(1.0 - 2.0 * p0)


def accuracy_candidates_from_ece(e0: float, p0: float) -> tuple[float, float]:
    """The two roots of ``ECE0 = |a - p0|``: ``(p0 - ECE0, p0 + ECE0)``."""
    return p0 - e0, p0 + e0


def floor_from_feedback(b0: float, e0: float) -> float:
    """Pair-constant floor ``a (1 - a) = B0 - ECE0^2`` from the feedback alone (no ``p0`` needed; A1-A2)."""
    return b0 - e0 * e0


def feedback_only_solutions(b0: float, e0: float) -> list[tuple[float, float]]:
    """All ``(a, p0)`` in [0, 1]^2 consistent with (B0, ECE0) when ``p0`` is unknown.

    Solves ``a (1 - a) = B0 - ECE0^2`` (two roots ``a``, ``1 - a``) and ``p0 = a +/- ECE0``. Returns up to four points,
    closed under the reflection ``(a, p0) -> (1 - a, 1 - p0)``.
    """
    f = floor_from_feedback(b0, e0)
    disc = max(1.0 - 4.0 * f, 0.0)
    roots = sorted({(1.0 - np.sqrt(disc)) / 2.0, (1.0 + np.sqrt(disc)) / 2.0})
    out: list[tuple[float, float]] = []
    for a in roots:
        for p in (a - e0, a + e0):
            if -1e-12 <= p <= 1.0 + 1e-12 and (a, p) not in out:
                out.append((float(a), float(p)))
    return out


def reflect(a: float, p0: float) -> tuple[float, float]:
    """The symmetry of the feedback: ``(a, p0) -> (1 - a, 1 - p0)`` leaves B0 and ECE0 unchanged."""
    return 1.0 - a, 1.0 - p0


@dataclass(frozen=True)
class PairInversion:
    """Result of :func:`invert_pair`.

    a           success rate from identity (1), clipped to [0, 1]
    a_raw       unclipped identity (1)
    a_ece       the ECE0 root nearer to ``a_raw``
    disagreement  ``|a_raw - a_ece|``
    tol         A3 bound on ``disagreement``: ``half_ulp / |1 - 2 p0| + half_ulp``
    consistent  ``disagreement <= tol`` (a necessary, not sufficient, check: it cannot detect a reflected p0)
    floor       ``a (1 - a)``
    """

    a: float
    a_raw: float
    a_ece: float
    p0: float
    disagreement: float
    tol: float
    consistent: bool
    floor: float

    @property
    def k_tol(self) -> float:
        """Residual tolerance for k-recovery (see :mod:`.krecovery`).

        The k-residual ``||p_k - a| - ECE_b|`` has rounding error at most ``tol`` (error of ``a`` plus half an ULP
        of ``ECE_b``); a safety factor of 2 is applied.
        """
        return 2.0 * self.tol


def invert_pair(b0: float, e0: float, p0: float, *, decimals: int = DEFAULT_DECIMALS) -> PairInversion:
    """Recover a pair's success rate from reported (B0, ECE0) and the archive's budget-0 prediction ``p0``."""
    a_raw = accuracy_from_brier(b0, p0)
    lo, hi = accuracy_candidates_from_ece(e0, p0)
    a_ece = lo if abs(lo - a_raw) <= abs(hi - a_raw) else hi
    tol = accuracy_tolerance(p0, decimals) + half_ulp(decimals)
    dis = abs(a_raw - a_ece)
    a = float(min(max(a_raw, 0.0), 1.0))
    return PairInversion(a=a, a_raw=float(a_raw), a_ece=float(a_ece), p0=float(p0), disagreement=float(dis),
                         tol=float(tol), consistent=bool(dis <= tol), floor=a * (1.0 - a))


def constancy_residual(brier_b: float, ece_b: float, a: float) -> float:
    """``|B_b - a (1 - a) - ECE_b^2|``: zero (to rounding) when the prediction is constant on the pair at budget b.

    A residual above the rounding tolerance proves the prediction varied within the pair at budget b (A1 fails);
    a zero residual is consistent with, but does not prove, constancy.
    """
    return abs(brier_b - a * (1.0 - a) - ece_b * ece_b)
