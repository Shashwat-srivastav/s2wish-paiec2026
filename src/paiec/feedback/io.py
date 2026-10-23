"""Read saved per-pair formative feedback (the "Logs -> Scoring output" rows) into typed records.

Expected JSON: ``{"pairs": [{"subject_id", "benchmark_id", "n_unique_pairs" | "n_unique_subject_item_pairs",
"brier": {"0": ..., "31": ...}, "ece": {...}}, ...]}``. IDs are kept as opaque strings; nothing here (or anywhere in
this package) maps an anonymous ID to a real subject or benchmark.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .scoring import BUDGETS


@dataclass(frozen=True)
class PairFeedback:
    subject_id: str
    benchmark_id: str
    n: int | None
    brier: dict[int, float]
    ece: dict[int, float]


def load_feedback(path: str | Path) -> list[PairFeedback]:
    """Load the per-pair rows of one saved feedback file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for p in data["pairs"]:
        n = p.get("n_unique_pairs", p.get("n_unique_subject_item_pairs"))
        out.append(PairFeedback(subject_id=str(p["subject_id"]), benchmark_id=str(p["benchmark_id"]),
                                n=None if n is None else int(n),
                                brier={b: float(p["brier"][str(b)]) for b in BUDGETS},
                                ece={b: float(p["ece"][str(b)]) for b in BUDGETS}))
    return out


def benchmark_multiplicity(pairs: list[PairFeedback]) -> dict[str, int]:
    """How many pairs of the draw share each benchmark ID (1 = singleton pair, see A4 in :mod:`.krecovery`)."""
    out: dict[str, int] = {}
    for p in pairs:
        out[p.benchmark_id] = out.get(p.benchmark_id, 0) + 1
    return out
