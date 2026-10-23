"""Exact inversion of per-pair formative feedback (Brier and ECE per label budget).

Modules
-------
scoring     platform Brier / ECE / ALC and the constant-prediction identities
inversion   pair success rate ``a`` from (B0, ECE0, p0); identifiability and the reflection pitfall
krecovery   successes ``k_b`` among the first b revealed labels (singleton pairs)
replay      exact zero-upload counterfactual replay of pair-level rules
bounds      pair-constant floor, label-removable excess, ceiling on pair-level gains
io          loading saved feedback rows

Pure NumPy. Every function states its assumptions (A1-A5) in its module docstring. The package only reads the
submitter's own feedback and never maps an anonymous ID to a real subject or benchmark.
"""
from .bounds import PairLevelBounds, pair_constant_floor, pair_level_bounds
from .inversion import (NotIdentifiable, PairInversion, accuracy_candidates_from_ece, accuracy_from_brier,
                        accuracy_tolerance, constancy_residual, feedback_only_solutions, floor_from_feedback,
                        half_ulp, invert_pair, reflect)
from .io import PairFeedback, benchmark_multiplicity, load_feedback
from .krecovery import KPath, KRecovery, is_nested, recover_k, recover_k_path
from .replay import Replay, budget0_delta, replay, replay_curve
from .scoring import ALC_WEIGHTS, BUDGETS, alc, brier, brier_constant, ece, ece_constant

__all__ = [
    "ALC_WEIGHTS", "BUDGETS", "KPath", "KRecovery", "NotIdentifiable", "PairFeedback", "PairInversion",
    "PairLevelBounds", "Replay", "accuracy_candidates_from_ece", "accuracy_from_brier", "accuracy_tolerance", "alc",
    "benchmark_multiplicity", "brier", "brier_constant", "budget0_delta", "constancy_residual", "ece", "ece_constant",
    "feedback_only_solutions", "floor_from_feedback", "half_ulp", "invert_pair", "is_nested", "load_feedback",
    "pair_constant_floor", "pair_level_bounds", "recover_k", "recover_k_path", "reflect", "replay", "replay_curve",
]
