"""Test-collection rules for checkouts that lack the organisers' starting kit.

The evaluator replica imports the official streaming client from ``external/starting_kit/tools``. The starting kit
belongs to the competition organisers and is not redistributed in this repository. Download it from
https://www.codabench.org/competitions/17828/ (Participate -> Files) and unpack it to ``external/starting_kit/``.
Without it, the modules below are skipped with a clear reason instead of failing at import time; every other test
still runs.
"""
from __future__ import annotations

from pathlib import Path

KIT_CLIENT = Path(__file__).resolve().parents[1] / "external" / "starting_kit" / "tools" / "streaming_ingestion.py"
NEEDS_KIT = ["test_simulator.py", "test_submission_equivalence.py", "test_exp_0049_variants.py",
             "test_correlated_hierarchical.py"]

collect_ignore = [] if KIT_CLIENT.exists() else NEEDS_KIT


def pytest_report_header(config):
    if not KIT_CLIENT.exists():
        return ("starting kit not found (external/starting_kit): skipped " + ", ".join(NEEDS_KIT) +
                "; download it from https://www.codabench.org/competitions/17828/")
    return None
