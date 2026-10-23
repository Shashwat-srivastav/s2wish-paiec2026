"""Measure v1_1 engine cost for the preregistered factorial resource pilot.

This pilot intentionally records no prediction-quality metrics.
"""
from __future__ import annotations

import copy
import ctypes
import json
import platform
import time
from pathlib import Path

from paiec import paths
from paiec.evaluation import episodes as E
from paiec.models import v1_1 as V


OUTPUT = paths.RUNS / "preregistration" / "resource_pilot_2026-10-14.json"
REGIMES = (("R1", (0,)), ("R2", (0,)), ("R3", (0,)))
POLICIES = ("A1", "A2", "A3", "A5", "A8")


def _configs() -> dict[str, dict]:
    base = dict(V.BASE2)
    configs = {
        "BASE2": base,
        "w_od=0.7": dict(base, w_od=0.7),
        "w_od=0.85": dict(base, w_od=0.85),
        "w_od=1.2": dict(base, w_od=1.2),
        "tau_scale=0.8": dict(base, tau_scale=0.8),
        "tau_scale=1.2": dict(base, tau_scale=1.2),
    }
    return dict(sorted(configs.items()))


def _peak_working_set_mb() -> float | None:
    if platform.system() != "Windows":
        return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    process = kernel32.GetCurrentProcess()
    get_info = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
    get_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessMemoryCounters), ctypes.c_ulong]
    get_info.restype = ctypes.c_int
    if not get_info(process, ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.PeakWorkingSetSize / (1024 * 1024)


def run() -> dict:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    configs = _configs()
    cells = []
    setup_seconds = 0.0
    total_start = time.perf_counter()

    for regime, seeds in REGIMES:
        for benchmark in E.EVAL_FOLDS:
            load_start = time.perf_counter()
            data = E.compact(benchmark)
            hyper = V.H_fold(benchmark)
            episode = E.build(data, regime, seeds)
            setup_seconds += time.perf_counter() - load_start

            for config_name, config in configs.items():
                for policy in POLICIES:
                    cell_start = time.perf_counter()
                    policy_episode = copy.copy(episode)
                    V.simulate_policy(policy_episode, hyper, config, policy)
                    predictions = V.run_cfgs(policy_episode, hyper, {config_name: config})
                    del predictions
                    cells.append({
                        "regime": regime,
                        "benchmark": benchmark,
                        "config": config_name,
                        "policy": policy,
                        "seconds": time.perf_counter() - cell_start,
                    })

    total_seconds = time.perf_counter() - total_start
    report = {
        "experiment_id": "EXP-0048",
        "preregistered": True,
        "quality_metrics_recorded": False,
        "engine": "paiec.models.v1_1 exact-grid x Gauss-Hermite",
        "configuration_count": len(configs),
        "regimes": {name: list(seeds) for name, seeds in REGIMES},
        "folds": list(E.EVAL_FOLDS),
        "policies": list(POLICIES),
        "budgets": list(E.BUDGETS),
        "setup_seconds": setup_seconds,
        "cell_seconds": {
            "count": len(cells),
            "sum": sum(cell["seconds"] for cell in cells),
            "median": sorted(cell["seconds"] for cell in cells)[len(cells) // 2],
            "p95": sorted(cell["seconds"] for cell in cells)[int(len(cells) * 0.95) - 1],
            "max": max(cell["seconds"] for cell in cells),
        },
        "total_seconds": total_seconds,
        "peak_working_set_mb": _peak_working_set_mb(),
        "cells": cells,
        "limitations": [
            "Timing only; prediction-quality metrics are discarded.",
            "Does not measure Gumbel, horseshoe, spike-and-slab, t3 pair effects, Laplace, or MCMC.",
            "One seed per regime estimates throughput, not score uncertainty.",
        ],
    }
    for _ in range(3):
        OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        report["output_bytes"] = OUTPUT.stat().st_size
    print(f"EXP-0048 metrics-only report: {OUTPUT}")
    print(f"cells={len(cells)} seconds={total_seconds:.2f} output_bytes={report['output_bytes']}")
    return report


if __name__ == "__main__":
    run()
