"""Rebuild submissions/v1_1/v1_1_params.json from the engine (src/paiec/models/v1_1.py).

Shipped config = ``v1_1.BASE2``: prior ('t', 3, 'half', 'learned') + momd merge guard
(rho0 = BASE2['rho'], k0 = mom_rho default 2.0). Fit on ALL public benchmarks:

    H            = ladder.fit_hyper(E.BENCHES)
    prior_center = G / 2                                       ('half')
    prior_scale  = max(0.5, rms(per-benchmark training means - prior_center))  ('learned')

Usage: python -m paiec.build.v1_1_params [--write]
"""
from __future__ import annotations

import argparse
import inspect
import json
import math

import numpy as np

from paiec import paths
from paiec.evaluation import episodes as E
from paiec.models import ladder as Lad
from paiec.models import v1_1 as V11

PARAMS_PATH = paths.SUBMISSIONS / "v1_1" / "v1_1_params.json"
SOURCE = "src/paiec/models/v1_1.py (BASE2: prior t3 half/learned + momd merge guard)"


def build_params(benches=None) -> dict:
    benches = list(E.BENCHES if benches is None else benches)
    cfg = V11.BASE2
    assert cfg["prior"] == ("t", 3, "half", "learned") and cfg["rho_pool"] == "momd"
    H = Lad.fit_hyper(benches)
    means = np.array([Lad.bench_stats(b)["mean"] for b in benches])
    G = float(H["G"])
    center = G / 2
    scale = float(max(0.5, math.sqrt(np.mean((means - center) ** 2))))
    k0 = float(inspect.signature(V11.mom_rho).parameters["k0"].default)
    return {"G": G, "prior_center": center, "prior_scale": scale, "tau": float(H["tau"]),
            "sigma_beta": float(H["sigma_beta"]), "rho0": float(cfg["rho"]), "rho_k0": k0,
            "clip": [0.005, 0.995], "_fit_on": benches, "_source": SOURCE}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help=f"overwrite {PARAMS_PATH}")
    a = ap.parse_args()
    prm = build_params()
    if a.write:
        with open(PARAMS_PATH, "w", encoding="utf-8") as fh:
            json.dump(prm, fh, indent=1)
    print(json.dumps(prm, indent=1))


if __name__ == "__main__":
    main()
