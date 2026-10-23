"""Rebuild submissions/v1_2/v1_2_params.json from the engines.

Shipped config = ``v1_2.cfg_of(facets="all", guard=1)`` = v1_1 ``BASE2`` (all v1_1 fields,
from paiec.build.v1_1_params) plus the E4 constants of src/paiec/models/v1_2.py:

    facet_pis     = v1_2.FACET_PIS
    facet_prior_w = exp(v1_2.FACET_PW)         (0.4 / 0.4 / 0.2)
    guard         = 1,  guard_rho = v1_2.GUARD_RHO

Usage: python -m paiec.build.v1_2_params [--write]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from paiec import paths
from paiec.build.v1_1_params import build_params as build_v1_1
from paiec.models import v1_2 as V12

PARAMS_PATH = paths.SUBMISSIONS / "v1_2" / "v1_2_params.json"
SOURCE = ("src/paiec/models/v1_2.py cfg_of(facets='all', guard=1) "
          "(v1_1 BASE2 + all-field facets + G1 key guard)")


def build_params(benches=None) -> dict:
    cfg = V12.cfg_of(facets="all", guard=1)
    assert cfg["facets"] == "all" and cfg["guard"] == 1 and not cfg["shuffle"]
    prm = build_v1_1(benches)
    fit_on = prm.pop("_fit_on")
    prm.pop("_source")
    prm.update({
        "facets": True,
        "facet_pis": [float(x) for x in V12.FACET_PIS],
        "facet_prior_w": [round(float(x), 12) for x in np.exp(V12.FACET_PW)],
        "guard": 1,
        "guard_rho": float(V12.GUARD_RHO),
        "_fit_on": fit_on,
        "_source": SOURCE,
    })
    return prm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help=f"overwrite {PARAMS_PATH}")
    a = ap.parse_args()
    prm = build_params()
    if a.write:
        with open(PARAMS_PATH, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(prm, fh, indent=1)
    print(json.dumps(prm, indent=1))


if __name__ == "__main__":
    main()
