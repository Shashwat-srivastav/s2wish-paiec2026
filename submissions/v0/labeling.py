"""v0 acquisition hook: representative (random-equivalent) sampling by default.

Policy is chosen by ``labeling_config.json`` -> {"policy": ...} (default
"accept_all"):

* ``accept_all`` — accept every candidate while budget remains. Candidates
  arrive in a fixed *random* order, so the first n accepted items form a
  uniformly random size-n subset of the pair's acquisition pool, with nested
  prefixes across caps 1/3/7/15/31. This is distributionally identical to the
  platform's default random policy (which accepts with probability
  labels_remaining / items_remaining, i.e. selection sampling = uniform random
  subset). Labels are therefore unbiased draws for the pair mean q, which is
  exactly what the Beta-Binomial pair-mean estimator in model.py assumes.

* ``uncertainty`` — accept iff |prediction - 0.5| <= band (default 0.15), with
  forced acceptance when items_remaining <= labels_remaining. NOTE: with the
  v0 predictor the prediction is constant across items within a pair, so this
  decision does not depend on the candidate item and still yields a random
  subset (it only changes *which* random items, and may delay acquisition).
  It becomes biased once the predictor is item-aware (see README).

The hook never raises; it only reads ``context`` (and ``prediction`` for the
alternative policy) and returns a native bool.
"""

from __future__ import annotations

import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_policy():
    cfg = {"policy": "accept_all", "band": 0.15}
    try:
        path = os.path.join(_HERE, "labeling_config.json")
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                cfg.update(data)
    except Exception:
        pass
    return cfg


CONFIG = _load_policy()


def acquisition_function(input: list, prediction: float | None = None,
                         labeled: list | None = None, context: dict | None = None):
    if context is None:
        # Legacy pool-ranking evaluator: equal priority -> ties broken by the
        # platform's fixed random order -> random sampling.
        return 0.0
    try:
        remaining = int(context.get("labels_remaining", 0))
        items_left = int(context.get("items_remaining", 0))
    except Exception:
        return False
    if remaining <= 0:
        return False
    if CONFIG.get("policy") == "uncertainty":
        if items_left <= remaining:
            return True
        try:
            band = float(CONFIG.get("band", 0.15))
            return bool(prediction is not None and abs(float(prediction) - 0.5) <= band)
        except Exception:
            return True
    return True
