"""v1_1 acquisition hook (unchanged from v1; Phase-3 policies A2-A8 were tested and dropped, report 17): accept every candidate while budget remains.

Candidates arrive in the platform's fixed random order, so the first n accepted
items are a uniformly random, nested subset of the pair's acquisition pool --
distributionally identical to the default random policy, and exactly the policy
simulated in src/paiec/evaluation/episodes.py. Declaring only `context` (keyword-only)
avoids an extra predictor call and a deep copy of `labeled` per candidate.
Never raises; returns a native bool (0.0 priority for a legacy one-arg caller).
"""


def acquisition_function(input, *, context=None):
    if context is None:
        return 0.0
    try:
        return bool(int(context.get("labels_remaining", 0)) > 0)
    except Exception:
        return False
