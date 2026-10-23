# paiec.feedback: exact inversion of per-pair formative feedback

`src/paiec/feedback/` is a small pure-NumPy package. It reads the per-pair formative feedback that the platform
shows for your **own** submission (Brier score and ECE at budgets 0, 1, 3, 7, 15 and 31, reported to 6 decimals) and
recovers three things:

1. **Pair accuracy** `a`, using `a = (B0 - p0^2) / (1 - 2 p0)` and cross-checking it with `ECE0 = |a - p0|`.
2. **Success counts** `k_b` among the first b revealed labels. This works on singleton pairs only.
3. **Exact replays and bounds.** It scores any pair-level rule exactly with no upload, and reports the floor
   `a(1-a)`, the label-removable excess per budget and the ceiling on gains.

The derivations and proofs are in `appendix_inversion.tex` (report Appendix "Exact inversion of per-pair formative
feedback"). The tests are in `tests/test_feedback_inversion.py`.

## Assumptions (checked, not assumed silently)

| | Assumption | How it is checked |
|---|---|---|
| A1 | The budget-0 prediction is constant on the pair. Get `p0` by calling **your archive's** `predict()` with no labels, never from the feedback. | Call `predict()` on several items and subjects. |
| A2 | The published scoring applies: binary outcomes, a fixed evaluation pool, and ECE over 10 equal-width bins weighted by count. | `PairInversion.consistent` |
| A3 | Values are rounded to 6 decimals (error at most 5e-7). The error of `a` is at most `5e-7 / abs(1 - 2 p0)`. | `PairInversion.tol` |
| A4 | For `k` only: the prediction at budget b depends only on the success count. This holds on singleton pairs with no item-level terms. | The residual must be within `PairInversion.k_tol`; also see `constancy_residual`. |
| A5 | For `k` only: label prefixes are nested and the order is fixed. | `KPath.nested`, plus identical paths across runs. |

**Pitfall.** `(a, p0)` and `(1 - a, 1 - p0)` give identical B0 and ECE0, so the ECE check cannot tell them apart.
If you use `1 - p0` by mistake, every accuracy is silently reflected. We made this mistake once. At `p0 = 0.5`,
`a` is not identifiable at all (`NotIdentifiable`). The floor `a(1-a) = B0 - ECE0^2` does not need `p0`.

## Usage

```python
from paiec.feedback import load_feedback, invert_pair, recover_k_path, replay, pair_level_bounds, BUDGETS

pairs = load_feedback("my_feedback.json")              # {"pairs": [{subject_id, benchmark_id, brier{}, ece{}}]}
p0 = my_model.predict([subject, item], [])             # from YOUR archive; check it is item-independent
inv = invert_pair(pairs[0].brier[0], pairs[0].ece[0], p0)
assert inv.consistent                                  # necessary, NOT sufficient (see pitfall)

p_of = lambda k, b: my_model.predict([subject, item], fake_labels(k, b))   # k successes in b labels
path = recover_k_path(inv.a, pairs[0].ece, p_of, tol=inv.k_tol)
if path.exact and path.nested:
    r = replay(inv.a, path.ks, my_new_rule, observed=pairs[0].brier)       # exact delta ALC, no upload
    print(r.delta_alc)

import numpy as np
bd = pair_level_bounds(np.array([inv.a]), np.array([[pairs[0].brier[b] for b in BUDGETS]]))
print(bd.floor, bd.ceiling_gain)
```

Run the tests with `python -m pytest -q tests/test_feedback_inversion.py`. The regression part skips itself if our
saved feedback and archives are not present.

## Scope and conduct

- Use the package only on your own submissions' feedback.
- Treat anonymous IDs as opaque labels and never try to identify the subject or benchmark behind them.
- All replays are in-sample descriptions of one draw, not predictions of the final score.
- Turning recovered levels into model parameters keyed by benchmark ID is a modelling choice. Clear it with the
  organisers and disclose it; we did both.
