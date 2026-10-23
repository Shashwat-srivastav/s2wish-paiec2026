# v0: Beta-Binomial shrinkage + random-equivalent acquisition

Zip contents (root): `model.py`, `labeling.py`, `prior.json`, `labeling_config.json`.
`analysis.py` and this README are dev-only (not zipped). Built zip: `../v0.zip`.

## Predictor (`model.py`)

For target `[subject, item]`, collect acquired labels with **identical full
subject dict** (all 8 fields) and identical `benchmark_id`; with `S` successes in `n` labels:

    p = clip((k*m + S) / (k + n), 0.02, 0.98)

* `(m, k)` comes from `prior.json`. Lookup order (most specific wins):
  `by_benchmark[benchmark_id]` > `by_subject_name[normalized_name]` >
  `by_subject_field[field][value]` for `field` in `field_order`
  (harness, reasoning_effort, provider) > `default`. These tables are empty
  until training data exists, so for now every pair uses the default.
* **Default `m=0.5, k=2`** is the uniform Beta(1,1) prior (Laplace's rule of
  succession). With no data, pair accuracies are assumed roughly uniform on
  [0,1], and the Bayes-optimal strength for that is `k = a+b = 2`. The MC
  results below show ALC is flat for k in [2,4] and that k=2 is the robust
  choice when accuracies are bimodal (near 0 or 1).
* `clip` is configurable. 0.02/0.98 is never binding for the Beta-Binomial with
  k=2 at n≤31 (the extremes are 1/33 and 32/33). It guards the IRT path.
* **Optional IRT** (`irt.enabled`): `P = sigmoid(theta - b_i)` with item
  difficulties from `irt.difficulties_file` (JSON `{item_key: b}`, where
  `item_key = sha1(benchmark_id + "\x1f" + item_content)[:16]`). Theta is the
  MAP under a `N(logit m, prior_sd^2)` prior (Newton). The prediction uses the
  Laplace posterior and the probit approximation
  `sigmoid((theta - b)/sqrt(1 + pi*V/8))`. Observed items with no known
  difficulty use `default_difficulty`. If the file or the target's difficulty
  is missing, the predictor falls back to Beta-Binomial.
* Robustness: `predict` never raises. Malformed inputs, labels, or config fall
  back to the prior. Labels that are not 0/1 are ignored. The output is always a
  finite float. No state carries across calls; only read-only config is cached
  at import.

## Acquisition (`labeling.py`)

Default `policy: "accept_all"` accepts every candidate while `labels_remaining > 0`.

**Equivalence to the platform default.** In `streaming_ingestion.py`, the
no-hook mode accepts candidate `i` with probability
`labels_remaining / items_remaining`, seeded by a hash. That is selection
sampling, which draws a uniformly random subset of the acquisition pool.
Candidates also arrive "in a fixed random order", so accepting the first n
gives a uniformly random size-n subset as well, with nested prefixes. The two
policies have the same distribution. Only the realised subsets differ.
Accept-all costs one cheap `predict` call per candidate (the signature accepts
`prediction`), which is negligible. Pools have at least 40 acquisition items,
so all 31 labels are always filled. Under the legacy evaluator (no `context`)
the hook returns a constant priority, and ties fall back to the platform's
random order.

Alternative `policy: "uncertainty"` accepts when `|prediction-0.5| <= band`
(forced near the end of the stream).

**Why random is the right policy here.** The Beta-Binomial estimator targets
the pair mean `q`. It is unbiased only when the acquired items are a
representative sample of the pair's items. Uncertainty sampling with an
*item-aware* acquisition model (e.g. BLE) favours items whose difficulty sits
near the subject's 50% point. For a strong subject those are its hard items,
and for a weak subject its easy ones. The acquired mean is then pulled toward
0.5, a bias that does not shrink with n. With v0's item-constant predictor,
the "uncertainty" policy happens to be harmless, because the decision does not
depend on which item is offered. It only changes which random items are taken.
Uncertainty or active sampling only pays off once the predictor models
item-level variation (IRT/embeddings) and re-weights to correct the selection.

## Expected Brier: empirical mean vs shrinkage

A predictor `p̂` that is constant within a pair is evaluated on items whose mean
success probability is `q`. Its expected Brier is
`E[q(1-q)] + E[(p̂-q)^2]`, which is irreducible noise plus squared estimation
error. Item heterogeneity only enters through `q`. Under random sampling each
label is marginally Bernoulli(q). Sampling without replacement from the finite
pool slightly reduces the variance, which is ignored here.

With `q ~ Beta(a,b)`, `μ = a/(a+b)`, `σ² = Var q`, `E q(1-q) = μ(1-μ) - σ²`:

* Empirical mean, n≥1: `E q(1-q) · (1 + 1/n)`. At n=1 it predicts exactly 0 or 1,
  doubling the noise term (e.g. 0.333 vs 0.167 for uniform q).
* Beta-Binomial: `E q(1-q) + [k² E(m-q)² + n E q(1-q)] / (k+n)²`. The bias
  term decays as `k²/(k+n)²` and the variance term is `n/(k+n)²` < `1/n`.
  Excess risk is minimised at `k* = E q(1-q) / E(m-q)²`, which equals `a+b`
  when `m=μ`. The gap to the empirical mean is
  `E q(1-q)/n - [k²E(m-q)² + nE q(1-q)]/(k+n)²`. That gap is O(1/n), so it
  is largest at n=1,3, exactly the budgets that carry 0.2 weight each, and it
  vanishes by n=31.

Results come from `python analysis.py` (20k pairs per scenario; 80 items, 40/40 split;
item logits `logit(q0) + 1.5·z`; ALC weights .1/.2/.2/.2/.2/.1).

| q0 dist (MC)        | predictor   | B0    | B1    | B3    | B7    | B15   | B31   | **ALC** |
|---------------------|-------------|-------|-------|-------|-------|-------|-------|---------|
| Beta(1,1)           | emp. mean   | .2500 | .3822 | .2552 | .2189 | .2045 | .1979 | **.2570** |
| Beta(1,1)           | BB k=2      | .2500 | .2385 | .2237 | .2110 | .2025 | .1974 | **.2199** |
| Beta(1,1)           | BB k=4      | .2500 | .2364 | .2222 | .2104 | .2023 | .1974 | **.2190** |
| Beta(.5,.5)         | emp. mean   | .2500 | .3013 | .2008 | .1726 | .1606 | .1556 | **.2076** |
| Beta(.5,.5)         | BB k=2      | .2500 | .2115 | .1846 | .1688 | .1597 | .1553 | **.1855** |
| Beta(2,2)           | emp. mean   | .2500 | .4379 | .2906 | .2496 | .2329 | .2252 | **.2897** |
| Beta(2,2)           | BB k=2      | .2500 | .2571 | .2497 | .2389 | .2301 | .2245 | **.2426** |
| Beta(4,2)           | emp. mean   | .2500 | .4227 | .2822 | .2418 | .2257 | .2184 | **.2813** |
| Beta(4,2)           | BB k=2      | .2500 | .2520 | .2431 | .2317 | .2231 | .2178 | **.2367** |
| Beta(4,2)           | BB oracle m=.67,k=6 | .2338 | .2323 | .2293 | .2252 | .2206 | .2170 | **.2266** |
| Beta(1,1), item-aware uncertainty sampling | emp. mean | .2500 | .4355 | .2780 | .2297 | .2066 | .1978 | **.2747** |
| Beta(1,1), item-aware uncertainty sampling | BB k=2    | .2500 | .2563 | .2432 | .2234 | .2056 | .1975 | **.2304** |

Closed-form values for q ~ Beta(1,1) itself (no item layer), m=.5:
emp. mean ALC .2270 (B1 .3333) vs BB k=2 ALC .1989 (B1 .2222).
Item heterogeneity at τ=1.5 pulls the effective `q` toward 0.5. That raises the
noise floor and makes slightly larger `k` (about 4) marginally better.

Takeaways:
* Shrinkage cuts ALC by 0.02–0.05 versus the official empirical-mean baseline.
  Almost all of the gain comes from B1 and B3.
* Most of the remaining headroom is in B0/B1, which need a better prior `m`
  (oracle m: -0.007 ALC). Per-benchmark, per-subject or feature priors from
  training data will close it.
* Item-aware uncertainty sampling costs about 0.01 ALC through bias.

## Validation

* `python external/starting_kit/tools/check_submission_zip.py submissions/v0.zip` → OK
* `python external/starting_kit/tools/run_smoke_test.py submissions/v0` → OK (2 predictions)
* `--concurrency 4` cannot run on Windows: the kit's `prediction_workers.py`
  uses `pass_fds`, which Windows does not support. This is a tooling
  limitation, not a problem with the submission.
