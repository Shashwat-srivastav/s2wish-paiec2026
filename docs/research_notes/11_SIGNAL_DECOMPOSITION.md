# 11 — Signal decomposition (§7 Components A–E), Level 1/2 evidence

Personas: **MT** = Measurement Theorist, **BS** = Bayesian Statistician. Code: `src/paiec/analysis/decomp.py`
(about 5 min, numpy/scipy only). Raw numbers: `artifacts/decomp_results.json`. Priors: `artifacts/priors_v1.json`.

**Data scope.** Binary benchmarks: matharena (rows that are not 0/1 were dropped; there are 57),
multi_swebench, real_webagents, researchcodebench and swe_rebench. swe_rebench has 1 subject.
**mmdocrag is excluded** from fits and priors. Its responses are continuous quality scores, mostly
0.64–0.84, and only 5% are binary. Any threshold would set the benchmark base rate by hand, and the
benchmark base rate is exactly the quantity whose prior we are estimating. It appears only in a
sensitivity check on transfer, as the logit of each subject's mean score.
**Data issue found:** 52 of 104 matharena subjects have `normalized_name` = null and provider = null.
They are treated as unseen names, which is what the runtime will see (`""`).
All Brier numbers weight pairs equally. Within a pair, each trial counts once. Only pairs with at
least 80 distinct items are used.

**Fitting method.** Crossed-random-effects logistic models (binomial cells), fitted by MAP with
empirical-Bayes variance updates. The variance update uses a diagonal Laplace approximation.

## 1. Variance decomposition (logit scale)

| benchmark | Rasch μ | var θ (subject) | var β (item) | var cell (s×i) | subjects / items |
|---|---|---|---|---|---|
| matharena | 1.02 | 7.00 | 9.68 | **0.63** (identified, ~4.6 trials/cell) | 104 / 1751 |
| multi_swebench | −2.63 | 2.57 | 4.43 | not identified† | 82 / 2126 |
| real_webagents | −1.02 | 2.10 | 4.11 | not identified† | 33 / 233 |
| researchcodebench | −1.45 | 2.43 | 7.06 | not identified† | 31 / 212 |
| swe_rebench | −0.23 | — (1 subject) | 11.05 | 0.20 (10.6 trials/cell) | 1 / 6306 |

† With one trial per cell, an s×i effect cannot be told apart from Bernoulli noise. The EB estimate
stays near its 0.5 starting value. This is an identifiability point from **MT**.

- **Benchmark level.** The Rasch μ values (conditional, median-item scale) have sd 1.36 across the 5 points.
  On the marginal pair-mean scale (mean logit pair accuracy), the benchmark means are 0.48, −1.62, −0.94,
  −0.74 and −0.08: G = −0.58, sd_B = 0.81 (n = 5). With the 4 eligible benchmarks only: G = −0.85, sd_B = 0.63.
- **Subject level.** The within-benchmark sd of logit pair accuracy, after removing binomial noise, is τ = 1.13.
- **Trial noise.** In repeated cells, within-cell variance is 27% of total Bernoulli variance in
  matharena and 20% in swe_rebench. 72–78% of repeated cells are unanimous.
- **Subject_id vs normalized_name vs provider (joint model).** The EB name variance fell to about 0.003
  and the pair variance to 4.25, with provider at about 0.001. **This collapse is an artefact**: 113 of
  132 names appear in only one benchmark, so name and pair are almost confounded, and the diagonal
  Laplace approximation ignores their posterior correlation. Use the moment-based transfer estimates
  in §2 instead. subject_id is shared only among mmdocrag, real_webagents and researchcodebench, so
  subject_id adds nothing beyond name.

**Brier explained (5-fold CV over cells, within each benchmark; these are ceilings, not cold-start numbers):**

| benchmark | 0.5 | bench mean | + pair mean | item mean only | additive Rasch | noise floor |
|---|---|---|---|---|---|---|
| matharena | .250 | .238 | .187 | .141 | **.091** | .059 |
| multi_swebench | .250 | .177 | .145 | .104 | **.084** | (n/a) |
| real_webagents | .250 | .229 | .188 | .147 | **.112** | (.081, 9% cells) |
| researchcodebench | .250 | .229 | .207 | .140 | **.103** | (.064, 3% cells) |

**C1 (MT).**
- **CLAIM:** Item difficulty is the largest predictable component. On every benchmark the item-only
  predictor beats the pair-only predictor by 0.04–0.07 Brier. Subject + item is close to the noise floor.
- **MECHANISM:** var β is 1.7–2.9× var θ.
- **EVIDENCE:** the table above. In matharena, Rasch reaches 0.091 against a floor of 0.059, and the
  interaction variance (0.63) is about 4% of the latent variance.
- **COUNTEREVIDENCE:** At test, item difficulty for unseen benchmarks is available only through
  content or through other pairs' labels on the same item.
- **COMPETING EXPLANATIONS:** Part of the "item" signal may be the item's benchmark-specific
  difficulty for the frontier models that dominate each pool.
- **FALSIFICATION TEST:** Add a 2PL or rank-1 interaction. If CV Brier drops by more than 0.005 on 3
  or more benchmarks, additivity is falsified.
- **DECISION:** ADVANCE additive Rasch.
- **CONFIDENCE:** strongly supported (within-benchmark).

**C2 (MT veto).**
- **CLAIM:** Ability is not on a common logit scale across benchmarks.
- **EVIDENCE:** sd θ is 2.6 on matharena and 1.4–1.6 elsewhere.
- **DECISION:** Any cross-benchmark ability must carry a benchmark-specific discrimination λ_b, or be
  used only as a standardized deviation.
- **CONFIDENCE:** moderately supported.

## 2. Cross-benchmark ability transfer (names in ≥2 benchmarks: 19 names, 53 observations)

Pairwise Pearson correlations of θ (n is the number of shared names):

| benchmark pair | n | r |
|---|---|---|
| multi_swebench – researchcodebench | 9 | **.81** |
| multi_swebench – real_webagents | 11 | .78 |
| real_webagents – researchcodebench | 12 | .68 |
| matharena – researchcodebench | 8 | .64 |
| matharena – multi_swebench | 8 | .60 |
| matharena – real_webagents | 6 | .16 |
| mmdocrag – multi_swebench | 8 | .00 (sensitivity only) |
| mmdocrag – researchcodebench | 17 | .49 (sensitivity only) |

**Predictive (absolute) transfer is much weaker than the pairwise correlations:**

| measure | value |
|---|---|
| ICC of standardized θ | 0.22 (bootstrap 90% CI −0.07 to 0.45) |
| ICC of logit-accuracy deviation from the benchmark mean | 0.09 |
| LOO r (predict a name's deviation from its other benchmarks), standardized θ | 0.31 |
| LOO r, logit-accuracy deviation | 0.13 |
| Additive two-way model, with the benchmark offset taken relative to the shared names | pooled r = 0.35, optimal slope 0.73 |
| Same additive model, per held-out fold | r = 0.41, 0.51, 0.26, 0.56 |
| Provider share of z variance (within benchmark) | 0.38 (descriptive; confounded with name) |

**C3 (MT vs BS: explicit disagreement).**
- **CLAIM (MT):** Ability transfers. Ranks correlate 0.6–0.8 among the agentic/coding benchmarks, so
  same-subject labels on other benchmarks are informative.
- **CLAIM (BS):** For an absolute prediction the useful signal is small. The ICC 90% CI includes 0,
  LOO r is 0.13–0.31, and in §4 the name prior gains only 0.0015 Brier at B0.
- **MECHANISM of the gap:** The shared names are a selected, mostly frontier subset. Their deviation
  from each benchmark's population mean shifts with the population, while their rank order holds.
  Correlation ignores that shift; ICC and LOO do not.
- **FALSIFICATION TEST:** In episodes, estimate the name effect *jointly* with the pooled benchmark
  offset (the additive model, slope about 0.7). Compare it against the population-relative prior with
  ICC shrinkage (k = 10.4, data weight 0.09 for one benchmark). If the joint version does not beat P1
  by more than 0.002 ALC, drop name transfer.
- **DECISION:** MODIFY. Keep a shrunk name effect inside the joint model, not a stand-alone prior.
- **CONFIDENCE:** weak (19 names).

## 3. Empirical-Bayes priors and shrinkage (`priors_v1.json`)

**(a) Benchmark mean logit (pair-mean scale).**
- N(−0.58, 0.81²) from 5 points, or N(−0.85, 0.63²) from the 4 eligible benchmarks.
- **BS:** With 4–5 points, the sd has a 90% CI of roughly ×0.6 to ×2.9. Treat it as a lower bound and
  use Student-t(3).

**(b) Subject deviation within a benchmark.**
- τ = 1.13 on the logit pair-mean scale.
- Rasch θ sd: 1.45 to 1.60 on the non-math benchmarks, 2.65 on matharena.

**(c) Item difficulty.**
- σ_β = 2.51 pooled.
- By benchmark: 2.03–3.32.
- **MT:** Probit-approximation scaling means a conditional intercept equals the marginal intercept
  × √(1 + πσ²/8), which is about ×1.6. Confusing the two scales caused a P3 failure (see below).

**Analytic Bayes-optimal shrinkage for the pair-mean predictor.**
- The estimator is p̂ = (κ·m + k)/(κ + n), with κ moment-matched from the hierarchy.
- When the benchmark is unknown, κ ≈ 2.8 (prior worth about 3 labels).
- When the benchmark offset is known or pooled, κ ≈ 4.0.

| budget n | 1 | 3 | 7 | 15 | 31 |
|---|---|---|---|---|---|
| data weight, unknown benchmark | .26 | .51 | .72 | .84 | .92 |
| data weight, known benchmark | .20 | .43 | .64 | .79 | .89 |

**Gain from cross-pair pooling.**
Predictors:
- **P0:** each pair is updated on its own (no pooling).
- **P1:** the benchmark offset is pooled across pairs, by exact grid and quadrature over a logit-normal hierarchy.
- **P2:** P1 plus the name prior.
- **P3:** full Rasch fitted on all pairs' labels, so item difficulty is pooled too, with posterior-predictive integration.
- **PX:** oracle benchmark mean.

| simulated from the fitted hierarchy (ALC) | P0 | P1 | PX | P3 |
|---|---|---|---|---|
| S = 5 subjects per benchmark | .2393 | .2378 | .2345 | .2314 |
| S = 20 | .2394 | .2365 | .2333 | .2176 |
| S = 80 | .2345 | .2305 | .2290 | .1927 |

**Level 2: real-data episodes.** Hyperparameters are leave-one-benchmark-out (LOBO), with a random
50/50 split per pair, nested random acquisition, and 6–12 repetitions.

| held-out benchmark (ALC) | P0 | P1 | P2 | P3 | PX |
|---|---|---|---|---|---|
| multi_swebench (82 pairs) | .1733 | .1642 | .1637 | .1613 | .1591 |
| real_webagents (26 pairs) | .1991 | .1995 | .1993 | .1921 | .1982 |
| researchcodebench (31 pairs) | .2263 | .2251 | .2250 | **.1929** | .2229 |
| swe_rebench (1 pair) | .2706 | .2706 | .2706 | .2615 | .2597 |
| matharena (81 pairs) | .2412 | .2297 | .2297 | **.2085** | .2225 |
| eligible-4, pair-weighted | .1905 | .1850 | .1847 | **.1747** | .1812 |

Learning curves for multi_swebench at n = 0, 1, 3, 7, 15, 31:
- P0: .220, .194, .175, .161, .153, .149
- P1: .220, .167, .162, .156, .152, .148
- P3: .218, .175, .166, .155, .140, .123

**C4 (BS).**
- **CLAIM:** Pooling the benchmark offset across pairs is worth −0.005 to −0.011 ALC. Almost all of it
  arrives at n = 1–3, where it closes more than 85% of the gap to the oracle offset.
- **MECHANISM:** 82 pairs × 1 label is enough to pin μ_b.
- **EVIDENCE:** multi_swebench B1 drops from .194 to .167 (PX is .167). matharena B1 drops from .271 to .234.
- **COUNTEREVIDENCE:** No gain when the benchmark mean is close to G (real_webagents,
  researchcodebench), or when there is only 1 subject.
- **DECISION:** ADVANCE.
- **CONFIDENCE:** strongly supported.

**C5 (MT favors, BS skeptical).**
- **CLAIM:** Pooling item difficulty across pairs (P3) is the single largest lever at n ≥ 7, worth
  −0.01 to −0.03 ALC.
- **MECHANISM:** var β exceeds var θ, and other pairs' acquisition labels land on this pair's evaluation items.
- **CRUX:** This holds only if the official per-pair splits are independent, *and* if `labeled`
  carries other pairs' item contents or identities that can be matched to this item.
- **COUNTEREVIDENCE:** P3 is worse than P1 at n = 1 on multi_swebench and real_webagents (.175 vs .167).
  The diagonal-Laplace predictive variance is too wide. A first version with no predictive
  integration and a near-flat μ prior was catastrophic on swe_rebench (B1 = .49), but slightly better at
  n = 31 (multi_swebench .117 vs .123). So the MT and BS versions trade off across budgets.
- **FALSIFICATION TEST:** In the official streaming simulator, measure the fraction of evaluation
  items that have at least one cross-pair label at n = 31. If it is near 0, P3 falls back to P1.
- **DECISION:** REPLICATE on the official harness.
- **CONFIDENCE:** moderately supported, and conditional on the crux.

## 4. LOBO budget-0 Brier (non-adaptive; pairs with ≥80 items; equal pair weight within benchmark)

| held-out benchmark | 0.5 | global pair-mean | global bench-mean | hier. marginal | name prior | provider prior | name→provider | oracle bench mean |
|---|---|---|---|---|---|---|---|---|
| matharena (81 pairs; 15% of names seen) | **.2500** | .3493 | .3053 | .3064 | .3056 | .3074 | .3066 | .2387 |
| multi_swebench (82; 73%) | .2500 | .2495 | .2222 | .2198 | .2177 | .2185 | **.2164** | .1717 |
| real_webagents (26; 58%) | .2500 | .2284 | .2300 | .2295 | .2287 | .2281 | **.2280** | .2245 |
| researchcodebench (31; 45%) | .2500 | .2309 | .2318 | .2312 | .2304 | **.2291** | .2293 | .2284 |
| swe_rebench (1; 0%) | .2500 | .2567 | .2595 | .2605 | .2605 | .2688 | .2688 | .2496 |
| **eligible-4 mean over benchmarks** | .2500 | .2414 | .2359 | .2353 | **.2343** | .2361 | .2356 | .2186 |
| **all-5 mean over benchmarks** | **.2500** | .2630 | .2498 | .2495 | .2486 | .2504 | .2498 | .2226 |

**C6 (both personas).**
- **CLAIM:** B0 is dominated by not knowing the benchmark offset. The oracle offset is worth 0.017 to
  0.027. Subject identity is worth 0.001–0.002 (name) and −0.002 to +0.001 (provider).
- **CLAIM:** Weighting the global mean by benchmark beats weighting it by pair. The pair-weighted mean is
  dominated by high-accuracy matharena pairs (on matharena it scores .349 vs .305).
- **CLAIM:** On a high-accuracy outlier benchmark (matharena), every learned prior loses to 0.5.
- **FALSIFICATION TEST:** Score a mixture predictor, p = w·0.5 + (1−w)·p_prior, under LOBO. If w* > 0.3,
  the benchmark prior is mis-specified, not just too wide.
- **DECISION:** MODIFY. Use a heavy-tailed prior on μ_b.
- **CONFIDENCE:** moderately supported (only 5 folds).

## Persona hypotheses and kill tests

**MT.**
- **Hypothesis H-MT:** logit p = μ_b + λ_b·a_name + θ_pair + β_i is sufficient, and s×i interaction is
  negligible (var 0.63 against 17 of latent variance).
- **Kill test:** low-rank interaction CV, plus the matharena noise floor (.059) against Rasch (.091).
  Is the remaining 0.03 interaction or estimation error?

**BS.**
- **Hypothesis H-BS:** Optimal ALC is reached with a pseudo-count κ of about 3–4, plus an exactly
  integrated, pooled benchmark offset, plus a heavy-tailed μ_b. Anything beyond that (the name prior,
  item pooling) yields less than 0.003 ALC unless split semantics allow item overlap.
- **Kill test:** the P3-vs-P1 overlap audit on the official simulator.

## Recommended model structure

Hierarchical logistic model (pseudo-code):

```
logit p = μ_b + a_name·λ + θ_pair + β_i
  μ_b      ~ t3(G, s_B)                   # pooled across all pairs' labels on the benchmark
  θ_pair   ~ N(0, τ²), τ ≈ 1.1–1.3 (marginal scale)
  a_name   strongly shrunk                # slope ≤ 0.7 jointly; ≤ 0.1 population-relative
  β_i      ~ N(0, σ²), σ ≈ 2.5            # pooled only if cross-pair item labels exist; content prior is future work
```

Predictions are posterior-predictive: integrate over the posterior instead of plugging in point
estimates. Exact grid or quadrature integration is enough for the pair and benchmark levels.
