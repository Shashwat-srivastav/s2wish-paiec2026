# Review round 2 (2026-10-22)

The code is `src/paiec/analysis/review_round2.py`; run it with `python -m paiec.analysis.review_round2` (it takes
seconds). It reuses the replay helpers from `review_round1.py`. Bootstraps use 10,000 pair resamples with seed 0. Sign
tests are exact two-sided binomial tests.

Platform values are in-sample descriptions of our own formative draws. Every run is a fresh draw, so comparisons
between versions on the platform are unpaired.

## 1. Per-budget table for the final model (v1_3): `per_budget_final.json`

Only one final-model run, 967620 (9 pairs), has per-pair and per-budget feedback. The other three runs report ALC only.
That means the between-draw SD per budget **cannot be estimated** for the final model. The column below gives the
within-draw SE across pairs instead.

| | B0 | B1 | B3 | B7 | B15 | B31 | ALC |
|---|---|---|---|---|---|---|---|
| Run 967620 | 0.2369 | 0.1903 | 0.1613 | 0.1445 | 0.1463 | 0.1412 | 0.1663 |
| Within-draw SE (pairs) | 0.0106 | 0.0241 | 0.0268 | 0.0277 | 0.0292 | 0.0290 | 0.0243 |
| Local R2 (EXP-0044) | 0.2456 | 0.2285 | 0.2123 | 0.1995 | 0.1894 | 0.1832 | 0.2088 |

**ALC over all four runs:**
- Mean 0.1839, one-draw SD 0.0240, SE 0.0120, t 95% CI [0.146, 0.222].
- 1 of 4 runs lies above the local value (968785, 0.2156); sign test p = 0.625.

**Pooled v1_1–v1_3 (labelled pooled).** This uses one per-pair run per version: 955381, 955660 and 967620. The three
versions share the budget-0 prediction (0.448722), so B0 comes from the same model in all three. Budgets 1–31 differ
slightly between versions, so each run is compared with its own version's local value.

| | B0 | B1 | B3 | B7 | B15 | B31 | ALC |
|---|---|---|---|---|---|---|---|
| Mean of 3 runs | 0.2317 | 0.1880 | 0.1718 | 0.1580 | 0.1556 | 0.1516 | 0.1730 |
| One-draw SD | 0.0051 | 0.0026 | 0.0092 | 0.0123 | 0.0083 | 0.0094 | 0.0058 |

- None of the 3 runs is above its own local value at any budget (sign test p = 0.25, the minimum possible with n = 3).
- Of the 26 slots, these are above the local value: B0 7 (p = 0.029), B1 8 (p = 0.076), B3 11, B7 12, B15 12, B31 12,
  ALC 10 (p = 0.33). The slot-level test treats pairs within a draw as independent, so it is descriptive only.

## 2. Local vs platform per version: `local_vs_platform.json`

| Version | Local R2 ALC | Platform: n, mean [95% t CI] | Replay ALC, 18 exact pairs (strict 8) |
|---|---|---|---|
| v0 | 0.2114 | not submitted | 0.1978 (0.1888) |
| v1 | 0.2098 | n=1: 0.2157 (no interval) | 0.1945 (0.1909) |
| v1_1 | 0.2097 | n=5: 0.1923 [0.176, 0.208] | 0.1954 (0.1896) |
| v1_2 | 0.2089 | n=4: 0.1909 [0.170, 0.212]; n=5 with the re-package: 0.1886 | 0.1954 (0.1896) |
| v1_3 (final) | 0.2088 | n=4: 0.1839 [0.146, 0.222] | 0.1954 (0.1896) |

**Does the ranking transfer?**
- The platform means fall in the same order as the local ALCs (Spearman 1.0, n = 4). This is not evidence that the
  ranking transfers.
- The local spread is only 0.0010, while each platform CI is 0.03–0.08 wide, and adjacent CIs overlap.
- The platform therefore cannot resolve the local differences, so the order of the platform means is uninformative.

**Paired replay differences on the 18 exact pairs.**

| Step | Mean difference | 95% CI |
|---|---|---|
| v0 → v1 | −0.0033 | [−0.0092, +0.0028] |
| v1 → v1_1 | +0.0009 | [−0.0020, +0.0039] |

- v1_1 → v1_2 → v1_3 is exactly 0. On singleton pairs replayed with synthetic labels that carry no item features, the
  item-group terms and the guard never fire. **The replay cannot see the item-group step.**
