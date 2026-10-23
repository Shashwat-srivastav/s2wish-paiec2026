# Review round 1 (2026-10-19)

Code: `src/paiec/analysis/review_round1.py` (`counts`, `platform`, `floor`, `ladder`). The `floor` and `ladder` parts
were run through `scripts/run_slot.py`. Bootstraps are percentile intervals with seed 0: 10,000 resamples for the
platform results and 2,000 for the hierarchical ladder intervals. ΔALC is later minus earlier, so a negative value is
an improvement.

All platform numbers are **in-sample** on our own formative draws. They come only from our own feedback and archives,
and anonymous IDs are used only as opaque labels.

| File | Item | Contents |
|---|---|---|
| `pair_counts.json` | #11 | Singleton and exact pair counts per set of runs |
| `platform_baseline_replay.json` | #4 | Exact replay of the official empirical-mean baseline, v0 and the final model on the exact singleton pairs |
| `ladder_intervals.json` | #1 | Pair, hierarchical and t intervals for every ladder step, recomputed on common episodes (R2: 10 seeds; R1: 100 seeds) |

## Key results

**#11 Pair counts.**
- The three later runs alone have 15 singleton pairs, 12 of them exact.
- Adding the first model's run gives 22 singleton pairs, 18 of them exact.
- The phrase "17 singleton pairs of the three later models (14 exact)" is therefore inaccurate.

**#4 Platform replay.** Within each draw, all three models are scored on the same revealed labels.

| Pair set | n | Benchmarks | Baseline ALC | v0 ALC | Final ALC | Baseline − final [95% CI] | v0 − final [95% CI] |
|---|---|---|---|---|---|---|---|
| All exact singletons | 18 | 7 | 0.2276 | 0.1978 | 0.1954 | +0.0322 [+0.0140, +0.0515] | +0.0023 [−0.0008, +0.0055] |
| Strict | 8 | 4 | 0.2062 | 0.1888 | 0.1896 | +0.0165 [−0.0032, +0.0352] | −0.0009 [−0.0052, +0.0029] |

- The baseline and v0 rules are count-only, so their replays are exact on every exact singleton pair.
- The final model's replay is exact only on the strict set. That set is the pairs where a run with item-group code

**#7 Floor and ceiling.**

| Quantity | Value | Slot CI | Benchmark-cluster CI |
|---|---|---|---|
| Floor, earlier draws (26 slots, 24 distinct pairs, 7 IDs) | 0.1511 | [0.117, 0.183] | [0.090, 0.205] |
| Ceiling, earlier draws | −0.0218 | [−0.0287, −0.0154] | [−0.0327, −0.0130] |

The same floor computed locally on R2 is 0.1797 (folds 0.143, 0.173, 0.203, 0.200; t interval [0.135, 0.224]). The
final model scores a local ALC of 0.2088 there, so its local ceiling is −0.0291.

**#1 Ladder (R2, ×1e-3).**

| Step | ΔALC | Hierarchical CI | t CI |
|---|---|---|---|
| Baseline → v0 | −33.70 | [−42.78, −23.41] | [−50.59, −16.81] |
| v0 → v1 | −1.62 | [−6.43, +6.41] | [−14.08, +10.85] |
| v1 → v1_1 | −0.09 | [−5.19, +3.31] | [−8.14, +7.97] |
| v1_1 → v1_2 | −0.83 | [−1.40, −0.21] | [−1.90, +0.25] |
| v1_2 → final | −0.09 | [−0.26, +0.00] | [−0.33, +0.15] |
| v0 → final | −2.62 | [−4.47, −0.26] | [−6.33, +1.09] |

The recomputation reproduces two quoted intervals exactly: e5's v0 vs v1_1 hierarchical interval, [−1.28, +3.65] with
the sign flipped, and e4's item-group t interval, [−2.16, +0.33]. The baseline ALC with 10 seeds is 0.2451; Table 3
gives 0.2433 from 5 seeds in phase 2.
