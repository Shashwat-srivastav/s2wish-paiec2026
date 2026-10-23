# 19 — E1: where does v1_1's budget-0 / budget-1 loss come from?

Personas:
- **CP**: Calibration Purist (lead).
- **BS**: Bayesian Statistician.
- **MM**: Mathematical Minimalist (veto).

Pre-registration: [`docs/preregistrations/E1_E2_preregistration.md`](../experiments/E1_E2_preregistration.md) (written before any result).
The decision boundaries below are copied from it and applied mechanically.

Code: `src/paiec/analysis/e1_b0_decomp.py`. Run it with `python -m paiec.analysis.e1_b0_decomp` (about 2 min, under 0.5 GB).
Raw output: `experiments/runs/e1/e1_results.json` holds every number in this report; `e1.log` holds the run log.
Registry: EXP-0020 and EXP-0021.

## Setup

- **Champion.** The predictor is `v1_1.BASE2`, the engine of `submissions/v1_1` (unmodified). Regenerated ALC is .20886 in R2 and .21505 in R1, identical to report 17.
- **Folds.** Leave-one-benchmark-out over the 4 evaluation benchmarks (swe_rebench is train-only).
- **Seeds.** Same as report 17: R2 uses seeds 0–4 (one pair per episode); R1 uses 100 draws (seeds 0–99).
- **Weighting.** Every Brier is pair-equal within a fold, then averaged over the 4 folds.
- **Leakage conditions (report 12).**
  - The offset prior, τ, σ_β and the metadata ridge are fit per fold on training benchmarks only.
  - No feature uses the benchmark name: pooling groups are opaque integers.
  - Item keys are the runtime content dict, so duplicate content is merged (MathArena duplicates). ResearchCodeBench papers never cross a fold boundary, because folds are whole benchmarks.
  - Oracle quantities use the held-out benchmark's full data on purpose. That is what an oracle is.

**Counterfactual predictors.** Each differs from v1_1 in exactly one thing.

| name | what is replaced |
|---|---|
| `oracle_offset` | μ_b is fixed at the held-out benchmark's full-data mean pair logit; θ_pair is still integrated and updated |
| `oracle_subject` | θ_pair is fixed at (full-data pair logit − benchmark mean); μ_b is still inferred under v1_1's prior |
| `oracle_item` | Item difficulty is the full-data Rasch β (centred); the pair level c solves mean_k σ(c+β_k) = q over the benchmark's keys |
| three-oracle floor (`oracle_all`) | All three oracles together |
| `metadata_prior` | θ_pair ~ N(m_meta, τ_meta²). m_meta is a ridge (α = 3) on: release date (clipped to the training range) with a missing flag; provider one-hot (providers with ≥ 3 training pairs); size tokens (small tier, big tier, log10 params, params known). It is fit on training benchmarks with weight 1/P per benchmark. τ_meta comes from leave-one-training-benchmark-out residuals. |

## R2 (primary): one subject per benchmark

**Brier at B0 and B1.** The mean is over the 4 folds. Pairs per fold: mswe 82, rwa 26, rcb 31, math 81.

| predictor | B0 mswe | rwa | rcb | math | **B0 mean** | B1 mswe | rwa | rcb | math | **B1 mean** |
|---|---|---|---|---|---|---|---|---|---|---|
| v1_1 | .2342 | .2383 | .2381 | .2719 | **.2456** | .2002 | .2179 | .2406 | .2507 | **.2274** |
| oracle_offset | .1714 | .2268 | .2273 | .2384 | **.2160** | .1683 | .2131 | .2245 | .2331 | **.2097** |
| oracle_subject | .2137 | .1885 | .2159 | .2375 | **.2139** | .1979 | .1890 | .2220 | .2300 | **.2097** |
| oracle_item | .1904 | .1430 | .1448 | .2114 | **.1724** | .1517 | .1332 | .1483 | .1776 | **.1527** |
| three-oracle floor | .0760 | .1007 | .0901 | .1033 | **.0925** | .0760 | .1007 | .0901 | .1033 | **.0925** |
| metadata_prior | .2332 | .2343 | .2353 | .2706 | **.2434** | .2001 | .2161 | .2388 | .2500 | **.2263** |
| *suppl.* oracle_item, leave-pair-out β | .1985 | .1561 | .1513 | .2178 | .1809 | .1600 | .1456 | .1548 | .1843 | .1612 |
| *suppl.* floor, leave-pair-out β | .0844 | .1129 | .0982 | .1111 | .1016 | .0844 | .1129 | .0982 | .1111 | .1016 |

**Murphy decomposition.** It uses 10 equal-width bins, computed per fold and then averaged.
- "Offset term" is the pair-level calibration-in-the-large, (p̄−ȳ)² per pair; "rest" = Brier − offset term.
- "bench CITL" is the same quantity at benchmark level.
- The residual Brier − (REL − RES + UNC), the within-bin term, is ≤ 0.002 everywhere.

| predictor | B0 REL | RES | UNC | offset term | rest | bench CITL | B1 REL | RES | UNC | offset term | rest |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v1_1 | .0300 | .0000 | .2157 | .0663 | .1793 | .0300 | .0189 | .0072 | .2157 | .0480 | .1793 |
| oracle_offset | .0003 | .0000 | .2157 | .0366 | .1793 | .0003 | .0013 | .0072 | .2157 | .0304 | .1793 |
| oracle_subject | .0336 | .0346 | .2157 | .0345 | .1793 | .0313 | .0259 | .0313 | .2157 | .0304 | .1793 |
| oracle_item | .0373 | .0795 | .2157 | .0617 | .1107 | .0285 | .0232 | .0853 | .2157 | .0461 | .1066 |
| floor | .0014 | .1237 | .2157 | .0052 | .0873 | .0004 | .0014 | .1237 | .2157 | .0052 | .0873 |
| metadata_prior | .0307 | .0012 | .2157 | .0640 | .1793 | .0302 | .0206 | .0093 | .2157 | .0470 | .1793 |

**Headroom shares.** share_X = (B_v1_1 − B_oracle_X) / (B_v1_1 − floor). The mean row is computed from the fold-mean Brier values.

| scope | B0 offset | B0 subject | B0 item | B0 headroom | B1 offset | B1 subject | B1 item | B1 headroom |
|---|---|---|---|---|---|---|---|---|
| **mean** | **0.194** | **0.207** | **0.478** | .1531 | **0.131** | **0.131** | **0.554** | .1349 |
| multi_swebench | 0.397 | 0.130 | 0.277 | .1582 | 0.257 | 0.019 | 0.391 | .1242 |
| real_webagents | 0.084 | 0.362 | 0.693 | .1376 | 0.041 | 0.247 | 0.723 | .1172 |
| researchcodebench | 0.073 | 0.150 | 0.630 | .1480 | 0.107 | 0.124 | 0.613 | .1505 |
| matharena | 0.199 | 0.204 | 0.359 | .1686 | 0.119 | 0.141 | 0.496 | .1475 |
| *suppl.* mean, leave-pair-out β floor | 0.206 | 0.220 | 0.449 | .1440 | 0.140 | 0.140 | 0.526 | .1257 |

**metadata_prior − v1_1.** Values are ×1e-3 ALC (negative = better). `[ ]` is a fold-stratified pair bootstrap; `h[ ]` is hierarchical over benchmarks, then pairs.

| quantity | value |
|---|---|
| ΔALC | **−0.58 [−1.00, −0.17]** h[−1.12, −0.00] |
| per fold | mswe −0.06, rwa −1.00, rcb −0.86, math −0.39 → **4 of 4 negative**, worst fold −0.06 |
| ΔB0 … ΔB31 | −2.25 [−3.93,−0.69], −1.09, −0.39, −0.21, −0.05, −0.03 |
| ALC threshold (−1.0, CI excludes 0, ≥ 3/4 folds negative, no fold > +1.0) | **FAILS**: point −0.58 > −1.0; the other three conditions pass |

## R1 (secondary): realistic ~1,000-item draw, 100 draws

- Pairs drawn per fold: mswe 53, rwa 26, rcb 31, math 78.
- The pair composition differs from R2, so B0 values are not identical even though B0 sees no labels.

| predictor | B0 mswe | rwa | rcb | math | **B0 mean** | B1 mswe | rwa | rcb | math | **B1 mean** |
|---|---|---|---|---|---|---|---|---|---|---|
| v1_1 | .2391 | .2380 | .2384 | .2705 | **.2465** | .2271 | .2238 | .2339 | .2522 | **.2342** |
| oracle_offset | .2192 | .2254 | .2281 | .2402 | **.2282** | .2127 | .2142 | .2238 | .2375 | **.2220** |
| oracle_subject | .2741 | .1903 | .2159 | .2358 | **.2290** | .2481 | .1897 | .2189 | .2229 | **.2199** |
| oracle_item | .1687 | .1416 | .1459 | .2069 | **.1658** | .1501 | .1380 | .1375 | .1790 | **.1511** |
| three-oracle floor | .0966 | .0996 | .0907 | .1015 | **.0971** | .0966 | .0996 | .0907 | .1015 | **.0971** |
| metadata_prior | .2378 | .2341 | .2356 | .2694 | **.2442** | .2253 | .2215 | .2325 | .2506 | **.2325** |
| *suppl.* oracle_item, leave-pair-out β | .1795 | .1549 | .1525 | .2136 | .1751 | .1608 | .1506 | .1445 | .1862 | .1605 |
| *suppl.* floor, leave-pair-out β | .1080 | .1120 | .0989 | .1094 | .1071 | .1080 | .1120 | .0989 | .1094 | .1071 |

| predictor | B0 REL | RES | UNC | offset term | rest | bench CITL | B1 REL | RES | UNC | offset term | rest |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v1_1 | .0205 | .0000 | .2260 | .0556 | .1909 | .0205 | .0138 | .0060 | .2260 | .0436 | .1906 |
| oracle_offset | .0022 | .0000 | .2260 | .0373 | .1909 | .0022 | .0022 | .0061 | .2260 | .0315 | .1906 |
| oracle_subject | .0371 | .0334 | .2260 | .0382 | .1909 | .0355 | .0222 | .0278 | .2260 | .0293 | .1906 |
| oracle_item | .0278 | .0870 | .2260 | .0531 | .1127 | .0202 | .0153 | .0890 | .2260 | .0424 | .1088 |
| floor | .0017 | .1298 | .2260 | .0055 | .0916 | .0006 | .0017 | .1298 | .2260 | .0055 | .0916 |
| metadata_prior | .0212 | .0013 | .2260 | .0533 | .1909 | .0209 | .0130 | .0068 | .2260 | .0419 | .1906 |

| scope | B0 offset | B0 subject | B0 item | B0 headroom | B1 offset | B1 subject | B1 item | B1 headroom |
|---|---|---|---|---|---|---|---|---|
| **mean** | **0.122** | **0.117** | **0.540** | .1494 | **0.089** | **0.105** | **0.606** | .1371 |
| multi_swebench | 0.140 | −0.245 | 0.494 | .1425 | 0.110 | −0.161 | 0.590 | .1304 |
| real_webagents | 0.091 | 0.344 | 0.697 | .1384 | 0.078 | 0.275 | 0.691 | .1242 |
| researchcodebench | 0.069 | 0.152 | 0.626 | .1477 | 0.071 | 0.105 | 0.673 | .1432 |
| matharena | 0.179 | 0.205 | 0.376 | .1690 | 0.098 | 0.194 | 0.485 | .1507 |
| *suppl.* mean, leave-pair-out β floor | 0.131 | 0.125 | 0.512 | .1394 | 0.096 | 0.113 | 0.580 | .1272 |

| metadata_prior − v1_1, R1 | value |
|---|---|
| ΔALC | **−0.91 [−1.38, −0.44]** h[−1.46, −0.44] |
| per fold | mswe −0.81, rwa −1.21, rcb −0.85, math −0.76 → 4 of 4 negative |
| ΔB0 … ΔB31 | −2.26 [−3.99,−0.60], −1.76 [−2.75,−0.82], −0.96, −0.46, −0.19, −0.10 |
| threshold | Not the adoption regime; the point estimate (−0.91) would also miss −1.0 |

## Pre-registered decision table (R2 = primary regime)

| Outcome | R2 value | Fired? | Action |
|---|---|---|---|
| share_offset ≥ 0.50 at B0 | 0.194 | no | (close metadata question; base-rate prior H5) — **not triggered** |
| share_subject ≥ 0.20 at B0 **and** metadata meets ALC threshold | 0.207; metadata −0.58 fails | no | — |
| **share_subject ≥ 0.20 but metadata fails threshold** | 0.207; fails | **YES** | **Record as limitation; no change.** |
| **share_item ≥ 0.20** | 0.478 | **YES** | **Raises the value of E2 (Jev / distilled prior).** |
| none of the above | — | no | — |

**Resulting action.**
- v1_1 is unchanged and the metadata prior is not adopted. The limitation is recorded: subject differences carry about 21% of B0 headroom, and metadata recovers only −0.58e-3 ALC.
- E2 (a content-based item-difficulty prior) is the priority B0/B1 investment.
- H-E1 (offset ≥ 50%) is **rejected**, so the metadata question is *not* closed.

R1 is concordant on the item row (0.540) and the offset row (0.122, fails). R1 does **not** reach the subject row (0.117).

## Claims (claim contract)

**C1 (CP, lead) — the B0 headroom is mostly item resolution, not offset.**
- **CLAIM:** Against the three-oracle floor, offset error explains 19% (R2) / 12% (R1) of v1_1's B0 headroom. Subject explains 21% / 12%, and item difficulty 48% / 54%. At B1 the item share rises to 55% / 61%.
- **MECHANISM:**
  - v1_1 at B0 issues one probability per benchmark, so RES = 0 exactly.
  - All of its loss above UNC is REL, and REL equals the benchmark-level calibration-in-the-large (.0300 R2).
  - The oracle offset removes that term almost entirely (REL .0003) but cannot add resolution.
  - Most of the gap to the floor is RES that only item (RES .080) or subject (RES .035) information can create.
- **PRIOR:** Report 14 measured offset 0.031 against subject 0.005–0.008. Here offset is worth 0.0296 (R2), which agrees. But the pre-registration expected share_offset ≈ 0.6–0.8; that expectation forgot the item axis.
- **EVIDENCE:**
  - The tables above.
  - The item share is ≥ 0.28 on every fold in both regimes.
  - With a leave-pair-out β (the item oracle no longer sees the target pair's own responses), the item share is 0.449 (R2) / 0.512 (R1), so in-sample optimism is small.
- **COUNTEREVIDENCE:**
  - Measured against the *climatological* forecast (UNC) rather than the floor, 100% of v1_1's B0 excess is offset (REL = CITL). Under that reading H-E1 would hold. The pre-registered metric is the floor-relative share, and it was applied as written.
  - The floor is an oracle on full data (Rasch β plus pair logit). No runtime predictor can approach it.
  - The offset share varies across folds (0.07–0.40). multi_swebench is the hardest benchmark, and offset carries 40% of its R2 headroom.
- **COMPETING EXPLANATION:** The item headroom could be uncapturable at runtime. Report 12 C3 found TF-IDF item content reaching AUC 0.79 within a benchmark, and report 15 found Jev worth −5 to −6e-3 ALC, so part of it is reachable from content.
- **FALSIFICATION TEST:** If E2's distilled prior retains ≥ 70% of Jev's ρ but yields < −1e-3 ALC in R2, item resolution exists but is not reachable at runtime. The share would then be a ceiling, not a target.
- **DECISION:** ADVANCE E2 (pre-registered row 4).
- **CONFIDENCE:** strongly supported for "offset < 50% of floor headroom" (every fold in both regimes is below 0.40); moderately supported for the item ranking (4 folds).

**C2 (BS) — the metadata prior helps a little, consistently, but below threshold.**
- **CLAIM:** The training-only metadata prior improves v1_1 by −0.58e-3 ALC in R2, CI [−1.00, −0.17], with 4 of 4 folds negative. In R1 the gain is −0.91e-3 [−1.38, −0.44], also 4 of 4.
- **MECHANISM:** It adds subject resolution at B0. RES goes from .0000 to .0012, which recovers ~7% of the oracle-subject gap (−2.2e-3 of 31.7e-3 Brier at B0). Once labels arrive the gain fades to ~0 by B15.
- **EVIDENCE:** The tables above. The R2 hier CI upper bound is −0.00, and the R1 hier CI excludes 0.
- **COUNTEREVIDENCE:** The point estimate misses the −1.0e-3 adoption bar. The B0 gain (−2.25e-3) is only 7% of the subject headroom, so most subject differences are invisible to metadata. This matches report 14, where R² was about 0.16.
- **MM:** The prior adds a 17–19-coefficient ridge with release extrapolation risk (report 14 §3) for a gain below the bar. Veto on shipping; no veto on recording.
- **DECISION:** Record as a limitation; no change (pre-registered row 3).
- **CONFIDENCE:** strongly supported that the effect is real and small; the shipping decision is mechanical.

**C3 (CP) — v1_1 at B0 is calibration-only.**
- **CLAIM:** Every v1_1 B0 forecast within a benchmark is the same number. Its whole B0 loss is calibration-in-the-large: REL equals bench CITL (.0300 R2, .0205 R1).
- **COUNTEREVIDENCE:** None. This holds by construction, because the R1 item-pooling path has no labels at B0.
- **DECISION:** Use this as the frame for any future B0 work: any gain must come from RES (item or subject information) or from a better offset (REL ≤ 0.030).
- **CONFIDENCE:** established.

## Deviations and implementation choices

1. **DEVIATION (addition):** I added supplementary `*_lpo` rows (item β re-estimated without the target pair's responses) to measure the oracle's in-sample optimism. They were not pre-registered and are not used by the decision table.
2. **DEVIATION (minor):** The release date is clipped to the training range [min, max], not only at the training max. The lower clip affects exactly one evaluated subject (one researchcodebench pair released before the training minimum); the upper clip affects 4 matharena pairs. The provider category set is fit per fold (≥ 3 training pairs), and "Unknown" counts as a category. The ridge excludes the name-transfer feature, following report 14 rec. 3, whose listed features are release, provider and size.
3. **Implementation:**
   - In `oracle_item`, c solves mean_k σ(c+β_k) = q over the benchmark's true β distribution, not v1_1's N(0, σ_β) assumption. Otherwise the oracle would also change the pair-level calibration.
   - Known β replaces v1_1's cross-subject item update at B1. The pair-level posterior is unchanged, so it does not adjust for the labelled item's β.
4. **Implementation:** The "true pair logit" is the smoothed full-data logit((S+½)/(n+1)) from `ladder.bench_stats`.
5. **Rows are not mutually exclusive.** Two rows fired, and both actions were applied. The decision was taken on R2, the primary regime under the shared conventions.
