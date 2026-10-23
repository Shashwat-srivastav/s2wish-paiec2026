> **Correction (2026-10-02, found in E2 / report 20):** `predict()` sees only the target item and `labeled`, never the hidden benchmark's full item pool. Standardising item features *within the benchmark* is therefore not possible at budget 0 at scoring time, contrary to what this report assumed. Gains that rely on it (the Jev prior L5: −6.2e-3 R1) are an offline upper bound, not a deployable result. A deployable variant using fixed training statistics is evaluated in report 20.

> **Erratum (Phase 4 red team, report 16):** the headline "v1 beats v0 by 5.7e-3 ALC" is withdrawn. With 200 draws the pair-level gap is −3.8e-3 [−5.2, −2.5]; over the 4 held-out benchmarks the CI is [−15, +7.6] (sign test p = 0.31), so a benchmark-level advantage is not established. v1 loses to v0 on easy hidden benchmarks at budget 0.

# 15 — Phase 2: realistic episodes, baseline ladder, submission v1

Personas: **BS** Bayesian Statistician (lead), **ML** Meta-Learning Scientist (episodes),
**CP** Calibration Purist (calibration veto), **MM** Mathematical Minimalist.
Code: `src/paiec/evaluation/episodes.py` (episodes and draw regimes), `src/paiec/models/ladder.py` (`--run`, `--table`, `--build-v1`, `--check`).
Raw results: `experiments/phase2_results.npz`, `phase2_table.txt`, `phase2_temps.json`, `phase2_run.log`.
A full run takes about 2 minutes and under 300 MB of RAM. Benchmarks are processed one at a time, and compact caches are kept in `artifacts/episode_cache/` on E:.

## 1. Episode generator (ML)

- **Folds.** Leave-one-benchmark-out (LOBO) over multi_swebench, real_webagents and researchcodebench, with matharena as an extra fold. swe_rebench is used only for training.
- **Training-fold fitting.** Every hyperparameter is fit only on the training benchmarks of a fold:
  - G and s_B: mean and sd of the benchmark mean logits. The prior is t3(G, s_B), with s_B ≥ 0.5.
  - τ: pooled pair-deviation sd, after removing binomial noise.
  - σ_β: empirical-Bayes Rasch item sd, from each benchmark's own data.
  - Name and metadata ridge: residual τ is estimated by LOBO within the training benchmarks.
  - Jev slope γ.
  - Temperature T: fit on inner-LOBO R1 episodes with disjoint seeds.
- **Pair splits, stream order and revealed trial.** These use the simulator's own hash seeds. `--check` runs the packaged v1 `model.py` through `simulator.run_direct`, the official client semantics, on real_webagents and researchcodebench. Per-pair ALC matches the vectorised engine to within **2e-16**.
- **Leakage veto conditions.**
  - Benchmarks are integers and are used only as pooling groups. No name is visible.
  - The subject key is the exact 8-field dict (1:1 with subject_id).
  - Items are keyed by the runtime dict (content, features, interactors), so collisions merge as they do at runtime.
  - Splits are per pair.
  - No hidden fields are read. Jev was computed from content only.
  - Only the held-out benchmark is scored, so a CV fold grouping is not needed.
- **Draw regimes.**
  - **R1:** B's eligible groups plus K−1 phantom hidden benchmarks, with K ~ U{1..4]. Phantom group sizes are resampled from the public pair sizes. Groups are selected with the simulator's `draw_groups` rule: fewest groups per benchmark first, then fewest per subject, then a random tie-break, skipping groups that do not fit, up to a 1000 subject-item cap. This gives a **median of 2 B-pairs per draw** (max 4–9). There are 40 draws per fold.
  - **R2:** one pair per episode, 5 seeds.
  - **R3:** all pairs, 5 seeds.
- **Weighting.** Pairs are weighted equally inside a fold. Headline numbers are the mean over the 4 folds, so benchmarks are balanced.
- **Bootstrap.** Paired Δ use a bootstrap over pairs, stratified by fold, with 2000 replicates.

## 2. Ladder (cumulative). Brier per budget; ALC is mean±sd over seeds; ECE is ALC-weighted. Δ is ×1e-3, vs the previous rung

**R1 (realistic draw)**

| rung | B0 | B1 | B3 | B7 | B15 | B31 | ALC | ECE | Δ vs prev [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| L0 empirical mean (official) | .2500 | .3842 | .2501 | .2150 | .2014 | .1952 | .2546±.032 | .191 | — |
| L1 v0 Beta-binomial (.5, k=2) | .2500 | .2392 | .2210 | .2073 | .1988 | .1944 | .2177±.021 | .137 | −39.4 [−43.9,−34.7] |
| L1f BB, m and k fit on training folds | .2486 | .2353 | .2188 | .2062 | .1978 | .1938 | .2159±.022 | .134 | −1.3 [−3.9,+1.4] |
| L2 + pooled offset (t3), exact grid×GH | .2449 | .2330 | .2166 | .2057 | .1982 | .1941 | .2146±.022 | .131 | −1.6 [−3.4,+0.1] |
| L3 + name link (λ·a_name) | .2436 | .2325 | .2163 | .2056 | .1981 | .1941 | .2143±.022 | .130 | −0.4 [−0.8,+0.0] |
| L4 + cross-pair item pooling | .2436 | .2322 | .2153 | .2032 | .1931 | .1845 | .2116±.022 | .136 | −2.8 [−3.3,−2.2] |
| L5 + Jev prior (**offline only**) | .2389 | .2262 | .2091 | .1971 | .1876 | .1795 | .2058±.021 | .151 | −6.2 [−7.3,−5.2] |
| L6 + metadata prior | .2389 | .2259 | .2088 | .1969 | .1875 | .1794 | .2057±.021 | .150 | −0.0 [−0.4,+0.3] |
| L6nj = L6 without Jev | .2436 | .2318 | .2150 | .2029 | .1931 | .1845 | .2114±.022 | .136 | −0.1 vs L4 |
| **L2i = L2 + item pooling (v1)** | .2449 | .2328 | .2155 | .2033 | .1932 | .1845 | **.2119±.022** | .137 | −2.8 [−3.3,−2.2] vs L2; **−5.7 [−7.5,−3.8] vs v0** |
| L2i+T, L4+T, L6nj+T (temperature) | | | | | | | ±0.0004 | +0.001–0.002 | −0.3 to −0.4, all CIs include 0 |

**R2 (1 pair, no cross-pair help).** ALC values:

| rung | ALC |
|---|---|
| L0 | .2433 |
| L1 | .2105 |
| L1f | .2111 |
| L2 | .2092 |
| L4 | .2087 |
| L2i | .2090 (−1.5 [−2.8,−0.1] vs v0) |
| L5 | .2033 |

**R3 (all pairs).** ALC values:

| rung | ALC |
|---|---|
| L1 | .2105 |
| L2 | .2046 (−6.6 vs L1f) |
| L4 / L2i | .1855 (item pooling −19.0 [−20.8,−17.1]) |
| L5 | .1812 |
| L6 | .1805 |

**Item-evidence overlap.** This is the share of eval targets that have at least one label on the same item dict, at budget 31:
- R1: 0.05 (multi_swebench), 0.45 (real_webagents), 0.20 (researchcodebench), 0.12 (matharena).
- R3: 0.66–0.99.
- R2: 0.

This share explains why the item rung gives −2.8 in R1 but −19 in R3.

**Per-fold R1 ALC.**

| rung | multi_swebench | real_webagents | researchcodebench | matharena |
|---|---|---|---|---|
| L1 | .2109 | .2141 | .2162 | **.2296** |
| L2i | .2014 | .2053 | .2079 | .2317 |

On the high-accuracy outlier (matharena) the learned offset prior loses to the plain v0 prior (m = 0.5). Inner-LOBO selection of the prior width kept s_B×1.0. Widening it did not help on any fold.

## 3. Marginal-value decisions (applied in R1; threshold 2e-3 ALC, from report 11 C3)

| component | R1 Δ | decision |
|---|---|---|
| Pooled offset (L2) | −1.6 vs L1f; −3.1 vs v0 | RETAIN. Robust in R2 and R3. |
| Name link (L3) | −0.4, CI touches 0 | **DROP** |
| Item pooling (L4) | −2.8, CI excludes 0 | RETAIN. It falls back exactly to L2 when there is no overlap. |
| Jev (L5) | −6.2 | Strongest single lever, but **not shippable**: api.typesafe.ai is not allowlisted. DEFER until a local distillation exists. |
| Metadata (L6, or L6nj vs L4) | −0.0 / −0.1; L6p vs L2 −0.5 | **DROP** |
| Temperature | about −0.4, CI includes 0. Fitted T 0.8–0.95 | **DROP** |

## 4. Claims (contract)

**C1 (BS).**
- **CLAIM (withdrawn — see erratum at top):** Pooling the offset plus item difficulty, with exact integration, beats v0 by 5.7e-3 ALC in R1. The gain is 1.5e-3 in R2 and 25e-3 in R3.
- **MECHANISM:** Other pairs' labels pin μ_b. Their labels on shared item dicts identify β_i.
- **EVIDENCE:** Table above. The gain holds in 3 of 4 folds. Engine and packaged model agree exactly.
- **COUNTEREVIDENCE:** matharena loses 2e-3 to v0. The R1 seed sd (0.022) is much larger than the Δ, but the paired CI excludes 0.
- **COMPETING EXPLANATION:** The R1 gain comes from shrinking toward the training base rate. The fitted BB (L1f) captures only −1.3, so most of the gain is pooling.
- **FALSIFICATION TEST:** Formative feedback at budget 1 vs budget 0 on many-pair benchmarks.
- **DECISION:** ADVANCE (v1).
- **CONFIDENCE:** strongly supported for R2/R3 direction, moderately supported for R1 magnitude.

**C2 (MM vs BS, disagreement).**
- BS would keep the metadata prior (L6p vs L2: −0.5 [−0.8,−0.2] in R1, −0.7 in R3; the CI excludes 0).
- MM: the gain is below 2e-3, it adds 11 provider coefficients, and it brings release-date drift risk (report 14 §3).
- **DECISION:** DROP from v1. Re-test after a forward-in-time holdout.

**C3 (CP).**
- **CLAIM:** v1 is not made better calibrated by temperature.
- **EVIDENCE:** T < 1 was chosen in every fold. The ΔBrier CI includes 0, and ECE does not improve.
- **OBSERVATION:** The item rung *raises* ECE at B15/B31. For L2i in R3, ECE at B31 is 0.100 vs 0.071 for L1, while Brier is far better.
- **RELIABILITY:** At B31, bins 0.8–0.9 are over-confident (predicted 0.84, observed 0.77).
- **CP veto status:** no veto, because Brier is the metric and ECE is diagnostic. CP flags the upper tail for Phase 3, for example a σ_β shrink or a per-budget T.

**C4 (ML).**
- **CLAIM:** Regime matters more than rung.
- **EVIDENCE:** The R3 vs R1 gap for the same model is 0.026 ALC.
- **IMPLICATION:** The usual all-pairs LOBO evaluation (report 11) overstates the hidden-test gain of item pooling by about 7×.

## 5. Submission v1 (`submissions/v1/`, zip `submissions/v1.zip`, 6.6 KB)

**Files.**
- `model.py`: the L2i model, numpy only. It never raises; every path is wrapped and finite values are clipped to [0.005, 0.995].
- `v1_params.json`: fit on all 5 public benchmarks.
  - G = −0.58, s_B = 0.81, τ = 1.13, σ_β = 2.70, T = 1.
  - Name and metadata fields are present but unused (`prior_mode: none`).
- `labeling.py`: accept-all, with the keyword-only `context` argument. This is random-equivalent and avoids a prediction call and a deepcopy of `labeled`.

**Checks.**
- `check_submission_zip.py`: OK.
- `run_smoke_test.py submissions/v1`: OK.
- `--concurrency 4` cannot run on Windows. The kit asserts `pass_fds`, so this path is untested locally.
- Garbage inputs and labels (54 combinations): all return finite values in [0,1].

**Latency per call.**

| case | latency |
|---|---|
| budget 0 | 0.26 ms |
| 3 pairs × 31 labels | 0.8 ms |
| 30 pairs × 31 labels | 3.2 ms |
| 930 labels | under 4 ms |

The 20 ms/call measured inside the simulator is the per-call JSON copy of `labeled`, which holds 78k-char researchcodebench contents.

## 6. Red-team risks
1. **Draw interpretation.** R1 assumes the 1000 cap counts subject-item pairs additively, as in the formative-feedback wording. If it counts *distinct source items*, the real regime is close to R3, which is better for v1. If hidden benchmarks have very large groups, it is close to R2, where v1 gains only 1.5e-3.
2. **Outlier base rate.** The t3 offset prior (G = −0.58) loses to the 0.5 prior on the high-accuracy matharena fold. A hidden easy benchmark would cost about 2e-3 at B0–B1. There are only 5 points behind the prior.
3. **Item pooling assumptions.** It assumes additive Rasch structure and identical item dicts across subjects of a hidden benchmark. If the platform varies the prompt per subject or harness, or uses non-empty `interactors`, the overlap goes to 0. v1 then degrades gracefully to L2, but the R1 gain disappears. The upper-tail over-confidence (C3) also grows with overlap.
