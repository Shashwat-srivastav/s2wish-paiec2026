# 16 — Phase 4 red team of submission v1

Personas:
- **RT**: Red-Team Researcher (#16, lead; assumes v1 is overfit).
- **LH**: Statistical Leakage Hunter (#11, veto).
- **MS**: Mechanistic Skeptic (#15).
- **RH**: Reviewer From Hell (#19, veto).

Code: `src/paiec/analysis/redteam.py`, run with `python -m paiec.analysis.redteam --section A1,…,A9`.

Raw output is in `experiments/runs/phase4_redteam/`:
- `redteam_results.json`: all sections.
- Logs: `redteam_A2.log`, `redteam_A3A5A6A8.log`, `redteam_A4A7.log`, `redteam_A9.log`.

The earlier run was interrupted. Its sections A1, A2 and A8(R1) were complete and valid, and they were reused.

Method notes:
- Every engine number uses hyperparameters re-fit per LOBO fold (`ladder.hyper_for(B)` excludes B). The engine is verified equal to the packaged `model.py` to 2e-16.
- Δ is ×1e-3 ALC or Brier. Negative means v1 is better.
- CIs are a stratified pair bootstrap (2000 reps) unless marked *hier*. Hier means a bootstrap over benchmarks, then pairs.
- `withB` is the R1 draw rule with 200 draws instead of 40.

Item-pooling regimes and draw rules:
- `real_pair` / `real_distinct`: the 1000 cap counted per subject-item vs per distinct source item.
- `large`: hidden groups of 400–900 items.
- `many_small`: 7–13 benchmarks of 80–150 items.
- `noleak`: phantom sizes drawn from other benchmarks only.

## 1. Attack table

**A1. Fold leakage — PASS.**
- **Hypothesis.** The reported R1 numbers use `v1_params.json`, which was fit on all 5 benchmarks.
- **Test.** Recompute with strictly per-fold parameters, then with the shipped parameters.
- **Result.**
  - The report-15 numbers *are* per-fold: −5.65 [−7.51, −3.83].
  - With the shipped all-5 parameters the result would read −7.41, a leak gain of −1.76 [−2.36, −1.16].
  - Phantom-size leakage is nil: noleak −3.89 vs withB −3.81.
- **Verdict.** Never quote numbers from the shipped parameters on public folds.

**A1b. Draw-count optimism — WARN.**
- **Hypothesis.** The 40-draw R1 headline is lucky.
- **Test.** Repeat with 200 draws under the same rule.
- **Result.** v1−v0 is **−3.81 [−5.15, −2.48]**, against −5.65 at 40 draws. The pair bootstrap understates draw variance.
- **Verdict.** The headline must be restated.

**A2. Draw regime — PASS (WARN at the extremes).**
- **Hypothesis.** v1 loses to v0 under other plausible draw rules.
- **Test.** Run v1−v0 over 8 regimes. Matharena is reported per fold.

  | regime | v1−v0 | matharena (per fold) |
  |---|---|---|
  | real_pair | −4.10 | +6.4 |
  | real_distinct | −24.2 | −16.9 |
  | large | −3.23 | +6.2 |
  | many_small | −1.17 [−2.66, +0.29] | +10.2 |
  | R2 | −1.48 [−2.98, 0.00] | +10.4 |
  | R3 | −24.0 | |

- **Result.** v1 never loses on aggregate. It loses on the easy benchmark (matharena) in every regime with few B-pairs.
- **Verdict.** WARN. The loss is driven by B0: matharena B0 Brier is +49 to +56 worse than v0.

**A3. Easy/hard shift — FAIL for easy benchmarks.**
- **Hypothesis.** A held-out base-rate shift breaks v1's offset prior.
- **Test.** Resample the Rasch model with a logit shift δ from −3 to +3.

  | δ | withB | R2 | B0 Brier (withB) | wide-sB×1.5 vs v1 (withB) |
  |---|---|---|---|---|
  | −3 | −17.1 | −10.6 | | −1.10 |
  | 0 | −2.7 | −2.2 | | +0.93 |
  | +1 | +0.95 | +2.73 | | |
  | +2 | +4.42 | +7.54 | | |
  | +3 | **+6.66** | **+12.40** | +54 | −2.13 |

  - Under R2, matharena alone is +33.0 at δ = +3.
  - Averaged over symmetric δ, the wide-sB variant is about −0.03 vs v1.
- **Result.** v1 wins big on hard benchmarks. It loses to v0 once the base rate is ≥ ~0.5. Widening sB only trades δ = 0 performance for performance at the extremes.
- **Verdict.** FAIL, documented. No fix is shipped (see §3).

**A4. Stateful execution and order — PASS.**
- **Hypothesis.** Predictions depend on the order of `labeled`, the targets or the pairs, on repeated calls, or on process state.
- **Test.** Shuffle `labeled` 5×; reverse the targets; repeat calls; run in a fresh process; interleave empty-`labeled` calls; reverse the pair order through `run_direct`.
- **Result.**
  - The maximum difference is 3e-16 (floating-point summation order) and 0 elsewhere.
  - No arguments are mutated.
  - Candidate order is the platform's, under accept-all.
- **Verdict.** PASS. Duplicate `labeled` entries are counted twice (shift 0.018). That is correct for genuinely distinct colliding item_ids.

**A5. Item identity — PASS for exact keys, WARN for false merges.**
- **Hypothesis.** The item key merges unrelated items, or misses true matches.
- **Test.**
  - Probes: trailing whitespace, CRLF, or a changed `item_features` value stop the match. `interactors=None` vs `""`, an int vs str `benchmark_id`, and dict key order still match.
  - Kangaroo: under the runtime key, the 336 Kangaroo items stay distinct because `problem_idx` is in `item_features`. A content-only key would merge them.
  - Random false merges of m unrelated items per key.

  | key variant | v1−L2 (withB) | v1−L2 (R3) |
  |---|---|---|
  | content-only (Kangaroo 1511→854 keys) | −2.70 | −19.5 |
  | random merges, m = 2 | −0.04 [−0.36, +0.26] | −10.6 |
  | random merges, m = 5 | **+5.09** | +4.9 |
  | random merges, m = 20 | **+11.79** | +18.2 |

- **Result.** The content-only merge does no harm. Random merges are neutral at m = 2 and harmful from m = 5.
- **Verdict.** WARN. If a hidden benchmark has template content and empty features, item pooling is harmful.

**A6. Calibration — WARN (report only).**
- **Hypothesis.** Upper-tail overconfidence is fixable cheaply.
- **Result.**
  - The 0.8–0.9 bin at B7 predicts 0.84 and observes 0.72. This holds in withB *and in R2*, where item pooling has no evidence. So it is a **pair-level posterior** effect, not item pooling.
  - Shrinking toward q (α 0.85 / 0.7), σ_β×0.8, and clips at 0.02 or 0.05 all give a ΔBrier between −0.00 and +1.6. None helps.
- **Verdict.** WARN. No fix lowers Brier; temperature was already rejected (EXP-0007).

**A7. Runtime robustness — PASS (WARN on scale).**
- **Test.** 52 malformed cases: labels as bool, str, float, NaN, inf, numpy types, lists or bytes; entries with missing fields; non-ASCII text and lone surrogates. Plus 6 malformed `context` values for the labeling hook.
- **Result.**
  - No exceptions; every output is in [0, 1].
  - Latency with 10k labels: 78 ms (1 subject), 156 ms (300 subjects), 666 ms and 191 MB peak (2000 subjects).
- **Verdict.**
  - The formative cap (≤ 1000 subject-items) is far inside 8 h.
  - The per-call cost is O(|labeled|) in Python, a risk for a very large summative set.
  - `--concurrency 4` is still untested on Windows.

**A8. Reviewer from Hell — VETO on the claim, not on the submission.**
- **Hypothesis.** "v1 beats v0" cannot be established with 4 benchmark folds.
- **Result (withB).**
  - Per fold: −7.7, −7.0, −7.5, +6.9.
  - t-CI over folds: [−15.2, +7.6]. Hier: −3.81 [−8.29, +3.38]. Sign test p = 0.31.
  - Leave-one-fold-out: −2.5 to −7.4.
- **Verdict.** The offset component vs v0 has a hier CI of [−6.43, +5.58]. The item-pooling component vs L2 has a **hier CI of [−3.16, −1.24]**, negative in 4/4 folds.

**A9. Codabench formative reproduction — PASS (sampling noise).**
- **Hypothesis.** The formative failures come from a mechanism in v1. The formative showed:
  - mean B1 > B0 (+5.5);
  - one pair at 0.36 at B3/B7;
  - one 2-subject benchmark at 0.27–0.31 at every budget.
- **Test.** R2, 5 seeds × 4 folds, on real data and on δ ∈ {−1, 0, +1, +2}. Monte Carlo of 20k formative-like 8-pair samples, plus a 2-subject regime.
- **Result.** See §2.
- **Verdict.** There is no systematic overreaction. All three are within the local spread, at probabilities 0.20, 0.25 and 0.03–0.05.

**Leakage Hunter.** No veto. Per-fold fitting is verified, no UNSAFE field is read, and the runtime-dict key is used (report 12 conditions a–e hold).

**Reviewer From Hell.** Vetoes the claims "v1 beats v0 by 5.7e-3" and "significant at the benchmark level". It does not veto the upload.

## 2. A9 — formative failure modes (MS: Level-1 evidence, mechanism stated with its test)

All rows are local R2, real data. P8 is the probability over 8-pair samples.

| model | B1−B0 | B1−B0 if first y = 1 | B1−B0 if first y = 0 | frac pairs B1 > B0 | P8(mean B1 > B0) | P8(any pair ≥ .36 at B3/B7) | P8(any pair > .27 at all budgets) |
|---|---|---|---|---|---|---|---|
| v0 | −20.2 | **+7.9** | −44.8 | 0.38 | 0.22 | 0.30 | 0.00 |
| v1 | −19.1 | −32.5 | −20.4 | 0.35 | 0.20 | 0.25 | 0.03 (0.05 at δ = +1) |
| v1 wide1.5 | −17.0 | −27.7 | −22.4 | 0.37 | 0.27 | 0.34 | 0.03 |

- **One-label overreaction is not a v1 property.**
  - v1's expected B1 − B0 is −19. It is negative whichever value the first label takes.
  - v0, with its Beta(1,1) prior, is the model that overreacts to a success.
  - On any single pair, B1 > B0 happens 35% of the time for every model. This is an aleatoric effect of one Bernoulli label on a heterogeneous item set.
  - A formative mean of B1 > B0 over 8 pairs has local probability 0.20.
- **Non-monotone curves.** These are draws where the first labels are atypical. Some pair among 8 hits ≥ 0.36 with probability 0.25.
- **The 2-subject benchmark stuck above 0.27.**
  - This is not caused by cross-pair pooling. In the 2-subject regime v1 − L2 ≤ 0 at every budget in every fold, and v1 is never worse than L2 by > 0.01 ALC.
  - The local probability is 0.03, rising to 0.05 for an easy benchmark (δ = +1).
  - The most likely mechanism is an offset-prior miss on an extreme or heterogeneous benchmark. That is the A3 failure: v1's B0 is about 0.36, and with a base rate of 0.65–0.7 it scores about 0.31. It is combined with near-0.5 within-pair heterogeneity, which no item-agnostic model can beat at B31.
  - Falsification test: if a later formative draw puts many-subject benchmarks next to it, v1 − L2 per pair on that benchmark should stay ≤ 0.
- **Should v1_1 fix these?** No fix reproduces an improvement.
  - wide-sB makes A9 worse: P8(B1 > B0) 0.20→0.27, ALC .2090→.2103.
  - A6 shows α, σ_β and clip variants do not help.

## 3. Item-pooling (L4/L2i) verdict — **KEEP**

v1 − L2 Brier by budget (B0 is identically 0, because there is no item evidence at B0):

| regime | B1 | B3 | ALC |
|---|---|---|---|
| R1 (40) | −0.30 [−0.51, −0.08] | −1.12 [−1.55, −0.72] | −2.76 [−3.33, −2.21]; hier [−4.24, −1.39] |
| R1 withB (200) | | | −2.23 [−2.59, −1.87] noleak; hier −2.29 [−3.16, −1.24] |
| R2 | 0.00 [−0.00, +0.00] | −0.03 [−0.06, −0.00] | −0.21 [−0.37, −0.09] |
| large groups | −0.12 [−0.20, −0.04] | −0.35 [−0.48, −0.22] | −1.32 [−1.59, −1.08] |
| many_small | −0.02 [−0.06, +0.01] | −0.11 [−0.22, −0.02] | −0.37 [−0.55, −0.21] |
| real_distinct | −3.37 | −9.48 | −19.4 |
| 2-subject (A9) | −0.02 to −0.12 | −0.08 to −0.42 | 0 / 1200 draws worse by > 0.01 |

**False-match harm (A5):**
- m = 2: neutral.
- m = 5: +5.1.
- m = 20: +11.8.
- Content-only Kangaroo merge: −2.7, so no harm.

**Decision.** The Jev decision layer feared low-budget harm. That harm is **not found**: no budget, regime or fold has a positive point estimate on real keys. The gain is small in harsh regimes (−0.2 to −0.4) but never negative, and it is the only component whose benchmark-level CI excludes 0.

The residual risk is specific: false merges on template-content benchmarks. A guard addresses it (fix #3). Dropping the component does not.

CONFIDENCE: strongly supported (direction), moderately supported (R1 magnitude).

## 4. Ranked required fixes

1. **Restate claims** in reports 15 and README (RH veto).
   - R1 v1 − v0 is about **−3.8 [−5.2, −2.5]** on 200 draws, not −5.7.
   - The benchmark-level CI includes 0; v1 wins 3 of 4 folds.
   - Only the item-pooling gain is benchmark-robust.
2. **The easy-benchmark B0/B1 loss** (A3, matharena) is the largest known risk, up to +12 ALC in R2 at δ = +3. Widening sB has ≈0 expected value. Re-open it only with more benchmarks behind the prior, or with a content-based base-rate prior (the Jev distillation). Do not tune to formative feedback.
3. **False-merge guard for item pooling** (A5). Disable pooling for a target key when one subject has ≥ 2 labels on it with conflicting outcomes. Implement and test it in v1_1 (Phase 3). It was not built this round, because no observed failure requires it.
4. **Pair-level upper-tail overconfidence** (A6: 0.84 predicted → 0.72 observed at B7, including R2). Investigate an overdispersion or heterogeneity model; simple fixes do not lower Brier.
5. **Runtime at summative scale** (A7). Per-call cost is O(|labeled|): 0.67 s / 191 MB at 2000 subjects. Cache the parsed evidence by an evidence fingerprint if the summative set is large.
6. Duplicate-entry double counting (A4). Low priority; it is correct for distinct colliding item_ids.

## 5. Upload candidate

**v1 stays the candidate.** No v1_1 is built, because no tested patch fixes A9 or A3 without regressing the δ = 0 regime.

The frozen archive was re-verified today: `check_submission_zip.py submissions/dist/v1.zip` OK, and `run_smoke_test.py submissions/v1` OK.
