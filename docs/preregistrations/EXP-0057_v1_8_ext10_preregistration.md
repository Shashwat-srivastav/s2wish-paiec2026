# EXP-0057 — v1_8: easy-benchmark mixture prior (formative-guided candidate)

**Registered:** 2026-10-15, before any v1_3-engine or platform result for this candidate.

## Why

Formative feedback (allowed to "guide development" by the competition rules) shows the hidden pool contains easy
benchmarks: platform median pair accuracy 0.72 vs 0.41 in our local folds, and ~10% near-perfect pairs where v1_3's
whole loss (~0.05 per pair) is avoidable because it predicts 0.55 at B0 and ~0.68 after one correct label
(`experiments/pattern_mining/patterns/platform_findings.md`). Internal pattern mining found extreme-component mixtures pay
off once ≥ 6–8% of hidden benchmarks sit near a 0.98 base rate. **No benchmark ID, recovered pair accuracy, or other
hidden-test datum enters the model**; the prior is general and adapts only through labels revealed at runtime.

## Candidate (frozen, no tuning)

v1_3 archive (SHA `a4b0e841…`) with the offset prior replaced by E5's **X1-ext10**, unchanged from
`src/paiec/models/x1_prior.py`: 0.90 × t3(G/2, learned scale) + 0.05 × t3(+3, 1) + 0.05 × t3(−3, 1), each component
normalised on the 121-point μ grid. Every other parameter is identical to v1_3. Archive:
`submissions/dist/v1_8_ext10.zip`, built by `tools/build_v1_8_ext10_zip.py`.

## Hypothesis vs competing

- **H:** on an easy benchmark mix, X1-ext10 lowers ALC (faster adaptation on near-perfect benchmarks) by more than
  its known cost on hard folds (E5: +0.50e-3 [+0.20, +0.78] vs v1_1 on the four hard folds).
- **Competing:** the extreme component rarely fires (the +3 logit centre ≈ p 0.95 is below near-perfect rates), so
  the cost on hard benchmarks dominates.

## Procedure and decision rule (fixed)

1. **Equivalence:** the package must match the engine (`v1_2.cfg_of(facets="all", guard=0)` with prior
   `X.ext(CORE_H, 0.10)`) to max |Δp| < 1e-9 on all four folds, 3 seeds, all budgets. Fail → fix before anything else.
2. **Safety (local, 4 hard folds, R2 seeds 0–9, paired vs v1_3):** abort if ΔALC > +1.5e-3 or ΔB1 > +1.0e-3.
3. **Easy-benchmark evidence (decisive local test; needs EXP-0039R's harness):** paired vs v1_3 on the repaired
   enlarged-pool harness with equal benchmark weights. Adopt as the local winner if ΔALC ≤ −1.0e-3 with the
   hierarchical interval upper bound < 0 and P(Δ_new < 0) ≥ 0.8 (programme gate as modified by Jev).
4. **Official runs (safety only):** official repeats can only detect a disaster, not a ~3e-3 gain (draw SD 0.016).
   Stop uploading if the mean of the first 4 runs exceeds v1_3's 4-run mean (0.1839) by > 0.02.
5. **Final selection** of v1_8 over v1_3 requires both (3) and (4) to pass; otherwise the selection stays 971666 (v1_3).

Kit checks (`check_submission_zip`, `run_smoke_test`) before any upload. Uploads are done by the user.

**CORRECTION 2026-10-15 (sign error):** platform pair accuracies here were recovered with p0 = 0.5513, but the archives' own predict() gives p0 = 0.448722 (v1: 0.400099; E1 B0 mean prediction 0.451 agrees). B0 Brier and ECE cannot tell (a, p0) from (1-a, 1-p0), so every recovered accuracy was flipped. Corrected: platform median pair accuracy **0.27** (harder than local 0.41), **14.7% of pairs at accuracy <= 0.05**, **0% near-perfect**; benchmark_<hidden> has accuracy ~0. ALC/Brier-based results are unaffected (symmetric in a); every 'easier' / 'near-perfect' interpretation in this document is wrong and reversed. Reproduce: `python scripts/reproduce_q4_estimate.py`.


## Outcome (2026-10-16)

- Step 1 (package vs engine): PASS. Max |Δp| is 3.2e-15 over 12 fold×seed checks.
- Step 2 (hard-fold safety, R2 seeds 0–9): no abort. ΔALC is +0.52e-3 with pair CI [+0.22, +0.80]. ΔB1 is +0.94e-3, just under the +1.0e-3 limit.
  The fold deltas are multi_swebench −0.57, real_webagents +0.67, researchcodebench +1.60 and matharena +0.38 (all ×1e-3). v1_8 is locally *worse* than v1_3.
- Platform: one finished run, 990888, at 0.1951. Run 990865 was still running when recorded.
- Decision: **RETIRE v1_8.** It is safe but worse locally; the single platform run is uninformative (draw SD ≈ 0.016–0.024). Note that v1_8 was registered under the old, sign-flipped "easy platform" motivation; its −3 component actually targets the near-zero pairs. That variant was not tested separately. The final candidate is v1_3 971666.
