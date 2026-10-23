# EXP-0039R — simulator repair with enlarged-pool folds (pre-registration)

**Registered:** 2026-10-15, before any repaired-harness result exists. **Trigger:** EXP-0039 failed its locked validity
gate (tail count 8 vs simultaneous range90 [1,7], p_upper 0.022; global envelope p 0.003; dossier
`docs/publication/dossiers/EXP-0039.json`). Jev decision `GATE-0039-next-step`: MODIFY (0.97), agreeing with the PI.

## Why

Every local comparison so far used four hard benchmark folds (median pair accuracy 0.41). The platform's draws are
much easier (median 0.72; 10% near-perfect pairs) and the summative keeps the same sampling design. A harness that
cannot represent easy benchmarks cannot be trusted to rank methods for the summative.

## Hypothesis vs competing

- **H:** the simulator's miss is a benchmark-composition shift. Adding held-out folds from the official public
  training release (which contains easy benchmarks) brings the predictive distribution in line with the 34 records.
- **Competing:** the miss comes from something composition cannot fix (pair-size/eval-pool differences — platform
  median 56 vs local 91 distinct items — or a mechanism difference such as the platform's draw rules).

## Procedure (fixed)

1. Data: `data/_main_54074453` (official public release, HF rev 54074453da). Inclusion by public rules only: binary
   responses, item-level, `release == public`, ≥ 1 eligible pair under the existing R1 episode builder. Expected
   ≈ 142 benchmarks. **No reweighting toward platform statistics; equal benchmark weights.**
2. Model: frozen v1_3 engine, `cfg_of(facets="all", guard=0)`, with hyperparameters frozen from revision
   `bc8204d811` (no refit). Package-equivalence assert on the shipped zip (SHA `a4b0e841…`) before any run.
3. Episodes: leave-one-benchmark-out over the enlarged set, R1 seeds 0–19, budgets 0–31, accept-all.
4. Re-run the EXP-0039 statistics unchanged (version-matched blind predictive distributions; run-level PI90;
   simultaneous tail-count range; global envelope) against the same 34 records, plus a 7-benchmark-stratified draw
   composition variant (platform_findings §4) reported as secondary.
5. Report the repaired harness's base-rate distribution next to the platform's (descriptive only).

## Decision rule (fixed)

| Outcome | Meaning | Action |
|---|---|---|
| All three EXP-0039 criteria pass | composition shift explains the miss | Adopt the repaired harness for all later local comparisons; re-open dependent experiments on it (each needs its own pre-registration) |
| Envelope passes but tail count fails, or vice versa | partial repair | Adopt as the better harness; report the residual miss as a caveat on every later claim |
| Both fail | composition is not the cause | Stop local model search; investigate the competing explanation (pair size / draw rules); claims stay local-only |

A pass here is a **calibration check against the same 34 records**, not independent confirmation; fresh platform
runs (EXP-0040) would be needed for that.

## Cost and limits

≈ 1–2 h in one compute slot; peak < 1 GB; cache 0.3–1 GB under `.tmp/` (E:). Personas: paiec-systems (owner),
paiec-redteam review before the decision is recorded.

**CORRECTION 2026-10-15 (sign error):** platform pair accuracies here were recovered with p0 = 0.5513, but the archives' own predict() gives p0 = 0.448722 (v1: 0.400099; E1 B0 mean prediction 0.451 agrees). B0 Brier and ECE cannot tell (a, p0) from (1-a, 1-p0), so every recovered accuracy was flipped. Corrected: platform median pair accuracy **0.27** (harder than local 0.41), **14.7% of pairs at accuracy <= 0.05**, **0% near-perfect**; benchmark_<hidden> has accuracy ~0. ALC/Brier-based results are unaffected (symmetric in a); every 'easier' / 'near-perfect' interpretation in this document is wrong and reversed. Reproduce: `python scripts/reproduce_q4_estimate.py`.

