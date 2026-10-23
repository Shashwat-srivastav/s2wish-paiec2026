# E7 — Priors from the enlarged training pool (281 benchmarks)

Pre-registered 2026-10-08, before any response table of the new revision was read.
Personas: Bayesian Statistician, LLM/Subject Evaluation Scientist, Cold-Start Specialist (run); Statistical Leakage
Hunter and Red-Team Researcher (review before any submission).

## Why (the decision it unblocks)
Every prior in v1_3 (benchmark-offset centre and scale, pair sd τ, item sd σ_β) is fitted on 5 benchmarks, and the
model has no usable knowledge of which AI systems are strong. On 2026-10-07 the public dataset grew to 281 benchmarks
(revision `54074453da`; report 25 §3): 212 binary/mixed item-level, 78 with 2025–26 subjects, 97 model names seen in
≥3 benchmarks. E1 (report 19) measured the subject oracle at −0.032 Brier at budget 0, of which the 5-benchmark
metadata prior recovered 7%. This experiment decides whether a v1_8 built on the enlarged pool replaces v1_3.

## Hypotheses
- H-subject: a model-name ability prior learned across benchmarks (empirical Bayes, shrunk, with provider /
  release-date regression as fallback for unseen names) lowers ALC by ≥0.002.
- H-hyper: refitting the benchmark-offset distribution and τ, σ_β on the enlarged pool lowers ALC by ≥0.001.
- Competing: the new benchmarks (mostly multiple-choice knowledge, safety, medical, multilingual) differ from the
  hidden agentic/maths-style benchmarks, so their level and ability structure do not transfer, or shift the starting
  level the wrong way.

## Current evidence
For: E1 subject oracle; other teams gain from attribute priors (report 25 §2). Against: report 14 found subject
metadata R² 0.16 on 5 benchmarks; EXP-0021 metadata prior −0.58e-3 (below bar); ability may not be one-dimensional
across domains.

## Data
`data/_main_54074453/<bench>/formatted_tables/` (benchmarks, subjects, responses; items only where needed). Binary
rows only (response ∈ {0,1}); item-level benchmarks; `release == public`. Subjects restricted to 2025–26 releases for
the primary fit (the competition's eligible population), all subjects as a sensitivity arm. Per benchmark × subject:
n, number correct. The four held-out folds below are matched to the new pool by benchmark name and removed from every
fit, together with any benchmark sharing a `source_url` or name stem with them (e.g. other SWE-bench variants are
reported separately as a leakage-sensitivity arm, not silently kept).

## Procedure
Primary evaluation is unchanged, so results are comparable with reports 15–24: leave-one-benchmark-out over
`multi_swebench`, `real_webagents`, `researchcodebench`, `matharena`; regime R2; seeds 0–9; budgets 0–31; same
episodes for all arms (paired); pair-bootstrap and hierarchical CIs.

Arms:
1. `base` — v1_3 (v1_2 named facets, no guard), hyper-parameters from the 5-benchmark pool.
2. `hyper` — same model, offset centre/scale, τ, σ_β refitted on the enlarged pool (minus held-out and relatives).
3. `subject` — base + ability prior θ(name) from the enlarged pool: θ = shrunk mean over training benchmarks of
   (pair logit − benchmark mean logit over eligible subjects), shrinkage weight from the between-benchmark variance of
   that residual; fallback regression on provider and release date for unseen names. Enters as the prior mean of the
   pair effect, with prior sd reduced according to the estimated reliability.
4. `subject+hyper`.
5. Sensitivity: `subject` fitted on all release years; `subject` with SWE-style relatives kept.
6. Control: `subject` with names permuted across subjects (must give Δ ≈ 0).

Secondary evaluation (power): leave-one-benchmark-out over every new binary benchmark with ≥5 eligible 2025–26
subjects and ≥80 items, budget 0 and budget 1 only, pair-level Brier of the pair-mean prediction with and without the
ability prior (no item model). Reported by domain.

## Decision table
| Outcome (primary R2 ΔALC vs base) | Action |
|---|---|
| `subject` or `subject+hyper` ≤ −0.002, CI excludes 0, ≥3 of 4 folds negative, no fold > +0.001, control ≈ 0, secondary agrees in sign | Build v1_8, red-team it, submit the zip five times and compare means with v1_3. |
| between −0.002 and −0.0005 | Keep as a documented option; adopt only if the secondary evaluation shows ≤ −0.003 at budget 0 on agentic/maths domains. |
| > −0.0005, or control also gains | Reject; record that cross-benchmark ability does not transfer; keep v1_3. |
| `hyper` alone ≤ −0.001 with the same conditions | Adopt the refit regardless of the subject arm. |
| Gain only with SWE-style relatives kept | Treat as leakage-like; do not ship that variant. |

Hold: no E7-based zip is selected as the final submission until the organisers confirm that the 281 benchmarks are
training pool.
(Resolved 2026-10-08 after the run, from the organisers' own tool and dataset card: `main` is the public training
release and 160 benchmarks pass its eligibility rule; see report 25 §8. The hold is lifted.)

## Expected result
`subject` −0.002 to −0.006 (mostly budgets 0–3); `hyper` within ±0.001; larger gains on matharena and SWE folds, where
model names overlap most.

## Cost
≈2.5 GB download to `data/` on E:, CPU only, a few hours. No submission.
