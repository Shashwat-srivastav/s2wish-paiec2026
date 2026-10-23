# E6 — Headroom: what would have to be true to score 0.15, 0.12 or 0.10?

Pre-registered 2026-10-08, before any run. Personas: Bayesian Statistician, Measurement Theorist, Competition Optimizer
(run); Red-Team Researcher (checks the oracles do not leak more than stated).

> **Scope narrowed 2026-10-08, before any result was seen.** Report 19 (E1) already measured the offset, subject and
> item oracles and the three-oracle floor (R2 mean ≈0.0925). Arms 2, 4 and 5 and `item_oracle(1.0)` are therefore not
> re-run; their numbers are quoted from E1. The only new measurement is `item_oracle(ρ)` for ρ ∈ {0.2, 0.3, 0.5, 0.7},
> built from E1's leave-pair-out item difficulty, with one ρ=1 run as a harness check against E1. The `pair_oracle`
> and `both(1.0)` rows of the decision table are read from E1's numbers.

## Why (the decision it unblocks)
The board's top cluster sits at 0.148–0.163 and #1 at 0.117; our multi-run mean is ≈0.19. The 10-08 parameter sweep
moved ALC by ≤0.0002. Before spending the remaining three weeks on a new method class (runtime LLM / HF model for item
difficulty, or a better label-free pair prior), we need to know which kind of information buys how much ALC, and
whether 0.10 is reachable at all on data like ours.

## Hypotheses
- H-item: label-free item-difficulty information of realistic quality (Spearman ρ≈0.3 with the benchmark's true item
  difficulty, the level Jev reached in report 13) lowers ALC by ≥0.02.
- H-pair: knowing each pair's base rate at budget 0 (a perfect benchmark × subject prior) lowers ALC by ≥0.02.
- Competing: neither does; the remaining loss is response noise and 0.15 needs near-perfect item knowledge.

## Current evidence
- Report 19 (E1): item difficulty is 48–61% of closable B0/B1 loss; pair-constant oracle gap ≈0.023 (report 21).
- Formative run 967620: two of nine pairs stay at ≈0.25 at every budget; three pairs are worse than 0.5-constant at B0.

## Procedure
Engine `src/paiec/evaluation/episodes.py`, base model `v1_2` with `facets="named", guard=0` (the shipped v1_3 config).
Leave-one-benchmark-out over `multi_swebench`, `real_webagents`, `researchcodebench`, `matharena`; regime R2 (one
subject per benchmark, as on Codabench); seeds 0–9; budgets 0, 1, 3, 7, 15, 31; ALC as in the repo. Same episodes for
every arm (paired).

Arms:
1. `base` — the shipped model.
2. `pair_oracle` — the pair's true base rate on the evaluation pool replaces the model's pair-level mean at every
   budget (item structure left as the model has it).
3. `item_oracle(ρ)` for ρ ∈ {0.2, 0.3, 0.5, 0.7, 1.0} — a per-item prior offset equal to the item's true difficulty
   (logit of its mean correctness over all *other* subjects in the held-out benchmark), mixed with Gaussian noise to the
   target Spearman ρ, standardised within benchmark and scaled by the within-benchmark sd; the subject's own responses
   are never used.
4. `both(ρ)` — arms 2 and 3 together, for ρ ∈ {0.3, 1.0}.
5. `floor` — per pair, the in-sample Rasch-style best constant-per-item prediction using the target subject's own
   outcomes (reported only as the noise floor).

Report mean ALC, per-budget Brier, per-benchmark ALC, and paired Δ vs `base` with a bootstrap-by-pair 95% CI.

## Decision table
| Outcome | Action |
|---|---|
| `item_oracle(0.3)` Δ ≤ −0.02 | Item route is the priority: build a runtime item-difficulty scorer (HF model or allowlisted API) and pre-register it as E7. |
| `item_oracle(0.3)` Δ between −0.02 and −0.005, `item_oracle(0.7)` ≤ −0.03 | Item route only pays with a strong scorer: test scorer quality offline first (ρ target ≥0.5) before any packaging. |
| `pair_oracle` Δ ≤ −0.02 | Pair-prior route: pre-register a label-free benchmark × subject prior (LLM-estimated benchmark difficulty, subject strength from release date / name). |
| Both item(0.3) and pair Δ > −0.005 | Stop method work on these lines; remaining effort goes to robustness and the report. |
| `both(1.0)` > 0.10 | State plainly that 0.10 is not reachable as a mean on public-like benchmarks; set the target from `both(0.3)`. |

## Expected result
Pair oracle ≈ −0.02 (report 21); item oracle at ρ=0.3 small (≈ −0.005); `both(1.0)` around 0.10–0.13.

## Cost
CPU only, minutes to an hour. No submission. New files only: `src/paiec/analysis/e6_headroom.py`,
`experiments/runs/e6/`.

## Limits
Oracles are upper bounds, not methods. Public benchmarks are harder than the hidden draw so far (local R2 ≈0.209 vs
formative ≈0.19), so absolute numbers will sit higher than on Codabench; the deltas are what transfer.
