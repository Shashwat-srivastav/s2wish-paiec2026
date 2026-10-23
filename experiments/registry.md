# Experiment Registry (append-only)

Every run gets `EXP-####`. Never overwrite. Fields: ID, hypothesis, motivation, dataset version (HF revision + eligible dirs), split (random/benchmark/subject/subject×benchmark/temporal/family/provider/harness + seed), features, model, hyperparams, seed, label budget(s), acquisition policy, Brier per budget, ALC, ECE, runtime, memory, result, CI, interpretation, decision, next.

## Log
| ID | Date | Hypothesis | Split | Model | ALC | Decision |
|---|---|---|---|---|---|---|
| EXP-0001 | 2026-09-17 | Metric re-implementation matches official ALC | n/a (unit) | metric.py | 0.140 illustrative | PASS → use for all eval |
| EXP-0002 | 2026-09-17 | Official baselines build + pass tests locally | synthetic/mock | mean + BLE smoke | n/a (no hidden labels) | PASS → proceed to internal simulator |
| EXP-0003 | 2026-10-01 | Vectorised episode engine == packaged model via official client | LOBO, rwa/rcb | v1 | max diff 2e-16 | PASS → use engine |
| EXP-0004 | 2026-10-01 | Ladder on realistic ~1000-item draw (R1) | LOBO 4 folds × 40 draws | L0..L6, L2i | L2i .2119 (v0 .2177) | ADVANCE L2i; drop name/meta/T |
| EXP-0005 | 2026-10-01 | Ladder, 1 pair per benchmark (R2) | LOBO × 5 seeds | L0..L6 | L2i .2090 (v0 .2105) | USE |
| EXP-0006 | 2026-10-01 | Ladder, all pairs (R3) | LOBO × 5 seeds | L0..L6 | L2i .1855 (v0 .2105) | upper bound |
| EXP-0007 | 2026-10-01 | Temperature toward 0.5 | inner-LOBO | +T | Δ ≈ −0.4e-3 n.s. | DROP |
| EXP-0008 | 2026-10-01 | v1 packaging | smoke/zip | v1 (L2i) | 0.3–3.2 ms/call | freeze candidate |
| EXP-0009 | 2026-10-01 | Red team A1: R1 gain not fold leakage | LOBO × 40/200 draws | v1 | v1−v0 −5.65 (40) / −3.81 (200) | PASS; restate headline |
| EXP-0010 | 2026-10-01 | Red team A2/A3: draw regime & base-rate shift | 8 regimes; δ −3..+3 | v1 | δ+3: +6.7 (withB), +12.4 (R2) | WARN easy benchmarks |
| EXP-0011 | 2026-10-01 | Item pooling (L2i vs L2), Jev disagreement | R1/R2/harsh/2-subject/false merges | v1 vs L2 | R1 −2.76, hier CI excl. 0; never >0 at B1/B3 | KEEP + false-merge guard |
| EXP-0012 | 2026-10-01 | Reproduce formative failures (A9) | R2 MC 8-pair | v1 | P(B1>B0)=0.20, P(bump)=0.25 | sampling noise; no fix |
| EXP-0013 | 2026-10-01 | Red team A4/A6/A7: state, calibration, runtime | packaged model | v1 | order diff 3e-16; 0 failures | PASS (WARN) |
| EXP-0014 | 2026-10-01 | Easy-benchmark prior repair (A3) | R2/R1 + δ sweep | prior variants | G/2+learned: max regret R2 12.6→7.7, real Δ −0.1/−0.3 | KEEP (b) |
| EXP-0015 | 2026-10-01 | False-merge guard (A5) | R1/R3 real + merges m=2/5/20 | ρ fixed/EM/mom/momd | momd: real +0.03; m20 −7.3/−10.1 | KEEP momd |
| EXP-0016 | 2026-10-01 | Pair overdispersion (A6) | R2/R1 | w_od, τ scale | w.85 −0.09/−0.32, tail unchanged | DROP |
| EXP-0017 | 2026-10-01 | Acquisition factorial (Phase 3) | R1×100, R2 | {v1,v1_1}×{A1,A2,A3,A5,A8} | best −0.27 (A2), R2 null | DROP (accept-all) |
| EXP-0018 | 2026-10-01 | Runtime cache (A7) | timing | v1_1 | ≤372 labels → ~4 ms/call | DROP |
| EXP-0019 | 2026-10-01 | v1_1 packaging + A4/A7 | kit tools, run_direct | v1_1 | zip/smoke OK; diff ≤2e-16 | replace v1 |
| EXP-0020 | 2026-10-01 | E1: B0 loss is ≥50% offset (H-E1) | R2×5 / R1×100, LOBO | v1_1 + oracles | B0 shares R2 off .19 subj .21 item .48 (R1 .12/.12/.54) | H-E1 REJECTED; item row → raise E2 |
| EXP-0021 | 2026-10-01 | E1: metadata prior on v1_1 | R2×5 / R1×100, LOBO | v1_1 + meta θ prior | R2 −0.58 [−1.00,−0.17] 4/4; R1 −0.91 | below −1.0 bar → limitation, no change |
| EXP-0022 | 2026-10-01 | E2a: Jev memorisation (release split + perturbation) | MathArena pre/post 2025-07; 300 perturbed | Jev pointwise | post ρ .455 [.35,.55]; pert/orig 1.03 | PROCEED (content-based) |
| EXP-0023 | 2026-10-01 | E2b: pairwise BT Jev | 39,164 pairs | BT vs pointwise | +≥.05 on 2/5 | KEEP pointwise |
| EXP-0024 | 2026-10-02 | E2c: MiniLM-L6 distillation | LOBO; R2×5, R1×100 | v1_1+student | retention −0.17; ΔR2 +0.18/+0.58 | FALLBACK |
| EXP-0025 | 2026-10-02 | E2c: mpnet-base fallback | as above | v1_1+student | retention 0.025; ΔR2 +0.01/+0.13 | STOP E2; no v1_2 |
| EXP-0026 | 2026-10-01 | Local simulator reproduces the official run_streaming exactly | synthetic | simulator.py vs official client | 9/9 tests; max |dB|=0.0 | use simulator for all evaluation (retro-registered) |
| EXP-0027 | 2026-10-01 | Beta-binomial shrinkage beats the empirical mean at low budgets | synthetic | v0 vs empirical mean | ALC .2199 vs .2570 (Beta(1,1)) | ship v0; keep random acquisition (retro-registered) |
| EXP-0028 | 2026-10-01 | Jev ranks item difficulty sensibly | n/a | Jev jev-1.13.0 | ordering sensible; $0.00004/item | offline use only (retro-registered) |
| EXP-0029 | 2026-10-01 | Variance/shrinkage decomposition of predictable signal | within-bench CV + LOBO | Rasch / hierarchical EB | item var > subject var; prior worth ~2.8 labels; LOBO B0 0.234 vs 0.25 | build pooled hierarchy (retro-registered) |
| EXP-0030 | 2026-10-01 | No runtime-visible field leaks the outcome | LOBO | per-field probes | no leak; dup groups; item_id folds inflate 0.014 | content/paper-grouped folds (retro-registered) |
| EXP-0031 | 2026-10-01 | Item content representations transfer difficulty across benchmarks | LOBO 5 folds | ridge on triv/TF-IDF/emb/Jev | only Jev transfers (rho .19-.34) | Jev prior offline only (retro-registered) |
| EXP-0032 | 2026-10-01 | Subject metadata predicts ability on unseen benchmarks | LOBO + leave-family-out | ridge on subject fields | R2 0.16; B0 gain 0.005-0.008 | exact-dict key; drop family (retro-registered) |
| _next_ | — | E01–E20 (see `FIRST_20_EXPERIMENTS.md`) | benchmark-holdout primary | — | — | — |
| EXP-0033 | 2026-10-02 | Item-group (facet) difficulty from revealed labels | LOBO 4 folds R2/R1 | v1_2 facets | R2 −0.91e-3, R1 −0.76e-3, shuffled +0.15 | ADOPT (report 23) |
| EXP-0034 | 2026-10-02 | Degenerate-key guard | LOBO + synthetic merges | v1_2 guard | real +0.09e-3; synthetic −2.2…−5.8e-3 | ADOPT as insurance (report 23) |
| EXP-0035 | 2026-10-02 | Robust offset prior X1 | LOBO + δ-sweep + extremes | x1_prior | none passes mean + regret + B1 rules | REJECT, stop prior-shape line (report 24) |
| EXP-0036 | 2026-10-08 | E6: an item-difficulty signal of realistic quality (ρ 0.3) buys ≥0.02 ALC | LOBO 4 folds R2, seeds 0–9 | v1_2 named/no-guard + degraded item oracle | ρ .2/.3/.5/.7: −2.3/−5.7/−17.6/−36.1e-3 | item route only with a scorer at ρ ≥ 0.5 shown offline; next E7 enlarged pool (report 25) |
| EXP-0037 | 2026-10-08 | E7: priors from the enlarged pool (281 benchmarks) lower ALC by ≥2e-3 | LOBO 4 folds R2, seeds 0–9; secondary 47 new benchmarks | v1_2 named/no-guard + name ability prior / refit hypers | subject −1.02e-3 [−1.54, −0.52]; hyper +3.55e-3; control +0.22 | below bar → documented option only; hyper REJECTED; keep v1_3 (report 26) |

## Submission log (Codabench formative — each submission = feedback event)
| Submission | Date | Hypothesis changed | Internal ALC Δ | Formative ALC | Notes |
|---|---|---|---|---|---|
| v1 (`submissions/dist/v1.zip`) | 2026-10-01 | pooled offset + cross-pair item pooling | local R1 0.2119 | **0.2157** (8 pairs, 7 benchmarks) | B1 > B0 on this draw; see experiments/platform_submissions/2026-10-01_v1.md |
| v1_1 (`submissions/dist/v1_1.zip`) | 2026-10-01 | prior centre to G/2 + false-merge guard (risk reduction) | local Δ vs v1 ≈ −0.0003 | **0.1761** (8 pairs; 7/8 subjects differ from v1's draw) | gap not attributable to v1_1 (Jev: noise p=0.94); see experiments/platform_submissions/2026-10-01_v1_1.md |
| v1_1 repeats | 2026-10-02 | same zip, 4 more fresh draws | — | runs 0.1761, 0.1812, 0.2046, 0.2030, 0.1963 → **mean 0.1923** (95% CI 0.176–0.208) | the board shows the best draw; see experiments/platform_submissions/2026-10-02_v1_1_repeats.json |
| v1_3 named facets, no guard (`submissions/dist/v1_3_named_facets_noguard.zip`) | 2026-10-07 | v1_2 with `facets="named"`, `guard=0` (params only; code identical) | local R2 ≈ −0.1e-3 vs v1_2 settings (not pre-registered) | **0.1663** (one draw, id 967620) | retro-logged 2026-10-08; draw noise, not attributable to v1_3; see experiments/platform_submissions/2026-10-07_v1_3_named_facets_noguard.md |

## Rules
- Frozen internal test set: benchmark-holdout split fixed at data landing (seed 0) + temporal split.
- EXP-XXXX: organiser question on an explored variant (not selected).
- Promote only: multi-seed + benchmark/subject/temporal holdouts + independent re-implementation (§49).
- CIs: bootstrap by pair (not by response) to respect hierarchy; paired comparisons across methods on same splits.
- `registry.csv` rows must have exactly the 26 header fields (empty = n/a). 2026-10-02 correction: EXP-0002 was short 4 fields (brier_alc, ece, runtime_s, memory_mb); padded empty, values unchanged.
