# Pre-registration — E1 and E2 (written 2026-10-01, before any E1/E2 result exists)

Format follows the spec's experiment template (`docs/spec/agents.md` §27): ID, why, hypothesis vs
competing hypothesis, current evidence, exact procedure, metrics, decision boundary for every
outcome, cost. Decision boundaries are fixed here and may not be moved after results arrive; any
deviation is reported as a deviation.

Shared conventions
- Champion: `submissions/v1_1` (predictor `src/paiec/models/v1_1.py`). Never modified by E1/E2.
- Folds: leave-one-benchmark-out over multi_swebench, real_webagents, researchcodebench, matharena
  (swe_rebench train-only). Folds grouped by content (MathArena duplicates) and paper (ResearchCodeBench),
  per the report-12 veto conditions. Benchmark names re-aliased in training.
- Regimes: **R2** (one subject per benchmark — what both Codabench draws looked like) is primary;
  R1 (realistic ≈1,000-item draw) secondary. Same seeds as report 17 (`experiments/runs/phase3_v1_1`).
- Δ is reported ×1e-3 ALC, negative = better, with a 95% CI over pairs AND the per-fold signs
  (only 4 folds, so "benchmark-level" means: how many of 4 folds agree).
- "ALC threshold" for adopting any component = **−1.0e-3 in R2 with the pair-CI excluding 0 and
  ≥ 3 of 4 folds negative**, and no fold worse by more than +1.0e-3.

---

## E1 — Where does the budget-0 / budget-1 loss come from?

**Why.** Budget 0 and budget 1 carry 30% of the ALC weight, and v1_1 is barely better than a coin at
budget 0 (B0 ≈ 0.246 vs 0.250). Two different defects can cause a poor B0, and they need opposite
fixes. We have only been able to guess which one dominates:
- *Offset error* (calibration-in-the-large): the model's average prediction for the hidden benchmark is
  wrong — it does not know how easy the benchmark is. Fix: a better benchmark base-rate prior (H5).
- *Missing resolution*: the average is right but the model cannot tell strong subjects or easy items
  apart before labels. Fix: a subject-metadata prior (dropped in Phase 2) or an item-difficulty prior.
Report 14 suggests offset dominates (unknown offset costs 0.031 Brier, subject features buy 0.005–0.008),
but that was measured with an oracle offset, not on v1_1's actual predictions.

**Hypothesis H-E1.** ≥ 50% of v1_1's excess budget-0 loss is offset error.
**Competing hypothesis.** Resolution (subject or item) is the larger share, so a metadata or item prior
should be re-tried.

**Exact procedure.**
1. Regenerate v1_1's predictions at B0 and B1 for every pair in R2 and R1 (same seeds as report 17).
2. For each pair compute four counterfactual predictors that differ from v1_1 in exactly one thing:
   - `oracle_offset`: v1_1 with the benchmark offset replaced by the true held-out benchmark mean logit
     (estimated on that benchmark's full data);
   - `oracle_subject`: v1_1 with the pair's subject ability set to its true value (pair logit accuracy
     minus benchmark mean, from full data);
   - `oracle_item`: v1_1 with each item's difficulty set to its full-data Rasch β (centred);
   - `metadata_prior`: v1_1 plus the report-14 metadata prior (release date clipped at training max,
     provider, size tokens), fit on training folds only.
3. Murphy decomposition of each predictor at B0 and B1 (10 equal-width bins, per fold, then mean):
   Brier = REL − RES + UNC. Also the pair-level split Brier = (p̄ − ȳ)² + rest, where (p̄ − ȳ)² is the
   calibration-in-the-large (offset) term.
4. Headroom shares: for X in {offset, subject, item}, share_X = (B0_v1_1 − B0_oracle_X) / (B0_v1_1 − floor),
   floor = Brier of the three-oracle predictor. Same at B1.

**Metrics.** B0, B1, ALC; REL, RES, UNC; offset term; headroom shares; per fold.

**Decision boundaries (fixed now).**
| Outcome | Interpretation | Action |
|---|---|---|
| share_offset ≥ 0.50 at B0 | H-E1 supported: the hidden base rate is the bottleneck | Close the metadata question for good; the next investment is a base-rate prior (H5, content or LLM forecast). |
| share_subject ≥ 0.20 at B0 **and** `metadata_prior` meets the ALC threshold | Subject resolution is recoverable from metadata | Add the metadata prior to a v1_2 candidate (red-team before shipping). |
| share_subject ≥ 0.20 but metadata fails the threshold | Subject differences matter but metadata cannot see them | Record as a limitation; no change. |
| share_item ≥ 0.20 | Item resolution matters at B0 | Raises the value of E2 (Jev/distilled prior). |
| none of the above | B0 loss is mostly irreducible noise | No change; stop optimising B0. |

**Expected result.** share_offset ≈ 0.6–0.8; metadata ΔALC 0 to −0.5e-3 (fails threshold).
**Cost.** < 1 day CPU, reuses cached episodes; no API calls. **Personas.** Calibration Purist (lead),
Bayesian Statistician, Mathematical Minimalist.

---

## E2 — Is Jev's difficulty signal real, and can a local model reproduce it?

**Why.** Adding Jev's item-difficulty score as a prior is the largest untapped gain we have measured
(−6.2e-3 ALC R1, −5.4e-3 R2; report 15), but Jev cannot be called at scoring time. Before spending days
on a local replacement we must rule out that Jev's signal is **memorisation**: if Jev recognises public
problems and recalls how hard models found them, its ρ would vanish on genuinely new hidden benchmarks,
and a distilled copy would be worthless. The literature (report 18, D5) flags contamination as a live risk.

### E2a — Memorisation tests

**Hypothesis H-mem (null we try to reject).** Jev's difficulty–Rasch correlation comes from recall of
published items. **Competing hypothesis.** It reflects reading the problem (content-based judgement).

Two independent tests:
1. *Release-date split (MathArena only — the one benchmark with dated items).* Map each competition to
   its public release month (e.g. aime_2025 → 2025-02, imo_2025 → 2025-07, putnam_2025 → 2025-12,
   arxivmath_0126 → 2026-01, aime_2026/hmmt_feb_2026 → 2026-02; competitions without a clear date are
   excluded in a sensitivity run). Compute Spearman(jev_difficulty, Rasch β) separately for items released
   **before 2025-07** and **from 2025-07 on**, with bootstrap 95% CIs (resampling items, duplicate content
   grouped), and the difference.
2. *Surface-perturbation test (all benchmarks).* Sample 300 items (stratified by benchmark). Create a
   perturbed copy that preserves the task but breaks verbatim recall: rename identifiers and entities,
   change numeric constants where the answer does not depend on them, reorder independent clauses (rules
   written as code, deterministic, logged). Re-score originals and copies with Jev (same pinned model).
   Measure (a) mean |Δ difficulty| and (b) Spearman with β for originals vs copies on the same items.

**Decision boundaries.**
| Outcome | Interpretation | Action |
|---|---|---|
| post-2025-07 ρ ≥ 0.15 with CI excluding 0 **and** perturbed-copy ρ ≥ 0.8 × original ρ | Signal is content-based | Proceed to E2b/E2c. |
| post ρ CI includes 0 **or** (pre − post) ≥ 0.15 with CI excluding 0 **or** perturbed ρ < 0.6 × original | Memorisation explains much of the signal | **Stop E2.** Record Jev's prior as unusable for unseen benchmarks; remove L5 from future plans. |
| anything in between | Partial memorisation | Proceed to E2c but evaluate the student only on post-2025-07 and perturbed items. |

### E2b — Pairwise (Bradley–Terry) Jev scoring

**Why.** Pointwise difficulty scores compress to a 5-level scale; pairwise "which is harder?" judgements
are often more reliable (report 18: Ballon 2025). **Procedure.** Within each benchmark, sample 8 random
partners per item, ask Jev a Choice question "which task is harder for a frontier model", fit
Bradley–Terry scores. **Decision.** Use BT scores as the distillation target iff their LOBO ρ with β
exceeds pointwise ρ by ≥ 0.05 on ≥ 3 of 5 benchmarks; otherwise keep pointwise. Cost ≈ 40k pairs ×
~1k tokens ≈ $2.

### E2c — Distil Jev into a local model that can run at scoring time

**Procedure.** Student = frozen small sentence encoder declared in `models.txt` (start with
`sentence-transformers/all-MiniLM-L6-v2`, 22M params; one larger fallback only if the small one fails),
plus a ridge head mapping embeddings → Jev score, standardised within benchmark at prediction time
(label-free). Train the head per LOBO fold on **training-fold items only**. Model cache goes to E:
(`HF_HOME` on E:), never C:. Evaluate:
1. Student–Jev agreement on held-out benchmarks (Spearman).
2. Student ρ with held-out β (must retain ≥ 70% of Jev's own ρ on that fold).
3. End-to-end: v1_1 + student prior (β prior = slope × standardised student score, slope fit on
   training folds) vs v1_1, in R2 and R1.
4. Runtime: per-item embedding time on CPU and package size; must fit the 8-hour limit at the
   ≈1,000-item draw with margin (target < 50 ms/item) and the platform's model-packaging rules.

**Decision boundaries.**
| Outcome | Action |
|---|---|
| Student meets the ALC threshold in R2, retains ≥ 70% of Jev's ρ, runtime OK | Build `submissions/v1_2` = v1_1 + student prior; red-team (A3, A4, A7) before any upload. |
| Student retains ≥ 70% of Jev's ρ but misses the ALC threshold | Keep as research result; no ship. |
| Student retains < 70% | Distillation failed with this encoder; one fallback encoder, then stop. |

**Expected result.** E2a passes with partial memorisation (post ρ ≈ 0.15–0.25); student retains
60–80% of Jev's ρ; end-to-end −1 to −3e-3 in R2 if it ships.
**Cost.** E2a ≈ ½ day + < $1 API; E2b ≈ ½ day + ≈ $2; E2c 1–2 days CPU + one ≈ 90 MB model download
(to E:). **Personas.** Representation Scientist (lead), Statistical Leakage Hunter (memorisation),
ML Systems Architect (packaging), Mathematical Minimalist.
