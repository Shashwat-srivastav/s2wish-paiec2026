# Pre-registration — E4: item-group facet random effect (P1) + degenerate-key guard

Written 2026-10-02 **before any E4 result exists** (Panel A's scratch prototype numbers, report 21 §3,
are the only prior evidence and motivated the test; they are not re-used). Decision boundaries below
are fixed now; any deviation is reported as a deviation in report 23.

Personas: Measurement Theorist (facet model), Bayesian Statistician (π-averaging, marginal
likelihoods, guard), Mathematical Minimalist (veto: exact fallback, ≤ 1 new free parameter per
component), Cold-Start Specialist (veto: B0 must be untouched, hidden-benchmark generality).

Code: `src/paiec/models/v1_2.py` (extends `paiec.models.v1_1`; v1_1 is not edited).
Outputs: `experiments/runs/e4/`. Report: `docs/research_notes/23_E4_FACETS.md`.

## Shared conventions
- Champion / reference: v1_1 = `v1_1.BASE2` (prior t3(G/2, learned), ρ0 = 0.05 momd, item level on).
- LOBO folds: multi_swebench, real_webagents, researchcodebench, matharena (swe_rebench train-only;
  it has no item_features). Hyperparameters re-fit per fold (`v1_1.H_fold`).
- Regimes: **R2 primary** (one pair per episode; what both Codabench draws looked like), seeds 0..9
  (10 seeds, ≥ 5 as required). **R1 secondary**, draws 0..99 (report-17 protocol).
- Acquisition: accept-all random stream (v1_1 default). Δ = (candidate − v1_1) ALC ×1e-3, negative =
  better, pair-averaged within fold, fold means averaged with equal weight.
- Statistics reported for every Δ: per-fold Δ; per-budget Δ; fold-stratified pair bootstrap CI;
  hierarchical (benchmark→pair) bootstrap CI; **t3 interval on the 4 fold means**
  (mean ± t_{.975,3}·sd/√4); **new-benchmark predictive interval** (mean ± t_{.975,3}·sd·√(1+1/4)).

## (a) P1 — item-group facet random effect

**Why.** Report 21 §2.1–2.2: on the leaderboard 7/8 pairs were alone on their benchmark, so v1_1's
cross-subject item pooling is inactive and it predicts a constant per pair. `item_features`
(runtime-visible) carries group fields (competition / lang / website / paper, plus problem_idx in
matharena) with between-group shares of item-difficulty variance 0.35 / 0.02 / 0.23 / 0.38. A facet
effect learnt from the subject's *own* revealed labels works with one subject per benchmark.

**Hypothesis H-P1.** Adding a centred, π-averaged facet random effect improves R2 ALC by ≥ 0.5e-3 vs
v1_1, with the gain at B3–B31 and none at B0.
**Competing hypotheses.** (i) Groups are too small at budgets ≤ 31 for the effect to be learnt; Δ ≈ 0.
(ii) Any gain is a mechanism artefact (extra variance / centring changes calibration), not group
information — then the **shuffled-facet** control shows a similar Δ.

**Model.** For target item t of pair n with facet values v_f(t):
logit p = c_n + Σ_f γ_f[v_f(t)] − s_{n,f} + β_res,
γ_f[v] ~ N(0, π σβ²), β_res ~ N(0, (1−π) σβ²) (total item variance σβ² unchanged), c_n = v1_1's
pair-level logit (`_invert_c(q_n, σβ)`). γ posterior per (pair, field, value) cell from the pair's own
labels only (20-node Gauss–Hermite, β_res integrated). Centring: s_{n,f} = label-count-weighted mean
of the posterior-mean γ over the pair's observed cells of field f (so facets redistribute, not shift,
the pair mean). Per-field contributions add on the logit scale. π ∈ {0 (implemented 1e-3), 0.1, 0.3}
with prior weights 0.4/0.4/0.2, combined by posterior weight ∝ prior × marginal likelihood of the
pair's labels on facet cells (per pair). Fields: **every** `key=value` field in item_features
(generic rule, no per-benchmark selection). Exact fallback: target with no features, no labels, no
labelled cell sharing a value, or whose key is itself labelled (item level handles it) → v1_1's
prediction bit-for-bit. B0 is therefore identical to v1_1 by construction.
Free parameters: none fitted (π grid and weights fixed above, from Panel A's proposal).

**Shuffled-facet control.** Same code, but the key→feature-dict map is randomly permuted across keys
within the held-out benchmark (seed 0), preserving group sizes and destroying the item↔group link.

**Secondary (reported, not used for the decision):** named fields only (competition/lang/website/paper,
i.e. matharena problem_idx and prompt_source dropped).

## (b) Degenerate-key guard

**Why.** v1_1's merge prior ρ comes from the momd estimator, which needs cross-subject collisions.
In R2 there are none, so ρ = ρ0 = 0.05 and two same-subject labels on one key are treated as the
same item with prior 0.95. On a content-poor hidden benchmark (many item_ids → one runtime dict)
that is wrong (report 21 plan #3: up to −0.04 B1).

**Rule (primary, G1).** For a target key with **no labels from another subject** in its pooling group,
use ρ_k = max(ρ_k, 0.5) (the shared-vs-independent likelihood ratio still decides).
**Secondary (G2).** If the target key has < 2 cross-subject labels, ignore same-key labels (p = q).
**Hypothesis.** G1 costs ≤ +0.2e-3 on real data (matharena genuine duplicates still agree, so the LR
keeps them) and gains ≥ 0.5e-3 R2 ALC on synthetic random merges (`redteam.rekey`, m = 5 and 20).
**Competing.** Matharena duplicates (14 ids on one key) lose their item-level gain → real regression.

## Ablation arms (all on real data, R2 and R1)
`v1_1`, `+facets`, `+guard` (G1), `+both`, `shuffled-facets`, secondary: `+facets_named`, `+guard2` (G2).
Synthetic merges (R2 and R1, m = 5, 20): `v1_1`, `+guard`, `+guard2`.

## Implementation checks (must pass before any number is read)
1. facets=off, guard=off reproduces `v1_1.predict(…, BASE2)` to < 1e-12 on all folds/budgets.
2. B0 Δ of every facet arm is exactly 0.
3. With facets on and the feature map emptied, predictions equal v1_1 to < 1e-12.

## Decision table (fixed now)

| Component | Adopt if ALL hold | Otherwise |
|---|---|---|
| facets | R2 Δ ≤ −0.5e-3; no R2 fold > +0.5e-3; shuffled-facet R2 |Δ| < 0.2e-3; R1 Δ ≤ 0 and no R1 fold > +0.5e-3 | DROP |
| guard (G1) | real R2 and R1 Δ ≤ +0.2e-3; matharena (any regime) ≤ +0.5e-3; synthetic m=5 or m=20 R2 Δ ≤ −0.5e-3 | DROP (or G2 if G2 meets the same row; G2 is secondary so it is flagged as a post-hoc choice) |
| both | ship the union of components that pass; check `+both` vs sum of singles within 0.3e-3 (no harmful interaction) and `+both` itself passes the facets row's fold guard | ship passing single |

**Evidence grade (reported, not gating):** "verified" if the t3 fold-mean interval upper bound < 0;
otherwise "adopted on pre-registered rule, unverified at 4 folds" — with only 4 folds an MDE of
0.7–3.5e-3 means a −0.5…−1e-3 effect is not verifiable (report 21 §2.4).

## Kill tests
- Shuffled-facet |Δ| ≥ 0.2e-3 → facets killed regardless of the real Δ (artefact).
- Any facet arm B0 Δ ≠ 0 or check 1/3 fails → bug, fix before reading results.
- Guard: matharena regression > +0.5e-3 → guard killed.

## Cost
≈ 2 h compute budget. If a run is slow, R1 draws are reduced (reported as a deviation).
