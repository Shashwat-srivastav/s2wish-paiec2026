> **Historical (written 2026-09-17, before the training data was available).** Kept for the record; facts about data, baselines and results are superseded by reports 10–20 and `docs/publication/research_report.html`.

# Formal Statistical Formulation

## Outcome
For subject `s` and item `i` (with trial/repetition index `r` where applicable):

`Y_{s,i,r} ∈ {0,1}` — observed binary response (1 = correct/success).

Repeated runs are separate draws; they may share `P` but score separately.

## Prediction target
Estimate:

`p_{s,i} = P(Y_{s,i} = 1 | X_s, X_i, E)`

where:
- `X_s` = subject attributes (8 strings): normalized_name, provider, release_date, access_date, harness, harness_version, reasoning_effort, subject_features_extra.
- `X_i` = item attributes (4 strings): item_content, item_features, interactors, benchmark_id (anonymous, stable within pair).
- `E` = revealed evidence at current budget: nested sets `E_0 ⊆ E_1 ⊆ E_3 ⊆ E_7 ⊆ E_15 ⊆ E_31`, each a list of `[[X_s', X_i'], y]` with `y ∈ {0,1}` from acquisition pool only. `|E_n| = n` per subject–benchmark pair (per-pair budgets). `E_0 = ∅`.

No other inputs. In particular: no grading criterion/verifier/reference answer, no hidden-test outcomes, no test-pool benchmark names, no cross-budget memory beyond `E`.

## Decision-theoretic framing
- Loss per response: squared error `(p − y)²`. Bayes-optimal prediction = true conditional probability. Brier rewards both discrimination and calibration.
- Budgeted sequential design: acquisition policy `π` maps `(candidate, prediction, E, context)` → {reveal, skip}; predictor `f` maps `(X_s, X_i, E)` → p. Joint objective = expected Brier ALC over hidden subject–benchmark pairs, equal-weighted.
- Cold-start hierarchy: test benchmarks unseen ⇒ `benchmark_id` is a *grouping* key, not a memorizable effect; generalization must come from item/benchmark features, subject features/history, and label adaptation.

## Hierarchical data-generating view (working model)
```
pair mean θ_{s,b} ~ prior(subject family/provider/generation, benchmark family/domain)
item difficulty δ_i | benchmark b
subject ability α_s | family/provider
interaction η_{s,i} (low-rank / IRT discrimination / semantic)
logit P(Y=1) = μ + α_s − δ_i + η_{s,i} (+ harness/reasoning effects)
Y ~ Bernoulli(logit⁻¹(...)), repeated trials conditionally independent given (s,i)
E_n = n acquired labels from acquisition half (random or π-selected)
```
Predictor approximates posterior predictive mean `E[P(Y=1|·) | X_s, X_i, E_n]` under sparse `E_n`.

## What the metric implies
- `B_n` per pair → ALC per pair → mean over pairs. Optimize *pair-level* learning curves, not global pooled Brier.
- Weights (0.1,0.2,0.2,0.2,0.2,0.1): half the mass on budgets 0/1/3. Low-budget priors + 1-shot updating dominate. Do not optimize only B_31.
- Equal pair weighting ⇒ shrinkage toward hierarchical priors; variance control on small pairs; per-pair calibration.

## Baselines in this notation
- Constant: `p = 0.5` ∀ (s,i,E). Brier 0.25 flat, ALC 0.25.
- Empirical mean: `p = mean{y : (s',b')=(s,b) in E}` else 0.5. Pair-constant, unsmoothed, no item discrimination.
- BLE: LLM-agent posterior `p = LLM(X_s, X_i, E_same-subject, retrieved public evidence)` in [0.02,0.98], up to 9 retrievals, medium reasoning; acquisition = |p−0.5|≤0.15.
