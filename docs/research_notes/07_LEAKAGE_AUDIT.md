> **Historical (written 2026-09-17, before the training data was available).** Kept for the record; facts about data, baselines and results are superseded by reports 10–20 and `docs/publication/research_report.html`.

# Leakage Audit Plan + Regime Determination

## Generalization regime (from spec — authoritative)
**Benchmark-level cold start.** Test items are drawn from benchmarks assigned to the private test pool (random assignment before curation). Subjects are platform-selected (may overlap training subjects or not; shared-subject illustration is explicitly "illustrative; actual coverage may differ"). Per-pair 50/50 acquisition/eval split; acquisition items never scored; eval outcomes never revealed; nested budgets; equal pair weighting.

This is **NOT** a random-split regime. A method that only wins on random splits proves nothing. Required holdouts:
- random (sanity/upper bound only)
- **benchmark holdout (primary)** — entire benchmarks unseen
- subject holdout, subject×benchmark holdout
- model-family / provider / harness holdouts
- temporal holdout (train past → test future by release_date/access_date)
- item holdout within benchmark (secondary; tests item discrimination given benchmark exposure)

Promote only methods that survive benchmark + subject + temporal holdouts across seeds.

## Leakage tests (run once gated data lands; code stubs in `src/leakage_checks.py`)
1. **Item leakage**: exact `content` / `content_hash` duplicates across train/test; near-dup via normalized edit distance + embedding cosine ≥0.98. Verdict: any test item with train dup must be flagged; report Brier with/without dups.
2. **Benchmark leakage**: `benchmark_id` overlap train↔test (must be zero by design); benchmark-name/domain leakage via `item_features` text (e.g. benchmark name in features) — scan for benchmark tokens in content/features.
3. **Subject leakage**: `subject_id`/normalized_name overlap stats; same base model under different harness counted as distinct — verify no accidental pooling across harnesses in our code (exact-dict matching).
4. **Response duplication**: duplicate `(subject,item,trial,test_condition,interactors)` keys; repeated-trial handling (reuse prediction, score each trial).
5. **Semantic duplicates**: embedding NN across pools; cluster-based audit; if test item has train NN with same subject outcome, measure NN-predictability (is it transfer or leakage?).
6. **Temporal leakage**: release_date/access_date vs benchmark release_date; train-future→test-past forbidden in temporal holdout; check `access_date` leakage (measurement occasion carrying performance signal).
7. **Release-date leakage**: does release_date alone predict success (generation trend)? If yes, it is legitimate metadata signal, but must generalize forward (rolling-window validation), not memorize dates.
8. **Benchmark-ID leakage**: anonymous ID must be grouping-only. Test: shuffling IDs within pool should not change cold-start predictions except via grouping; any model using ID as categorical effect on unseen IDs is broken by construction.
9. **Hidden-evaluation leakage**: never use formative `subject_id/benchmark_id` mappings, per-pair Brier tables, or acquired-outcome patterns to alter training (no fitting to formative feedback; frozen internal test set; log every Codabench submission per §47).
10. **Target-encoding leakage**: all target encodings (benchmark means, subject means, pair means) must be computed **inside** training folds only, with smoothing + holdout validation; no pooling of eval labels into priors.
11. **Normalization leakage**: scalers/embeddings/IDFs fit on train only; content_sha1/hash checks are for dedup, never label lookup.
12. **Feature-extraction leakage**: text embeddings precomputed offline are fine (content-only), but any supervised fine-tuning must respect pool splits; `grading_criterion/verifier/reference_answer` must never enter features (they are not available at runtime).
13. **Acquisition leakage**: acquisition decisions may use only current candidate + previously acquired labels + context; never future candidates' labels, never eval-pool labels, never cross-pair labels (subject+benchmark match). Simulate streaming order exactly.
14. **Public-leaderboard overfitting**: formative resampling limits per-submission signal, but repeated submissions still adapt. Mitigations: frozen internal benchmark-holdout suite; pre-registered hypotheses; submission log (count, hypothesis changed, score Δ, validation Δ); promote only with internal significance + multi-seed stability.

## Decision rules (from prompt §49, adopted)
- Random-split win + benchmark-holdout fail → **reject** as generalization solution.
- Strong at 31, weak at 0–7 → **reject** as low-budget evaluator.
- Acquisition helps selected item only, not eval items → **reject** as information-efficient.
- ECE↓ but Brier↑ → do not claim calibration win.
- Internal win + hidden loss → investigate shift before adding complexity.
- IRT wins over neural → keep IRT (simplicity is a virtue).
- Statistical tie → simpler/reproducible wins.
- Promote only after: multi-seed + benchmark/subject/temporal holdouts + independent re-implementation.
