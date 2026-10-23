> **Historical (written 2026-09-17, before the training data was available).** Kept for the record; facts about data, baselines and results are superseded by reports 10–20 and `docs/publication/research_report.html`.

# First 20 Experiments (prioritized, high information gain)
**Order = run order. Do not run sweeps before E01–E20 are reviewed. Primary split = benchmark holdout; secondary = subject / temporal. Metric = Brier per budget + ALC (equal pair weighting) + ECE diagnostic. CIs by pair-bootstrap, paired comparisons, ≥3 seeds for promoted claims.**

Conventions: Control = what we compare against; Treatment = single change; Falsification = what kills the hypothesis; Compute = CPU-only unless noted (GPU only for E08-embeddings/E10-neural/E20-fusion if justified).

---
## E01 — Official baseline reproduction (internal simulator)
- Hypothesis: Empirical mean + random acquisition yields ALC < 0.25 internally and matches streaming semantics (50/50 split, nested budgets, exact subject+benchmark matching).
- Control: constant 0.5 (ALC 0.25 flat).
- Treatment: official `empirical_mean` + simulated random acquisition.
- Split: benchmark holdout (leave-1-or-2-benchmarks-out, all 5 eligible) + random sanity; per-pair 50/50 acquisition/eval, ≥80-item pairs only, nested 0/1/3/7/15/31; 3 seeds for split + acquisition order.
- Metric: Brier 0/1/3/7/15/31, ALC, ECE; per-pair curves.
- Expected: ALC ≈ 0.18–0.23 (below 0.25); B0=0.25 exactly; steep 0→1 drop, diminishing after 7.
- Falsification: ALC ≥ 0.25 on benchmark holdout → simulator or eligibility bug; fix before any E02+.
- Compute: CPU minutes (pandas/numpy).
- Interpretation: sets the floor every method must beat; validates simulator vs official streaming rules.
- Decision: PASS → freeze simulator + splits as the internal suite; FAIL → debug, do not proceed.

## E02 — Global-rate baseline
- Hypothesis: A smoothed global success rate beats 0.5 at budget 0.
- Control: E01 (0.5 at B0, unsmoothed mean after).
- Treatment: `naive_baselines: global` (Beta(5,5) prior + pair-label posterior).
- Split: same as E01.
- Metric: ALC + B0 + ECE.
- Expected: B0 < 0.25 iff class balance ≠ 50%; small ALC gain; ECE improves (less 0/1 overconfidence at n=1).
- Falsification: B0 ≥ 0.25 → global balance is ~50% or prior mis-specified; keep 0.5 fallback.
- Compute: CPU minutes.
- Interpretation: measures pure class-balance signal.
- Decision: keep as prior component iff B0 improves on benchmark holdout; else retain 0.5.

## E03 — Benchmark-difficulty baseline (cold-start prior problem)
- Hypothesis: Benchmark identity is strongly predictive, but anonymous unseen IDs force feature-based difficulty — historical benchmark means fail cold start.
- Control: E02.
- Treatment: (a) oracle benchmark mean (train benchmarks only, to quantify ceiling); (b) cold-start: global + benchmark-family/domain prior (no test-ID effect).
- Split: benchmark holdout (critical) + random (ceiling contrast).
- Metric: ALC/B0; report random-vs-holdout gap.
- Expected: (a) wins big on random, collapses on holdout; (b) small but real gain via domain/family.
- Falsification: if (b) ≈ (a) on holdout → benchmark difficulty is captured by coarse features (good); if (b) ≈ global → need item-text difficulty (→E08).
- Compute: CPU minutes.
- Interpretation: quantifies benchmark-effect size and cold-start penalty.
- Decision: adopt family/domain priors iff holdout gain; never ship ID-memorization.

## E04 — Subject-ability baseline
- Hypothesis: Subject means transfer across benchmarks (strong models stay strong).
- Control: E02/E03-cold.
- Treatment: subject / provider / family / harness / reasoning / release-bucket priors (+ hierarchy blend), updated with pair labels.
- Split: benchmark holdout + subject holdout + subject×benchmark holdout.
- Metric: ALC/B0; per-subject Brier.
- Expected: subject prior wins when subjects overlap; family/provider backoff wins when unseen; harness matters most on agentic benches.
- Falsification: no gain over global on subject×benchmark holdout → ability does not transfer (unlikely); then rely on pair labels only.
- Compute: CPU minutes.
- Interpretation: decomposes subject-side transfer (identity vs family vs provider vs harness).
- Decision: keep hierarchy levels that survive subject×benchmark holdout; drop the rest.

## E05 — Subject×benchmark interaction baseline (pair history + shrinkage)
- Hypothesis: Pair-specific history with shrinkage beats marginal means and unsmoothed empirical mean.
- Control: E01 unsmoothed mean + E04 hierarchy.
- Treatment: hierarchical pair posterior (prior = E04 blend, pseudo-count tuned 2–20) vs unsmoothed mean.
- Split: benchmark holdout (pair history unavailable for test benches → tests shrinkage) + random (history available → ceiling).
- Metric: ALC + low-budget Brier (1/3) + ECE (overconfidence check).
- Expected: shrunk ≥ unsmoothed everywhere; biggest win at n=1 (0/1 → shrunk).
- Falsification: unsmoothed wins → pseudo-count too large or prior badly centered; retune, do not discard shrinkage.
- Compute: CPU minutes (grid over pseudo-count).
- Interpretation: establishes that adaptation = Bayesian updating, not raw averaging.
- Decision: replace unsmoothed mean with tuned shrunk posterior as the new "mean" control for all later E.

## E06 — IRT baseline (Rasch/1PL + MAP adaptation)
- Hypothesis: A unidimensional ability–difficulty model with MAP theta updates beats shrunk means by modeling item difficulty within pair.
- Control: E05 shrunk mean (pair-constant).
- Treatment: `Rasch1PL` (bench-level difficulty prior + MAP theta from acquired labels; item difficulty = bench prior until E08 features arrive).
- Split: benchmark holdout primary; item holdout secondary (within-bench discrimination).
- Metric: ALC + per-item ranking (within-pair Brier gap mean-vs-IRT) + ECE.
- Expected: modest ALC win; larger win on within-pair item ranking; calibrated posteriors.
- Falsification: IRT ≤ shrunk mean on both ALC and ranking → unidimensional structure absent or difficulty priors too weak; try 2PL/discrimination (E10) or MF (E07) before discarding.
- Compute: CPU (numpy, minutes–hour).
- Interpretation: answers H1 (low-dimensional structure?) for 1PL.
- Decision: promote IRT as structural backbone iff holdout win; else keep as ensemble member only.

## E07 — Matrix-factorization baseline (logistic MF + biases, cold-start)
- Hypothesis: Low-rank subject×benchmark factors capture interactions beyond 1PL.
- Control: E06 (or E05 if E06 failed).
- Treatment: `LogisticMF` rank {4,8,16}, biases, cold-start (unseen bench factor = prior/0 + domain backoff), pair-label bias update.
- Split: benchmark holdout + subject holdout.
- Metric: ALC; rank ablation; latent-dimension stability across seeds.
- Expected: rank 4–8 ≈ or slightly above 1PL; higher ranks overfit (holdout ALC worsens).
- Falsification: all ranks ≤ 1PL → interactions are not low-rank-linear (try 2PL/neural interaction E10 or retrieval E08).
- Compute: CPU (numpy) hours for rank×seed grid; GPU optional.
- Interpretation: quantifies low-rank explainability (H1, second leg).
- Decision: keep best rank iff holdout win; freeze embeddings for E10/E20 fusion.

## E08 — Item-text embedding model (semantic difficulty + similarity)
- Hypothesis: Item semantics predict difficulty/success beyond benchmark identity (H2).
- Control: E05/E06 (no text).
- Treatment: (a) TF-IDF/logreg difficulty; (b) frozen sentence-embeddings + ridge/logreg; (c) precomputed `measurement-db-embed` (text-embedding-3-small) NN + regression; all benchmark-holdout-trained, never test-fit.
- Split: benchmark holdout (primary — the only valid test of H2) + item holdout.
- Metric: B0 (pure text prior, no labels) + ALC; NN/label-correlation analysis.
- Expected: B0 improves over global; ALC improves modestly; NN transfer positive but weaker than pair labels.
- Falsification: no B0 gain on benchmark holdout → semantics add nothing beyond coarse features (kill text-heavy plans, keep cheap length/tier features).
- Compute: CPU for TF-IDF/ridge; embedding inference CPU/GPU hours (cache; models.txt-prefetchable).
- Interpretation: decides the text-investment budget for the whole project.
- Decision: promote cheapest text encoder that survives holdout; park LLM-embedding fine-tuning unless B0 gain is large.

## E09 — Subject-metadata model (identity → priors)
- Hypothesis: Metadata-only subject model (provider/family/release/harness/reasoning) predicts B0 without history (cold subject).
- Control: E04 hierarchy.
- Treatment: GBM/logreg on subject metadata → B0 prior; then E05-style updating.
- Split: subject holdout + provider/family/harness holdouts + temporal.
- Metric: B0 + ALC; ablation by feature group.
- Expected: provider/family/release dominate; harness/reasoning matter on agentic; temporal holdout shows drift (recent > old).
- Falsification: metadata ≈ global on unseen-provider/family → identity memorization only (keep hierarchy, drop ML-metadata).
- Compute: CPU minutes (LightGBM/XGB or sklearn GBM).
- Interpretation: maps what subject information actually transfers.
- Decision: keep feature groups with holdout gain; encode temporal drift handling (recalibration by release bucket).

## E10 — Subject+item interaction model (additive vs bilinear vs neural)
- Hypothesis: Explicit interaction (H3) beats additive subject+item (E08+E09).
- Control: additive (subject prior + item difficulty, logistic).
- Treatment: FM / bilinear / MLP-cross / 2PL-discrimination; same features, same splits.
- Split: benchmark holdout + subject×benchmark holdout.
- Metric: ALC + within-pair ranking; interaction-ablation (Δ over additive).
- Expected: small but consistent win for bilinear/FM; MLP wins only with enough data (watch overfit).
- Falsification: no interaction wins on subject×benchmark holdout → ship additive (simpler, §49).
- Compute: CPU (FM/2PL) to GPU-hours (MLP/cross-attention).
- Interpretation: settles H3 and the model-class ceiling.
- Decision: promote simplest winning interaction; freeze for E20 fusion.

## E11 — Benchmark holdout stress (transfer limits)
- Hypothesis: Transfer varies by domain/task family; same-family transfer >> cross-family (H7).
- Control: E05–E10 winners.
- Treatment: leave-one-benchmark-out for every eligible bench; group by domain/family; report per-bench ALC/B0/transfer gap.
- Split: exhaustive benchmark LOO.
- Metric: per-bench ALC/B0, transfer matrix, failure taxonomy (easy/hard/predictable/unpredictable).
- Expected: 1–2 benches much harder; cross-domain transfer weak.
- Falsification: uniform transfer → benchmarks are interchangeable (unlikely; would simplify everything).
- Compute: CPU (reuse fitted models; LOO loop).
- Interpretation: produces the per-benchmark failure analysis (§42) and tells where acquisition/calibration must compensate.
- Decision: prioritize hard-benchmark remedies (priors, calibration, acquisition) in E14/E16–E19.

## E12 — Subject holdout stress (unseen-model generalization)
- Hypothesis: Family/provider/generation priors carry unseen-model prediction (H9).
- Control: winners so far.
- Treatment: leave-family/provider/harness-out; nearest-family/provider transfer; full-config vs model-only ablations.
- Split: family/provider/harness/subject holdouts.
- Metric: B0/ALC by holdout type; config-ablation (model vs +harness/version/reasoning).
- Expected: family transfer works; harness hurts most on agentic; full config > model-only.
- Falsification: unseen-subject ≈ global → no transfer (then budget-0 is hopeless for new models; invest in 1-shot updating instead).
- Compute: CPU.
- Interpretation: per-subject failure analysis (§43) + config-value decomposition.
- Decision: lock subject-backoff chain; feed OOD thresholds to E18/uncertainty.

## E13 — Temporal holdout (forward generalization)
- Hypothesis: Past predicts future with drift; recalibration by recency is required.
- Control: winners trained on all past.
- Treatment: train-past→test-future (release/access-date cutoffs), rolling windows, recency-weighted priors.
- Split: temporal.
- Metric: ALC/B0 vs time gap; drift slope; harness-version drift.
- Expected: forward ALC degrades with gap; recency weighting + recalibration recovers part.
- Falsification: no degradation → stationarity (simpler); degradation unrecoverable → cap historical window.
- Compute: CPU.
- Interpretation: answers "do old evals predict new systems?" (§27).
- Decision: set training window + temporal recalibration policy.

## E14 — Calibration ablation (ranking vs probability)
- Hypothesis: Ranking wins do not imply Brier wins; budget/benchmark-specific calibration is required (H11).
- Control: best predictor uncalibrated.
- Treatment: global Platt/temperature/isotonic/beta; + benchmark-specific; + budget-specific (esp. n=1/3); hierarchical shrinkage as calibration.
- Split: benchmark holdout (calibrators fit on train-pool only, validated on holdout).
- Metric: Brier + ECE + reliability curves + Brier decomposition.
- Expected: isotonic/Platt helps pooled; budget-specific shrinkage helps most at n=1/3; ECE↓ + Brier↓ together.
- Falsification: ECE↓ but Brier↑ → reject that calibrator (§49); no Brier gain → ship uncalibrated + shrunk means.
- Compute: CPU minutes.
- Interpretation: separates accuracy from calibration (§17).
- Decision: lock calibrator(s) that improve Brier on holdout; always report ECE alongside.

## E15 — 0-vs-31 learning curve + label efficiency
- Hypothesis: Low-budget evaluation is fundamentally different from high-budget (H6); marginal gains diminish steeply.
- Control: random acquisition + winners.
- Treatment: full curves 0/1/3/7/15/31; marginal-gain-per-label, uncertainty-reduction-per-label, curve CIs.
- Split: benchmark holdout.
- Metric: Brier_n, ΔBrier per label, ALC decomposition by weight.
- Expected: 0→1 ≫ 1→3 > 3→7 > 7→15 > 15→31; half the ALC decided by budgets 0/1/3.
- Falsification: flat or increasing gains → acquisition/eval bug or pathological benchmark (investigate before E16–E19).
- Compute: CPU (reuse predictions).
- Interpretation: tells where the budget is most valuable (§23) and weights E16–E19 effort toward low budgets.
- Decision: freeze low-budget-first optimization policy (never optimize B31 alone).

## E16 — Random acquisition baseline (formal)
- Hypothesis: Random + shrunk updating is a strong, unbiased information baseline that uncertainty methods must beat on *eval* items (not selected items).
- Control: E15 curves.
- Treatment: multiple random trajectories (≥5 seeds) → mean/CIs; per-pair variance.
- Split: benchmark holdout, streaming-order simulation, nested sets.
- Metric: ALC + per-budget variance + eval-item Brier (never selected-item Brier).
- Expected: moderate variance at n=1/3, shrinking by 15/31.
- Falsification: near-zero variance → pairs too homogeneous (check simulator); huge variance → need variance-aware acquisition (E18).
- Compute: CPU.
- Interpretation: the acquisition control (§21).
- Decision: all acquisition claims must beat this on eval items with paired CIs.

## E17 — Uncertainty acquisition (|p−0.5|, entropy, disagreement)
- Hypothesis: Uncertainty sampling beats random on eval ALC.
- Control: E16.
- Treatment: |p−0.5| (BLE rule ≤0.15), entropy, ensemble-disagreement; forced-take logic preserved; streaming order.
- Split: benchmark holdout.
- Metric: ALC + low-budget Δ + selected-vs-eval gap (detect unrepresentative selection).
- Expected: small win at 3–15; risk of bias at 31 (unrepresentative subset hurts mean estimation — baseline README warns of this).
- Falsification: eval-ALC ≤ random → reject uncertainty-only (per §49); try diversity/hybrid (E18).
- Compute: CPU (+ ensemble cost if disagreement).
- Interpretation: tests H4/H5 first leg.
- Decision: keep iff eval-ALC win with CIs; never on selected-item metrics.

## E18 — Diversity / difficulty / hybrid acquisition
- Hypothesis: Uncertainty+diversity (or difficulty+diversity) beats uncertainty-only by covering the pair distribution.
- Control: E17 winner (or E16 if E17 failed).
- Treatment: diversity-only (embedding farthest-point), difficulty-only (predicted-hard mix), uncertainty+diversity (E18 hybrid), benchmark-balanced, subject-conditioned.
- Split: benchmark holdout.
- Metric: ALC + coverage diagnostics (selected difficulty/embed spread vs eval).
- Expected: hybrid ≥ uncertainty-only ≥ diversity-only; difficulty-mix helps calibration.
- Falsification: all ≤ random → acquisition does not matter given strong predictor/updater (ship random + invest in predictor).
- Compute: CPU (+ embeddings if diversity).
- Interpretation: ablates *why* acquisition succeeds (§24).
- Decision: lock simplest winning policy; implement streaming `labeling.py` for it.

## E19 — Information-gain acquisition (EIG / expected Brier reduction)
- Hypothesis: Expected information gain / expected Brier improvement beats heuristic uncertainty/diversity (H5).
- Control: E18 winner.
- Treatment: EIG (IRT/MF posterior entropy reduction), expected Brier reduction, EVI; greedy sequential with nested-set compatibility.
- Split: benchmark holdout.
- Metric: ALC + per-label information (nats) + cost (compute per decision).
- Expected: EIG ≥ hybrid at low budgets if posteriors are well-specified; compute-heavy.
- Falsification: EIG ≤ hybrid or too slow for 8-h limit → reject EIG for competition (keep as paper analysis).
- Compute: CPU-hours (posterior updates per candidate); must fit streaming timeouts.
- Interpretation: decision-theoretic test (§38) + answers H5.
- Decision: ship EIG only if ALC win + runtime fits; else ship E18 policy.

## E20 — Unified predictive + acquisition model (fusion)
- Hypothesis: A fused meta-model (subject + item + bench + history + IRT/MF + semantic + uncertainty + labels) with budget-aware calibration + winning acquisition beats all components.
- Control: best component stack (E05+E06/E07+E08+E09+E10+E14) with E16 random.
- Treatment: fusion {linear stacking, MLP, attention, Bayesian} × acquisition {E18/E19 winner}; budget-aware heads; OOD-gated fallbacks.
- Split: benchmark holdout (primary) + subject×benchmark + temporal; multi-seed; frozen test split untouched until final.
- Metric: ALC + B0/1/3 + ECE + per-bench/per-subject tables + runtime/memory/params.
- Expected: fusion ≥ best single by small margin; acquisition adds low-budget gain; OOD gate prevents catastrophic pairs.
- Falsification: fusion ≤ best single on holdouts → ship best single (simpler, §49); complexity must earn its place.
- Compute: GPU-hours (fusion training) + CPU (eval); must package within 8-h / 5-repo / 1024-GB limits.
- Interpretation: the competition submission candidate + paper method.
- Decision: promote to Codabench formative iff internal ALC win with CIs across benchmark/subject/temporal holdouts; else iterate on failed holdout.

---
## H. Expected information gain (per experiment)
E01: simulator validity (gating). E02–E05: prior hierarchy + shrinkage value (low-budget core). E06–E07: low-dimensional structure (H1). E08: semantic transfer (H2) — go/no-go for text budget. E09: subject transfer (H9). E10: interaction value (H3). E11–E13: transfer limits (benchmark/subject/time) + failure taxonomies. E14: calibration separability (H11). E15: label-efficiency curve (H6, budget prioritization). E16–E19: acquisition value + mechanism (H4/H5). E20: fusion + shippability.

## I. Compute requirements (total, excl. BLE API)
CPU-only through E16 (laptop/workstation hours). E08-embeddings + E10-neural + E19-EIG + E20-fusion: single-GPU hours (cache embeddings; prefetch via models.txt). BLE formative runs: paid API (Luna medium, ≤5–10 turns × eval+acquisition calls) — budget before submitting; mock first. Codabench: 8 h/run incl. setup; design for streaming timeouts (per-call + acquisition + total).

## J. Global decision rules (apply to every E)
§49 verbatim: random-only wins rejected; 31-only wins rejected; selected-item-only acquisition rejected; ECE-only calibration rejected; internal-vs-hidden gaps trigger shift analysis before complexity; IRT ties broken toward simplicity; ties broken toward reproducibility; promotion requires multi-seed + benchmark/subject/temporal holdouts + independent re-implementation.
