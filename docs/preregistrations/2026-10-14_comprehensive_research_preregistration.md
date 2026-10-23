# PAIEC research programme — pre-registration

**Registration date:** 2026-10-14  
**Status:** EXP-0048 metrics-only resource pilot and EXP-0044 tempering replay completed; EXP-0044 was a null/no-promotion result. The staged EXP-0049 amendment is locked; all other model-quality work remains gated.  
**Primary aim:** determine whether any reproducible, runtime-legal method can improve the current predictor on unseen benchmarks. A score below 0.11 on one official draw is an aspiration, not a tuning objective. No formative score, per-pair outcome, anonymous identifier, or leaderboard rank is used as a training label, feature, hyperparameter target, or model-selection signal.

## 1. Programme rules

1. The reference archive is `submissions/dist/v1_3_named_facets_noguard.zip`, SHA-256 `a4b0e841d4a344ff207bed2c855342212e95f7d550195b667446ea4e92685b86`. Its frozen parameters are `G=-0.5804949802000193`, prior centre `-0.29024749010000966`, prior scale `0.7781603399165817`, `tau=1.1298321532152709`, `sigma_beta=2.701516919156687`, `rho0=0.05`, named facets, and guard off. The archive's single official score is not used for fitting or selection.
2. Primary local validation uses identical, paired episodes for every arm: leave one benchmark out over `multi_swebench`, `real_webagents`, `researchcodebench`, and `matharena`; `swe_rebench` remains training-only. The regimes are R1 (seeds 0–99), R2 (seeds 0–9), and R3 (seeds 0–9). R2 is primary because it is closest to one pair per benchmark; R1 and R3 are stress tests. Nothing is promoted on R3 alone.
3. Use the saved eligible public data revision `bc8204d811` for the five benchmark folds. The enlarged public pool `54074453da` is used only where a registered experiment names it. Record data revision and archive hash with every run.
4. The outcome is pair-weighted Brier ALC: `0.1 B0 + 0.2 B1 + 0.2 B3 + 0.2 B7 + 0.2 B15 + 0.1 B31`, lower is better. Compare arms on the same episodes. Report paired pair-bootstrap 95% intervals conditional on the four folds, four-fold t intervals with 3 degrees of freedom, per-fold deltas and signs, and a new-benchmark predictive interval. Do not treat a pair bootstrap as benchmark-generalization uncertainty.
5. The primary minimum practically important difference (MDE) is `0.001 ALC`; `0.0005` is the minimum screening effect and is not an adoption threshold. Adoption requires mean paired ΔALC `<= -0.001`, pair-bootstrap 95% CI upper endpoint `< 0`, at least 3/4 fold means `< 0`, and no fold delta `> +0.0005`. If benchmark-level intervals include zero, state that transfer is uncertain even when the pair interval excludes zero.
6. All model selection is inside training folds. Outer held-out benchmarks are not used to choose settings. Group folds by canonical item content/runtime item dict, not `item_id`; identify subjects by the full eight-field runtime dictionary; exclude benchmark identity and hidden IDs as predictors. Only fields available to `predict()` can be used in a proposed submission.
7. Every experiment has a dossier after execution. Record hypothesis, competing explanation, data, procedure, settings, formulas, metrics/MDE, preregistered rule, result, decision, caveats, cost, files, and exact commands. Keep null results. Report deviations. No ZIP is changed or promoted without an explicit final gate.
8. Current available evidence—including E5 rejection of prior-shape candidates, the earlier failure of metadata/name links, and the correlated-model pilot—is acknowledged. This programme is a new, bounded attempt to close the open research record and test the named combinations; it does not erase earlier null results or treat them as fresh evidence.

## 2. Open research-record items (§27 close-out)

The experiment IDs below are registered in `experiments/registry.csv`. Run in this order. A failed validity gate stops dependent work.

| ID | Question and hypothesis | Fixed procedure and analysis | Decision / kill rule |
|---|---|---|---|
| EXP-0039 F2 | Does the local simulator reproduce the distribution of observed platform pair scores? | Compare all 34 saved official per-pair records (8 v1 + 8 v1_1 + 9 v1_2 + 9 v1_3) to blind, version-matched R1 predictive distributions produced only from public training data. Do not match or infer anonymous pair identities. Report each saved score, run-level ALC, predictive ranks, and a simultaneous 90% predictive envelope; account for multiplicity rather than applying 34 unadjusted 90% intervals. No fitting. | Any run outside its prespecified run-level 90% predictive interval, or observed tail count outside the simultaneous 90% predictive range, triggers a simulator/draw audit before model search. An isolated point outside an unadjusted marginal interval is not by itself a failure. |
| EXP-0040 Repeat | Does the same v1_1 archive reproduce its prior platform distribution? | If the platform accepts submissions and quota/access are available, submit the byte-identical `v1_1.zip` four additional times, for nine total runs including the five already recorded. Do not change the archive. Report new run scores, mean, SD, t interval, and predictive interval, then compare the new mean with the frozen historical v1_1 interval and the separately labelled v1_2 interval. No anonymous-ID analysis or tuning. | If the new mean is outside either frozen 95% interval, investigate draw composition and simulator assumptions. A platform upload is an external action and remains blocked until the user confirms the exact account/action and available submission budget. |
| EXP-0041 Item scorer | Can an offline item-demand scorer provide transferable item ranking? | On training benchmarks only, score items with the registered ADeLe-style demand annotator plus item length, format flags, code-block count, step count, and LLM-estimated human minutes. Before scoring, freeze the annotator implementation/checkpoint or approved API model, prompt, decoding settings, output schema, and content hashes; if these cannot be fixed without selecting against evaluation results, mark the arm blocked. Fit/choose the scorer only inside training folds. Evaluate Spearman `rho` against leave-subject-out item difficulty on each of the four outer folds. Jev is a separately labelled offline upper bound, never a runtime feature. Use content-grouped folds. An API-derived feature is blocked until organizer policy is known; no API calls before that answer. |
| EXP-0042 A2 | Does lexical similarity to already-observed items improve the current item's residual prediction? | At each prediction, use only the current item and items already revealed for that subject–benchmark pair; never inspect the hidden benchmark's item pool. Fit TF-IDF vocabulary on training folds only. Compare with frozen v1_3 on R2 primary and R1 secondary, budgets 0–31, paired episodes. Include a shuffled-neighbour control. | Drop if ΔALC improvement is less than `0.001`, or if the benefit occurs only at B31. |
| EXP-0043 Level | Can runtime item-text summaries improve the budget-0 benchmark level without benchmark identity? | Fit a label-free text-statistic level predictor using only runtime-visible item content/feature summaries. Evaluate budget-0 Brier and worst-case regret over the preregistered training-fold base-rate sweep `δ ∈ {-3,-2,-1,0,+1,+2,+3}`; use the same synthetic seeds across arms. | Drop if worst-case regret does not improve over frozen v1_3. No platform outcomes used. |
| EXP-0044 Tempering | Do `w_od=0.8`, `tau_scale=0.8` improve the likelihood-tempered pair update? | Full replay of four arms: baseline, `w_od=0.8`, `tau_scale=0.8`, and both together. Use R2 10 seeds and R1 100 seeds; same episodes; compute paired CIs, per-fold signs, B1 delta, and fit/runtime diagnostics. This is a confirmatory replay of a previously screened line, not an invitation to search a new grid. | Leave the setting unchanged if gain is `<0.0005` ALC, CI includes zero, B1 worsens by `>0.0005`, or the paired fold decision is inconsistent. |
| EXP-0045 X3 | Can four prior hyperparameters be learned by simulated ALC rather than likelihood? | Nested leave-one-benchmark-out: inner folds optimize only prior centre, prior scale, `tau`, and `sigma_beta` using synthetic episodes drawn from training folds; outer fold evaluates once. Fixed inner grid: centre `G/2 + {-0.5,0,+0.5}` logits; each of the three scales is its training-fold value times `{0.8,1,1.2}`. Common seeds; compare with frozen v1_3. | Drop if outer ΔALC improvement is `<0.001`, CI includes zero, fewer than 3/4 folds improve, or any fold is worse by `>0.0005`. |
| EXP-0046 E3 | How does tinyBenchmarks gp-IRT compare in this setting? | Implement the published gp-IRT estimator in the same LOBO code, with randomly selected benchmark anchors because no anchors can be chosen for an unseen benchmark. Same R1/R2 episodes and budgets; report paired CIs and per-fold results. This is an external baseline, not a leaderboard-tuning route. | Report regardless of sign. Do not promote unless it independently meets the general adoption bar. |
| EXP-XXXX Organizers | Which inference APIs are allowed? Are board rows before 2026-09-16 comparable? Does the summative subset match the formative protocol? | Submit these three exact questions to the organizers via an authorized human channel. Archive the answer and date. Until answered, do not call an external inference API in an experiment or submission and do not treat pre-2026-09-16 rows as comparable. | Answers gate every API-derived scorer and any comparison that assumes the summative subset equals the formative draw. No answer means blocked/unknown, not permission. |

### §27 experiment-priority record

Costs below are pre-run planning classes, not measured runtimes: low `<1 CPU-hour`, medium `1–8 CPU-hours`, high `>8 CPU-hours`; external platform/API costs are separate. Information gain is about resolving the stated competing explanations, not a promise of score improvement.

| ID | Competing hypothesis / current evidence | Expected information gain | Expected objective impact | Compute / engineering cost |
|---|---|---|---|---|
| EXP-0039 | Platform draw variation may be wider or differently shaped than the simulator; 34 official pair records and repeat-run summary exist, but they have not been matched to draws. | High; a validity gate for all local comparisons. | No direct gain; prevents selecting a simulator artifact. | Medium / low |
| EXP-0040 | The observed archive variance may reflect draw composition rather than predictor differences; five repeats exist. | Medium; estimates a direct same-archive replicate distribution. | No direct gain; improves interpretation of the 0.1172 result. | External quota-dependent / low |
| EXP-0041 | Item content may not transfer; E6 found weak realistic item-signal value and E7 did not establish a runtime scorer. | High; distinguishes missing signal from a closed content route. | Only a scorer reaching `rho>=0.30` continues; high-quality scorer could create a new signal. | Medium / high; API gated |
| EXP-0042 | Lexical neighbours may not predict residual difficulty; earlier E6 indicates low-rho item signal has limited ALC value. | Medium; cleanly tests whether online revealed-item context adds signal. | Adoption-sized gain is at least `0.001` ALC. | Low / medium |
| EXP-0043 | Item-text statistics may not predict benchmark level; E7's enlarged-pool level refit was harmful. | Medium; tests a different, label-free query-text signal without pool aggregation. | Must reduce worst-case regret; no gain expected from benchmark IDs. | Low / medium |
| EXP-0044 | Previously tested tempering values may have small or seed-dependent effects; prior sweeps were not decisive. | Low–medium; closes the exact requested replay. | At most screening-scale gain is plausible; no adoption below the fixed bar. | Low / low |
| EXP-0045 | Four meta-learned priors may overfit four training benchmarks; E7 hyperparameter refit was harmful. | Medium; tests whether simulated ALC selects useful settings under nested holdout. | Adoption-sized only if outer ΔALC reaches `-0.001`. | Medium / medium |
| EXP-0046 | gp-IRT may be a useful external baseline or may fail without hidden-benchmark anchors. | Medium for comparative evidence; low expected direct gain. | None assumed; reportable baseline. | Medium / high |
| EXP-XXXX | API permission, temporal comparability, and formative/summative match are unknown. | Very high policy information; unlocks or blocks API routes. | No direct ALC effect. | No compute / low |
| EXP-0048 | The factorial's runtime and memory may be far above naive vectorized estimates. | High operational information, no outcome selection. | Avoids committing to an infeasible run. | Low pilot / medium |
| EXP-0049 | One or more interactions among model components may improve ALC, but prior/metadata evidence is mostly null or negative. | Potentially high scientific information, with substantial multiplicity and compute cost. | Improvement is exploratory; adoption still needs `-0.001` and nested confirmation. | Very high / very high |
| EXP-0050 | Synthetic draw-noise distribution may be mischaracterized; prior estimate is `±0.013`. | High precision for one frozen finalist only. | No model selection; quantifies single-draw risk. | Very high / medium |
| EXP-0051 | Sensitivity may be concentrated in a small subset of finalist settings. | Medium; identifies synthetic-DGP variance drivers. | No direct gain; supports robustness claims. | High / medium |
| EXP-0052 | The current prior's effective shrinkage may not match empirical learning; budget steps are nested. | Medium; distinguishes prior scale mismatch from absence of signal without a false stationarity claim. | Could motivate a bounded correction only if paired Brier also improves. | Low / medium |
| EXP-0053 | `T=1` may not minimize held-out log loss; earlier calibration evidence was ordinary. | Low–medium; tests one well-defined calibration diagnostic. | Reopen E6 only if stable `T` lies outside `[0.9,1.1]`. | Low / low |
| EXP-0054 | Remaining loss may be sensitive to prior-centre or shrinkage misspecification. | Medium; bounds vulnerability without tuning. | Stress diagnostic only, not a direct gain target. | Low / low |
| EXP-0055 | A low-complexity residual expression may add transferable signal; E6's value curve makes low-quality expressions unlikely to matter. | Medium only if an upstream signal gate passes. | No adoption unless its held-out ALC clears the general rule. | Medium / high |
| EXP-0056 | A chosen improvement may be a fold-specific winner rather than a transferable method. | High; final confirmation and bound. | Determines whether the target is defensible; negative result remains valuable. | Medium / medium |

### Execution addendum — EXP-0044 (2026-10-14)

The preregistered four-arm replay completed against the hash-verified v1_3 archive. Package-vs-engine prediction checks (all six budgets) established that the archive parameter `"facets":"named"` is treated as a truthy enable flag by its packaged model, which consequently uses all `item_features` fields; `facets="all", guard=0` is the faithful engine setting. It matched the package to <`2e-15` on the checked real_webagents and matharena pairs. A preliminary run with the analysis engine's genuinely named-only mode differed by `0.05195` on matharena and was discarded before interpretation. In the corrected primary R2 replay, paired ALC deltas were `-0.170e-3` (`w_od=0.8`, 95% pair CI `[-0.538,+0.236]e-3`), `-0.123e-3` (`tau_scale=0.8`, `[-0.428,+0.209]e-3`), and `+0.090e-3` (both, `[-0.581,+0.839]e-3`). None meets the `0.0005` screening effect or `0.001` adoption MDE; fold diagnostics also fail for the tau-only and combined arms, and the w_od-only arm has a `+1.305e-3` worst-fold regression. The R1 w_od-only pair interval excludes zero narrowly, but the effect is below the screening threshold and its benchmark-level intervals include zero. **Decision: retain v1_3; no candidate ZIP was built.** Full per-fold values, intervals, commands, and caveats are in [EXP-0044.json](../report/dossiers/EXP-0044.json) and the raw replay at `experiments/runs/preregistration/exp_0044_tempering_replay_2026-10-14.json`. This records the completed experiment without changing its locked procedure or authorizing other experiments.

## 3. Bayesian component screen (amended scope)

### Factors and fixed levels

The originally registered full cross-product had 63,360 configurations. Its execution scope is superseded by the approved nested staged design in [2026-10-14_staged_factorial_amendment.md](./2026-10-14_staged_factorial_amendment.md), incorporated into this preregistration. Its 16 interaction rows, candidate gates, nested folds, seed counts, and stop conditions are fixed before any model-quality result. The full cross-product remains an unrun historical proposal and is not an authorized fallback.

The staged design contains up to 58 one-component configurations plus 16 nominated interactions, evaluated first with seed 0 in each regime and then with a bounded inner-fold stability screen. Only the inner-selected configuration-policy pair is evaluated on each untouched outer fold. Full details and the exact interaction table are in the incorporated amendment.

**Offset prior (11 unique configurations; prior densities are normalized on the existing `mu` grid):**

1. `t3(G,sB)`; 2. `t3(G/2,s_half)`; 3. `t1(G/2,s_half)`; 4–8. the five fixed E5 mixtures (`X1-bma`, `X1-ext10`, `X1-ext20`, `X1-bma-ext10`, `X1-Gext10`) exactly as `docs/preregistrations/E5_X1_preregistration.md`; 9. type-I Gumbel with location `G` and scale `sB*sqrt(6)/pi`; 10. regularized horseshoe centered at `G/2`, global scale `s_half`, local scale `HalfCauchy(0,1)`, slab scale 3 logits; 11. spike-and-slab with point mass 0.5 at `G/2` and mass 0.5 on `t3(G/2,2*s_half)`.

Here `G` and `sB` are calculated from training benchmarks only; `s_half=max(0.5, RMS(training means-G/2))`. The two labels `t1` and `Cauchy` are the same probability distribution; Cauchy is an alias/check, not a duplicated factorial level. For horseshoe marginalization use deterministic numerical quadrature over the local scale and the same `mu` grid; no method may silently change the prior support.

**Item factors (24 settings):** the Gaussian item residual `beta ~ N(0,sigma_beta^2)` is present in every arm; facets are `off`, `all`, or `named`; degenerate-key guard is off/on; false-merge prior `rho0 ∈ {0,0.05,0.1,0.2}`. Here `rho0` is the item-key merge prior, not the facet guard. Item keys use only the runtime-visible item dictionary.

**Pair factors (24 settings):** pair effect is either `theta ~ N(0,(tau_scale*tau)^2)` or a centered `t3` scale mixture with the same scale; likelihood tempering `w_od ∈ {0.7,0.85,1,1.2}`; `tau_scale ∈ {0.8,1,1.2}`. All scales and training-fold fits are learned without the held-out benchmark.

**Integration (2 primary settings):** exact `mu` grid × Gauss-Hermite integration; Laplace approximation to the same model. MCMC is not a factorial arm: use it only as an independent numerical check on the frozen baseline and the top eligible candidate, if one exists. Compare posterior means and predictive probabilities; absolute probability difference above `1e-3` triggers an integration audit.

**Acquisition (5 settings):** A1 (accept-all), A2, A3, A5, A8, using their existing implementation in `src/paiec/models/v1_1.py`. A1 is the implementation's accept-all policy. These acquisition-policy levels are distinct from §27 experiment labels such as EXP-0042/A2 (lexical smoothing). Do not add a new acquisition policy. If a policy is not valid under a regime, record it as not applicable rather than substituting another policy.

### Execution order and decision rules

Generate and validate the amended staged candidate manifest without fitting. Confirm unique encodings, no duplicate interaction rows, and equality of duplicate aliases (`t1 == Cauchy`). EXP-0048 measured the existing engine but does not authorize a model-quality run; any newly implemented variant requires a separate metrics-only pilot.

For each stage, use identical seeds and episode data across arms. Report ALC, paired Δ vs frozen v1_3, pair-bootstrap interval, four-fold t interval, per-fold delta/sign, budget curve, ECE (secondary), convergence/failure, runtime, and peak memory. Preserve failed configurations and null results in the result table.

Apply the fixed stage-specific screen and stability thresholds in the amendment. Stage 3 estimates the nested selection procedure; outer results cannot change settings, acquisition policy, or selection rules. Apply the programme adoption bar in §1 and report pair-level and benchmark-level uncertainty. **No stage is authorized to run until the user separately approves its execution and resource plan.** The original full cross-product is not a fallback.

## 4. Monte Carlo and sensitivity

Only one frozen eligible candidate may enter this stage. If none passes the Bayesian gate, do not run 10 million draws to search for a winner; report that the stage is not reached.

1. Simulate `10^7` synthetic episodes separately for R1, R2, and R3 using a vectorized NumPy generator, common random numbers for candidate and baseline, and fixed `PCG64DXSM` streams `20261014`, `20261015`, and `20261016`. Fit the synthetic data-generating parameters from training folds only. Do not use formative scores to calibrate the generator.
2. Report mean ALC, standard deviation, median, 5th/50th/95th percentiles, tail probabilities, empirical Monte Carlo SE `sd(ALC)/sqrt(N)`, and quantile uncertainty. Separate Monte Carlo uncertainty from the four-fold benchmark-level uncertainty. The stated draw-noise value `±0.013` is a comparison reference, not an error bar to add to the MC SE.
3. One-at-a-time sensitivity uses the exact factor levels above and a five-point grid for continuous scales (`0.75,0.875,1,1.125,1.25` times the selected value); the requested four `rho0` values remain four levels. Estimate Sobol first- and total-order indices with independent discrete-uniform factor sampling, Saltelli estimators, and a preregistered base size of 131,072; report bootstrap intervals. Rank the top eight factors by total-order index. These are sensitivity indices for the stated synthetic distribution only.
4. Do not start 10^7 draws or the Sobol run until the measured resource estimate is reviewed and approved.

## 5. Mean-reversion and Kelly diagnostics

### Shrinkage / mean reversion

Label budgets are nested posterior updates, not equally spaced observations of a time series. Therefore a stationary AR(1) / OU fit to the six budget means is not a valid primary test and will not be described as evidence of stationarity. Instead, measure the empirical shrinkage curve of the pair logit prediction versus acquired-label count on held-out episodes. Fit a monotone exponential curve `w(n)=1-exp(-n/h)` to training-fold-only synthetic/episode updates and estimate half-life `h` with a pair bootstrap. Convert the t3 prior's local log-density curvature at its centre into equivalent Bernoulli pseudo-counts: `n_eff = 4 * [-d² log prior(mu)/d mu²]` at the centre, under the `p=0.5` binomial curvature `n/4`. Check whether fitted `h` differs from this value. A shorter fitted half-life indicates the prior/update is too conservative only if the paired Brier evidence also shows over-shrinkage. Report all folds and uncertainty.

### Kelly / log utility

Kelly maximizes expected log wealth given probabilities, prices/odds, and a specified stake rule. This task provides no wager price or bankroll, and "expected log growth of Brier" is not a defined objective. Therefore no value will be called a Kelly-optimal temperature. The valid forecast diagnostic is to estimate temperature `T` by minimizing held-out log loss, report that value and Brier ALC at `T=1` and at the fitted `T`, with nested folds and paired CIs. If `T` is stable and outside `[0.9,1.1]`, rerun the registered E6 headroom analysis with that temperature as the new base; otherwise stop. A separate literal Kelly test requires a user-specified betting market and payoff schedule and is not part of this registration.

### Deliberate prior mis-specification

For the selected baseline/candidate, shift the prior centre by `-1` and `+1` logit and multiply the fitted half-life by `0.5` and `2`. Use identical R2/R1 seeds, report ΔALC and regret with all intervals and fold signs. This is a stress test, not a parameter-tuning grid. The same general adoption bar applies; any failing stress arm is reported as sensitivity, not silently repaired.

## 6. Symbolic regression

Run only if EXP-0041 or EXP-0042 demonstrates a runtime-legal signal with held-out rank correlation `rho >= 0.30`; otherwise stop the symbolic search. Do not use benchmark identity, anonymous identifiers, Jev, formative per-pair scores, or outcomes from the outer held-out benchmark in expression fitting. Jev can appear only in a clearly labelled non-deployable upper-bound analysis.

Features are extracted from fields visible to `predict()`: item length, format flags, code-block count, step count, and within-training-fold TF-IDF projections (top 50 fitted on training content); subject release date, provider, name tokens, reasoning effort, and harness fields only when they are part of the eight-field runtime subject dictionary. Benchmark features are prohibited. LLM-estimated human-minutes are blocked until API rules are confirmed and the feature-generation model/version is frozen.

Fit expressions to out-of-fold residuals `r = y - sigmoid(mu_b + theta_pair - beta_item + gamma_group)` using training benchmarks only; `gamma_group` may be included only when runtime-visible. Group examples by content/runtime item key and full subject dictionary. Use a complexity-penalized objective `Brier_ALC + lambda * node_count`, with `lambda` selected in inner folds from `{0,1e-4,5e-4,1e-3}` and a maximum expression size of 20 nodes. Compare to no-expression and shuffled-feature controls. Report symbolic form, node count, held-out-fold Spearman rho, ΔALC with paired intervals, fold signs, and whether its gain is compatible with the E6 `rho^2` value curve.

## 7. Final confirmation, target, and deliverables

1. A grid-selected or symbolic candidate must be frozen before a final nested confirmation on untouched outer benchmark episodes. It must meet the adoption rule in §1 and improve over the exact v1_3 archive implementation. No result selected on the same outer folds is called independent confirmation.
2. Headroom plot: show the frozen best validated candidate, the user-specified `0.102` floor only after tracing it to a saved E6 oracle output, and the public `0.1172` board entry as context only. Mark the public entry as not used for fitting. If the saved source does not support 0.102, plot its verified value and document the discrepancy; do not hard-code a convenient floor.
3. The E6 `rho^2` estimate implies an item scorer around `rho≈0.85` may be needed to approach 0.11; this is a preregistered expectation to verify, not a result. If no legal new signal reaches the required quality, the correct conclusion is a negative result and an explicit bound, not a tuned claim.
4. Final deliverables after all gates: this registration, one `pre_registered=true` registry record per executed run/configuration, a dossier per experiment, a headroom plot built from saved results, and a one-page result summary. Do not create placeholder dossiers or invent results. A blocked/unrun experiment is labelled blocked/unrun with reason.

## 8. Resource estimate and run gate

The original full factorial contained **63,360 model configurations** and implied **30,412,800 fold-seed configuration evaluations**, with six budget predictions per evaluation. This scope is superseded and must not be run. The approved staged design has at most **30,480** nested cells; the EXP-0048 mean-throughput estimate is 34.8 serial hours, with a **70-hour stop-and-review planning allowance** before implementation and unsupported-variant overhead.

### EXP-0048 pilot locked protocol

Run a timing-only pilot with the existing exact-grid × Gauss-Hermite `v1_1` engine, its `BASE2` prior, six configurations (`BASE2`; `w_od={0.7,0.85,1.2}`; `tau_scale={0.8,1.2}`), all five acquisition policies (A1/A2/A3/A5/A8), all four outer folds, and seed 0 once in each regime R1/R2/R3. Run every configuration-policy-fold-regime cell at all six budgets in fixed lexical order. The pilot does not implement or estimate the incremental cost of the not-yet-implemented Gumbel/horseshoe/spike-slab priors, t3 pair effects, Laplace integration, or MCMC; report that limitation as an engineering/resource add-on, not as zero cost.

Measure wall time per cell and in total, operating-system peak working set, and the bytes written by the pilot report. Do not persist or inspect ALC/Brier outputs; only the timing and memory measurements are written. Data loading and episode construction are measured separately from the per-cell policy replay plus prediction. Use only the fixed training-fold data and the predefined seed-0 episodes; no settings or code are selected from scores. Save the metrics-only JSON to `experiments/runs/preregistration/resource_pilot_2026-10-14.json`. The resulting estimate is for the implemented engine only and is not sufficient by itself to authorize the amended staged design.

Historical reference (superseded estimate): report 24 recorded about 13 minutes for nine E5 candidates including real R1/R2 and synthetic sensitivity work. Its 63.6-day R1/R2 and 69.4-day R1/R2/R3 linear estimates were based on per-candidate scaling from a different workload and are contradicted by the representative pilot below. The earlier 20–63 MB compact-summary estimate was also only a rough projection; it is not a measured full-run output estimate.

**EXP-0048 measured pilot (metrics-free):** the preregistered runner completed 360 cells (6 configurations × 5 policies × 4 folds × 3 regimes; seed 0 in each regime) in 1,481.15 seconds total, including 1,479.06 seconds in measured cells and 2.09 seconds in setup. Cell time was median 1.11 seconds, p95 20.59 seconds, and maximum 23.52 seconds. Peak Windows working set was 95.02 MiB; the JSON report was 60,935 bytes. No prediction-quality metrics were retained or inspected. The measured mean of 4.1085 seconds per cell gives a crude linear serial extrapolation of **124,951,189 seconds, or about 1,446 days / 3.96 years**, for the registered 30,412,800 fold-seed configuration evaluations. At the pilot report's bytes-per-cell, a similarly detailed report would be about 5.15 GB. These projections assume the pilot average transfers unchanged; the large median/mean gap shows cell costs are highly skewed. They are throughput warnings, not a runtime promise.

This pilot measures only the implemented `v1_1` exact-grid × Gauss-Hermite engine and does not estimate the engineering or runtime cost of the unimplemented priors/integrators or MCMC. It also does not measure a maximum safe batch size. Six budget predictions are produced per cell, so the original full design represents 182,476,800 budget predictions before any extra checks. The pilot result and its limitations are in `experiments/runs/preregistration/resource_pilot_2026-10-14.json` and dossier `docs/publication/dossiers/EXP-0048.json`.

**Hard stop:** EXP-0048 is complete and the staged design is preregistered, but no stage is authorized to run. Obtain explicit user approval of the specific stage and its resource plan before execution. Stop and seek renewed approval if measured wall time exceeds the 70-hour allowance or if memory/output approaches the safe operational limit. Any new optimized/batched runner requires its own metrics-only profiling and must not be chosen by outer-fold scores. EXP-0050 and EXP-0051 remain gated to one frozen eligible finalist and separate resource approval. Platform submissions still require the explicit external-action gate in EXP-0040; API calls require the organizer answer in EXP-XXXX.

## 9. Execution addendum — 2026-10-15 (no procedure changes)

User instruction (2026-10-15): execute the programme with parallel persona agents. Nothing above is altered; this
records scope only. Agents follow `docs/spec/lite_agent_contract.md` and the lite persona definitions (internal, not released).

| Run now | Agent persona | Scope |
|---|---|---|
| EXP-0039 | paiec-systems | as registered |
| EXP-0041 | paiec-representation | runtime-legal structural features only; ADeLe annotator arm run only if a frozen local offline checkpoint fits this machine, else `blocked`; LLM-estimated minutes `blocked` (API, EXP-XXXX) |
| EXP-0042 | paiec-representation | as registered |
| EXP-0043 | paiec-coldstart | as registered |
| EXP-0045 | paiec-coldstart | as registered |
| EXP-0046 | paiec-bayesian | as registered |
| EXP-0049 | paiec-bayesian | **Stage 0 only**: implement and unit-test the new variants, freeze the candidate manifest, metrics-only resource pilot. Stages 1–3 still need explicit approval of the measured resource plan (§8 hard stop stands). |
| EXP-0052–0054 | paiec-calibration | on the frozen v1_3 baseline (no candidate exists yet) |

Still gated: EXP-0040 (user must upload; external action), EXP-XXXX (user must send the organiser questions),
EXP-0050/0051 (need one eligible finalist), EXP-0055 (needs ρ ≥ 0.30 from EXP-0041/0042), EXP-0056 (needs a frozen
candidate). After results, a paiec-redteam agent reviews every dossier before the PI records decisions.
Machine limits: 5.8 GB RAM, ~11 GB free on E:; heavy jobs share two compute slots via `scripts/run_slot.py`.

**Gate correction (2026-10-15, after critic review):** §2 makes EXP-0039 a validity gate for dependent work, but the
§9 launch ran dependent experiments in parallel with it. To honour the gate, runs may finish, but **no decision from
EXP-0041–0046, 0049 or 0052–0054 is recorded or acted on until EXP-0039 reports**; if EXP-0039 fails, those decisions
are void pending the simulator audit.

**EXP-0039 outcome (2026-10-15): gate FAILED** under the locked rule — observed tail count 8 vs simultaneous range90
[1,7] (p_upper 0.022); global envelope p 0.003; run v1_3 967620 outside its PI90 (PIT 0.017). The simulator is
pessimistic (pooled pair obs 0.183 vs pred 0.211) and over-disperses runs (sd 0.013 obs vs 0.021 pred). Descriptive
platform forensics: platform pairs are far easier (median subject accuracy 0.72 vs 0.41 local; 10% near-perfect pairs
vs 0.5%). Per §2 and the gate correction, decisions from EXP-0041–0046, 0049, 0052–0054 are **void pending a simulator
audit**; their runs are paused. The audit/repair must use public data only (no formative scores as fit targets).

## 10. Rule amendment — 2026-10-15 (user instruction: align with the organisers' rules)

The rule in the header ("No formative score, per-pair outcome, anonymous identifier, or leaderboard rank is used as a
training label, feature, hyperparameter target, or model-selection signal") and the matching line in the staged
amendment were stricter than the competition requires. From today the governing rules are the organisers' own,
recorded verbatim in `docs/spec/COMPETITION_RULES.md`: **formative feedback may guide development**; prohibited are
extracting/sharing private test data, trying to identify the subjects or benchmarks behind anonymous feedback IDs,
data outside the benchmark inventory, and non-approved network access at runtime. Unchanged: no `benchmark_id`-keyed
behaviour (an open organiser question), no outer-fold selection in LOBO (a validity rule), and every completed
result above keeps the rules it was registered under.

**CORRECTION 2026-10-15 (sign error):** platform pair accuracies here were recovered with p0 = 0.5513, but the archives' own predict() gives p0 = 0.448722 (v1: 0.400099; E1 B0 mean prediction 0.451 agrees). B0 Brier and ECE cannot tell (a, p0) from (1-a, 1-p0), so every recovered accuracy was flipped. Corrected: platform median pair accuracy **0.27** (harder than local 0.41), **14.7% of pairs at accuracy <= 0.05**, **0% near-perfect**; benchmark_<hidden> has accuracy ~0. ALC/Brier-based results are unaffected (symmetric in a); every 'easier' / 'near-perfect' interpretation in this document is wrong and reversed. Reproduce: `python scripts/reproduce_q4_estimate.py`.

