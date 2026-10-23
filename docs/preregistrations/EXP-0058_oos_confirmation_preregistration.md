# EXP-0058: out-of-sample confirmation on held-out public benchmarks (pre-registration)

**Registered:** 2026-10-22. No EXP-0058 result has been computed. **Owner:** paiec-systems.
**Approval:** the PI requested this registration and it is approved (Jev 0.98). It is locked by the PI's commit of this file; nothing runs before that commit.

## Why

Every benchmark-level claim in the report rests on four LOBO folds. This experiment bounds the effect sizes of the two
modelling steps (shrinkage over the raw mean, and the hierarchical model over shrinkage) on about 90 public benchmarks
that the shipped hyperparameters never saw.

## Question

On held-out public benchmarks, does (a) Beta-binomial shrinkage (v0) beat the official empirical mean, and (b) the
final model's hierarchical predictor (no-facet path) beat v0, in pair-weighted Brier ALC on identical episodes?

## Arms (no refitting; every hyperparameter frozen)

| Arm | Definition | Source |
|---|---|---|
| `official` | Official empirical-mean baseline: S/N over acquired labels of the same (full subject dict, benchmark_id); 0.5 with no labels; unsmoothed, unclipped. | `external/official_baselines/empirical_mean/model.py` (= `ladder` rung L0) |
| `v0` | Beta-binomial shrinkage `clip((k*m + S)/(k + N), 0.02, 0.98)` with the shipped prior `m=0.5, k=2`. | `submissions/dist/v0.zip` (SHA-256 `57ad9d98…`), `prior.json` default; IRT disabled (= `ladder` rung L1) |
| `final` | v1_3 engine `v1_2.cfg_of(facets="all", guard=0)` with the parameters in the shipped archive (G, prior centre/scale, tau, sigma_beta, rho0). | `submissions/dist/v1_3_named_facets_noguard.zip` (SHA-256 `a4b0e841…92685b86`) |

- **Facets are off.** The enlarged release ships no items table, so items are keyed by `item_id`, `item_features` is
  unavailable, and the final model runs on its no-facet path (with item_id keys the guard is also inert). This tests
  **shrinkage vs the hierarchical model**, not the facet step.
- **Acquisition:** accept-all over the simulator's fixed random stream for every arm, so labels are identical across
  arms. The official baseline ships no `labeling.py` and would get the evaluator's random policy. That policy also
  selects a uniformly random subset of the stream, so the two are distributionally identical but not label-identical.
  Accept-all is used to keep the comparison paired.
- **Asserts before any scoring:** (i) package-vs-engine equality for `v0.zip`, `v1_3_named_facets_noguard.zip`, and
  the official `empirical_mean/model.py`, via `simulator.run_direct` on bc8204d811 real_webagents seed 0 (4 pairs, all
  six budgets; max |Δp| < 1e-9, exactly 0 for the two closed-form arms); (ii) the cell-episode builder equals
  `episodes.Episodes` (as in EXP-0039R).

## Benchmark set (fixed)

The EXP-0039R R1-drawable list (92), minus the training benchmarks of revision bc8204d811 (`multi_swebench`,
`real_webagents`, `researchcodebench`; `matharena` and `swe_rebench` are not in the list): **89 benchmarks** from
`data/_main_54074453` (rev `54074453da7b9911be0f48e4c560be547bc84d21`), using the EXP-0039R cache:

afrieval, agentic_review_perturbation, algotune, alignment_faking, appworld, averimatec, babilong, beavertails, benger,
bird_sql, braveguard, bridging_gap, cellverse, critic_discernment_game, crow, devbench, editbench, edu_circuit_hw,
egoschema, elicitation_game, engdesign, enginemt_qa, exploitgym, faithcot, fewshot_ttt_bbh, financebench,
finegrain_t2i, hallusionbench, harmbench, healthadminbench, helm_bbq, helm_thaiexam, hle, igakuqa119, ikp,
jailbreakbench, kernelbench, legal_rag_bench, llmail_inject, llms_lean_formalization, mathconstraint, mathvista_mini,
medhelm_ehr_sql, medhelm_head_qa, medhelm_med_mcqa, medhelm_med_qa, medhelm_medbullets, medhelm_medcalc_bench,
medhelm_medec, medhelm_medhallu, medhelm_pubmed_qa, medhelm_race_based_med, mj_bench, mmbench_v11, monkey_power_laws,
morphkv, mtbbench, naep_llm_students, naturebench, nester, nyu_ctf_bench, openbiorq, os_harm, perfcodebench,
planbench, proofaug, pxplore, realtimeqa, reliancescope, revolve, risebench, robust_reasoning_benchmark,
safeagentbench, sib200, software_world_models, sustainable_food, swe_smith, swebench_live, swepolybench, tau2_bench,
taubench, truthfulqa_mc, tulu_human_eval, tumlu, visual_riddles, visualwebarena, vl_rewardbench, workarena,
workbench_revisited.

**Disclosure:** EXP-0039R already computed absolute v1_3 ALCs for these benchmarks in R1 (seeds 0–19). No `official`
or `v0` arm, and no paired delta, has been computed on this set.

## Regimes and seeds (fixed)

- **R2 (primary):** one pair per episode (no cross-pair pooling). Seeds 0–4. Eligible pairs have 80–1000 distinct
  items (1000 is the platform's draw limit). Benchmarks with more than 200 such pairs keep the first 200 by
  `sha256(subject_id)` order. That gives 2,610 pairs, each run on all 5 seeds.
- **R1 (secondary):** the EXP-0039R episodes (seeds 0–19, existing R1 draw rule), with cross-pair pooling within a
  draw for `final`.
- Budgets 0, 1, 3, 7, 15, 31. ALC = 0.1 B0 + 0.2 (B1+B3+B7+B15) + 0.1 B31; Brier is exact over all trials
  (sufficient statistics).

## Outcomes

**Primary (R2), for each contrast** `final − v0` and `v0 − official`:
1. For each benchmark, the paired ΔALC is the mean over its pair instances (all seeds). The **effect** is the
   unweighted mean of these over the 89 benchmarks.
2. **95% interval:** percentile benchmark-level bootstrap (10,000 resamples of the 89 benchmarks, `numpy`
   `default_rng(58)`). This is benchmark-generalisation uncertainty, not a pair bootstrap.
3. **Sign count:** the number of benchmarks with Δ < 0 out of 89, with an exact two-sided sign-test p-value.

**Secondary:**
- The same statistics for each budget (B0, B1, B3, B7, B15, B31).
- The same statistics restricted to benchmarks with pooled accuracy ≤ 0.15 (all eligible-pair responses in the
  cache; label-free, fixed now). This subgroup has **5 benchmarks**, so it is reported descriptively and
  underpowered.
- R1 secondary: the same statistics on the R1 episodes.
- Sensitivity: excluding the name-family relatives of the training benchmarks (`swe_smith`, `swebench_live`,
  `swepolybench`, `visualwebarena`; the EXP-0037 stem rules), 85 benchmarks.

MDE for interpretation is 0.001 ALC (programme rule).

## Outcome → interpretation (fixed; applied to each primary contrast)

| 95% benchmark-bootstrap CI | Interpretation |
|---|---|
| upper < 0 | The refinement is **confirmed out of sample** at the benchmark level. Report the effect and CI as its bound. |
| lower > 0 | The refinement is **reversed out of sample**. It is harmful on new public benchmarks. |
| includes 0, within [−0.001, +0.001] | **Equivalent within the MDE.** No practically important difference either way. |
| includes 0, extends beyond ±0.001 | **Indistinguishable, bounded by the CI.** State the CI as the bound on the effect. |

The sign count is reported next to the CI. A significant sign test with a CI that includes 0 is described as
"consistent in direction, magnitude uncertain". Secondary outcomes never override the primary row.

## Rules

- No tuning, refitting, re-selection of benchmarks, or change of seeds after any result is seen. Every deviation goes
  in the dossier's caveats.
- No formative scores, platform statistics, or anonymous IDs are used. The arms are the shipped archives as they are.
- Compute runs only through `scripts/run_slot.py`, in the background, with peak RSS < 1 GB and caches under `.tmp/`.
- Outputs: `experiments/runs/preregistration/exp_0058_oos_<date>.json`, `docs/publication/dossiers/EXP-0058.json`
  (the EXP-0044 keys), and a proposed registry row.
- This is evidence on **public** benchmarks. It does not establish transfer to the hidden summative benchmarks.
