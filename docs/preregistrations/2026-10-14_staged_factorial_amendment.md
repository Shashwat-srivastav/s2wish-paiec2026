# Staged alternative to the full Bayesian factorial

**Status:** Approved as a preregistration amendment on 2026-10-14; no experiment run is authorized.  
**Related registration:** [Comprehensive research preregistration](./2026-10-14_comprehensive_research_preregistration.md)  
**Resource evidence:** [EXP-0048 dossier](../report/dossiers/EXP-0048.json)

## Why propose a change

The registered full factorial has 63,360 model configurations and 30,412,800 fold-seed configuration cells. EXP-0048 measured a mean of 4.1085 seconds per cell in the implemented engine. A simple serial extrapolation is about 1,446 days (3.96 years), before engineering or runtime for unimplemented priors, Laplace integration, MCMC, Monte Carlo, and symbolic regression. The pilot is strongly skewed (1.11-second median; 20.59-second p95), so the extrapolation is a planning warning, not a runtime promise.

This amendment reduces cross-component combinations while retaining a nested held-out evaluation. It is not equivalent to the full factorial: it can miss interactions outside those explicitly selected below. It was approved before any quality results were inspected and replaces the full-factorial execution scope for EXP-0049.

## Proposed design

### 0. Implementation and analysis freeze

Before generating any model-quality result:

- Implement and test the candidate variants on deterministic synthetic inputs and unit tests.
- Freeze each prior/integration definition, parameterization, numerical tolerances, candidate table, episode seeds, bootstrap procedure, and output schema.
- Keep the existing leakage controls: content-grouped folds, complete eight-field subject keys, runtime item dictionaries, no benchmark identity, no leaderboard/formative feedback, and no Jev/runtime feature.
- Run a fresh metrics-only throughput/memory/output pilot for newly implemented variants. EXP-0048 does not price those variants.
- Do not include MCMC in the search grid. If the screening identifies one eligible finalist, use MCMC only as a separately budgeted posterior/integration check.

### 1. Nested main-effect and nominated-interaction screen

Within each of the four existing LOBO outer folds, use only its three remaining benchmark folds as inner folds. Evaluate:

- Up to 58 unique main-effect settings: the reference plus each of the 11 offset-prior alternatives, 24 item settings, 24 pair settings, and 2 integration methods varied one component block at a time around the reference, deduplicating the reference setting.
- Up to 16 unique, predeclared interaction settings, frozen in a design table before any scores are produced. The nominated interaction families are offset-prior × pair-prior and item-facet/guard/ρ₀ × pair-prior. Choose the actual combinations by a balanced covering rule over the registered levels, with duplicate configurations removed. No interaction is added after seeing a score.
- All five acquisition policies for each candidate.
- Seed 0 in each of R1, R2, and R3, with all six registered budgets.

Freeze these 16 unique interaction rows in the amendment. All unmentioned factors remain at the frozen BASE2 reference; the item residual remains `N(0, sigma_beta²)`, and each row is evaluated under all five acquisition policies.

| Row(s) | Offset setting | Item setting | Pair setting |
|---|---|---|---|
| I01–I02 | `t3(G,sB)` | BASE2 reference | P1; P2, respectively |
| I03–I04 | `t1(G/2,s_half)` | BASE2 reference | P1; P2, respectively |
| I05–I06 | Type-I Gumbel, location `G`, scale `sB*sqrt(6)/pi` | BASE2 reference | P1; P2, respectively |
| I07–I08 | Spike-and-slab: mass 0.5 at `G/2`, mass 0.5 on `t3(G/2,2*s_half)` | BASE2 reference | P1; P2, respectively |
| I09–I10 | BASE2 reference | facets off; guard off; `rho0=0` | P1; P2, respectively |
| I11–I12 | BASE2 reference | facets all; guard on; `rho0=0.1` | P1; P2, respectively |
| I13–I14 | BASE2 reference | facets named; guard on; `rho0=0.2` | P1; P2, respectively |
| I15–I16 | BASE2 reference | facets off; guard on; `rho0=0.05` | P1; P2, respectively |

`P1` is a normal pair effect `theta~N(0,(0.8*tau)^2)`, `w_od=0.7`, `tau_scale=0.8`. `P2` is a centered `t3` pair scale mixture with scale `1.2*tau`, `w_od=1.2`, `tau_scale=1.2`. These choices intentionally cross components that differ from the reference; they are nominated interaction probes, not a representative estimate of every possible interaction.

The screen contains at most 74 model configurations. Its nested cell count is `74 configurations × 5 policies × 4 outer folds × 3 inner folds × 3 regimes = 13,320`. Compare every candidate-policy pair with frozen v1_3 under A1 on identical inner-fold episodes. A candidate-policy pair passes the screen if pooled inner ΔALC is `<= -0.0005`, at least 2/3 inner-fold means are negative, and no inner-fold mean is `> +0.0005`. Rank passing configurations by their best passing policy, then by the fixed candidate-table order; advance the baseline plus at most eight passing configurations. Within a configuration, break policy ties in fixed order A1, A2, A3, A5, A8. The screen threshold is only a pruning rule, not an adoption rule.

### 2. Inner-fold stability screen

For each outer fold, re-evaluate the baseline and up to eight advancing configurations on the same three inner folds, all five acquisition policies, and ten common seeds in each regime (R1/R2/R3). Compare each candidate-policy pair with frozen v1_3 under A1 on identical episodes. A pair passes if pooled inner ΔALC is `<= -0.0005`, at least 2/3 inner-fold means are negative, and no inner-fold mean is `> +0.0005`. Select the passing configuration-policy pair with the lowest pooled inner ΔALC; break ties by candidate-table order, then fixed policy order A1/A2/A3/A5/A8. If none passes, select no candidate for that outer fold. The outer fold remains untouched until Stage 3.

Worst-case cell count is `9 settings × 5 policies × 4 outer folds × 3 inner folds × 30 seeds = 16,200`.

### 3. Untouched outer-fold evaluation

For each outer fold, compare frozen v1_3 under A1 with the single configuration-policy pair selected using that fold's inner data. Evaluate only that selected policy on the untouched outer benchmark and use the registered full seed counts: R1 100, R2 10, R3 10. Do not revise settings or policy after seeing the outer result. If no candidate passed Stage 2 for an outer fold, record that fold as no selection; do not substitute a candidate after examining the outer fold.

Worst-case cell count is `2 arms × 1 selected policy × 4 outer folds × 120 seeds = 960`. The outer-fold results estimate the nested selection procedure, not an unconditionally preselected configuration. Apply the registered adoption rule without relaxing it: mean paired ΔALC ≤ −0.001, pair-bootstrap 95% CI upper endpoint < 0, at least 3/4 fold means < 0, no fold delta > +0.0005, with benchmark-level uncertainty reported separately. Require a candidate-policy selection in at least 3/4 inner loops; otherwise the route is unstable and not promoted. Preserve the null result if the procedure does not clear the adoption bar.

The nested estimate validates the selection procedure; it does not imply that a different configuration may be selected after inspecting the outer results. Any single final configuration for a submission must be selected by the frozen inner procedure on all available public training data, not by official scores.

## Resource estimate

Using EXP-0048's measured mean of 4.1085 seconds per cell:

| Stage | Maximum cells | Linear serial estimate |
|---|---:|---:|
| Nested initial screen | 13,320 | 15.2 hours |
| Inner-fold stability screen | 16,200 | 18.5 hours |
| Untouched outer evaluation | 960 | 1.1 hours |
| **Total** | **30,480** | **34.8 hours** |

Use **70 hours** as a conservative planning allowance (2× the linear estimate), before implementation, test, process-startup, data-transfer, or unsupported-variant overhead. Pilot-proportional timing output would be about 5.2 MB; a result-bearing report will be larger. The measured peak working set of 95.02 MiB is only a reference point for the existing engine, not a guarantee for new variants.

The proposed maximum cell count is about 998× smaller than the full registered design, but it still requires explicit resource approval. The mean is sensitive to the slow tail; the 2× allowance is not a hard upper bound.

## Work deliberately excluded or gated

- Do not run the full cross-product or add interactions after observing results.
- Do not run the 10⁷-episode Monte Carlo, Sobol sensitivity, symbolic regression, or full MCMC check as part of this screen. Each needs its own frozen candidate and resource approval; Monte Carlo/Sobol only proceed for one eligible frozen finalist.
- Do not make API calls or submit repeated platform archives until the existing organizer-policy and external-action gates are satisfied.
- Do not claim this staged screen answers every combination in the original preregistration. State that it is a lower-cost, interaction-limited alternative.

## Approval and amendment gate

The parent preregistration and EXP-0049 registry row are amended to identify this as the replacement design, carry forward the 16 interaction rows and stage-specific rules above, and record the 70-hour planning allowance as a stop-and-review cap. This amendment is not authorization to run; obtain separate explicit approval before starting any stage.
