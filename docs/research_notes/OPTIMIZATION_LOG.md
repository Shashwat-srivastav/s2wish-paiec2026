# PAIEC Optimization Log

Updated: 2026-10-08

## Evaluation protocol

- Local public-pool replay using `src/paiec/evaluation/episodes.py`.
- Held-out folds: `multi_swebench`, `real_webagents`, `researchcodebench`, `matharena`.
- R2 regime, seeds `0..9` for the core parameter sweep.
- Budgets: `0, 1, 3, 7, 15, 31`.
- Metric: ALC as implemented by the repository. Lower is better.
- Every configuration used the same folds and seeds; these results are comparative, not leaderboard guarantees.

## Core parameter sweep

Baseline was the v1.3-style named-facet/no-guard core: `facets="named"`, `guard=0`, BASE2 prior, `w_od=1`, `tau_scale=1`, `kappa=1`.

| Configuration | Mean ALC | Effect vs baseline |
|---|---:|---:|
| Baseline named/no-guard | 0.208722 | reference |
| `w_od=0.8` | **0.208563** | small improvement |
| `tau_scale=0.8` | 0.208611 | small improvement |
| `guard=1` | 0.208811 | worse |
| `guard=2` | 0.208982 | worse |
| `facets="all"` | 0.208824 | slightly worse |
| `facets=False` | 0.209738 | worse |
| `kappa=0.8` | 0.208769 | worse |
| `kappa=0.6` | 0.208836 | worse |
| `rho=0` | 0.208715 | effectively neutral |
| `rho=0.02` | 0.208718 | effectively neutral |
| `rho=0.10` | 0.208730 | effectively neutral |
| `rho=0.20` | 0.208747 | effectively neutral |
| prior `G/sB` | 0.208820 | mixed, worse overall |
| prior `half/sB` | 0.208831 | mixed, worse overall |
| prior `zero/learned` | 0.210563 | clearly worse |
| `tau_scale=1.2` | 0.209175 | worse |
| `w_od=1.2` | 0.209186 | worse |

## Packaged-variant results

- `submissions/dist/v1_3_named_facets_noguard.zip`: current safest offline candidate.
- v1.4 lexical kernel: experimental; not promoted without stronger validation.
- v1.5 metadata/IRT hybrid: rejected; replay worsened `real_webagents`.
- v1.6 sparse-label damping: rejected after comparison.
- v1.7 adaptive jump cap: rejected after same-trajectory public screen; mean delta was only `+0.001717` and it lost on several benchmarks.
- Mistral BLE bundle: built and mock-tested, but requires an organizer-approved API and a locally supplied key; no key is embedded.

## Interpretation

The measurable offline gains are small. The most promising unvalidated direction is a mild likelihood tempering around `w_od=0.8`, possibly combined with `tau_scale` near `0.8`. It is not yet strong enough to replace v1.3 without a full public replay and Codabench confirmation.

## Files and methods

- Core engines: `src/paiec/models/v1_1.py`, `src/paiec/models/v1_2.py`, `src/paiec/models/v1_3_kernel.py`.
- Public comparison: `tools/compare_variants_public.py`.
- Experimental builders: `tools/build_v1_4_kernel_zip.py`, `tools/build_v1_5_hybrid_zip.py`, `tools/build_v1_6_damped_zip.py`, `tools/build_v1_7_adaptive_zip.py`.
- Mistral packaging: `tools/build_mistral_ble_bundle.py`.

