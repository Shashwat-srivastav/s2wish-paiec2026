# 26 — E7: priors from the enlarged training pool (EXP-0037)

Date: 2026-10-08. Pre-registration: `docs/preregistrations/E7_expanded_pool_preregistration.md`.
Personas: Bayesian Statistician, LLM/Subject Evaluation Scientist, Cold-Start Specialist.
Code `src/paiec/analysis/e7_expanded_pool.py`; outputs `experiments/runs/e7/`, `artifacts/e7/`.

## Result in one line
The model-name ability prior helps a little (−1.0e-3 ALC), below the −2e-3 bar; refitting the benchmark level on the
enlarged pool hurts (+3.6e-3). **v1_3 stays.** The pre-registered expectation (−2 to −6e-3) was wrong.

## Pool actually usable
281 benchmarks listed, 203 with binary rows. Only 930 of 14,201 pairs with ≥80 items are 2025–26 releases (69% of
pool pairs have no release date, 57% no `normalized_name`). The primary fit per fold used 46 benchmarks, ≈870 pairs,
≈220 names (64–65 in ≥3 benchmarks) — well under the 78 benchmarks / 97 names assumed from the subject tables.
Name ability variance 0.09–0.13 vs between-benchmark residual 0.45–0.47: about 0.2 of the variance is shared across
benchmarks. Provider/release-date fallback: cross-validated R² 0.02.

## Primary (R2, 4 folds, seeds 0–9; base 0.208722 reproduced; ΔALC ×1e-3)
| Arm | Δ [pair-bootstrap 95% CI] | Per fold (mswe / rwa / rcb / math) | ΔB0 / ΔB1 |
|---|---|---|---|
| hyper (refit level, τ, σ_β) | +3.55 [+2.74, +4.29] | +4.2 / +5.1 / +6.9 / −2.0 | +8.9 / +6.4 |
| subject (name ability prior) | −1.02 [−1.54, −0.52] | −1.8 / −0.7 / −1.5 / −0.1 | −3.7 / −2.0 |
| subject + hyper | +2.51 [+1.64, +3.34] | +2.8 / +4.3 / +5.4 / −2.5 | +4.5 / +4.4 |
| subject, all release years | +0.38 [−0.67, +1.42] | +4.1 / +1.4 / −2.5 / −1.5 | |
| control (names permuted) | +0.22 [−0.06, +0.48] | | |

Supplementary, not pre-registered (chosen after seeing results; not usable as evidence without fresh folds):
two-way fit −1.33 [−1.95, −0.75], 4/4 folds negative; name matches only −0.74; priors reassigned within fold −0.45.

Name-match coverage: 37–48% of fold subjects; correlation of prior with true pair effect among matched subjects
0.22–0.26 (0.54 on matharena); prior sd ≈0.15 logit against a true sd of 0.9–1.3.

## Secondary (47 new benchmarks, 896 pairs; ΔBrier ×1e-3)
All: B0 −2.36 [−5.50, +0.38], B1 −1.16 [−2.41, −0.12]. Agentic + maths (16 benchmarks): B0 −2.33 [−6.3, +2.2].
Knowledge/reasoning: −0.07. The adoption condition (≤ −3 at B0 on agentic/maths) is not met.

## Reading
- **CLAIM:** cross-benchmark model-name ability transfers, but weakly: ≈ −1e-3 ALC, of which about 40% is a small
  level shift (reassigned priors still gain −0.45) and ≈ −0.6e-3 is name-specific — the same size as EXP-0021.
- **CLAIM:** the enlarged pool is much easier than our folds (pool level +0.25 to +0.30 logit vs −0.3 to −0.85), so
  a level refit moves the starting point the wrong way. The competing hypothesis on level is supported.
- **Why the subject oracle (−0.032 at B0) is not reachable this way:** ability is mostly benchmark-specific (shared
  variance ≈0.2), fewer than half the evaluation subjects have a name match, and the pool's metadata is sparse.
- **Leakage check is vacuous, not passed:** keeping SWE and matharena relatives changes nothing because they contribute
  no eligible 2025–26 pairs.
- Decision-table row fired: "between −0.002 and −0.0005: keep as a documented option". `hyper`: rejected.

## Deviations
Pool pairs need ≥80 items and benchmarks ≥5 eligible pairs; key is lower-cased `normalized_name`; σ_β from a Rasch fit
on `item_id` with 20k-item subsample; prior sd τ·√(1−reliability); control uses one permutation seed; secondary
budget 1 computed exactly; domain bucketing by first tag misfiles a few benchmarks; supplementary arms added.

## Consequence
Three independent lines now agree that the remaining gap cannot be closed by priors: prior shape (E5), metadata (E1),
and a 40-times larger pool (E7). What is left is item-level information (E6: needs ρ ≥ 0.5) or accepting ≈0.19 as
the method's level and competing on robustness, the common summative subset and the report.
