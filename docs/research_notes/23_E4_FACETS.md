# 23 — E4: item-group facet effect (P1) and the degenerate-key guard (2026-10-02)

Personas: Measurement Theorist, Bayesian Statistician, Mathematical Minimalist; Cold-Start Specialist veto.
Pre-registration: `docs/preregistrations/E4_facets_preregistration.md`, written before any run.
Code: `src/paiec/models/v1_2.py`. It imports v1_1, which is unchanged.
Runs: `experiments/runs/e4/{check,real,merge}.{json,log}`. Run with `python -m paiec.models.v1_2 --step check,real,merge`. Total compute was about 6 min.

## Protocol (as pre-registered, no deviations)
- LOBO over 4 folds.
- Regimes: R2 is primary (10 seeds, one pair per episode). R1 uses draws 0–99.
- Δ = candidate − v1_1, in ALC ×1e-3. Negative is better.
- Intervals reported:
  - `pair` is a fold-stratified pair bootstrap.
  - `hier` is a benchmark→pair bootstrap.
  - `t3` is the t-interval on the 4 fold means.
  - `PI` is the predictive interval for a new benchmark.
- v1_1 reference ALC: R2 0.20974, R1 0.21505. These differ from report 17 because the R2 seeds differ (10 here vs 5 there).
- Implementation checks all passed with max |diff| = 0 (`check.json`):
  - facets and guard switched off reproduce v1_1.
  - an emptied feature map reproduces v1_1.
  - B0 is unchanged in every arm.

## Results — real data

| arm | R2 Δ | R2 folds (mswe / rwa / rcb / math) | R2 t3 / PI | R1 Δ | R1 folds | R1 t3 / PI |
|---|---|---|---|---|---|---|
| **facets** (all fields) | **−0.91** pair[−1.10,−0.74] hier[−1.61,−0.24] | +0.02 / −0.93 / −0.85 / −1.89 | [−2.16,+0.33] / [−3.69,+1.86] | **−0.76** hier[−1.49,−0.14] | +0.09 / −0.69 / −0.68 / −1.76 | [−1.97,+0.45] / [−3.46,+1.94] |
| guard G1 | +0.09 pair[+0.04,+0.15] | +0.04 / 0 / 0 / +0.31 | [−0.15,+0.33] | +0.09 | +0.04 / 0 / 0 / +0.30 | [−0.14,+0.32] |
| **both** | **−0.83** hier[−1.40,−0.21] | +0.06 / −0.93 / −0.85 / −1.58 | [−1.90,+0.25] / [−3.23,+1.58] | **−0.67** hier[−1.27,−0.11] | +0.13 / −0.69 / −0.68 / −1.46 | [−1.71,+0.36] / [−2.98,+1.64] |
| shuffled facets (control) | +0.15 pair[+0.09,+0.22] | +0.16 / −0.10 / +0.31 / +0.24 | [−0.13,+0.44] | +0.16 | +0.21 / −0.08 / +0.22 / +0.27 | |
| facets_named (secondary) | −1.02 | +0.02 / −0.93 / −0.85 / −2.30 | [−2.54,+0.51] | −0.87 | +0.09 / −0.69 / −0.68 / −2.22 | |
| guard2 G2 (secondary) | +0.26 | +0.11 / 0 / 0 / +0.92 | | **+2.11** | +0.62 / +2.72 / +2.68 / +2.42 | [+0.51,+3.70] |

**Per budget, mean over folds (Δ Brier ×1e-3)**

| arm | regime | B0 | B1 | B3 | B7 | B15 | B31 |
|---|---|---|---|---|---|---|---|
| facets | R2 | 0 | +0.01 | −0.29 | −0.72 | −1.76 | −3.61 |
| facets | R1 | 0 | +0.02 | −0.24 | −0.71 | −1.44 | −2.83 |
| guard | R2 | 0 | +0.01 | +0.03 | +0.08 | +0.18 | +0.30 |
| both | R2 | 0 | +0.02 | −0.27 | −0.65 | −1.58 | −3.31 |
| both | R1 | 0 | +0.03 | −0.20 | −0.61 | −1.26 | −2.63 |
| shuffled | R2 | 0 | +0.01 | +0.06 | +0.16 | +0.32 | +0.44 |

**Per fold × budget, `both` R2 (B0…B31)**

| fold | B0…B31 |
|---|---|
| mswe | 0, +.01, +.02, +.12, +.09, +.13 |
| rwa | 0, 0, −.33, −.66, −1.81, −3.74 |
| rcb | 0, 0, −.28, −.63, −1.65, −3.36 |
| math | 0, +.05, −.47, −1.41, −2.94, −6.27 |

The facets arm in R2 by fold:
- multi_swebench (one field, `lang`, 2% variance share) is neutral: +0.09 at B7, −0.03 at B31.
- matharena B31 −7.31.

R2 curve, v1_1 → both: .2456/.2285/.2126/.2002/.1912/.1868 → .2456/.2285/.2123/.1995/.1896/.1835.

**Facet coverage and π weights.**
- Share of targets whose prediction changes: R2 0.75 / 0.50 / 0.40 / 1.00. R1 0.90 / 0.34 / 0.34 / 0.94.
- The posterior π weights stay near the prior. The mean over pairs is (0.40–0.43, 0.39–0.40, 0.18–0.20). The pair-level marginal likelihood barely separates π at budgets ≤ 31, so in practice the averaging is a fixed mixture.

## Results — synthetic merges (`redteam.rekey`, random m items → 1 key; guard only)

| m | regime | guard G1 Δ | folds | guard2 Δ |
|---|---|---|---|---|
| 5 | R2 | **−2.18** pair[−2.46,−1.91] hier[−4.21,−0.26] | −0.29 / −3.63 / −4.53 / −0.25 | −3.07 |
| 5 | R1 | −0.65 | −0.37 / −0.72 / −1.46 / −0.04 | −1.71 |
| 20 | R2 | **−5.76** hier[−10.0,−1.7] | −1.41 / −8.57 / −10.85 / −2.20 | −8.49 |
| 20 | R1 | −1.37 | −0.95 / −1.14 / −2.71 / −0.67 | −4.29 |

In R2, G2 is identical to switching the item level off (ALC 0.21000 on real data and on every merge level). That is why it helps more on merges and costs +2.1e-3 on real R1.

## Decision (pre-registered table)

| component | criterion | value | pass |
|---|---|---|---|
| facets | R2 Δ ≤ −0.5 | −0.91 | ✓ |
| | no R2 fold > +0.5 | worst +0.02 | ✓ |
| | shuffled \|Δ\| < 0.2 (R2) | +0.15 (R1 +0.16) | ✓ (margin small) |
| | R1 Δ ≤ 0 and no R1 fold > +0.5 | −0.76, worst +0.09 | ✓ |
| **→ facets ADOPT** | evidence grade | t3 UB +0.33 > 0 | **unverified at 4 folds** (hier UB −0.24 < 0 passes report 21's kill test) |
| guard G1 | real R2 / R1 ≤ +0.2 | +0.09 / +0.09 | ✓ |
| | matharena ≤ +0.5 | +0.31 / +0.30 | ✓ |
| | synthetic m=5 or 20 R2 ≤ −0.5 | −2.18 / −5.76 | ✓ |
| **→ guard G1 ADOPT** (insurance) | | | |
| both | interaction within 0.3 | −0.83 vs sum −0.83 (≈ 0 interaction) | ✓ |
| | fold guard | worst R2 +0.06, R1 +0.13 | ✓ |
| **→ SHIP `both`** | | | |

Not shipped:
- `facets_named` is better (−1.02) but is a secondary arm. Choosing it now would be post-hoc. Dropping matharena `problem_idx` helps only on matharena.
- G2 fails on real R1 (+2.11).

**Shipping config** (`v1_2.cfg_of(facets="all", guard=1)`):
- v1_1 `BASE2`, which is prior t3(G/2, learned), ρ0 = 0.05 momd, κ = 1, item level on, plus:
  - facets on every `key=value` field of item_features, own-subject labels, centred.
  - π ∈ {1e-3, 0.1, 0.3} with prior weights 0.4/0.4/0.2.
  - guard: ρ_k ← max(ρ_k, 0.5) when the target key has no cross-subject label.
- Exact fallback to v1_1 at B0, for items without features, and for labelled keys.

## Caveats
- The shuffled control is positive (+0.15) rather than ≈ 0. Non-informative facets cost a little, because they add noisy redistribution. The real effect sits about 1.07e-3 below that control.
- 4 folds give a t3 interval and a new-benchmark PI that both include 0. The expected gain on the 8-pair final is about −0.8e-3, which is a coin flip against noise (report 21 §2.6).
- Matharena carries the largest gain, and the guard's whole real-data cost is matharena duplicates plus mswe.
- The guard's benefit is shown only on synthetic merges.
- Before upload this needs the standalone `submissions/` port and an engine↔package equivalence test. Both are for the systems agent and are not done here.
