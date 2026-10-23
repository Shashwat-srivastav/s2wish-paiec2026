# Pre-registration — E5 / X1 robust offset prior (written 2026-10-02, before any X1 result exists)

Personas: Bayesian Statistician (BS), Cold-Start Specialist (CS), Red-Team Researcher (RT),
Competition Optimizer (CO). Source of the task: report 21 §2 item 3, §3 row 1, §4 item 2.
Decision boundaries below are fixed now; any deviation is reported as a deviation in report 24.

## Why
v1 centres the hidden-benchmark offset prior at the training mean G; v1_1 centres it at G/2 with a
re-learned width. Report 21 showed this is a symmetric trade, not insurance (R2 Δ = −0.07e-3 on fresh
seeds; matharena −7.3e-3, the other three folds +1.1…+3.9e-3), and the only exactly paired formative
pair (subject_<hidden> / benchmark_<hidden>, an extreme near-deterministic benchmark: B31 Brier 0.001) got
worse under v1_1 (ALC 0.041 → 0.052, B0 0.160 → 0.201). The centre is a point bet on the hidden base
rates. We want a prior that (a) does not depend on that bet and (b) moves quickly to an extreme
benchmark after 1–3 labels.

## Hypothesis vs competing hypothesis
- **H-X1.** A mixture offset prior (components at G, G/2 and/or near logit ±3) is non-inferior to v1_1
  on real LOBO data and has strictly lower worst-case regret on the A3 δ-sweep than both v1 and v1_1,
  because after 1–3 labels the posterior mass moves to the right component (the mixture *is* Bayesian
  model averaging over centres, weights ∝ prior weight × marginal likelihood of the revealed labels,
  so candidates (i) and (iii) of the brief are the same model).
- **Competing H0.** With only 4–5 training benchmarks and pair-ability spread τ ≈ 1.1, 1–3 labels do
  not discriminate components; any mass placed away from G costs B0/B1 on real data (as the report-17
  wide-mixture candidates did: +0.26…+0.94e-3) and the worst-case gain does not beat v1_1.

## Candidates (fixed list; all keep every other v1_1 setting: ρ0 = 0.05 "momd", κ = 1, w_od = 1)
Notation: t3(c, s) Student-t ν = 3 on the 121-point μ-grid, each component normalised on the grid;
sB = training sd of benchmark means (v1); s½ = max(0.5, rms(means − G/2)) (v1_1).
| name | prior | new hyper-parameters |
|---|---|---|
| v1 | t3(G, sB) (+ v1 item settings) | — (reference) |
| v1_1 | t3(G/2, s½) | — (champion) |
| X1-bma | ½ t3(G, sB) + ½ t3(G/2, s½) | 0 |
| X1-ext10 | 0.9 t3(G/2, s½) + 0.05 t3(−3, 1) + 0.05 t3(+3, 1) | ε = 0.10 |
| X1-ext20 | 0.8 t3(G/2, s½) + 0.10 t3(−3, 1) + 0.10 t3(+3, 1) | ε = 0.20 |
| X1-bma-ext10 | 0.45 t3(G, sB) + 0.45 t3(G/2, s½) + 0.05 t3(±3, 1) each | ε = 0.10 |
| X1-Gext10 | 0.9 t3(G, sB) + 0.05 t3(±3, 1) each | ε = 0.10 |
| X1-t1 | t1(G/2, s½) (heavy tail) | ν = 1 |
Extreme location ±3 and scale 1 are fixed a priori (logit ±3 ≈ base rate 0.05/0.95 before the
pair-ability spread); they are not tuned. The only tuned quantity is ε ∈ {0.1, 0.2} (2-point grid,
disclosed), and the choice among the fixed list is made by the decision rule below.

## Procedure
1. Implement as a drop-in prior kind `("x1", components)` in `src/paiec/models/x1_prior.py`;
   `install()` wraps `paiec.models.v1_1.log_prior` so both the vectorised predictor and the streaming
   `_Group` use it; old prior kinds are delegated unchanged (equivalence check: v1_1 config through the
   wrapper reproduces v1_1 ALC bit-exactly).
2. Real data, LOBO over the 4 eval folds, hyper-parameters re-fit per fold (ladder.hyper_for):
   R2 with 10 episode seeds (0–9; ≥ 5 required), R1 with 100 seeds (0–99).
3. A3 δ-sweep: synth_D(B, δ, s), δ ∈ {−3,−2,−1,0,+1,+2,+3}, synthetic seeds s ∈ {0,1,2};
   R2 episode seed [0], R1 seeds 0–49 per synthetic seed. ALC averaged over synthetic seeds.
4. Extreme benchmarks: per fold, δ chosen by bisection so the synthetic base rate is 0.02 and 0.98
   (seed 0 data), synthetic seeds {0,1,2}, R2 and R1 as in step 3.
5. Regret at each δ: ALC_m(δ) − min over the full candidate set {v0, v1, v1_1, all X1}. Worst-case
   regret = max over δ ∈ {−3…+3}. Same reference set for every row.

## Metrics
- Real: mean ALC; Δ vs v1_1 (×1e-3) with pair-bootstrap CI, per-fold Δ, **t3 interval on the 4 fold
  means** (mean ± t_{0.975,3}·sd/√4) and **predictive interval for a new benchmark**
  (mean ± t_{0.975,3}·sd·√(1+1/4)); Δ per budget (B0…B31).
- Sweep: ALC by δ, regret by δ, worst-case and mean regret (R2 primary, R1 secondary).
- Extreme: ALC and B0/B1/B3 at base rates 0.02 / 0.98, Δ vs v1 and v1_1.

## Decision rule (fixed now)
A candidate is **eligible** iff all of:
1. R2 mean ΔALC vs v1_1 ≤ +0.2e-3 (non-inferior), and R1 mean ΔALC vs v1_1 ≤ +0.2e-3;
2. R2 worst-case δ-sweep regret strictly lower than both v1 and v1_1 (R1 worst-case regret must not
   exceed v1_1's — secondary guard);
3. no B1 regression vs v1_1 > +0.5e-3 (R2 or R1, mean Δ at budget 1).
Among eligible candidates: fewest new hyper-parameters first (Minimalist), then lowest R2 worst-case
regret, then lowest R2 mean ΔALC.

| outcome | action |
|---|---|
| ≥ 1 eligible | ADOPT the selected X1 as the offset prior of the next candidate (v1_2 combines it with the facet change); v1_1 stays shipped until the combined candidate is red-teamed |
| none eligible, some X1 fails only rule 2 by < 0.5e-3 | REJECT; record as "no robust gain"; keep v1_1 prior; stop the prior-shape line (report 21 stop list) |
| none eligible because of rule 1 or 3 | REJECT; H0 supported (mass away from G costs real data); stop the line |
| X1 wins only at |δ| = 3 or base rate 0.02/0.98 and loses at |δ| ≤ 1 | report as insurance trade, not adoption, unless eligible by the rule |

## Cost
≤ 2 h compute (one sweep ≈ 20 min for 9 configs × 3 synthetic seeds; real R1/R2 ≈ 5 min).
