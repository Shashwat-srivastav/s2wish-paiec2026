# 21 — Persona critique of the whole programme (2026-10-02)

Four read-only panels, routed per orchestrator_merged.md §4 activation rule. Scratch evidence:
a temporary scratch folder (`panel{A,B,C,D}/`, not kept). No project file was modified by the panels.

| Panel | Personas | Scope |
|---|---|---|
| A Model | Measurement Theorist, Bayesian Statistician, Mathematical Minimalist, Meta-Learning Scientist, Active-Learning Theorist | v1_1 internals, where loss sits |
| B Validation | Cold-Start Specialist, Competition Optimizer, Experimental Designer, Calibration Purist | proxy fidelity, power, headroom, strategy |
| C Red team | Red-Team Researcher, Reviewer From Hell, Leakage Hunter, Mechanistic Skeptic | break submission, claims, leakage |
| D Systems | ML Systems Architect, Experimental Designer, Literature Hunter, PI-efficiency | reproducibility, directory, process, literature |

## 1. What survived attack (no veto)
- v1_1 zip: kit check OK, smoke OK (0.2095), zip byte-identical to folder, finite outputs on all adversarial inputs
  (unseen benchmark, empty/garbage labels, extreme benchmarks), order-invariant, 0.55 s/call, no network (C, D).
- Params regenerate bit-exactly; no eval-derived content in params/priors (C, D).
- Simulator vs formative (F2): v1 at 61st pct, v1_1 at 7.6th pct of simulated 8-pair draws; hidden B0 easier than local
  but no gross mismatch (B).
- Posterior update is calibrated: post-label slopes 0.95–1.04 at every budget; per-budget recalibration null/harmful (A).

## 2. Key facts the critique established
1. **Item pooling was effectively OFF on the leaderboard.** 7/8 formative pairs were alone on their benchmark, so
   v1_1 predicted one constant per pair (A F1). The pair-constant floor is 0.179 of R2's 0.209 (A) and 0.153 of the
   formative 0.176 (B). Item-level ideas that rely on cross-subject pooling (e.g. the extreme-difficulty mixture from D)
   therefore act on almost nothing in the hidden draw.
2. **Runtime item groups are unused.** `item_features` carries competition / lang / website / paper; between-group share
   of item-difficulty variance is 0.35 / 0.02 / 0.23 / 0.38 (A F2). This works with one subject per benchmark.
3. **v1_1 is a symmetric trade, not insurance.** Fresh seeds: v1_1−v1 = −0.07e-3 (R2), CI spans 0; better only on
   matharena (−7.3e-3), worse on the other three (+1.1…+3.9e-3) (C). The only exactly paired formative pair (an extreme
   benchmark, the case v1_1 targeted) got worse 0.041→0.052 (B F3). Hidden assumption: hidden base rates centre on 0.5.
4. **The adoption rule's "pair CI" ignores between-benchmark variance**; with 4 folds the real MDE is 0.7–3.5e-3, and
   "≥3/4 folds" passes 31% under the null (B F1). Bars were applied inconsistently (v1_1 adopted at −0.11e-3, metadata
   prior rejected at −0.58e-3 with 4/4 folds) (B F2).
5. **Yield ≈ 1 in 12 experiments** (D). Root cause: 4 LOBO folds cannot verify effects < ~5e-3.
6. **Final evaluation ≈ 8 pairs**: a 1e-3 gain is a coin flip on the final; summative also grades code/report (B F4).
7. **Matharena score target ≠ label target** (trial-weighted eval mean 4.9 pts below item mean) — data artefact, do not
   tune on it (A F5).

## 3. Disagreements (no-consensus format)

| Question | Position A | Position B | Evidence | Source of disagreement | Decisive experiment | Decision |
|---|---|---|---|---|---|---|
| Prior centre G/2 (v1_1) vs G (v1) | C/B: G/2 is a bet on easy hidden benchmarks; G is +2e-3/bench better if hidden ≈ public | v1_1 rationale: halves worst-case regret | Seeds: symmetric trade; paired formative pair worse | Unknown hidden base-rate distribution | Not decidable from 4 folds → replace the point bet with X1 (heavy-tailed / mixture offset) that adapts fast after 1–3 labels | Pre-register X1; keep v1_1 shipped until then |
| Sub-threshold gains | B (Optimizer): bundle metadata prior + A3/A8 ≈ −1e-3 | D (PI-efficiency): stop micro-tuning | Each component 4/4 folds or CI≠0 but < bar | Power vs research efficiency | One pre-registered bundle test, then stop the line either way | Run once |
| Facet effects P1 | A: −1.02e-3 R2, −1.05e-3 R1, works with 1 subject | D: effects < 5e-3 unverifiable | Matharena ≈ half the gain; mswe neutral only with π-averaging + centring | Fold count | Shuffled-facet control (|Δ|<0.2e-3) + no fold > +0.5e-3 | Top research priority |
| Extreme-difficulty item mixture | D: −1…−4e-3 (21–52% never-solved items) | A: acts through cross-subject pooling, absent in hidden draw | A F1 | Regime (R1/R3 vs R2) | Measure in R2 only | Deprioritised |
| 2PL / guessing | Measurement Theorist | Minimalist veto | Constant per pair in R2; slopes unidentifiable | — | — | Dead |

## 4. Ranked plan to improve the score
| # | Item | Expected effect | Cost | Kill test |
|---|---|---|---|---|
| 0 | **Engineering freeze (Architect veto for next upload):** git init + .gitignore (.hf_cache, runs/*.parquet/npz, __pycache__, zips), `v1_1 --build/--check`, `scripts/build_zip.py` + SHA256SUMS, pytest engine↔package equivalence (<1e-9), pinned deps | protects every later gain; summative grades code | ½ day | — |
| 1 | **P1 item-group (facet) random effect**, own-subject, centred, π-averaged {0,.1,.3}, exact fallback | −1.0e-3 (B15/B31) | 1 day | hier-bootstrap UB<0 in R2; no fold > +0.5e-3; shuffled facets |Δ|<0.2e-3 |
| 2 | **X1 robust offset prior** (t-tail / mixture incl. near-0 and near-1 components), replaces G vs G/2 bet | up to −3…−4e-3 on draws with an extreme benchmark; ≈0 mean | 1 day | R2 ΔALC>0, or worse max regret on δ-sweep |δ|≥3, or B1 regression >0.5e-3 |
| 3 | **Degenerate-key guard** (no cross-subject collisions → don't treat same-subject same-key as same item) | up to −0.04 B1 on content-poor benchmarks | ½ day | matharena duplicates regress |
| 4 | **One bundle test**: metadata prior + A3/A8 (+ items 1–3) vs v1_1 | −0.8…−1.2e-3 | ½ day | bundle Δ > −0.5e-3 or any fold > +0.5e-3 |
| 5 | **Decision statistics repair**: t3 interval on fold means, new-benchmark predictive interval, worst-fold/worst-δ guard; one threshold everywhere | prevents false adopts/rejects | few hours | — |
| 6 | **Check measurement-db for a new revision** (pinned bc8204d811) — more benchmarks is the highest-value input | unlocks verification | minutes | — |
| 7 | Claim/doc fixes (below) | report credibility (graded) | 2 h | — |

**Stop list:** acquisition policies (except P2 after P1), Jev/embedding distillation, prior-shape micro-tuning,
per-budget recalibration, 2PL, unpaired formative comparisons.

**Submission use:** only for method classes expected ≥ ~3e-3 or engineering verification; track recurring
subject/benchmark IDs as paired comparisons.

## 5. Claim corrections required (Reviewer From Hell)
- README:15–16 and registry EXP-0004 quote the withdrawn 40-draw R1 numbers (v1 0.2119, v0 0.2177) next to v1_1's
  100-draw 0.2151; same-protocol values are v1 0.2154, v0 0.2203.
- "Halves worst case at no mean cost": cut is 39–50% depending on table; v1_1 is worse on 3/4 R2 folds. Reword to
  "reduces worst-case regret by ~40–50%; mean-neutral overall, trades matharena gain for small losses elsewhere".
- "Safer on easy hidden benchmarks" contradicted by the only paired formative pair — mark untested.
- build_report.py:596/:911 metadata prior magnitude stale (should be −0.58e-3 R2, −0.91e-3 R1, 4/4 folds).
- build_report.py:355 compares v1 formative to R1 0.212; R2-like local was 0.209.
- Report 18:62 "Codabench forbids an LLM at runtime" — false (HF models via models.txt, allowlisted APIs).
- Report 20 C2 heading overclaims (rewrite-ratio CI lower bound 0.76 < 0.8 bar); "domain-normalised question" is
  untested mechanism. Report 17 C1/C3 mechanism lines are Level 1.
- Thresholds: 2e-3 in reports 15/17 vs 1e-3 in pre-registration — unify.
- Registry EXP-0002 has 22/26 fields; EXP-0026..0032 retro-registered (flag); decisions.jsonl lacks experiment_id;
  E1/E2 STOP decisions not logged; PI override rationale for P3-v1_1-replace missing.
- README stale (reports 00–16, submissions v0/v1, test time); root `S2WISH Research Report.html` stale duplicate.
