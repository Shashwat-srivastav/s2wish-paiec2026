# 20 — E2: is Jev's difficulty signal memorisation, and can a local model reproduce it?

Pre-registration: [`docs/preregistrations/E1_E2_preregistration.md`](../experiments/E1_E2_preregistration.md) §E2
(written before any result; decision boundaries applied mechanically below).
Code: `src/paiec/analysis/e2_jev.py` (`python -m paiec.analysis.e2_jev e2a|e2b|e2c [--encoder REPO]`).
Raw output: `experiments/runs/e2/` (`e2a.json`, `e2a_diag_textless.json`, `e2a_perturb_*`, `e2b.json`,
`e2b_pairs.parquet`, `e2b_bt_*.parquet`, `e2c_<encoder>.json`, `e2c_emb_*.npz`, `*.log`).

Personas:
- **RS** Representation Scientist (lead).
- **LH** Statistical Leakage Hunter (memorisation verdict).
- **SA** ML Systems Architect (packaging and runtime).
- **MM** Mathematical Minimalist (internal sceptic).

Conventions are the same as report 13:
- β is the per-benchmark Rasch difficulty (`item_repr.rasch`), centred, with higher meaning harder.
- Jev is the pinned `jev-1.13.0` `jev_difficulty`, scored from item content only. No benchmark name and no `item_features` were sent.
- CIs are 95% bootstraps (2000 replicates) that resample duplicate-content groups.

## Summary

| step | decision row that fired | action |
|---|---|---|
| E2a memorisation | **"post ρ ≥ 0.15, CI excludes 0, and perturbed ρ ≥ 0.8 × original" → content-based** | Proceeded to E2b/E2c. The sensitivity run (uncertain dates excluded) gives the same row. |
| E2b pairwise BT | **BT beats pointwise by ≥ 0.05 on 2 of 5 benchmarks (needs 3)** | Kept the pointwise target. |
| E2c distillation, MiniLM-L6 | **retains < 70% of Jev's ρ** (aggregate −0.17, 0 of 5 folds) | One fallback encoder. |
| E2c distillation, fallback mpnet-base | **retains < 70%** (aggregate 0.025, 0 of 5 folds); the one fallback is used up | **STOP E2.** No ship; v1_2 not built |

**Bottom line.**
- Jev's signal is **not** explained by memorisation:
  - It is at least as strong on problems released after mid-2025 as on earlier ones.
  - It survives surface rewrites.
- The offline gain from Jev reproduces on the v1_1 engine: −5.8e-3 in R2, −6.4e-3 in R1. Both pass the ALC threshold.
- A frozen sentence encoder with a ridge head **cannot** reproduce that signal across benchmarks. No v1_2 was built.

---

## E2a — Memorisation tests

### Procedure as run
1. **Release-date split (MathArena).**
   - Each competition is mapped to its public release month (table below).
   - A duplicate-content group takes the earliest month among its competitions. For example, 13 apex_2025 items are verbatim SMT 2025 problems, so they count as 2025-04.
   - Items are split into pre (< 2025-07) and post (≥ 2025-07).
   - Spearman(jev, β) is computed separately for each half, with a cluster bootstrap over content groups. The pre − post difference uses independent bootstraps, paired by replicate index.
   - Sensitivity run: competitions flagged *uncertain* are removed.
2. **Surface perturbation (all benchmarks).**
   - Sample: 300 items, 60 per benchmark. Seed 0. One item per distinct content.
   - Each item is rewritten by deterministic rules (`RULES` in the code), and every application is logged in `e2a_perturb_rules_log.csv`.
   - Two Jev runs, each against a fresh cache: the original text again (a **retest**, to measure Jev's own noise) and the perturbed copy.
   - Metrics: mean |Δ difficulty|, and the mean over benchmarks of within-benchmark Spearman with β (the same statistic as report 13's "mean ρ") for original, perturbed and retest scores.
   - Paired bootstrap over items, stratified by benchmark.

**Release-month map** (✱ = uncertain; excluded in the sensitivity run).

| competition | n | month | post? | note |
|---|---|---|---|---|
| aime_2025, aime_2025_I, aime_2025_II | 155 / 46 / 16 | 2025-02 | pre | AIME I 6 Feb, II 12 Feb 2025 |
| hmmt_feb_2025 | 94 | 2025-02 | pre | HMMT February |
| usamo_2025 | 46 | 2025-03 | pre | 19–20 Mar 2025 |
| kangaroo_2025_* (6 levels) ✱ | 336 | 2025-03 | pre | contest 20 Mar 2025; public posting date of the problem text undocumented |
| smt_2025 | 168 | 2025-04 | pre | Stanford Math Tournament |
| brumo_2025 | 93 | 2025-04 | pre | Brown Univ. Math Olympiad |
| cmimc_2025 ✱ | 209 | 2025-04 | pre | spring 2025, month not verified (pre either way) |
| imo_2025 | 141 | 2025-07 | post | 15–16 Jul 2025 |
| imc_2025 ✱ | 47 | 2025-08 | post | late Jul / early Aug (post either way) |
| apex_2025 ✱ | 25 | 2025-08 | post | MathArena Apex set; compiled from 2025 contests with varying dates |
| apex_shortlist ✱ | 114 | 2025-08 | post | compilation; publication month not verified |
| miklos_2025 ✱ | 13 | 2025-10 | post | autumn 2025, month not verified |
| hmmt_nov_2025 | 61 | 2025-11 | post | |
| putnam_2025 | 24 | 2025-12 | post | 6 Dec 2025 |
| arxivmath_1225 / _0126 / _0226 | 17 / 23 / 32 | 2025-12 / 2026-01 / 2026-02 | post | arXiv-derived monthly sets |
| aime_2026, aime_2026_I | 31 / 31 | 2026-02 | post | identical content |
| hmmt_feb_2026 | 33 | 2026-02 | post | |

**Perturbation rules.** Benchmark identity only selects rules locally and is never sent to Jev. A guard asserts that no benchmark name appears in any state.

| pool | rules (in order) |
|---|---|
| matharena | Person names → fixed disjoint list. All-caps letter runs inside math mode relabelled by a fixed shift-15 bijection, with `\mathbb{}`, `\text{}` etc. protected. Leading instruction paragraph moved to the end. **No constants changed**, because they determine the answer. |
| multi_swebench, swe_rebench | snake/camel/PascalCase identifiers → synthetic names from a word list (consistent within an item; error classes and well-known names on a stoplist). Names. Integers 1–999 in prose → n+1 (code spans, versions, decimals, times and `#refs` untouched). Title line moved after the body. |
| researchcodebench | Identifiers; names; leading instruction paragraph moved to the end. Paper numbers left intact. |
| real_webagents | Person and clone-site names (Zilloft→Homefinder, …). Integers → n+1, with ordinals fixed. Short coordinated lists reversed. Sentence-initial instruction verbs → fixed synonyms (Find→Locate, Book→Reserve, …). |

280 of 300 items (93%) changed: matharena 95%, multi_swe 100%, webagents 73%, rcb 100%, swe_rebench 98%. Example before/after pairs (`e2a_perturb_examples.json`):

1. **matharena** (math labels + preamble move). Jev 0.740 → 0.810 (retest 0.735).
   - Before: `You are given a difficult question. … Put the final answer … \boxed{}.` ⏎⏎ `Let $p$ and $q$ be twin primes with $q=p+2$. … \min\{n\in\mathbb N_{>0} …`
   - After: `Let $p$ and $q$ be twin primes with $q=p+2$. … \min\{n\in\mathbb O_{>0} …` ⏎⏎ `You are given a difficult question. …`
   - This example also shows a **rule defect**: an unbraced `\mathbb N` was relabelled (see deviations).
2. **multi_swebench** (8 identifiers, 5 numbers, title move). Jev 0.608 → 0.535 (retest 0.603).
   - Before: `` Add `JsonParser.getNumberTypeFP()` `` ⏎ ``Currently `JsonParser` has method `getNumberType()` … `float` (32-bit binary FP), `double` (64-bit …``
   - After: ``Currently `DeltaLichen` has method `jasperJasperMarble()` … `float` (33-bit binary FP), `double` (65-bit …`` … title at end.
3. **real_webagents** (list reversal + verb). Jev 0.260 → 0.255 (retest 0.260).
   - Before: `Create a job listing … required skills (MySQL, Docker, Kubernetes, and Jenkins) …`
   - After: `Set up a job listing … required skills (Jenkins, Kubernetes, Docker, and MySQL) …`
4. **researchcodebench** (64 identifiers, 3 names, preamble move). Jev 0.315 → 0.320 (retest 0.320).
   - The role line `You are an expert in reproducing research code from a paper.` moves from the head to the tail.
   - Code identifiers such as `replay_buffer` are renamed throughout.
5. **swe_rebench** (2 identifiers, title move). Jev 0.418 → 0.553 (retest 0.428).
   - Before: `cast_to_a1_notation(method)   bugs?` ⏎ `… rowcol_to_a1()  does not want to take a list …`
   - After: `dune_indigo_indigo(method) I believe there are bugs here … birch_zephyr_cedar() does not want …` ⏎⏎ `dune_indigo_indigo(method)   bugs?`

### Results

**Test 1 — release-date split (MathArena), Spearman(jev, β).**

| run | pre (< 2025-07) | post (≥ 2025-07) | pre − post |
|---|---|---|---|
| main (all 27 competitions) | 0.108 [−0.045, 0.520] (1178 items / 530 groups) | **0.455 [0.347, 0.554]** (577 / 368) | −0.347 [−0.526, +0.092] |
| sensitivity (✱ excluded) | 0.162 [−0.124, 0.593] (618 / 364) | **0.608 [0.469, 0.718]** (393 / 238) | −0.445 [−0.748, +0.028] |
| *diagnostic, post hoc:* text-bearing items only | 0.481 [0.370, 0.577] (659 / 528) | 0.455 [0.345, 0.551] (577 / 368) | +0.026 [−0.126, +0.169] |
| *diagnostic:* within-competition (both variables demeaned by competition) | 0.007 | 0.253 | — |

Why the pre-period ρ is low and wide (LH):
- 519 of the 1178 pre items have **no problem text at all**.
  - 183 items (aime/brumo/cmimc/hmmt/smt) carry only the system prompt `You are ChatGPT, a large language model trained by OpenAI.`
  - 336 Kangaroo items share one image-only instruction.
- Those items get one Jev score each (ties), while their β spans the full range. That depresses ρ, and as single huge clusters it widens the CI.
- With them removed (diagnostic row; not pre-registered, so it does not drive the decision), pre ≈ post (0.48 vs 0.46, difference CI includes 0).
- Recall would show the opposite pattern: a **higher** pre-period ρ.

**Test 2 — surface perturbation (300 items).** Mean of within-benchmark Spearman with β:

| | original (cached) | retest (same text) | perturbed | perturbed / original |
|---|---|---|---|---|
| all 300 | 0.180 [0.065, 0.287] | 0.182 | 0.187 [0.070, 0.295] | **1.03 [0.76, 1.44]** |
| changed only (280) | 0.181 | 0.183 | 0.191 | 1.05 [0.79, 1.54] |

Score movement:
- Mean |Δ difficulty| is 0.033 [0.029, 0.038] on the 0–1 scale, about 0.13 of one rubric level. The retest gives 0.008.
- The perturbation therefore moves scores about 4× more than Jev's own noise, but it does not degrade the ranking.
- Spearman(original, perturbed) is 0.92.
- Mean signed Δ is +0.012: copies are judged marginally harder.

Per benchmark (60 items each; within-benchmark ρ is noisy at n = 60):

| benchmark | ρ orig | ρ pert | ρ retest | ρ(orig, pert) | mean \|Δ\| |
|---|---|---|---|---|---|
| matharena | .504 | .494 | .496 | .977 | .021 |
| multi_swebench | −.046 | −.111 | −.061 | .885 | .040 |
| real_webagents | .355 | .364 | .353 | .985 | .022 |
| researchcodebench | .159 | .248 | .185 | .859 | .045 |
| swe_rebench | −.070 | −.062 | −.062 | .905 | .037 |

On the full item sets, ρ is .20/.21 for the two SWE pools (report 13). Their negative values here are 60-item sampling noise, and they are the same for original, retest and perturbed.

### Decision (mechanical)
- Post ρ = 0.455, CI [0.347, 0.554]: ≥ 0.15 and excludes 0.
- Perturbed/original = 1.03 ≥ 0.8.
- STOP conditions: none holds. The post CI excludes 0; pre − post = −0.35 is < 0.15; and 1.03 is not < 0.6.
- **Row fired: "Signal is content-based → proceed to E2b/E2c."** The sensitivity run fires the same row.

**Claim (LH).**
- **CLAIM:** Recall of published items does not explain Jev's item-difficulty ρ.
- **EVIDENCE:**
  - Post-release ρ (0.46–0.61) ≥ pre-release ρ.
  - Recall-breaking rewrites keep 103% of ρ.
- **COUNTEREVIDENCE:**
  - The 2025-07 split is a proxy. Jev's real training cutoff is not published, so "post" items could still have been seen if the cutoff is later than early 2026. It is likely earlier, given the model's version date, but this is unverified.
  - Math rewrites change labels only, not constants. A model that memorised "this AIME problem is hard" could still match.
  - The ratio CI lower bound (0.76) sits slightly below 0.8.
- **COMPETING EXPLANATION:** Jev recognises the *competition style* (IMO vs AIME), not the item. The within-competition diagnostic answers part of this:
  - Post: ρ 0.25 remains after demeaning by competition.
  - Pre: ρ 0.01, driven by the text-free clusters.
  - So within-competition item ranking exists on new problems.
- **DECISION:** ADVANCE (E2b/E2c).
- **CONFIDENCE:** moderately supported.

---

## E2b — Pairwise Bradley–Terry Jev scoring

### Procedure as run
- **Sampling.** Within each benchmark, each distinct content draws 8 random partners (seeded). Unordered duplicates are removed, giving **39,164 pairs**. A/B positions are set by a seeded coin.
- **Question.** One Jev Choice question over the state `{task_a:{text}, task_b:{text}}`, each text head+tail-truncated to 8k chars as in report 13: "which task would be harder for a current frontier AI system … to complete fully correctly?"
- **Scoring.** P(task_a) is used as a soft outcome in a ridge-regularised (λ = 0.1) Bradley–Terry fit, per benchmark.
- **Comparison.** Spearman of BT scores vs β, against the pointwise jev ρ on the same items, with paired cluster-bootstrap CIs. Neither score has any cross-benchmark fit, so these are the LOBO ρ.

### Results

| benchmark | pointwise ρ | BT ρ | BT − pointwise | ρ(BT, pointwise) |
|---|---|---|---|---|
| matharena | .335 [.183, .563] | .412 [.256, .599] | **+.076 [+.011, +.139]** | .81 |
| multi_swebench | .197 [.154, .238] | .114 [.072, .155] | −.082 [−.103, −.061] | .86 |
| real_webagents | .186 [.056, .302] | .150 [.022, .271] | −.036 [−.099, +.022] | .90 |
| researchcodebench | .211 [.076, .343] | .312 [.174, .444] | **+.101 [+.002, +.200]** | .70 |
| swe_rebench | .207 [.158, .256] | .145 [.096, .197] | −.061 [−.088, −.037] | .88 |

- 0 failed calls.
- Position bias: mean P(task_a) = 0.475, a mild preference for calling B harder. It is balanced by the coin.
- Cost: 56.0M input tokens = **$2.35**.

**Decision.** BT ≥ pointwise + 0.05 on 2 of 5 benchmarks (the rule needs ≥ 3). **Row fired: keep pointwise.**
- RS note: pairwise judgement helps on long, heterogeneous items (rcb, math) and hurts on the two SWE pools, where both issues in a pair look alike.
- MM: a per-domain switch is not pre-registered and would be fitted to 5 points. Not pursued.

---

## E2c — Distil Jev into a local encoder

### Packaging constraints checked first (SA)
Sources: `external/starting_kit/README.md` and `templates/hf_submission`.
- Up to 5 public HF repos may be declared in `models.txt`. They are pre-downloaded, and the hub is offline at runtime.
- The template loads with `AutoTokenizer/AutoModel.from_pretrained(repo, cache_dir=HF_HOME|/app/hf_cache, local_files_only=True)`.
- CPU submissions run 4 workers, and the whole run must finish within 8 h.
- So the student is a frozen HF encoder loaded exactly that way, with mean pooling and L2 normalisation done in our own code (the `Encoder` class, also used offline). The head ships as a small JSON or npz.
- **Runtime limitation found:** `predict()` sees one `[subject, item]` plus `labeled`, never the hidden benchmark's item pool. The pre-registered "standardise within benchmark at prediction time" (variant **A**, which is what L5 did) therefore cannot be computed at B0.
- Deployable variant **B**: standardise with fixed training-fold statistics (benchmark-balanced mean and mean within-benchmark sd of inner-LOBO predictions). Both were evaluated. Shipping would require **B**, because only B is the packaged predictor.

### Procedure as run
Model cache: `E:\NeurIPS\aims\.hf_cache`. Nothing was written to C:.
1. **Embedding.** Each distinct content becomes [head 1000 chars | tail 1000 chars]. Each chunk is ≤ 256 word pieces, mean-pooled and L2-normalised, giving 2×d features. This was fixed before any result.
2. **Head.** sklearn Ridge on raw embeddings. The target is the within-benchmark z-score of jev_difficulty, with benchmark-balanced weights.
   - α ∈ {0.1, …, 1000} is chosen by inner leave-one-benchmark-out on the **training folds only**.
   - The held-out benchmark gets out-of-fold predictions.
   - Each training benchmark also gets inner-LOBO predictions, which are used for the β slope and for variant-B statistics.
3. **Retention rule** (fixed in code before results; the pre-registration says "on that fold"): student ρ / Jev ρ must be ≥ 0.70 in aggregate over the 5 LOBO folds **and** on ≥ 3 of 5 folds.
4. **End-to-end.** The v1_1 engine (`BASE2`: G/2 prior and momd merge guard) gains an item prior logit p = c + μ_key + σ_r·x, with μ = γ·z(student) and γ = the mean over training benchmarks of the Rasch-easiness slope on z.
   - Training benchmarks use inner-LOBO student scores. σ_r = √(σ_β² − γ²), as in ladder L5.
   - With μ = 0 the engine reproduces v1_1. The max |ΔALC| vs `v1_1.run_cfgs` is in `e2c_*.json` (`max_abs_diff_vs_v1_1` < 1e-9).
   - Also run: **v1_1 + Jev** (offline reference).
   - Regimes and seeds are the same as report 17: R2 has 5 seeds, R1 has 100 draws.
5. **Runtime.** One thread, batch 1, 50 real items; plus the on-disk size of the model files.

### Results — MiniLM-L6-v2 (22M params; pre-registered student)

ρ with held-out β (LOBO), with student–Jev agreement:

| held-out | α | ρ(Jev, β) | ρ(student, β) | ρ(student, Jev) | retention |
|---|---|---|---|---|---|
| matharena | 0.1 | .335 | −.085 [−.158, −.008] | .289 | −0.25 |
| multi_swebench | 1 | .197 | .017 [−.026, .062] | .230 | 0.09 |
| real_webagents | 0.1 | .186 | −.131 [−.263, −.005] | .050 | −0.70 |
| researchcodebench | 0.1 | .211 | .006 [−.132, .146] | .100 | 0.03 |
| swe_rebench | 1 | .207 | .004 [−.049, .053] | .227 | 0.02 |
| **aggregate** | | .227 | −.038 | | **−0.17** (0/5 folds ≥ 0.70) |

End-to-end ΔALC vs v1_1, ×1e-3. The format is pt [pair CI] h[benchmark CI], followed by per-fold values (multi / webagents / rcb / matharena).

| regime | v1_1 + Jev (offline reference) | v1_1 + student A (oracle pool z) | v1_1 + student B (deployable) |
|---|---|---|---|
| R2 | **−5.78 [−6.64, −4.97]** h[−7.09, −4.20]; −6.5/−3.6/−7.0/−6.1 ✔ threshold | +0.18 [+0.06, +0.31]; +0.0/+1.0/+0.0/−0.3 ✘ | +0.58 [+0.22, +0.94]; −0.3/+3.4/−0.0/−0.8 ✘ |
| R1 | **−6.42 [−7.28, −5.57]** h[−8.48, −4.05]; −9.0/−3.2/−6.5/−6.9 ✔ | +0.21 [+0.09, +0.34] ✘ | +0.63 [+0.21, +1.03] ✘ |

Fitted parameters:
- The fitted student slope γ_s is ≈ 0 (−0.13 to +0.05 logits/SD), against γ_Jev ≈ −0.5. The engine correctly learns that the student carries no β information.
- Variant B's held-out mean z is −1.5 to +0.8: the head output shifts by benchmark. On real_webagents that offset costs +3.4e-3.
- Runtime: **180 ms/item** (1 thread, batch 1, 2 chunks), 92 MB of model files. This fails the < 50 ms target. With a per-worker content memo and 4 workers it would still fit 8 h (≈ 0.3 h for 6k calls), but the target as pre-registered is missed.

**Row fired: "Student retains < 70% → distillation failed with this encoder; one fallback encoder, then stop."**

### Results — fallback all-mpnet-base-v2 (110M params)

The protocol is identical to MiniLM's: same chunks, head, retention rule and engine.

| held-out | ρ(Jev, β) | ρ(student, β) | ρ(student, Jev) | retention |
|---|---|---|---|---|
| matharena | .335 | .017 [−.050, .099] | .371 | 0.05 |
| multi_swebench | .197 | −.020 [−.062, .022] | .360 | −0.10 |
| real_webagents | .186 | −.060 [−.196, .073] | .237 | −0.32 |
| researchcodebench | .211 | .069 [−.078, .207] | .218 | 0.33 |
| swe_rebench | .207 | .023 [−.030, .076] | .288 | 0.11 |
| **aggregate** | .227 | .006 | | **0.025** (0/5 folds ≥ 0.70) |

End-to-end ΔALC vs v1_1 (×1e-3; per fold: multi / webagents / rcb / matharena):

| regime | student A (oracle pool z) | student B (deployable) |
|---|---|---|
| R2 | +0.01 [−0.17, +0.20]; +0.0/+0.5/−0.3/−0.2 ✘ | +0.13 [−0.26, +0.57] h[−2.38, +2.08]; −3.5/+2.5/+0.6/+0.9 ✘ |
| R1 | +0.10 [−0.09, +0.28] ✘ | +0.40 [−0.07, +0.86]; −1.6/+2.3/+0.4/+0.5 ✘ |

What changed relative to MiniLM:
- The larger encoder agrees more with Jev on held-out benchmarks (.22–.37 vs .05–.29).
- It still carries no transferable β signal (ρ ≈ 0 on every fold).
- Runtime is 1395 ms/item (1 thread) and the model files are 439 MB. That fails the runtime target by 28×.

**Row fired: "Student retains < 70%". This was the one permitted fallback, so the rule is "then stop". E2 stops here; no v1_2 is built.**

---

## Claims (contract)

**C1 (RS + MM): local distillation of Jev does not transfer across benchmarks.**
- **CLAIM:** A frozen general-purpose sentence encoder with a linear head trained on 4 benchmarks does not reproduce Jev's difficulty ranking on a held-out 5th. Student–Jev agreement is .05–.29, and the student's ρ with β is ≈ 0.
- **MECHANISM:** This is the report-13 failure again: embedding directions that predict "hard" in one domain do not predict it in another. Jev answers a domain-normalised question; a linear map on generic embeddings learns domain-specific surface cues.
  - In-domain, embeddings carry difficulty (report 13: within-benchmark ρ .35).
  - Out of domain they carry none. With 4 training domains there is no way to learn the normalisation.
- **EVIDENCE:** The tables above. Variant A was given the oracle pool for standardisation and still gave Δ ≥ 0.
- **COUNTEREVIDENCE:** The student was not allowed to see hidden-benchmark items. A transductive or retrieval student was not tested and is outside the pre-registration.
- **COMPETING EXPLANATION:** The head is too small or the input too truncated. Against this:
  - mpnet, at 5× the size with the same protocol, gives retention 0.025.
  - Agreement with Jev is non-zero (.23–.29 on 3 folds), yet it still does not map onto β.
- **FALSIFICATION TEST:** A domain-adaptive student, for example a small instruction-tuned LM declared in `models.txt` that answers the same rubric, scored under the same LOBO.
- **DECISION:** ELIMINATE frozen-encoder distillation (per the pre-registered stop rule). The Jev prior stays offline-only.
- **CONFIDENCE:** strongly supported for this encoder class.

**C2 (LH): Jev's prior is not a contamination artefact.** See E2a. Its offline value is −5.8e-3 (R2) and −6.4e-3 (R1) on v1_1, and it meets the ALC threshold on 4 of 4 folds. It remains the largest single lever, but it is blocked at runtime: api.typesafe.ai is not on the allowlist, and the student failed.

**C3 (SA): a runtime defect in the pre-registered student design.** Within-benchmark standardisation is not computable in `predict()` at B0. Any future content prior must either be calibrated to absolute units (variant B) or centre on `labeled` items only, and the B-type benchmark-level offset error must be audited (+3.4e-3 on real_webagents here).

## Deviations from the pre-registration
1. **Variant B added.** The pre-registered "standardise within benchmark at prediction time" is not deployable (C3). Both variants are reported. The ship row would have required the deployable one. This did not affect the outcome: neither variant met the threshold.
2. **Retention criterion operationalised.** The pre-registration says "≥ 70% of Jev's ρ on that fold". The rule used was aggregate ≥ 0.70 and ≥ 3 of 5 folds, fixed in code before results. The outcome is unaffected, since the result was 0 of 5 folds and an aggregate of −0.17.
3. **Perturbation rule defects.**
   - Math items with *unbraced* font commands (`\mathbb N`) had that letter relabelled: 5 of 60 math items.
   - "n-bit"-style numbers were shifted (32-bit → 33-bit): up to 14 of 180 SWE/web items.
   - These changes alter meaning slightly. They make the perturbation stronger, not weaker, so the test is if anything conservative. The rules were not edited after seeing scores.
4. **Retest added** (re-scoring originals against a fresh cache) to measure Jev's own noise. The decision uses the cached original ρ, as pre-registered. The retest ρ is equal (0.182 vs 0.180).
5. **E2b cost** was $2.35 against ≈ $2 estimated. Pairs average about 1.4k tokens, not 1k, because rcb texts are 8k chars each.
6. **Post-hoc diagnostics** (text-free items removed; within-competition ρ) are labelled as such and were not used for decisions.
7. **Fallback encoder choice** (all-mpnet-base-v2) was not named in the pre-registration ("one larger fallback"). It is the starting kit's own HF-template example model.
8. **E2c runtime target** was measured at 1 thread, batch 1. MiniLM at 180 ms/item misses the < 50 ms target; this is moot, since retention failed first.

## Cost
- Jev:
  - E2a: 1.01M tokens, **$0.043**.
  - E2b: 56.0M tokens, **$2.35**.
  - Total ≈ **$2.39**.
- CPU:
  - MiniLM embedding ≈ 10 min.
  - mpnet ≈ 84 min (5254 s total run).
  - Engine runs ≈ 1 min per encoder.
- Downloads: 92 MB (MiniLM) + 439 MB (mpnet), both to `E:\NeurIPS\aims\.hf_cache`.
- Peak RAM < 1 GB, one process at a time.

## Action
- No `submissions/v1_2`. v1_1 stays the champion. `submissions/v1` and `v1_1` are untouched.
- Plans: remove "local distillation of Jev with a frozen encoder" (L5 via student). Keep Jev as an offline research aid and as the yardstick for any future runtime-legal difficulty source (the remaining route is an allowlisted LLM API answering the same rubric).
- Report 18's D5 question ("is Jev's difficulty real or contamination?") is answered: real, at the precision available.
