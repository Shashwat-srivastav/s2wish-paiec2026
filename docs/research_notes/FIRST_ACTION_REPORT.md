> **Historical (written 2026-09-17, before the training data was available).** Kept for the record; facts about data, baselines and results are superseded by reports 10–20 and `docs/publication/research_report.html`.

# FIRST ACTION REPORT — PAIEC 2026 (2026-09-17)
**Team role: principal research scientist. Status: steps 1–10 complete to the extent possible without gated data; large-scale experimentation NOT started (per §57).**

## A. Exact task formulation
Predict `P(Y=1)` for unobserved subject–item responses without running the subject. Subject = (model + harness + harness_version + reasoning_effort + extra + dates) (8 string fields); item = (content + features + interactors + anonymous benchmark_id) (4 strings); response ∈ {0,1} per (subject,item,trial,condition,interactors); repeated trials = separate targets sharing one prediction. Benchmark-level cold start: test items from held-out benchmarks (random pool assignment pre-curation). Adaptation via nested acquired labels at budgets 0/1/3/7/15/31 per subject–benchmark pair (≥80-item pairs, fixed 50/50 acquisition/eval split; acquisition never scored, eval never revealed; candidates streamed in fixed random order). Optional `labeling.py` acquisition hook; default random. Full spec: `reports/00_TASK_SPEC.md`; Feynman: `reports/01_FEYNMAN.md`; stats: `reports/02_FORMAL_STATS.md`.

## B. Exact evaluation mechanism
Streaming protocol (`streaming_alc_v1`): platform calls `predict(input, labeled)` per distinct [subject,item] (repeats reuse); `labeled` = budget-allowed `[[subject,item],y]` evidence; `acquisition_function(input, prediction, labeled, context)->bool` decides reveal/skip per candidate (`context`: subject/benchmark IDs, acquired/remaining, max 31, items_remaining). Six budgets, same eval items, nested sets. Formative: fresh ≤1,000-unique-pair sample per submission, private per-submitter feedback (ALC summary + per-budget Brier/ECE, stable anonymous IDs). Summative: common hidden subset (Brier ALC) + manual reproducibility/method/report grading; orals weigh ALC + rigor + curation/community + clarity. Runtime: 8 h incl. setup, auto GPU, approved-APIs-only network, no HF downloads, policy-gated requirements/trust_remote_code. Failure codes 10/11/12/40/41/42/50. Packaging: root `model.py` + optional `labeling.py`/`models.txt`/`requirements.txt` + artifacts; ≤5 HF repos (300B params/1,000 GB each, 1,024 GB total).

## C. Exact scoring formula
`B_n = mean((p−y)²)` per pair per budget; **`ALC = 0.1B0+0.2B1+0.2B3+0.2B7+0.2B15+0.1B31`** per pair (trapezoid on log2(n+1)), then **equal mean over pairs** (pair size ignored). Lower better; 0.5-flat → 0.25. ECE (10 equal-width bins, ALC-weighted) diagnostic only. Verified in `src/metric.py` (illustrative 0.140; flat→Brier; tests pass).

## D. Data structure
Bank schema v3 (6 fixed tables; see `reports/00_TASK_SPEC.md` §1–3 + `reports/08_DATA_FORENSICS.md`): subjects (subject_id hash + 8 metadata), items (item_id v4 hash incl. grading+verifier+scale; content_hash for dedup), benchmarks (response_scale structured; binary/item-granularity = eligible), responses (long-form, nullable float64, trial/condition/interactors), traces (raw text by response_id), assets (byte payloads). Runtime sees only 8+4 strings + acquired labels. HF HEAD lists 6 dirs (`matharena, mmdocrag, multi_swebench, real_webagents, researchcodebench, swe_rebench`); `mmdocrag` ineligible (fraction). **Downloads gated (401) — forensic battery + eligibility stats pending registration + HF approval.** Metadata audit complete from GitHub; catalog (61/425K/909/15.5M) is Data Bank ambition, not current gated snapshot; training pool may grow, assignments fixed.

## E. Baseline result
- Local: 19/19 unit tests pass; mean ZIP valid; BLE mock smoke OK (`reports/04_BASELINE_REPRO.md`).
- Empirical mean: B0=0.25 exactly (0.5); n>0 unsmoothed pair mean (1 obs → 0/1); random acquisition. BLE: LLM retrieve→revise→submit ([0.02,0.98], ≤9 retrievals, medium reasoning) + |p−0.5|≤0.15 streaming acquisition; mean+BLE-acquisition hybrid separates acquisition value (README warns uncertainty sampling can bias mean estimation).
- Hidden-test Brier/ALC/ECE/runtime/params/seeds: **not recordable without gated data + Codabench + paid API**. Floor to beat: ALC 0.25. No Codabench submission spent yet (correct per §47/§57).

## F. Major leakage risks
Benchmark-level cold start is the regime; random-split wins prove nothing. Top risks: (1) test-benchmark ID memorization (must be grouping-only); (2) train↔test item/semantic dupes (content_hash + NN ≥0.98 audit); (3) benchmark tokens/grading text in content/features; (4) harness pooling errors (exact-dict matching); (5) target-encoding/normalization outside folds; (6) temporal leakage (train-future); (7) acquisition using future/eval/cross-pair labels; (8) formative-ID re-identification or fitting to formative tables; (9) repeated Codabench adaptation without frozen internal suite. Full audit + split code: `reports/07_LEAKAGE_AUDIT.md`, `src/leakage_checks.py`. Decision rules §49 adopted.

## G. First 20 experiments
`reports/FIRST_20_EXPERIMENTS.md`: E01 simulator+mean repro → E02 global → E03 benchmark difficulty (cold-start) → E04 subject ability → E05 shrunk pair posterior → E06 1PL-IRT → E07 logistic MF → E08 text/semantic → E09 subject-metadata → E10 interaction (FM/bilinear/2PL/neural) → E11 benchmark LOO → E12 subject/family/provider/harness → E13 temporal → E14 calibration → E15 curves/efficiency → E16 random acquisition → E17 uncertainty → E18 diversity/hybrid → E19 EIG/EVI → E20 fusion+ship. Each has hypothesis/control/treatment/split/metric/expected/falsification/compute/interpretation/decision.

## H. Expected information gain
E01 gates everything (simulator validity). E02–E05 settle low-budget priors + shrinkage (half the ALC). E06–E07 test H1 (low-dimensional). E08 is the text go/no-go (H2). E09–E10 settle subject transfer (H9) + interaction (H3). E11–E13 map transfer limits + failures. E14 separates calibration (H11). E15 prioritizes budgets (H6). E16–E19 isolate acquisition mechanism (H4/H5). E20 yields the shippable fusion + paper method. Maximize info/experiment; no sweeps before review.

## I. Compute requirements
CPU-only through E16 (hours on workstation). GPU-hours only for E08-embeddings/E10-neural/E19-EIG/E20-fusion (cached; prefetchable). BLE/formative: paid Luna API — mock first, budget per-call × (eval×6 + acquisition candidates). Codabench: fit in 8 h streaming timeouts. Storage: payload GBs (data + optional 1536-d embeddings); submission ≤1024 GB / 5 repos.

## J. Decision rule for each
Per-experiment rules in `FIRST_20_EXPERIMENTS.md` + global §49: reject random-only / 31-only / selected-item-only / ECE-only wins; shift-analysis before complexity on hidden gaps; prefer IRT/simple on ties; promote only after multi-seed + benchmark/subject/temporal holdouts + re-implementation. Registry: `experiments/registry.{md,csv}` (EXP-0001/0002 logged; submission log empty by design).

## Artifacts delivered (reproducible)
`reports/00_TASK_SPEC.md, 01_FEYNMAN.md, 02_FORMAL_STATS.md, 04_BASELINE_REPRO.md, 07_LEAKAGE_AUDIT.md, 08_DATA_FORENSICS.md, FIRST_20_EXPERIMENTS.md`; `src/metric.py (verified), naive_baselines.py, irt_matrix.py, leakage_checks.py, forensics.py`; `experiments/registry.*`; clones `external/official_baselines/`, `measurement-db/`.

## Unblock request (to proceed past §57 gate)
1. Team registration (aimslab@cs.stanford.edu) + Codabench/HF accounts. 2. HF gated approval + `huggingface-cli login` token (or pre-downloaded `payload/`). 3. Optional: OpenAI API key + budget cap for BLE/formative runs. On receipt: run forensics → freeze splits → execute E01–E05 → report before E06+.
