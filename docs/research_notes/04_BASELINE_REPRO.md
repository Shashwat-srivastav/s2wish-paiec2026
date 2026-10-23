> **Historical (written 2026-09-17, before the training data was available).** Kept for the record; facts about data, baselines and results are superseded by reports 10–20 and `docs/publication/research_report.html`.

# Baseline Reproduction Report (local, 2026-09-17)

## What was reproduced
Cloned `aims-foundations/paiec_baseline` (depth 1, commit 82d330d "Add empirical mean baseline with BLE acquisition", 2026-09-16) and `aims-foundations/measurement-db` (schema v3).

### 1. Static + unit verification — PASS
- `python -m unittest discover -s tools -p 'test_*.py'` → **19 tests OK** (mean isolation, out-of-order checkpoints, missing-benchmark guard, zip independence, BLE-acquisition hook, public packaging).
- `python tools/build_mean_zip.py` → `dist/empirical_mean.zip` (1,935 bytes, `{model.py, README.md}` only).
- `python check_submission_zip.py dist/empirical_mean.zip` → **OK: valid submission ZIP** (structure + `predict` smoke + acquisition check).
- `python tools/smoke_test.py --mock` → **OK** (synthetic corpus + scripted LLM, no API calls; predict→0.62; acquisition uncertainty policy; bounds; failure surfacing; SDK-free HTTP; redaction; separate logs for repeated inputs).

### 2. Metric reproduction — PASS
`src/paiec/evaluation/metric.py` verified:
- Brier constant-0.5 → 0.25; perfect → 0.0.
- Illustrative curve [0.22,0.18,0.145,0.12,0.1,0.09] → **ALC 0.140** (matches competition page).
- Flat curve → ALC == Brier; weights sum to 1; ECE sanity; equal pair averaging.

### 3. Empirical-mean semantics (from `empirical_mean/model.py`, 39 lines)
- Budget 0 / no matching labels → **0.5**. Else **unsmoothed successes/count** for exact `(subject-dict, benchmark_id)` match. Other benchmarks/subject-settings ignored. Missing `benchmark_id` at positive budgets → `ValueError` (no silent pooling). Non-binary label → `ValueError`. No retained state; out-of-order checkpoints independent. ZIP omits `labeling.py` → evaluator default **deterministic random** acquisition, nested sets at 0/1/3/7/15/31.
- Bounds `[0,1]` unsmoothed: 1 obs → 0 or 1 (overconfident by design; beatable via shrinkage).

### 4. BLE semantics (from `ble/model.py`, `labeling.py`, agent loop)
- `predict`: builds question from all subject fields + item_content/features/interactors; appends same-subject revealed excerpts (verdict + first 300 chars, omits features/interactors); LLM estimate→retrieve→revise→submit loop, default 10 turns (≤9 retrievals), Luna Responses API medium reasoning, 32k→64k→128k token retry, per-request 120 s, question deadline 240 s (competition example: 5 turns/240 s). Returns float clipped to **[0.02,0.98]**. Failures raise (never silent fallback to 0.5). Retrieval over bundled public payload (`data/<bench>/{benchmarks,subjects,items,response[,traces]}.parquet` + optional `embeddings/items/<bench>.parquet`, text-embedding-3-small 1536-d, verified by content_sha1). 7 keyword/table tools + optional semantic search + submit.
- `acquisition_function`: streaming `|p−0.5| ≤ 0.15` else skip, unless must-take (`items_remaining ≤ labels_remaining`) or budget exhausted (`False`); no extra model call (reuses supplied `prediction`). Legacy 1-arg path returns `-abs(len−1200)` priority.
- Mean+BLE-acquisition variant: eval = empirical mean; acquisition calls BLE per candidate (signature omits `prediction` so evaluator skips mean call), then applies BLE uncertainty policy. API cost on acquisition path only.

## Brier scores at budgets — what can / cannot be recorded locally
| Budget | Empirical mean (theory) | BLE | ECE/runtime/params |
|---|---|---|---|
| 0 | 0.25 exactly (always 0.5) | unknown w/o hidden test + paid API | — |
| 1,3,7,15,31 | data-dependent; unsmoothed mean, random acquisition | data-dependent | — |
| ALC | 0.25·0.1 + Σ data terms | unknown | — |

**No hidden-test Brier numbers can be recorded without (a) gated HF training data access, (b) Codabench formative submission, (c) paid LLM API for BLE.** Local smoke/mock probabilities (0.62) are illustrative, not scores. Constant-0.5 ALC = 0.25 is the floor to beat at budget 0; any informative prior or shrunk adaptation must show ALC < 0.25 on a frozen internal split before a Codabench submission is spent.

## Seeds / determinism
- Empirical mean: no RNG in predictor; acquisition random is evaluator-side deterministic (sha256 of context+input). Tests enforce order-independence.
- BLE: stochastic LLM runs may select different labels; reuse one recorded trajectory for matched comparisons.

## Next to reproduce (blocked on access)
1. `tools/prepare_data.py --out payload` (needs HF auth + gated approval) → real eligible-benchmark payload.
2. Local 50/50 per-pair split simulator + nested budgets + random/uncertainty acquisition → internal Brier curves for mean / shrunk-mean / IRT / matrix baselines.
3. One Codabench formative submission of empirical mean → calibrate local simulator vs hidden distribution (record per-pair Brier/ECE, do not re-identify IDs).
