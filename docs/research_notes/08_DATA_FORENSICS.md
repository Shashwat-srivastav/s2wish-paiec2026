> **Historical (written 2026-09-17, before the training data was available).** Kept for the record; facts about data, baselines and results are superseded by reports 10–20 and `docs/publication/research_report.html`.

# Data Forensics — Status + Plan (2026-09-17)

## Access status (blocking)
- HF `aims-foundations/measurement-db` listing works (repo_type=dataset): **6 benchmark dirs** at HEAD `ad2f23b…`: `matharena, mmdocrag, multi_swebench, real_webagents, researchcodebench, swe_rebench` (each with benchmarks/subjects/items/response [+traces/assets]).
- **File downloads are gated (401):** "Access to dataset aims-foundations/measurement-db is restricted. You must have access and be authenticated." Same for `measurement-db-embed` (HEAD `fb3a155…`, 3 vectors: real_webagents, researchcodebench, swe_rebench).
- Competition page says verified training data (incl. participant contributions) are released through measurement-db and **may grow**; pool assignments fixed. The website catalog (61 benchmarks / 425K items / 909 subjects / 15.5M responses) describes the broader Data Bank ambition, not the currently downloadable gated snapshot. Treat the 6-dir snapshot as *examples*; expect growth after registration + HF approval.
- **Unblock steps:** (1) register team via aimslab@cs.stanford.edu (team name + members + Codabench/HF accounts); (2) accept HF gated conditions + get approval; (3) `huggingface-cli login`, then `python tools/prepare_data.py --out payload` from paiec_baseline (needs ble/requirements); (4) run `src/paiec/analysis/forensics.py` (ready, below) on the payload.

## What was audited WITHOUT gated files
- **Schema audit (complete, from `parquet_schemas.yaml` v3):** 6 fixed tables (subjects/items/assets/benchmarks/responses/traces), exact columns/order/types/nullability, PKs, `response_scale` structured metadata (discrete/interval/mixed + direction/meanings), trial/test_condition/interactors semantics, ID hashing (subject_id v-registry, item_id v4 incl. grading+verifier+scale, response_id v1 opaque). No extension columns. Builders enforce via validators.
- **Benchmark metadata audit (from `benchmarks/*/metadata.yaml`):** all 6 examples documented (response_type/granularity/domain/modality/release/license). Eligibility filter (`binary` + `item`) excludes `mmdocrag` (fraction). `researchcodebench` notes Gemini-2.5-Pro label collapse (trials 1/2, not distinct subjects) — a subject-dedup precedent.
- **Baseline input audit:** runtime `input` = 8 subject + 4 item strings; `""` = missing; exact-dict subject matching; anonymous `benchmark_id` grouping; `labeled` binary only.
- **Inventory signal:** benchmark name list + curation guide confirm 2025–2026 base-model restriction for competition curation, peer-review + characterization requirements, raw/`raw/` archiving (1,191 files / 10.4 GB across 6 archives; total repo 11.4 GB).

## Forensic battery (ready to run on payload; see `src/forensics.py`)
Schema/missing/duplicate/response-count audits; benchmark-size + subject-frequency + family/provider/release/harness/reasoning/item-length/benchmark distributions; class balance; success-rate; benchmark/subject difficulty; subject×benchmark interaction (coverage, per-cell n, mean heatmap, χ²/association); repeat-subject/item detection; content_hash cross-benchmark dupes; embedding NN dupes (≥0.98); harness-vs-success correlation; release-date trend; benchmark-ID structure (length/entropy/uniqueness); metadata-label leakage scans (benchmark tokens in text; grading text in content; length-vs-label correlation).

## Hypotheses to test first (once data lands)
H-F1: 1–2 benchmarks dominate response counts (weighting risk under equal pair averaging).
H-F2: provider/harness skew (few providers; harness correlates with success on agentic benches).
H-F3: release-date trend leaks performance (must validate forward, not memorize).
H-F4: repeated/near-dup items across benchmarks exist (content_hash + NN will find them).
H-F5: class imbalance varies strongly by benchmark (global 0.5 prior is weak; hierarchical priors win).
H-F6: subject×benchmark interaction is large (benchmark-specific subject rankings shuffle) — motivates IRT-discrimination / MF over additive means.
