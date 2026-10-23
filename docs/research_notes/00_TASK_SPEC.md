> **Historical (written 2026-09-17, before the training data was available).** Kept for the record; facts about data, baselines and results are superseded by reports 10–20 and `docs/publication/research_report.html`.

# PAIEC 2026 — Formal Task Specification
## Predictive AI Evaluation Competition at NeurIPS 2026 (Stanford AIMS)
**Authoritative sources (read 2026-09-17, Asia/Calcutta):**
- Competition page: https://aimslab.stanford.edu/competition
- Baseline repo: https://github.com/aims-foundations/paiec_baseline (cloned, commit 82d330d, 2026-09-16)
- Data bank: https://aimslab.stanford.edu/measurement-db + https://huggingface.co/datasets/aims-foundations/measurement-db (gated) + https://github.com/aims-foundations/measurement-db (schema v3, `parquet_schemas.yaml`)
- Codabench: https://www.codabench.org/competitions/17828/ (login + organizer approval required; starting kit inside)
- Curation guide / inventory construction: Google Doc `1LdUEiUG5tLhDdnIAtm4YKSf4w3LFb2HmI_ahiiuoRLU` + Benchmark Name List sheet `1T69QyIZJRWX3d-Tmuh_lcFwtpqv9ytcLxfLWKn6gopI`
- OpenReview venue: `NeurIPS.cc/2026/Workshop/PAIEC` (technical-report submission)

> This document is the authoritative local spec. Where the competition page and baseline README conflict, the competition page governs; baseline READMEs govern implementation details of the reference methods.

---

## 1. What constitutes a subject?
A **subject** = one fixed AI-system configuration = **(base model + access_date + harness + harness_version + reasoning_effort + subject_features_extra)**, identified at runtime by 8 string fields:
`normalized_name, provider, release_date, access_date, harness, harness_version, reasoning_effort, subject_features_extra`.
**Same model + different harness = two distinct subjects.** Same for different harness_version / reasoning_effort / extra features.
- In measurement-db, `subjects.parquet.subject_id` = truncated sha256 over canonical identity + every metadata column at observed value (nulls included). `display_name` preserves upstream label; `normalized_name/provider/release_date` come from `map_model_registry.json` (null if unmapped, never guessed).
- For competition curation, eligible base models are currently restricted to **2025–2026 releases** (organizer confirmation required for additions). This restriction does not apply to general Data Bank curation.
- Runtime rule: every field value is a plain string; missing = `""`. Subject matching in baselines is **exact dict equality** on the full visible subject dict.

## 2. What constitutes an item?
An **item** = complete stimulus + grading protocol + effective response scale, identified at runtime by 4 string fields:
`item_content, item_features, interactors, benchmark_id`.
- `item_content`: task text presented to subject (for agentic/coding tasks: full task spec incl. identifiers making it answerable).
- `item_features`: additional attributes (canonical `key=value;...` sorted encoding in the bank; e.g. `tier=medium;points=1`, `lang=...`, `paper=...`, `website=...`).
- `interactors`: tools/agents/other parties in the interaction (same `key=value;...` encoding; e.g. attacker, user_sim, opponent). Null when subject interacted with no one but the item.
- `benchmark_id`: **stable, anonymous** identifier for the item's benchmark, **shared by target items and acquired examples from the same benchmark**. This is the grouping key for adaptation.
- In measurement-db, `item_id` = 16-hex sha256 over (benchmark, normalized content, item_features, asset_manifest, grading_criterion, full verifier, effective response scale), v4 identity. Same stimulus under different grading = different items. `content_hash` (sha256 of normalized content alone) enables cross-benchmark duplicate detection. `grading_criterion` (reference_answer and/or rule) + `verifier` (Judge vs ExactMatcher) live in `items.parquet` but are **NOT** passed to `predict()` — see §6.

## 3. What constitutes a response?
A **response** = grade for one observed **(subject, item, trial, test_condition, interactors)** tuple.
- Competition scoring uses **binary** outcomes: `1` = correct/success, `0` = otherwise. A **target** is a response to be predicted; **a subject that attempted an item 3 times contributes 3 separate targets** (repeated responses with identical `[subject,item]` inputs reuse one prediction but score as separate outcomes).
- In the bank, `responses.parquet` = long-form table, `response: float64 nullable` (uniform storage even for integer-coded categories), must be finite and within declared `response_scale`. Explicit null = recorded attempt without usable grade (kept, never imputed as 0); no row = no observation. `trial` = 1-indexed positional repetition index. `test_condition` holds only true occasion settings (usually temperature or null). `response_id` (v1 identity payload, opaque) joins `traces.parquet.trace` (raw model output text, 0/1 rows per response) and is derived, not a hash of exported columns.
- Only **binary + item-granularity** benchmarks are competition-eligible (checked by `tools/prepare_data.py`: `response_type=="binary" and granularity=="item"`). E.g. among the 6 currently released bank examples, `mmdocrag` (fraction) is ineligible.

## 4. What information is known before prediction?
At every `predict(input, labeled)` call:
- `input[0]` = full subject dict (8 keys above), `input[1]` = full item dict (4 keys above). All values plain strings, `""` = missing.
- `labeled` = evidence allowed **at that budget**: list of `[[subject, item], label]` entries across sampled subject–benchmark pairs; dicts match `input` structure; `label ∈ {0,1}`. At budget 0, `labeled == []`.
- Anything bundled in the submission ZIP (`any_other_files/`: fitted state, templates, payload mirrors) and anything listed in `models.txt` (prefetched HF repos). Plus the predictor LLM / API behaviour configured privately (BLE) — but no live web.
- Public training corpus as released on HF `measurement-db` (+ optional `measurement-db-embed` vectors) **if bundled or prefetched** — the sandbox has no HF Hub access at runtime.

## 5. What information can be revealed after acquisition?
- Up to **31 labeled responses per subject–benchmark pair**, via one nested acquisition trajectory (labels at smaller budgets ⊆ labels at larger budgets). Each evaluation receives only its allowed evidence at budgets **0, 1, 3, 7, 15, 31** on the **same fixed evaluation items**.
- Acquisition mechanics: only pairs with **≥80 distinct items** are split; items randomly split **50/50 into acquisition vs evaluation pools**, fixed across submissions and budgets; all recorded responses for an item stay together. **Acquisition items are never scored; evaluation outcomes are never revealed.**
- Candidates arrive **one at a time in fixed random order**. Optional `labeling.py: acquisition_function(input, prediction, labeled, context) -> bool`. `prediction` = current candidate's prediction conditioned on previously acquired labels (computed by the platform calling `predict` — unless the hook signature omits `prediction`, in which case no prediction call is made, cf. mean+BLE-acquisition baseline). `context` = `{subject_id, benchmark_id, labels_acquired, labels_remaining, max_labels=31, items_remaining (incl. current)}`; counts are per current pair; `labels_remaining` = allowance left out of 31, not distance to next scoring budget. Return `True` = reveal one recorded response; `False` = skip permanently. Without `labeling.py`, platform selects **randomly** (deterministic uniform without replacement, reproducible across submissions).
- Legacy compatibility: one-argument `acquisition_function(input) -> float` = pool-ranking priority (numeric), evaluated in `rank` mode; streaming boolean hooks use `acquire` mode. Mode is locked at budget-0 checkpoint.

## 6. What information is never available?
- Evaluation-pool outcomes (hidden test labels), including for items textually identical to acquisition items (identical visible descriptions can be independent targets — no label directly supplies the answer).
- Private test subject/benchmark identities behind anonymous formative IDs (must not attempt to re-identify for development).
- Grading criteria / verifiers / reference answers as structured fields (they shape `item_id` in the bank but are not passed to `predict`); raw benchmark source files beyond what is bundled; hidden-test traces; future test sampling draws.
- Internet at scoring time except **approved inference APIs**. Web search/browsing prohibited. No HF downloads at runtime; no `trust_remote_code` / extra pip installs by default (organizer-policy-gated).

## 7. What are the label budgets?
**0, 1, 3, 7, 15, 31 per subject–benchmark pair**, nested, same evaluation items at every budget. `max_labels=31`. Sampling cap per formative submission: **≤1,000 unique subject–item pairs across acquisition + evaluation pools**. Note evaluating 6 budgets can require more predictor calls than the cap (each budget re-predicts eval items; acquisition candidates may also trigger `predict` calls when the hook needs `prediction`).

## 8. What is the exact metric?
- **Brier score** at budget n: `B_n = mean((p_i − y_i)^2)` over evaluation responses (all recorded responses count; repeated targets each count). 0 = perfect; constant 0.5 → 0.25.
- **Brier ALC** (primary, lower better): normalized area under learning curve on `x = log2(n+1)` (budgets equally spaced 0..5), trapezoid rule, width 5:
  **`Brier ALC = 0.1·B_0 + 0.2·B_1 + 0.2·B_3 + 0.2·B_7 + 0.2·B_15 + 0.1·B_31`**.
- **ECE** (diagnostic only): 10 equal-width bins, `Σ (n_b/n)·|conf_b − acc_b|`; summary ECE uses ALC weights. Sensitive to sample size; reported as diagnostic.
- Local re-implementation: `src/paiec/evaluation/metric.py` (verified: illustrative curve [0.22,0.18,0.145,0.12,0.1,0.09] → 0.140; flat → Brier; constant-0.5 → 0.25).

## 9. How is the metric aggregated?
**Brier ALC is computed per subject–benchmark pair, then averaged equally across pairs. Every pair has the same weight regardless of its number of responses.** Within a pair, Brier averages over all evaluation responses (repeated trials each count). `Number of Unique Subject-Item Pairs` in formative tables counts distinct evaluated pairs once (fixed across budgets) but scoring uses all responses.

## 10. How are subject–benchmark pairs weighted?
Equally (see §9). A pair with 46 eval pairs counts the same as one with 181. Implication: small-pair variance is not downweighted; hierarchical shrinkage and per-pair uncertainty matter; do not overfit to large pairs.

## 11. What is the private-test structure?
- **Benchmark-level cold start**: held-out items drawn from benchmarks assigned to the **private test pool**. Candidate benchmarks randomly assigned to **training vs test pools before curation**; pool assignments + sampling design fixed; collections may grow during competition.
- Platform **selects the subjects** for each submission (no `subjects.txt`; validator rejects it).
- Per-pair design: ≥80 distinct items → fixed 50/50 acquisition/eval split; all responses for an item stay together.
- **Formative**: each submission draws a **newly sampled subset** of hidden test data (cap 1,000 unique pairs); same eval responses across budgets within a submission; feedback private to submitter + organizers (Codabench Logs → Scoring output): ALC summary + per-budget tables (unique counts, Brier, ECE) with **anonymous, permanent, cross-submission stable** `subject_id`/`benchmark_id` (matching IDs = same entity). Aggregate leaderboard visibility announced separately.
- **Summative**: common hidden test subset for all teams (automatic Brier ALC) + manual grading of codebase + technical report (reproducibility review: rerun with reasonable effort, allowing sampling/random/API variation). Oral selection considers Brier ALC on common subset + scientific rigor + data-curation/community contributions + report clarity.

## 12. What data may be used for training?
- Benchmarks in the **public training pool** for competition-specific training and curation during development (verified training data incl. participant contributions, released through `measurement-db`; may grow; pool assignments fixed).
- **Pretrained models** explicitly allowed (must disclose in technical report).
- Public `measurement-db-embed` vectors (`text-embedding-3-small`, 1536-d, verified by `content_sha1`) for retrieval.
- Participant-contributed curation that is verified + released (coordinate with organizers; check curation status first).
- LLM-based coding assistants permitted for development (must disclose).

## 13. What external data may be used?
- Pretrained models (bundled locally or declared in `models.txt` for prefetch; ≤5 repos; 300B params + 1,000 GB per repo; 1,024 GB combined).
- Public evidence retrievable **before** submission and bundled (BLE pattern: public corpus payload + API-based LLM reasoning at runtime). At scoring, network only to approved inference APIs.
- Any training sources are allowed if **disclosed** — but competition-specific training/curation must stay within the public training pool (do not train on test-pool benchmarks).

## 14. What information is prohibited?
- Extracting/sharing private test data; attempting to identify subjects/benchmarks behind anonymous formative IDs **for use in development**.
- Using test-pool benchmarks for competition-specific training/curation.
- Internet-enabled tools at scoring (web search/browsing); unapproved network; runtime HF downloads; undisclosed extra pip installs / `trust_remote_code` (policy-gated).
- Sharing credential-bearing archives (BLE private ZIPs); publishing `submission_config.json` with keys.
- Cross-team private-test extraction; each participant only one team.

## 15. What are runtime constraints?
- **8 hours per run incl. setup**; GPU allocation automatic. Must budget: 6 budgets × eval predictions + sequential acquisition decisions (each may cost ≥1 LLM turns if hook needs BLE prediction).
- `predict` must return finite `float` in `[0,1]` (`float()` coercion; NaN/inf/out-of-range/conversion-failure rejected). BLE self-bounds to `[0.02,0.98]` (baseline choice; competition permits `[0,1]`). Empirical mean uses unsmoothed `[0,1]` (1 obs → 0/1).
- No state retained between calls/budgets by contract (evaluator recreates workers with frozen evidence; empirical-mean tests enforce out-of-order checkpoint independence). Do not rely on cross-call memory.
- Failure codes: `10/11` invalid `model.py`; `12` invalid `labeling.py`; `40` prediction exception; `41` call timeout; `42` invalid probability; `50` total timeout. Hosted logs hide stdout/stderr — debug via `tools/check_submission_zip.py` + `tools/run_smoke_test.py` (local: `check_submission_zip.py`, `smoke_test.py`).
- Packaging: `submission.zip/model.py` (required `predict(input, labeled)`), `labeling.py` (optional `acquisition_function`), `models.txt` (optional HF prefetch), `requirements.txt` (optional, policy-gated; named pip packages only, no `-` options/URLs/paths), `any_other_files/` (artifacts/fitted state/templates). ZIP root placement enforced; no nested ZIPs; no `subjects.txt`; `models.txt` entries = plain HF repo IDs (≤5, no dups); HF refs in code must be in `models.txt` or point to bundled files; bundled-file refs must exist. `SentenceTransformer` shorthand mapping handled. Local checks: `(cd my_submission && zip -r ../my_submission.zip .)`, `check_submission_zip.py`, `run_smoke_test.py`.
- Python 3.10+; preinstalled deps only guaranteed (BLE runtime boundary: numpy/pandas/pyarrow + stdlib for Luna path; litellm lazily for other providers).

## 16. What models can be packaged?
- Any Python predictor + optional acquisition hook satisfying the interfaces, incl. statistical (mean/IRT/matrix), classical ML (GBM), neural (NCF/transformer), retrieval-augmented, and LLM-agent (BLE) approaches — subject to runtime/packaging/network limits above.
- Local models bundled in ZIP or declared in `models.txt` (prefetch). Up to 5 repos within param/storage caps. API-backed models allowed via approved inference APIs with privately configured credentials (never published).
- Must supply own development compute + API keys; models cannot be downloaded at runtime.

## 17. What must be disclosed?
- **All training sources** in technical report (data + pretrained models + payload revisions/manifests).
- LLM-based coding-assistant use (kindly asked).
- Shared cross-team contributions acknowledged.
- For presentation consideration: technical report + release of code used to develop/run submission + any training data differing from `measurement-db`, with links via OpenReview by deadline; report must document **data, method, configuration, experiments, with instructions sufficient to rerun**.
- Final submission: codebase + technical report by **30 Oct 2026 23:59 AoE**, emailed selection of Codabench submission. Registration: email `aimslab@cs.stanford.edu` (team name + each member's name/affiliation/email/Codabench/HF accounts); free; organizer approval required. Competition session: **11 Dec 2026, Sydney, 08:00–10:15** (5 orals 60 min + posters 60 min + 15 min opening/transitions). Launch: 15 Aug 2026 (per News). Formative: multiple dev submissions welcome.
- Eligibility: one team per person; ≤6 members; cross-institutional encouraged; organizers/collaborators may submit for comparison. Conduct: discussion/code/permitted-data sharing encouraged with acknowledgement.

## Appendix: interface signatures (verbatim semantics)
```python
# model.py (required)
def predict(input: list[dict], labeled: list) -> float: ...
# input[0]: normalized_name, provider, release_date, access_date, harness,
#           harness_version, reasoning_effort, subject_features_extra
# input[1]: item_content, item_features, interactors, benchmark_id
# labeled:  [[[subject, item], label], ...], label in {0,1}

# labeling.py (optional, streaming)
def acquisition_function(input, prediction=None, labeled=None, context=None) -> bool: ...
# context: subject_id, benchmark_id, labels_acquired, labels_remaining,
#          max_labels=31, items_remaining (incl. current)
```
