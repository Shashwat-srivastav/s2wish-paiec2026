> **Superseded decision (2026-10-01):** the recommendation to ask the organizers to allowlist TypeSafe was dropped at the user's direction. Jev is used only offline — as the PI's Decision Layer (`src/paiec/jev/decide.py`) and as an item-difficulty research tool (reports 13, 20). It is never called inside a submission.

# 09 — TypeSafe Jev feasibility

**Tool:** `src/jev_item_features.py` sends 9 questions per item in one request: 3 Score questions (difficulty, expertise, scope) and 6 Noul questions (multistep, tool_use, verifiable, ambiguous, obscure_knowledge, refusal_risk). It outputs 12 numeric columns and caches results by content hash, so a rerun only sends new items. Smoke test: `experiments/jev_smoke.csv`. The 11 items were ranked sensibly. Difficulty went arithmetic 0.00 < GSM 0.28 < SWE-easy 0.15 / web-easy 0.31 < SWE-hard 0.70 / web-hard 0.72 < research math 0.88–0.93. tool_use was about 0.96 for SWE and web items and about 0.02 for the rest. refusal_risk was 0.98 on the harmful prompt. Each request cost about 950 input tokens, took about 1 s and cost about $0.00004, so 100k items would cost roughly $4.

**(a) Runtime use is blocked for now.** At scoring time the sandbox can only reach approved inference APIs, and `api.typesafe.ai` is not known to be on that list. **Recommendation: ask the organizers whether TypeSafe can be approved.** It is cheap, fast and deterministic enough to run on the fly inside `predict()`.

**(b) Legitimate offline uses:**
1. **Feature discovery.** Compute Jev features for the training items. Add them to the GBM/IRT model and measure the change in Brier under leave-benchmark-out CV, following the autoresearch cookbook. Find which semantic dimensions predict item difficulty and the subject × item residual. Then reproduce the useful dimensions with things we can ship: regexes and metadata (tool tokens in `interactors`, length, MCQ format), or a small local classifier distilled on the Jev scores.
2. **Label-free item characterization** for analysis: profiling benchmarks, finding clusters and spotting distribution shift between train and the unseen benchmarks.

**(c) Leakage caveats:**
- Requests contain only inputs that `predict()` also receives (content, features, interactors, benchmark_id). Never put grading criteria, reference answers, responses or labels in the state or the questions.
- Do not tune question wording against hidden-benchmark labels.
- If Jev is used at runtime, compute it on the fly for hidden items. Never precompute or bundle features for hidden items.
- Pin `jev-1.13.0`, because the `jev-latest` alias can change.
- `--no-benchmark` removes benchmark_id from the state, which gives purer item-intrinsic features.
