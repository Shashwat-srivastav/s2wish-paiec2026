# 10 — Real data forensics (measurement-db @ bc8204d811, 2026-10-01)

Core tables only (benchmarks/subjects/items/response); raw/traces not downloaded. Embeddings repo
(text-embedding-3-small, 1536-d) covers only real_webagents, researchcodebench, swe_rebench.

| Benchmark | response_type | official-eligible* | subjects | pairs ≥80 items | pair acc mean (sd) | range | trials/item |
|---|---|---|---|---|---|---|---|
| matharena | mixed (values all 0/1) | NO | 104 | 81 | 0.606 (0.194) | 0.03–0.99 | ~4 |
| mmdocrag | fraction (5% binary) | NO | 68 | 51 | — (not binary) | — | 1 |
| multi_swebench | binary | yes | 82 | 82 | 0.220 (0.167) | 0.01–0.57 | 1 |
| real_webagents | binary | yes | 33 | 26 | 0.340 (0.233) | 0.02–0.75 | 1 |
| researchcodebench | binary | yes | 31 | 31 | 0.353 (0.156) | 0.01–0.62 | 1 |
| swe_rebench | binary | yes | **1** | 1 | 0.479 | — | ~10.6 |

\* `external/official_baselines/tools/prepare_data.py::_elig` = `response_type=="binary" and granularity=="item"`.

## Findings (Level 1 — descriptive)
1. **The training pool is tiny at the benchmark level**: 4 officially eligible benchmarks, one of which
   has a single subject. Leave-one-benchmark-out has at most 4–5 folds → any benchmark-level model
   (δ_b predicted from content) is estimated from ≤5 points. Heavy cold-start architectures are
   unidentifiable here; strong priors + runtime evidence pooling is the realistic lever.
2. **Benchmark mean accuracy varies enormously** (0.22 → 0.61; mmdocrag ~0.89 if thresholded). At
   budget 0 on an unseen benchmark, the benchmark offset is essentially unknown → B0 is dominated by
   benchmark-level uncertainty, not item difficulty.
3. **Subject overlap across benchmarks is low**: 132 normalized names, 34 appear in ≥2 benchmarks,
   15 in ≥3. Subject-ability transfer across benchmarks is testable but on few subjects.
4. **Subject metadata is mostly empty**: harness empty for 237/319, reasoning_effort empty for 288,
   harness_version empty for 318. Provider is the only well-populated field besides name/dates.
5. Matharena is 0/1-valued despite `mixed` type and has many repeated trials — usable as extra
   *training* signal (public data), even though it is not an official eligible benchmark.

## Implication for method (hypothesis, to be tested)
H-POOL: at budgets ≥1 the dominant gain comes from pooling runtime `labeled` evidence across pairs
(other subjects on the same hidden benchmark → benchmark offset; same subject on other benchmarks →
ability) in a hierarchical logistic model `logit p = μ + a_s + d_b (+ β_i)` with priors fit on the
public pool. Depends on the exact `labeled` semantics — **resolved**: the simulator audit (Phase 0) and the starting-kit README confirm `labeled` carries evidence from all sampled pairs, with per-pair 50/50 splits. H-POOL was tested in Phase 2 (report 15): offset pooling helps mainly at B1; item pooling is the only benchmark-robust component.
