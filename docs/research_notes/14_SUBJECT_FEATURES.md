# 14 — Subject features & subject matching (persona 9, challenged by persona 3)

Code: `src/paiec/analysis/subject_feats.py` (prints every number below; writes nothing). Data: matharena +
4 binary benchmarks, pairs with ≥80 distinct items (221 pairs; swe_rebench = 1 pair, train-only).
Target `y` = logit(response-level pair accuracy) − benchmark mean logit. Equal pair weight.
Budget-0 Brier computed exactly per pair from response mean m: `B(q)=m(1-q)²+(1-m)q²`.

## 1. Accuracy table and cross-benchmark rank agreement

| bench | pairs | acc mean (sd) | sd(y) | distinct names |
|---|---|---|---|---|
| matharena | 81 | 0.606 (0.194) | 1.07 | 72 (incl. 52 name-less, keyed by `model_config`) |
| multi_swebench | 82 | 0.220 (0.167) | 1.19 | 20 (12 names × up to 7 harnesses) |
| real_webagents | 26 | 0.340 (0.233) | 1.41 | 26 |
| researchcodebench | 31 | 0.353 (0.156) | 0.96 | 31 |

Spearman of y between benchmark pairs, matched by key (Level 1, n = 6–12 per pair):

| key | keys in ≥2 benches | pooled ρ | range |
|---|---|---|---|
| exact 8-field dict (= subject_id) | 12 | 0.57 | only real_webagents×researchcodebench |
| normalized_name (fallback `model_config`) | 19 | **0.56** | −0.14 (math×web) … 0.93 (mswe×rcb) |
| coarse family (versions/sizes stripped) | 18 | 0.30 | −0.08 … 0.86 |

Family normalisation **corrupts meaning**: it adds almost no overlap (18 vs 19), roughly halves ρ,
and groups very different abilities together (e.g. "claude sonnet" = 3.5 + 4.5, within-group sd(y) = 3.1
on matharena). **Rejected.**

## 2. Features → ability on an unseen benchmark / unseen family

Ridge (α=3) on runtime-visible fields only. R² is measured against the subject-agnostic prior y=0.
B0 = budget-0 Brier with the benchmark offset given by an oracle (this isolates the subject term).

| feature set | LOBO R² | LOFO(family) R² | B0 Δ (oracle offset) |
|---|---|---|---|
| release_date | 0.06–0.10 | 0.05–0.10 | −0.0003…−0.005 |
| provider | 0.06 | 0.07 | −0.002 |
| effort + has_harness + has_extra + has_name | 0.00 | 0.00–0.03 | 0 |
| name tokens (mini/flash/pro/…, log params) | 0.02 | 0.03–0.04 | ≈0 (sometimes worse) |
| **all metadata** | **0.16** | **0.18–0.22** | **−0.005 … −0.008** (0.2125 → 0.204–0.207) |
| name transfer (0.7 × mean y on other benches) | 0.025 (46% coverage) | n/a (0 coverage by construction) | −0.001 |
| name transfer, metadata fallback | 0.11 | 0.22 | −0.002 / −0.008 |

Per-benchmark LOBO R² (all_meta): matharena 0.09, multi_swebench 0.13, real_webagents 0.32,
researchcodebench 0.22. Full-fit coefficients: +1.0 logit per year of release; Anthropic, Google and
OpenAI +0.4…0.5; Meta −0.6; small-tier tokens −0.3.

Scale check: if the benchmark offset is unknown (we use the mean of the benchmark logits), the
agnostic B0 is **0.2435**, against 0.2125 with an oracle offset. That gap of **0.031** is about 4×
the best subject-feature gain. Budget-0 error is dominated by the benchmark offset, not by the subject.

Name transfer is weaker than its ρ suggests. The OLS slope of target y on source y ranges from 0.36
(multi_swebench) to 1.71 (matharena), so the scale is not stable across benchmarks. On a
harness-collapsed name level, R² = −0.10 at shrink 0.7. Abilities rank alike across benchmarks but
do not share a scale.

**Harness is the largest subject factor we measured.** In multi_swebench (12 names × 7 harnesses,
n=73), harness alone explains R² = **0.77** of y, name alone 0.16, and both together 0.92.
Agentless/RepoRepair score about +1.3, MSWE-agent about −1.3 logit. Harness strings do not recur across
benchmarks, so this cannot transfer at budget 0. It does mean that "same name, different harness" is
a different subject. Reasoning effort is not monotone in matharena: o3-mini scores low 0.42,
medium 0.69, high 0.56, and o4-mini is similar. Treat these as separate configs, not as an ordinal.

## 3. Temporal

Within-benchmark Spearman ρ(release_date, y): matharena +0.61 (p=2e-5, n=43), real_webagents +0.48
(p=0.02), researchcodebench +0.22 (ns), multi_swebench +0.18 (ns; confounded by harness). Pooled ρ is
0.28. Measuring release relative to the newest subject in the benchmark ("frontier-ness") gives 0.29,
so it is no better. **Drift risk is real.** Only 12% of training pairs were released in 2026 or
later, and release_date is missing for all 52 name-less matharena subjects. Hidden subjects are
plausibly newer than everything in training. The absolute-release coefficient (+1 logit/yr) then
extrapolates, and centring by benchmark only partly protects against this. Cap or clip
`rel_years` at the training maximum, and prefer runtime evidence to the prior.

## 4. Identity and matching key

- The exact 8-field dict maps one-to-one onto subject_id (238 keys ↔ 238 ids), so it **is** the subject.
- Across benchmarks the same normalized_name differs in access_date (47% of 59 cross-bench pairs),
  harness (47%) and subject_features_extra (44%). Only 13 exact dicts recur across benchmarks,
  against 22 names.
- subject_features_extra contains only `model_config=…;source_model_name=…` (all 104 matharena subjects,
  plus 5 in mmdocrag). It is the **only** identity for 52 matharena subjects whose name and provider
  are empty, and it encodes the effort variant (`o3-mini--low`).

**Recommendations** (claim contract below):
1. Within a benchmark, pool labels using the **exact 8-field dict**. It is the true identity, and the
   harness/effort effects are large.
2. Across benchmarks, pool ability through a hierarchical key: name_key = normalized_name, or else
   `model_config` with any effort suffix stripped. Use a weak, shrunk coupling (slope about 0.4–0.7,
   not 1) and add harness and effort as separate offsets. **Do not use family**.
3. Budget-0 prior: ridge on release (clipped) + provider + small-tier tokens. Expect at most about
   −0.005 to −0.008 Brier. Most of the remaining effort should go to the benchmark offset.
4. Drop harness_version (1/238 non-empty) and access_date as *features*. Keep them inside the
   identity key. Has-extra, has-harness and effort-as-ordinal add nothing out-of-sample, so drop
   them as features. Keep extra for parsing the name.

| Claim | Evidence level | Confidence | Decision |
|---|---|---|---|
| Metadata prior gives R² ≈ 0.16–0.22 out of benchmark and out of family | L2 (LOBO with 4 folds, LOFO) | moderately supported (4 folds; n of 26–82 per fold) | ADVANCE as B0 prior |
| Budget-0 gain from subject features ≪ offset uncertainty (0.007 vs 0.031) | L2 | strongly supported | ADVANCE (prioritise offset) |
| Name ranks transfer (ρ ≈ 0.56) but scale does not | L1 (n=6–12 per pair) | weak/moderate | MODIFY → shrunk coupling |
| Harness is a large ability modifier | L1 within one benchmark | strongly supported *within multi_swebench*; untested elsewhere | ADVANCE as a per-dict identity |
| Family normalisation helps | L1 | contradicted | ELIMINATE |
| Release date predicts ability; newer hidden subjects | L1 (ρ 0.18–0.61) | moderately supported; extrapolation untested | REPLICATE with a time-split (train ≤2025-06 → test later) |

Cold-start veto check: every claim above uses LOBO/LOFO or is labelled L1. No claim rests on a
random split.
