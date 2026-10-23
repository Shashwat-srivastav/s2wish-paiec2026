> **Correction (2026-10-02, found in E2 / report 20):** `predict()` sees only the target item and `labeled`, never the hidden benchmark's full item pool. Standardising item features *within the benchmark* is therefore not possible at budget 0 at scoring time, contrary to what this report assumed. Gains that rely on it (the Jev prior L5: −6.2e-3 R1) are an offline upper bound, not a deployable result. A deployable variant using fixed training statistics is evaluated in report 20.

# 13 — Item representations for cold-start difficulty: what transfers across benchmarks?

Personas: Representation Scientist (4), with the Mathematical Minimalist (18) as internal skeptic.
Code: `src/item_repr.py`. Results: `experiments/item_repr_results.csv`, log `experiments/item_repr.log`.
Jev features: `experiments/jev_out_<bench>.parquet` (cache `data/jev_cache/jev_v1.parquet`).

## Setup (one downstream model, one split; only the representation changes)
- **Items.** matharena 1755, multi_swebench 2126, real_webagents 233, researchcodebench 212, and swe_rebench
  as a random sample of 1500 (seed 0, the same items for every representation). Total 5826.
- **Target.** Per-benchmark Rasch β_i, fit by L2-regularised JML on trial counts (prior N(0,1) on β). Each
  benchmark's β uses only its own responses, so a held-out benchmark's labels are used only for scoring.
  β is centred within its benchmark. A secondary target, the smoothed logit of mean correctness, gives the
  same conclusions (column `spearman_plogit`).
- **Features.** Each block is z-scored within its benchmark. This is label-free and legal at runtime,
  because predict() sees the hidden benchmark's items. The model therefore learns *within-benchmark
  ranking*. The benchmark offset is not modelled here: with only 5 benchmarks it cannot be identified.
- **Model.** Ridge with benchmark-balanced weights. Alpha is chosen by inner leave-one-benchmark-out (LOBO)
  CV, or by inner fold CV for the within-benchmark runs.
- **Representations:**
  - triv: 13 length and format features.
  - TF-IDF: 2000 uni- and bigrams, vocabulary fit on the training folds only.
  - Jev: 12 columns, i.e. the 9 questions plus 3 confidences. Jev saw content only; it was run with
    `--no-benchmark` and without `item_features`.
  - jevd: the single `jev_difficulty` scalar.
  - OpenAI embeddings: available for 3 benchmarks only. They are joined on sha1(content), because the embed
    item_ids differ from the items table.
- **Brier.** Scored on held-out (subject,item) pairs with logit p = θ_s(oracle) − [mean β_b(oracle) +
  f(x_i)].
  - Compared against **none** (f = 0, item-free) and **oracle β**, the in-sample ceiling.
  - **slope** also refits one scalar slope on the held-out benchmark. This separates ranking value from
    scale error.
  - Because the offset and θ are oracle, the absolute Brier values are optimistic. Only the Δ values are
    meaningful.
- **Within-benchmark CV** uses 5 folds grouped by exact content. matharena contains 857 exact-duplicate
  problems across competitions.

## Results — LOBO (primary). Spearman(pred, β) [95% bootstrap CI]; ΔBrier ×1000 vs item-free (negative = better)

| rep | matharena | multi_swe | webagents | rcb | swe_rebench | mean ρ | mean ΔBrier | ΔBrier, slope-oracle |
|---|---|---|---|---|---|---|---|---|
| triv | −.01 | .02 | −.01 | −.08 | .04 | **−.01** | +1.3 | −0.8 |
| TF-IDF | −.04 | .02 | −.06 | **−.34** | .03 | **−.08** | +2.0 | −2.6 |
| jevd (1 scalar, no fit needed) | **.34** [.30,.38] | .20 [.15,.24] | .19 [.07,.31] | .21 [.08,.35] | .21 [.16,.25] | **.23** | −5.9 | −9.3 |
| Jev (12) | .25 | **.24** | .19 | **.25** | .23 | **.23** | **−6.6** | −7.9 |
| triv+Jev | .25 | .24 | .19 | .19 | .23 | .22 | −3.9 | −6.4 |
| triv+Jev+TF-IDF | .10 | .24 | .19 | .16 | .23 | .18 | −3.0 | −4.8 |
| *oracle β ceiling* | | | | | | 1 | −109 | |

For context, the item-free Brier on each benchmark is 0.200 / 0.107 / 0.189 / 0.209 / 0.249.

The 3-benchmark subset with embeddings (LOBO, 2 training benchmarks) gives these mean ρ values:

| rep | mean ρ |
|---|---|
| emb | −.08 |
| TF-IDF | −.05 |
| triv | −.10 |
| jevd | **.20** |
| Jev | .17 |
| emb+Jev | .14 |

For emb alone, the inner CV picked α = 10⁴, which means a constant prediction.

**Within-benchmark CV (diagnostic only).** Content features do carry difficulty signal *inside* a benchmark.

| rep | mean ρ |
|---|---|
| TF-IDF | .36 |
| triv+Jev+TF-IDF | .41 |
| emb (3 benchmarks) | .35 |
| emb+Jev (3 benchmarks) | .42 |
| Jev | .33 |
| triv | .22 |

## Claim (contract §3)
1. **CLAIM.** Lexical, format and embedding representations carry within-benchmark difficulty signal
   (ρ ≈ .2–.55) that does **not** transfer across benchmarks (LOBO ρ ≈ 0, sometimes significantly
   negative). A single LLM-judged difficulty scalar (Jev) does transfer, at a modest level (LOBO ρ .19–.34,
   CI excludes 0 on 5/5 folds).
2. **MECHANISM.** Surface features map onto difficulty differently in each domain: a long matharena
   problem is hard, but a long web task is not. A ridge fit on other domains therefore learns sign-flipped
   or idiosyncratic weights. Jev answers a domain-normalised question ("how hard for a frontier model?").
3. **PRIOR.** Plausible. The smoke test in report 09 already ranked items sensibly.
4. **EVIDENCE.** Table above. Zero-fit `jev_difficulty` gives ρ = .34/.20/.19/.21/.21 directly, with no
   training at all.
5. **COUNTEREVIDENCE.** The Brier gain is small: about −0.006, roughly 5–10% of the oracle-β headroom.
   Adding triv or TF-IDF on top of Jev *hurts* under LOBO.
6. **COMPETING EXPLANATIONS.**
   - (a) Jev may recognise well-known public problems (contamination or memorised difficulty), for example
     IMO vs AIME. On new hidden benchmarks this would make the transfer optimistic.
   - (b) The Brier baseline uses oracle θ and oracle offset, so the Δ values are on an optimistic scale.
7. **FALSIFICATION TEST.** Run Jev on a benchmark released after the Jev training cutoff. Also distil
   jev_difficulty into a local model and check whether ρ survives.
8. **DECISION.**
   - The item-difficulty prior from Jev is **MODIFY/REPLICATE**. It is useful offline; it is blocked at
     runtime.
   - Content-embedding priors for cold start are **ELIMINATE**.
   - Within-benchmark content similarity is **DEFER**: it may help once labels exist, through
     kernel/retrieval smoothing of β among items of the same hidden benchmark, which is where its signal
     lives.
9. **CONFIDENCE.**
   - "Content/embeddings do not transfer": strongly supported (5 folds, consistent).
   - "Jev transfers": moderately supported.

## Recommendation
- **Cheapest representation for the gain:** `jev_difficulty` alone. It is 1 number, needs 0 fitted
  parameters beyond a slope, and costs about $0.00005 per item. The other 11 Jev columns add essentially
  nothing to the ranking.
- **Use in the model:** bring it in as a *weak* prior: β_i = c·z_b(jev_difficulty_i), with c ≈ 0.4
  logits/SD from LOBO. It must not replace label-driven β.
- **Runtime:** only if the organisers approve api.typesafe.ai. Otherwise, distil jev_difficulty into a
  local model and test it under LOBO before shipping. That model could be a small text regressor or an
  allowed inference API with the same rubric.
- **Do not ship** TF-IDF or embedding difficulty priors for cold start.
- **Item-free baseline:** the item-difficulty channel is worth at most −0.006 Brier at budget 0, compared
  with the benchmark offset (see report 10), which dominates. Prioritise the offset and ability pooling.

## Jev run
- 5826 items, 4914 unique states (matharena has duplicates).
- 6.86M input tokens, about **$0.29** total.
- Wall time about **2 min** (16 workers, median latency 0.32 s). researchcodebench is the costliest per
  item, since it is truncated to 8k characters (head plus tail).
