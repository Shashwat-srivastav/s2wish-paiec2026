> **Historical (written 2026-09-17, before the training data was available).** Kept for the record; facts about data, baselines and results are superseded by reports 10–20 and `docs/publication/research_report.html`.

# Feynman Explanation — Predicting AI Responses Without Running the AI

## The student-and-exam analogy
We have a **student** (an AI system: model + harness + version + reasoning setting) and a **new exam question** (a benchmark item: text + features + tools + anonymous benchmark ID).
We know some properties of the student (who they are, when released, what harness drives them).
We know some properties of the question (its text, its features, which exam it belongs to).
We may observe a **tiny number** of the student's previous answers **on the same exam** (0 to 31 revealed labels from a separate acquisition pool — never the target questions themselves).
We must estimate **P(student answers the new question correctly) WITHOUT asking the student the target question.**

That is the entire competition. Everything else is how to spend a tiny evidence budget wisely and how to be honest about uncertainty.

## Why this is difficult
- **Benchmark-level cold start.** Test questions come from **held-out benchmarks** — entire exams never seen in training. We cannot memorize question answers; we must transfer knowledge across exams.
- **Sparse evidence.** 0–31 labels per student–exam pair must inform predictions on dozens of unseen questions. At budget 0 we have only metadata.
- **Heterogeneity.** Students differ by model, provider, generation, harness, reasoning effort. Questions differ by domain, length, tool use, grading. Averages lie.
- **Repeated trials.** The same student–question pair can appear multiple times as separate targets (trial noise / nondeterminism). We predict a probability, not a fixed label.
- **Equal pair weighting.** A small exam counts as much as a large one. Overfitting to big benchmarks fails.

## Why benchmark difficulty matters
Exams differ wildly in hardness (e.g. agentic SWE tasks vs factual trivia). A strong student can fail a hard exam; a weak student can pass an easy one. Without a benchmark-difficulty prior, every prediction collapses to a global average and Brier suffers — especially at budget 0 where no labels calibrate us. Benchmark ID (anonymous but stable within a pair) lets us pool the few revealed labels into a pair-specific difficulty estimate (the empirical-mean baseline).

## Why model identity matters
Models have persistent capability differences (scale, generation, provider practices, release date). Knowing *which* student we predict — and which related students we saw in training — provides the prior before any labels arrive. When the exact model is unseen, family/provider/generation priors are the only transfer path.

## Why model–item interactions matter
Global easiness + global strength is not enough. A coding-specialized model may ace SWE items but fail trivia; a harness change can flip agentic performance while leaving QA unchanged. The residual after additive effects — *this student × this question* — is where IRT discrimination, matrix factors, and semantic retrieval win or lose. The competition explicitly rewards capturing it under sparse labels.

## Why a few observations can dramatically change belief
One label on the same student–exam pair collapses a wide prior (say 0.5 ± wide) into a much sharper posterior about that pair's success rate. Going 0→1 and 1→3 has outsized ALC weight (0.1+0.2+0.2 = half the ALC on budgets 0/1/3). A good updater (Bayesian/IRT posterior, shrunk mean) extracts maximum information; a bad one (unsmoothed 0/1 after one obs) becomes overconfident and Brier-explodes on the remaining eval items.

## Why calibration matters
Brier = calibration + sharpness + irreducible noise. Ranking correctly (AUC) while outputting 0.99 on 70%-correct groups is punished quadratically. ECE is reported as diagnostic; Brier ALC is the score. Benchmark-specific and budget-specific calibration (e.g. shrink 1-obs means toward prior) directly lowers Brier even when ranking is unchanged.

## Why uncertainty matters
With 0–31 labels we often *should* say "I don't know" (≈0.5, or a wide posterior mean near the prior). Uncertainty tells the acquisition function what to ask next (uncertain + representative + informative items) and tells decision-makers when *not* to trust extrapolation (OOD student/benchmark, sparse region). A model that knows when it doesn't know buys the right labels and avoids catastrophic 0/1 bets.

## Why predicting the mean performance is insufficient
The empirical-mean baseline predicts the *same* number for every unseen question in a pair. It captures pair difficulty but ignores *which* questions are hard. Two students with 50% mean can fail disjoint halves — mean-only prediction cannot distinguish them, cannot rank items, cannot select informative acquisitions, and leaves item-difficulty and interaction signal on the table. Beating the mean requires item representations (text/features/embeddings), subject representations (metadata/history/latent ability), and their interaction — calibrated and updated from sparse labels.
