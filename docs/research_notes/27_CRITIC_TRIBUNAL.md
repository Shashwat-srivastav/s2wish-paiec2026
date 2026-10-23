# 27 — Critic tribunal (2026-10-15)

Three brutal critics (methodology, strategy, engineering) each attacked the programme; three counterparts then
checked every claim against the files, conceded or rebutted, and ruled DO / DON'T / DO-MODIFIED. The PI verified
the load-bearing numbers independently. This report records the rulings; it changes no locked procedure.

## Verified facts (PI-checked)

- Code + technical report are due **30 Oct 2026 23:59 AoE** via OpenReview with rerun instructions
  (`00_TASK_SPEC.md:119`) — a day before the board closes.
- The shipped v1_3 archive treats `facets="named"` as truthy → **all facets** (`submissions/v1_2/model.py:184`).
  Shipped-equivalent R2 ALC 0.208824 vs named-only engine 0.208722 (Δ level 1.0e-4). E6 (`e6_headroom.py:53`) and
  E7 (`e7_expanded_pool.py:46`) used the named-only engine and labelled it "shipped".
- E6 with a perfect item oracle (ρ=1) reaches local ALC **0.1387** (`experiments/runs/e6/e6.json`), so the
  pre-registration's "ρ≈0.85 → 0.11" (prereg §7.3) is contradicted. Δ/ρ² = 58, 63, 71, 74, 70 for ρ=0.2…1.0: roughly
  quadratic, not a law. 0.102 is E1's leave-pair-out three-oracle (0.1016), which uses scored labels — an
  optimistic bound, not a floor.
- E7 name-ability prior: −1.02e-3 [−1.54, −0.52], 4/4 folds negative — passes today's bar by 0.02e-3, but failed
  its own pre-registered −2e-3 bar and was measured on the named-only engine.
- Board entry 971666 (0.1643, 2026-10-09) is logged nowhere; its archive is unknown locally.
- The repo has no git remote; HEAD is 2026-10-02; ~74 paths are uncommitted, including the shipped v1_3 zip.

## Rulings

| # | Issue (critic) | Ruling | Action | Cost |
|---|---|---|---|---|
| 1 | Report/dossiers say "named facets"; shipped is all-facet (C2, C3) | Honesty flaw, ~0.1e-3 performance | Erratum + relabel "named (effective: all)"; never edit the shipped params (would change SHA) | 3 h |
| 2 | No owner for the 30 Oct deliverable; work uncommitted, no remote (C2, C3) | Conceded; **biggest risk** | Snapshot commit + off-machine copy now; report owner from 10-19; freeze 10-24; clean-clone rerun by 10-28 | 3–4 d |
| 3 | ×30 lucky-draw resubmission (C2) | Killed. Odds 45–85% over 30 (not 25%), but the board is not the final ranking and our policy forbids showing the luckiest draw | ≤10 uploads total as variance data for EXP-0039/0040; log 971666 | minutes |
| 4 | Organiser email unsent; only big-upside route blocked (C2) | DO today. Realistic LLM-scorer value ≈ Jev's −5.8e-3, not −18e-3 (ρ 0.455 was pooled within MathArena) | Send; scorer proceeds only if an approved API shows held-out ρ ≥ 0.30 by 10-20 | 10 min |
| 5 | 1e-3 programme can't move the outcome (C1, C2) | Partly: worthless for the formative board, small but real for the summative (common subset, true means) | Do not run EXP-0049 Stages 1–3; finished experiments become report content; keep EXP-0050 for the final finalist only | saves ~5 d |
| 6 | E6/E7 on the wrong engine (C1, C3) | E6 deltas (−5.7…−70e-3) dwarf the 1e-4 level gap: no rerun, relabel as an upper bound. E7 subject arm is knife-edge | New pre-registered re-test of the name prior on `facets="all", guard=0` (+R1), after EXP-0039 passes | ~1 h + 1 d packaging |
| 7 | Adoption rule blind to benchmark variance (C1) | Conceded; a 3-df t gate would veto everything (needs Δ≈−2.6e-3 for 80% power) | Add: hierarchical interval upper < 0 **and** posterior-predictive P(Δ_new < 0) ≥ 0.8 | 0 |
| 8 | Multiplicity (C1) | Partly; null adoptions are cheap, the real bias is best-of-N selection | No Holm; confirm any single winner once on fresh seeds (R2 10–19) | one run |
| 9 | Simulator unrepresentative; EXP-0039 gate bypassed (C1) | Conceded | Done: decisions frozen until EXP-0039 reports. One base-rate-shift stress arm on the final candidate | one run |
| 10 | Compute gate racy and bypassed (C3) | Conceded (low RAM is mostly other apps; experiments ~100 MB) | Done: `run_slot.py` rewritten on OS locks, tested | — |
| 11 | No v1_3 equivalence test; v1_3 not reproducible from git (C3) | Partly: v1_3 = v1_2 code + 2 params; `build_zip --verify` compares bytes | Extract to `submissions/v1_3_named_facets_noguard/`, `--verify` only (never build without it), add a 4-fold package-vs-engine test | 15 min |
| 12 | Episodes cache non-atomic (C3) | Mostly rebutted: all caches built, nothing rebuilds today | Atomic write + revision-keyed cache after the freeze | 15 min |
| 13 | "0.03–0.07 behind" (C2) | Partly: best-of-~13 explains ~0.026; 0.117 is 4.5σ and not explained by luck | Restate the gap as ~0.02–0.05 in expectation | 0 |

**Most important methodology change (C1 counterpart):** every runner asserts it reproduces the shipped v1_3
(R2 base ALC 0.208824 ± 1e-6), and adoption gates on the hierarchical predictive probability, not the pair bootstrap.
Both errors so far (the false w_od lead, the knife-edge E7 call) came from the wrong baseline plus pair-level CIs.
