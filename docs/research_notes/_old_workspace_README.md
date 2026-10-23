> **Historical (written 2026-09-17, before the training data was available).** Kept for the record; facts about data, baselines and results are superseded by reports 10–20 and `docs/publication/research_report.html`.

# PAIEC 2026 — Team Workspace (First Action Complete)

Start here: `FIRST_ACTION_REPORT.md` (A–J), then `reports/00_TASK_SPEC.md`.

## Layout
- `FIRST_ACTION_REPORT.md` — §57 report (A–J).
- `reports/` — task spec, Feynman, formal stats, baseline repro, leakage audit, forensics, first-20 experiments.
- `src/` — `metric.py` (verified official Brier/ALC/ECE), `naive_baselines.py`, `irt_matrix.py`, `leakage_checks.py`, `forensics.py`.
- `experiments/` — append-only `registry.md` + `registry.csv`.
- `external/official_baselines/` — official baseline clone (19/19 tests pass, mean ZIP valid, BLE mock smoke OK).
- `../measurement-db/` — bank schema + builders clone (schema v3, 6 example benchmarks' metadata).

## Verify locally (no gated data needed)
```
python src/paiec/evaluation/metric.py
(cd paiec_baseline && python -m unittest discover -s tools -p 'test_*.py')
(cd paiec_baseline && python tools/build_mean_zip.py && python check_submission_zip.py dist/empirical_mean.zip)
(cd paiec_baseline && python tools/smoke_test.py --mock)
```

## Unblock data forensics (gated)
Register team → HF approval → `huggingface-cli login` → `(cd paiec_baseline && python tools/prepare_data.py --out payload)` → `python src/paiec/analysis/forensics.py --data payload/data --out docs/research_notes/forensics/`.

## Operating philosophy
MEASURE → HYPOTHESIZE → CONTROLLED EXPERIMENT → ELIMINATE → ABLATE → VALIDATE → GENERALIZE → REPRODUCE. No large-scale runs before E01–E20 review.
