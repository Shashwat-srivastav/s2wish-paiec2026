# Team S2WISH — AIMS Predictive AI Evaluation Competition (NeurIPS 2026)

Predict the probability that an AI system answers an item from an **unseen benchmark** correctly,
first with no labels and then from up to 31 labels it chooses to reveal. Score: **Brier ALC**
`0.1·B0 + 0.2·B1 + 0.2·B3 + 0.2·B7 + 0.2·B15 + 0.1·B31`, averaged over subject–benchmark pairs
(lower is better; always predicting 0.5 scores 0.25).
Competition: <https://aims.stanford.edu/competition> · Codabench competition: <https://www.codabench.org/competitions/17828/>

## At a glance

| Item | Where |
|---|---|
| Technical report (submission, NeurIPS template) | [`docs/publication/zenodo/S2WISH_PAIEC2026_technical_report.pdf`](docs/publication/zenodo/S2WISH_PAIEC2026_technical_report.pdf) |
| Submitted archives and their SHA-256 | [`submissions/dist/`](submissions/dist/) |
| Rerun every number in the report | [`REPRODUCE.md`](REPRODUCE.md) |

The full research record (every experiment, its question, method, result and decision) is kept privately by the team.

**Final model** (`submissions/dist/v1_3_named_facets_noguard.zip`): a small hierarchical Bayesian predictor (benchmark
offset with a heavy-tailed prior, pair ability, item difficulty pooled across subjects, item-group effects, exact
posterior integration). Pure Python and NumPy, about 5 KB.

| Result | Value |
|---|---|
| Local leave-one-benchmark-out ALC, one subject per benchmark (10 paired seeds) | official baseline 0.2451 → final model **0.2088** |
| Of that gain, plain Beta-binomial shrinkage | 33.7 of 36.3 ×10⁻³; the later refinements are small at the level of four held-out benchmarks |
| Official formative runs of the final model (4 runs; each is a fresh draw) | mean **0.1839** (SD 0.024) |

The report's limitations section states what these numbers do and do not show.

## Repository layout

```
├── src/paiec/              research package (pip install -e .)
│   ├── paths.py            every file location in one place
│   ├── evaluation/         metric, official-protocol simulator, LOBO episodes, synthetic data
│   ├── analysis/           forensics, leakage audit, signal decomposition, subject/item studies, red team
│   ├── models/             baseline ladder (empirical mean → hierarchical pooled model), IRT
│   └── jev/                offline TypeSafe Jev: item scoring + research Decision Layer
├── tests/                  simulator ↔ official run_streaming equivalence tests
├── submissions/            v0/, v1/ (self-contained Codabench folders) and dist/*.zip
├── experiments/            registry.md/csv (every run), decisions.jsonl (PI + Jev decisions), runs/<phase>/
├── artifacts/              fitted priors, cached episodes (not shipped in the public release)
├── docs/
│   ├── spec/               orchestrator_merged.md (the operating spec), prompt.md, agents.md
│   └── reports/            numbered research reports 00–16
├── data/                   measurement-db tables (gated; not shipped; see data/README.md)
└── external/               NOT SHIPPED. Organisers' code, unmodified, which you fetch yourself:
                            starting_kit (Codabench), official_baselines and measurement-db (pins in REPRODUCE.md)
```

## Install

CPU only. The paper's numbers were produced with CPython 3.12.6. Pinned versions are in `requirements-lock.txt`.

```bash
python -m venv .venv && .venv/Scripts/activate     # POSIX: source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
```

## Data

The training data is the gated Hugging Face dataset `aims-foundations/measurement-db`. It is not redistributed
here, and `data/` is gitignored. Request access, then place the core tables under `data/<benchmark>/` at revision
`bc8204d811823da849c6686bf124d4ca9f82e4de` (see `data/README.md`). The table in [`REPRODUCE.md`](REPRODUCE.md#2-data-gated-not-redistributed)
gives every revision hash: the expanded release `54074453da`, embeddings `fb3a155cce`, and the pinned
`external/official_baselines`. The code reads `<repo>/data` (`src/paiec/paths.py`). Jev tools need
`TYPESAFE_API_KEY`; they are offline research aids, and nothing needs them to reproduce the paper.

## Reproduce the report

```bash
python scripts/reproduce_all.py --smoke   # < 1 min, no gated data: archives, hashes, rebuild, saved results, figures
python scripts/reproduce_all.py --quick   # every table/figure/headline number vs the paper, ~2 min; PASS/FAIL table
python scripts/reproduce_all.py --full    # + reruns of the heavy experiments (~2.5 h, via scripts/run_slot.py)
python scripts/clean_clone_check.py       # fresh clone + venv + pytest + --quick, dated report
```

[`REPRODUCE.md`](REPRODUCE.md) maps each table, figure and number to its command, expected value, tolerance,
runtime and data. It also lists the numbers that cannot be regenerated yet.

## Quickstart

```bash
# 1. a fresh virtual environment first (a system interpreter with a broken fontTools cannot build the figures)
python -m venv .venv && .venv/Scripts/activate     # POSIX: source .venv/bin/activate
python -m pip install -r requirements-lock.txt && python -m pip install --no-deps -e .

# 2. checks
python scripts/reproduce_all.py --smoke   # < 1 min, no gated data
python -m pytest -q                    # whole test suite, about 7 minutes; tests whose inputs are absent skip
python scripts/build_v1_3_archive.py   # rebuild the final archive byte-identically from submissions/v1_2

# reproduce the Phase-2 ladder (leave-one-benchmark-out, R1/R2/R3 draw regimes; needs the gated data)
python -m paiec.models.ladder
```

The organisers' submission checks need the competition starting kit, which this repository does not ship. **Requires
the organisers' starting kit:** download it from <https://www.codabench.org/competitions/17828/> and unpack it to
`external/starting_kit/`. Then:

```bash
python external/starting_kit/tools/check_submission_zip.py submissions/dist/v1_3_named_facets_noguard.zip  # requires the starting kit
python external/starting_kit/tools/run_smoke_test.py submissions/v1_2                                      # requires the starting kit
```

## How the research is run

The operating spec is [`docs/spec/orchestrator_merged.md`](docs/spec/orchestrator_merged.md): a
Principal Investigator routes each question to the relevant expert personas, every claim carries
evidence, counter-evidence and a falsification test, and every method must pass gates A–L
(validity → cold start → Brier gain → calibration → … → frozen reproduction). Research decisions
are double-checked by the **Jev Decision Layer** (`python -m paiec.jev.decide`), which logs Jev's
calibrated verdict next to the PI's and flags disagreements. Jev is a research aid only; it is
never called inside a submission.

| Report | Topic |
|---|---|
| 00–08 | task spec, formal stats, baseline reproduction, leakage plan (pre-data, historical) |
| 09 | Jev feasibility |
| 10 | real-data forensics |
| 11 | signal decomposition (Measurement Theorist + Bayesian Statistician) |
| 12 | leakage forensics (Leakage Hunter + Benchmark Forensicist) |
| 13 | item representations incl. Jev (Representation Scientist + Minimalist) |
| 14 | subject features (LLM/Subject Scientist + Cold-Start Specialist) |
| 15 | Phase 2 model ladder → v1 |
| 16 | red team of v1 |
| 17 | v1.1 repairs + acquisition policies |
| 18 | verified literature review (29 papers) mapped to open decisions |
| 19 | E1: where the budget-0/1 loss comes from — item difficulty ≈ half; offset hypothesis rejected |
| 20 | E2: Jev is not memorising, but no local model reproduces it — stopped |

Pre-registrations with fixed decision rules: [`docs/preregistrations/`](docs/preregistrations/).

## Disclosure

Training data: public `aims-foundations/measurement-db` (rev `bc8204d811`) and
`measurement-db-embed`. Offline analysis used TypeSafe Jev (`jev-1.13.0`). An LLM-based coding assistant
was used during development.

## License

MIT License: see [`LICENSE`](LICENSE). The gated dataset is not included and remains under its own terms of access.
