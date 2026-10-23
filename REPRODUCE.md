# Reproducing the S2WISH technical report

One command checks every table, figure and headline number of
`docs/publication/neurips/main.tex` against code and raw files:

```bash
python scripts/reproduce_all.py --quick      # ~1-2 min of compute (+ ~1 min the first time, to build the episode cache)
python scripts/reproduce_all.py --full       # adds full reruns of the heavy experiments (~2.5 h, one at a time)
```

It prints one row per number (`PASS` / `FAIL` / `NOSRC` / `SKIP`) and exits non-zero if anything does not
reproduce. Nothing in the repository is modified; scratch output goes to `.tmp/reproduce_all/`
(figures in `.tmp/reproduce_all/figures/`, machine-readable results in `.tmp/reproduce_all/last_run.json`).

- **Tolerance:** "rounds to the printed value", i.e. half a unit in the printed last digit, unless the table says
  otherwise.
- **Live recomputations:** these are compared with the saved result files at 1e-12 (`--quick`) or 1e-9 (`--full`).
  The pipeline is deterministic: fixed seeds, no threading-dependent reductions.
- **`NOSRC`:** the paper number has no saved machine-readable source. The row says what does exist.
- **`SKIP`:** the check needs an input that is not in the public release. Exit code 1 means a `FAIL` or `NOSRC`
  row; add `--strict` to count `SKIP` rows as failures as well.

**Public release.** Some inputs are not distributed:
- the gated dataset (`data/`) and item-level files derived from it (`experiments/runs/**/*.parquet`, `*.npz`);
- our saved formative feedback (not released);
- `artifacts/` and `external/`.

Checks that need any of these are reported as `SKIP requires gated data / organiser-held feedback: missing <path>`.
To preview what a grader with only the public repository sees, run
`python scripts/reproduce_all.py --quick --simulate-public`. That run gives SKIP rows only for Table 1, live Table 2, the platform runs/Fig. 3 and the feedback-inversion (`Q4.*`) checks.
`--smoke` is the subset that needs neither data nor feedback.

To test a fresh checkout end to end (clone, venv, data link, pytest, `--quick`):

```bash
python scripts/clean_clone_check.py            # report -> experiments/runs/reproducibility/clean_clone_<date>.md
python scripts/clean_clone_check.py --no-venv  # offline variant: current interpreter + PYTHONPATH
```

## 1. Environment

- **Machine:** CPU only. The paper's numbers came from CPython **3.12.6** (64-bit) on Windows 11. Nothing is
  OS-specific except `scripts/run_slot.py`'s file lock, which has a POSIX branch.
- **Pins:** `requirements-lock.txt` pins every package that tracked code imports, taken from `pip freeze` of
  that environment.

```bash
python -m venv .venv && .venv/Scripts/activate        # POSIX: source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python -m pytest -q                                   # about 7 min; data-dependent tests skip without data/
```

The figures need a working `fontTools.subset`, which matplotlib's PDF backend uses with `pdf.fonttype 42`. If
`import fontTools.subset; fontTools.subset.Options` fails, reinstall fonttools in your venv. Our dev interpreter's
user-site copy was broken this way, and in that interpreter `reproduce_all.py` reports `FIG.env` as an environment
SKIP.

On a machine with little free memory (we saw this with 6 GB of RAM), a few subprocess checks can fail with
`OpenBLAS error: Memory allocation still failed`. That is the environment, not a reproduction failure: rerun with
one BLAS thread, `OPENBLAS_NUM_THREADS=1 python scripts/reproduce_all.py --quick` (all checks then pass).

## 2. Data (gated; not redistributed)

All training data comes from the gated Hugging Face dataset **`aims-foundations/measurement-db`**. Request access on
its Hugging Face page, then run `huggingface-cli login`. The code reads `data/` at the repository root
(`src/paiec/paths.py`, `DATA = ROOT / "data"`, no override).

| what | Hugging Face repo | revision (full hash) | where it goes | used for |
|---|---|---|---|---|
| core pool, five benchmarks | `aims-foundations/measurement-db` | `bc8204d811823da849c6686bf124d4ca9f82e4de` | `data/<benchmark>/{benchmarks,subjects,items,response}.parquet` + `metadata.yaml`, `characterization.yaml`; the hash goes in `data/REVISION` | every model and parameter, Tables 1-3, all programme experiments |
| expanded 281-benchmark release | `aims-foundations/measurement-db` | `54074453da7b9911be0f48e4c560be547bc84d21` | `data/_main_54074453/<benchmark>/formatted_tables/*.parquet`, with the hash in `data/_main_54074453/REVISION` | EXP-0039R simulator repair, E7 expanded-pool priors |
| embeddings | `aims-foundations/measurement-db-embed` | `fb3a155ccefe5b5ec6398d64329cfc569945d3c2` | `data/embed/<benchmark>.parquet` (3 benchmarks), with the hash in `data/EMBED_REVISION` | Table 5 embedding row only |
| offline LLM ratings | none (our TypeSafe Jev 1.13.0 outputs) | n/a | `data/jev_cache/jev_v1.parquet` | Table 5 LLM row (offline only) |

Download example (the five core benchmarks are `matharena multi_swebench real_webagents researchcodebench
swe_rebench`; MMDocRAG is downloaded but unused):

```python
from huggingface_hub import snapshot_download
snapshot_download("aims-foundations/measurement-db", repo_type="dataset",
                  revision="bc8204d811823da849c6686bf124d4ca9f82e4de", local_dir="data/_hf_core")
```

Then place each benchmark's four tables under `data/<benchmark>/` as listed in `data/README.md`, and write the
revision hash to `data/REVISION`. **This placement step is not scripted.** `reproduce_all.py` checks Table 1 (row
counts per benchmark) and the revision prefix, so a wrong placement shows up as `T1` failures.

The expanded release is about 2.4 GB in total with the core pool. A clean clone can reuse an existing copy without
copying it: `clean_clone_check.py` makes `<clone>/data` a directory junction (or a symlink) to an existing `data/`.

Third-party code: the organisers' starting kit is **not** redistributed here (it is theirs). Download it from <https://www.codabench.org/competitions/17828/> and unpack it to `external/starting_kit/`; without it, `tests/conftest.py` skips the three test modules that need the official streaming client, and every other check still runs. `external/official_baselines` is a nested clone that you
re-clone at `82d330ddcdb16016a3ae9e048db7e588ba2c6a39` (`external/PINS.txt`); `tests/test_simulator.py` needs it.

## 3. Command table (paper Appendix A)

The same table is in `docs/publication/neurips/appendix_repro_table.tex`.

Runtimes:
- "Check" runtimes were measured on 2026-10-19 in the clean-clone venv (Python 3.12.6, 3 BLAS threads).
- "Rerun" runtimes are the `elapsed_seconds` recorded by the original run, in its result JSON or log.

Data column:
- **G** means the gated `measurement-db` must be under `data/`.
- **F** means our own formative feedback (`experiments/platform_submissions/`), which is not in the public release.
- **–** means only public repository files are needed.

| item | command | runtime | data |
|---|---|---|---|
| everything, no data | `python scripts/reproduce_all.py --smoke` | 12 s | – |
| everything, saved results + live checks | `python scripts/reproduce_all.py --quick` | 82 s | G, F (rows SKIP without them) |
| Table 1 | `reproduce_all.py --only T1` | 2 s | G |
| Table 5 | check: `reproduce_all.py --only T2`; rerun: `python -m paiec.analysis.item_repr` | <1 s; rerun not timed | –; rerun G (+ `data/embed`, `data/jev_cache`) |
| Table 2, all rows R2 (seeds 0–9) | `reproduce_all.py --only T3,T3live` | 67 s | – (T3); G (T3live) |
| Table 2, official row on the paired seed 0–9 episodes | `python scripts/run_slot.py -- python scripts/table3_official_baseline.py` | 30 s | G |
| Table 2, R1 column (v1_3 0.214294) | `reproduce_all.py --full --jobs ladder_r1` | not timed (est. ~8 min) | G |
| Fig. 3 (platform runs) | `python docs/publication/neurips/make_figures.py`; checks `--only PLAT` | 3 s (all figures) | F |
| Fig. 4 (programme forest) | figure: `make_figures.py`; reruns: `reproduce_all.py --full --jobs exp0042,exp0046,exp0052,exp0053,exp0057` | 3 s; reruns 1971 + 353 + 727 + 1795 + 463 s | –; reruns G |
| Fig. 2a (value of ρ) | figure: `make_figures.py`; rerun `--full --jobs e6` | 3 s; rerun 452 s | –; rerun G |
| Fig. 2b (excess map) | figure: `make_figures.py` from `experiments/pattern_mining/meta_2026-10-16/m10_excess_map.csv`; source script `m10_excess_map.py` | 3 s | –; source F |
| Simulator check (EXP-0039 / 0039R) | checks `--only SIM`; rerun `python -m paiec.analysis.exp_0039r_simulator_repair` | <1 s; leave-one-out step 69 s, main run not timed | –; rerun G (`data/_main_54074453`) + F |
| Out-of-sample confirmation, EXP-0058 (§5.1, Limitations) | checks: `reproduce_all.py --only EXP0058` (saved `experiments/runs/EXP-0058/exp_0058_oos_2026-10-22.json`); rerun: `reproduce_all.py --full --jobs exp0058`, or `python scripts/run_slot.py -- python -m paiec.analysis.exp_0058_oos_confirmation` | <1 s; rerun ~90 s, 242 MB peak RSS | –; rerun G (public 281-benchmark release `54074453da` under `data/_main_54074453`, plus `external/official_baselines`) |
| Final archive, deterministic rebuild | `python scripts/build_v1_3_archive.py` (byte-identical, SHA-256 `a4b0e841…`) | <1 s | – |
| Full test suite | `python -m pytest -q` | about 7 min (419 s, 66 tests, clean clone) | see below |

Data needs of the test suite:
- `test_submission_equivalence.py` (v1_1/v1_2 vs engine) and the engine tests in `test_v1_3_archive.py` need **G**.
- The `regression` tests in `test_feedback_inversion.py` need **F**.
- `test_simulator.py` needs `external/official_baselines` at the pin in `external/PINS.txt`.
- `test_exp_0049_variants.py` reads episodes for a few tests (**G**).
- `test_correlated_hierarchical.py`, the synthetic tests in `test_feedback_inversion.py`, and the hash and contents
  tests in `test_v1_3_archive.py` need no data.

Tests whose inputs are absent skip rather than fail.

Test counts:
- **With data and the starting kit:** 66 tests, all pass (about 7 min).
- **Public clone** (no data, no feedback, no starting kit): 26 tests are collected, 18 pass and 8 skip in about
  4 s. `tests/conftest.py` does not collect the four kit-dependent modules. Measured on 2026-10-23 in a
  preview of the public release.

**Evaluator-replica equivalence in a public clone: not possible as the code stands.** The replica
(`src/paiec/evaluation/simulator.py`) imports the official client at import time. That includes
`prediction_workers.group_prediction_inputs` and `run_smoke_test._assert_probability` / `_build_input`, which exist
only in the unlicensed starting kit (the public MIT repository `aims-foundations/paiec_baseline` has only
`streaming_ingestion.py`). Its fast path, `run_direct`, also calls `group_prediction_inputs` and `AcquisitionHook`.
So no kit-free fixture test can drive the replica, and saved kit outputs would have nothing kit-free to be compared
with. Making this possible needs a change to `simulator.py`: lazy kit imports inside `run_official_client`, plus
kit-free equivalents of those four helpers for the fast path. A fixture of the official client's outputs on
synthetic episodes, generated with the kit, could then pin the fast path. Until then, the equivalence is verified
only where the kit is present (`tests/test_simulator.py`).

## 4. What reproduces what (check by check)

Runtimes are wall-clock seconds on the dev machine (3 BLAS threads). "Data" means the gated core pool under `data/`.
Check IDs refer to rows of `reproduce_all.py`.

| paper item | command (in `reproduce_all.py` unless noted) | expected | tolerance | runtime | data needed |
|---|---|---|---|---|---|
| Archive checksums (App. A) | `ARCH.sha_*` | v1_3 `a4b0e841…92685b86`, `submissions/dist/SHA256SUMS` | exact | <1 s | none |
| v1_3 = v1_2 code + facets/guard only (sec. 3) | `ARCH.code_identical`, `ARCH.param_diff` | byte-identical code; diff `{facets: true→"named", guard: 1→0}` | exact | <1 s | none |
| Fitted parameters (sec. 3) | `ARCH.G/s/tau/sigma_beta/centre` | G −0.58, s 0.78, τ 1.13, σβ 2.70, centre G/2 | printed digits | <1 s | none |
| v1_3 archive == engine (App. A) | `python -m pytest -q tests/test_v1_3_archive.py` (live); `ARCH.eq_saved` (EXP-0044 file) | max \|Δp\| < 1e-9 (saved 1.7e-15, printed 2e-15) | 1e-9 | ~20 s | data |
| Table 1 (data) | `T1.*`, recomputed from `data/*/response.parquet` | subjects, items, responses, pairs ≥ 80, mean pair accuracy | exact / printed digits | ~5 s | data |
| Table 5 (representations) | `T2.*` from `experiments/runs/phase1_item_repr/item_repr_results.csv`; rerun: `python -m paiec.analysis.item_repr` | per-row ρ means | printed digits | <1 s (rerun ~min, needs `data/embed`, `data/jev_cache`) | saved file |
| Table 2 (ladder, R2, seeds 0–9) | `T3.*` (saved: `review_round1/table3_baseline_seeds0-9.json` for the official row, `e5/real.json`, `e4/real.json`, EXP-0044) | all 42 cells; official row 0.250 0.360 0.243 0.209 0.194 0.189, ALC 0.2451 | printed digits | <1 s | none |
| Table 2 recomputed live | `T3live.*` (all rows on the same seed 0–9 episodes) | equals saved files | 1e-12 | ~67 s | data |
| v1_3 archive rebuild | `REBUILD.*` (`scripts/build_v1_3_archive.py`) | content- and byte-identical | exact | <1 s | none |
| Review round 1 numbers (ladder intervals, pair counts, platform replay of the official baseline, floor/ceiling CIs) | `REVIEW.*`: saved `experiments/runs/review_round1/*.json` plus cross-checks (ladder file vs baseline file vs Table 2 sources at 1e-12), cheap parts rerun via `paiec.analysis.review_round1`; heavy parts `--full --jobs review_ladder,review_floor` | 14 of 18 exact singletons; official − final +0.0322 [+0.0140, +0.0515]; floors 0.1511 / 0.137 / 0.1797; official→v0 −33.70 | printed digits; reruns 1e-9 | 6 s; full ~17 min | rerun F (+ `external/official_baselines`) |
| Review round 2 numbers (final-model per-budget run 967620 and within-draw SEs; 4-run mean/SD/SE/CI and sign test; local-vs-platform table; paired replays) | `REVIEW2.*`: saved `experiments/runs/review_round2/*.json`, cross-checked against Table 2 sources and the round-1 replay at 1e-12; full rerun of `python -m paiec.analysis.review_round2` (output to `.tmp`) | 0.1839 / 0.0240 / 0.0120 / [0.146, 0.222], p = 0.625; replay v1_3→table −0.0100 [−0.0155, −0.0054] | printed digits; rerun 1e-9 | 4 s | rerun F (+ `external/official_baselines`) |
| Synthetic inversion + archive unit tests | `INV.*` (pytest subsets, no data) | all pass | – | ~5 s | none |
| v1_3 R2 0.208824 / R1 0.214294 | `T3.v1_3.R2_exact`, `T3.v1_3.R1_exact`; live R1: `--full --jobs ladder_r1` | 0.208824, 0.214294 | 5e-7 | <1 s / ~8 min | data for live |
| Shrinkage 33.7 of 36.3 (×1e-3), 2.6 after | `T3.shrink/total/after` | 33.7, 36.3, 2.6 | ±0.05; printed digits | <1 s | none |
| Item-group step fold t-interval | `T3.fold_t_lo/hi` (`review_round1/ladder_intervals.json`, step v1_1→v1_2) | [−1.9, +0.3] ×1e-3 | printed digits | <1 s | none |
| Fig. 3 + platform run means | `PLAT.*` from `experiments/platform_submissions/*.json` | v1_3 runs 0.1663/0.1895/0.2156/0.1643, mean 0.1839, SD 0.024; v1_1 mean 0.1923, CI 0.176–0.208 | printed digits | ~4 s | none |
| Best-of-N for an archive like v1_1 | `PLAT.best_of_5/20` (exact normal order statistics) | 0.177, 0.168 | printed digits | <1 s | none |
| Fig. 4 programme forest + FACTS deltas | `PROG.00xx.*` from `experiments/runs/preregistration/exp_00{42,46,52,53,57}_*.json`; reruns: `--full --jobs exp0042,exp0046,exp0052,exp0053,exp0057` | +1.68 [+1.08,+2.30], +6.88 [+4.80,+8.94], +0.74 [−0.28,+1.85], +0.40 [+0.24,+0.56], +0.52 [+0.22,+0.80] | 0.005 | <1 s / 33, 6, 12, 30, 8 min | data for reruns |
| Half-life, temperature, stress | `PROG.halflife_*`, `PROG.T_*`, `PROG.stress_*`; rerun `--jobs exp0052,exp0053,exp0054` | 5.27 vs 4.30; T 0.945–1.05; +15.4; −0.26 | printed digits | <1 s / 12, 30, 5 min | data for reruns |
| Acquisition, prior lines, distillation | `PROG.acq`, `PROG.prior_lines`, `PROG.e7_hyper`, `T2.distil.*` | ≤ 0.27, ≤ 1.0 (×1e-3), hyper worse, ≤ 3% | printed digits | <1 s | none |
| Registry counts | `PROG.registry/prereg/post` (`experiments/registry.csv`) | 60, 22, 38 | exact | <1 s | none |
| Simulator validity (sec. 5) | `SIM.*` (EXP-0039/0039R JSONs) | 92 benchmarks, 0/11, tail 6 in [1,7], p 0.544, single drops pass, six-drop fails, SD 0.030 vs 0.013 | exact / printed | <1 s | none |
| Platform pairs harder | `Q4.median_acc`, `Q4.frac_low`, `Q4.none_high` (recovered from saved feedback) | median 0.27, 14.7%, none ≥ 0.95 | printed digits | ~3 s | F (not released) |
| Feedback inversion: p0, 34 pairs, 5.1e-6 | `Q4.p0_*`, `Q4.pairs`, `Q4.ece_agree` (internal script; needs our formative feedback, not released) | p0 0.448722 / 0.400099; 34 pairs; 5.1e-6 | printed digits | ~3 s | F (not released) |
| Floor 0.151, ceiling 0.022 (earlier draws) | `Q4.floor_saved`, `Q4.ceiling` | as printed | printed digits | – | F (not released) |
| Fig. 2a value of ρ_S | `HEAD.c/ratio/rho03/rho05/raw/oracle_item` (`e6/e6.json`); rerun `--jobs e6` | −71ρ_S², 58–74, 5.7e-3, 0.018, 0.1387 | printed digits | <1 s / ~8 min | data for rerun |
| Budget-0 bound, item share | `HEAD.b0_bound`, `HEAD.item_share` (`e1/e1_results.json`); rerun `--jobs e1` | 0.1016; about half | printed / [0.4, 0.6] | <1 s | data for rerun |
| Fig. 2b excess map, replay | `HEAD.excess_*`, `HEAD.slots`, `HEAD.replay_*`, `HEAD.public_bench` | 71%, 56%, 26 slots, 14 of 17 exact, 89 benchmarks | printed / exact | <1 s | none |
| Every figure | `FIG.*`: runs `docs/publication/neurips/make_figures.py` with output redirected to `.tmp` | six PDFs, FACTS asserts pass | – | ~5 s | none |

Records of explored, unselected variants are in the team's research record, which is kept privately.

## 5. Known gaps

As of 2026-10-19, `--quick` passes all 237 checks (clean venv), and `--smoke` passes 151 of 152 with one
`SKIP`: Fig. 3 needs feedback. The paper's earlier unsupported numbers (Laplace 0.013, damping +1.7e-3, the
after-success cell, the 0.016 lower SD bound and the "about 40%" regret cut) were corrected or removed in review
round 1, and their checks now test the corrected text. The following gaps remain:

- Claims carried over from research notes 10–17 are not machine-checked by any script:
  - re-run variance 20–27%;
  - metadata 16%;
  - ρ 0.6–0.9 and ICC 0.22;
  - duplicate inflation 0.014;
  - plug-in 0.49;
  - about 4 ms per call;
  - nine evaluator discrepancies.
  The evaluator replica itself is tested by `tests/test_simulator.py`.
- `HEAD.replay_exact` reads `experiments/pattern_mining/meta_2026-10-16/m2_k.csv`, which is derived from feedback.
  Where that file is not published, the same count (14 exact singleton pairs) is covered by `REVIEW.counts`, which
  reads `experiments/runs/review_round1/pair_counts.json`.
- EXP-0058's output went to `experiments/runs/EXP-0058/exp_0058_oos_<date>.json`, not to the output path named in
  its pre-registration. The run records the pre-registration's SHA-256 (`EXP0058.prereg_sha` checks it is
  unchanged). The module dates its output file, so a rerun writes a new file name, and `--full --jobs exp0058`
  compares that file with the saved one.
- On Windows, git's `core.autocrlf=true` checks text files out as CRLF. The zips are unaffected (they are binary
  blobs).
- The episode cache is rebuilt from `data/` on first use. `experiments/runs/jev/*.parquet` is gitignored, so
  `key_jev` is NaN in a clean clone. No model in the paper's tables reads it except the Jev rungs of the ladder.
