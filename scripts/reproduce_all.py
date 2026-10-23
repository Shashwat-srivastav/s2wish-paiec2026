"""Regenerate every number in the S2WISH technical report and check it against the paper.

    python scripts/reproduce_all.py --smoke          # <2 min, no gated data: archives, rebuild, synthetic tests, figures
    python scripts/reproduce_all.py --quick          # ~2-4 min: saved result files + cheap live recomputation
    python scripts/reproduce_all.py --full           # + reruns the heavy experiments (hours; one at a time via
                                                      #   scripts/run_slot.py), compared with the saved results
    python scripts/reproduce_all.py --quick --only T3,Q4   # restrict to some sections
    python scripts/reproduce_all.py --list           # print the check catalogue and exit

What a check is
    Each row regenerates one number (from code, raw data or a raw result file) and compares it with the value printed
    in docs/publication/neurips/main.tex (or FACTS_2026-10-16.md where the paper rounds further). The default
    tolerance is "rounds to the printed value": half a unit in the printed last digit. Live recomputations are compared
    with the saved result file at 1e-12 (the pipeline is deterministic). Tolerances are never widened to force a pass.

Statuses
    PASS   reproduced within tolerance
    FAIL   regenerated value disagrees with the paper (or a required file is missing/broken)
    NOSRC  the paper number has no saved machine-readable source; the row says what does exist
    SKIP   an input is absent (e.g. the gated data/), so the number could not be regenerated
Exit code is 1 if any FAIL or NOSRC row exists (SKIP rows also count with --strict).

Public release: the gated dataset (data/), item-level files derived from it (experiments/runs/**/*.parquet, *.npz),
the formative platform feedback (experiments/platform_submissions/, the Q4 feedback json), artifacts/ and external/
are not distributed. Checks that need them are reported as SKIP "requires gated data / organiser-held feedback"
with the missing path. `--simulate-public` pretends those paths are absent, to preview what a grader sees.

Nothing in the repository is modified: every output (figures, recomputed JSON, extracted archives) goes to
.tmp/reproduce_all/.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import glob
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

TMP = ROOT / ".tmp" / "reproduce_all"
TEX = ROOT / "docs" / "publication" / "neurips" / "main.tex"
FACTS = ROOT / "docs" / "publication" / "FACTS_2026-10-16.md"
PREREG = ROOT / "experiments" / "runs" / "preregistration"
RUNS = ROOT / "experiments" / "runs"
PLAT = ROOT / "experiments" / "platform_submissions"
DIST = ROOT / "submissions" / "dist"
DATA = ROOT / "data"
META = ROOT / "experiments" / "pattern_mining" / "meta_2026-10-16"

SHA_V13 = "a4b0e841d4a344ff207bed2c855342212e95f7d550195b667446ea4e92685b86"
W = {"0": .1, "1": .2, "3": .2, "7": .2, "15": .2, "31": .1}
BUDGETS = ["0", "1", "3", "7", "15", "31"]

# --------------------------------------------------------------------------------------------- result table
ROWS: list[dict] = []
WARN: list[str] = []
_TEX = None


def tex() -> str:
    global _TEX
    if _TEX is None:
        _TEX = TEX.read_text(encoding="utf-8") if TEX.exists() else ""
    return _TEX


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)


# Checks of an explored, unselected variant that the paper does not report. They run only where the internal file
# scripts/internal_variant_checks.py exists (it is not released), and are labelled "record only, not in paper".
RECORD_ONLY = re.compile(r"^(ARCH\.sha_q4|PLAT\.q4|PLAT\.welch|Q4\.(predicted|cf_|floor_run|run_alc|script_oos|"
                         r"folder_eq_zip|table_entries|rebuild_sha)|REVIEW\.variant_|"
                         r"REVIEW2\.(plat|replay8?)\.proto|REVIEW2\.paired\.(v1_3->proto|proto8))")
# Set by scripts/internal_variant_checks.py when present; neutral defaults otherwise.
VARIANT_DIR_REL, VARIANT_RUN_KEY = "explored_variant_feedback", None


def record(sec, cid, what, status, got="", expected="", tol="", note="", lit=None):
    if RECORD_ONLY.match(cid):
        what, lit = what + " [record only, not in paper]", None
    ROWS.append(dict(sec=sec, id=cid, what=what, status=status, got=_fmt(got), expected=_fmt(expected),
                     tol=str(tol), note=note))
    if lit and tex() and lit not in tex():
        WARN.append(f"{cid}: literal {lit!r} not found in main.tex (paper text changed? update this check)")


def printed_tol(s: str) -> float:
    """Half a unit in the last printed digit of a number written as text."""
    s = s.strip().lstrip("+-").replace("−", "")
    dec = len(s.split(".")[1]) if "." in s else 0
    return 0.5 * 10 ** (-dec) + 1e-12


def num(sec, cid, what, got, printed: str, scale=1.0, lit=None, note="", tol=None):
    """got*scale must round to `printed` (a string exactly as the paper prints it)."""
    exp = float(printed.replace("−", "-"))
    t = printed_tol(printed) if tol is None else tol
    g = got * scale
    ok = abs(g - exp) <= t
    record(sec, cid, what, "PASS" if ok else "FAIL", g, exp, f"±{t:.2g}", note, lit if lit is not None else printed)
    return ok


def exact(sec, cid, what, got, expected, note="", lit=None):
    record(sec, cid, what, "PASS" if got == expected else "FAIL", got, expected, "exact", note, lit)


def close(sec, cid, what, got, expected, tol, note=""):
    ok = abs(got - expected) <= tol
    record(sec, cid, what, "PASS" if ok else "FAIL", got, expected, f"±{tol:.0e}", note)


def nosrc(sec, cid, what, paper, note):
    record(sec, cid, what, "NOSRC", "", paper, "", note)


def J(p):
    d = json.loads(Path(p).read_text(encoding="utf-8"))
    if SIMULATE_PUBLIC:  # what the released, redacted copy contains (scripts/build_release.py TRANSFORMS)
        rel = Path(p).resolve().relative_to(ROOT).as_posix() if Path(p).resolve().is_relative_to(ROOT) else ""
        if rel == "experiments/runs/review_round2/local_vs_platform.json":
            d["table"].pop("proto", None)
            d["paired_replay"] = {k: v for k, v in d["paired_replay"].items() if "proto" not in k}
        elif rel == "experiments/runs/review_round1/floor_ceiling.json":
            d.pop(VARIANT_RUN_KEY, None)
    return d


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# Must mirror scripts/build_release.py: what the PUBLIC repository leaves out (2026-10-22: the earlier list was
# incomplete, so --simulate-public passed while a real clone of the release failed).
PUBLIC_EXCLUDED = ["data", "artifacts", "external", "explored_variant_feedback", "docs/internal",
                   "scripts/internal_variant_checks.py", "scripts/build_release.py", "src/paiec/_variant.py",
                   "experiments/runs/review_round1/platform_baseline_replay.json",
                   "docs/publication/record", "docs/publication/dossiers",
                   "experiments/platform_submissions", "experiments/leaderboard_snapshots",
                   "experiments/pattern_mining/meta_2026-10-16"]
PUBLIC_ALLOWED = ["experiments/pattern_mining/meta_2026-10-16/m10_excess_map.csv",  # aggregates behind Fig. 2b
                  "experiments/pattern_mining/meta_2026-10-16/m3_grid.csv",         # and the replay figure
                  "experiments/pattern_mining/meta_2026-10-16/m4_public_summary.json"]  # public-data replay summary
SIMULATE_PUBLIC = False
GATED = "requires gated data / organiser-held feedback"
KIT = "requires the organisers' starting kit in external/starting_kit (https://www.codabench.org/competitions/17828/)"
KIT_MODULES = {"streaming_ingestion", "prediction_workers", "run_smoke_test"}


def available(rel: str) -> bool:
    if SIMULATE_PUBLIC and rel not in PUBLIC_ALLOWED and any(rel == x or rel.startswith(x + "/")
                                                              for x in PUBLIC_EXCLUDED):
        return False
    return (ROOT / rel).exists()


def _excluded_path(path: str) -> str | None:
    """The release-excluded location a missing path belongs to, if any."""
    try:
        rel = Path(path).resolve().relative_to(ROOT).as_posix()
    except (ValueError, OSError):
        return None
    if rel in PUBLIC_ALLOWED:
        return None
    return next((x for x in PUBLIC_EXCLUDED if rel == x or rel.startswith(x + "/")), None)


def requires(sec, cid, what, *rels) -> bool:
    """Record one SKIP row and return False if any needed (non-public) input is absent."""
    missing = [r for r in rels if not available(r)]
    if missing:
        record(sec, cid, what, "SKIP", note=f"{GATED}: missing " + ", ".join(missing))
        return False
    return True


def have_data() -> bool:
    if SIMULATE_PUBLIC:
        return False
    return all((DATA / b / "response.parquet").exists() for b in
               ("matharena", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench"))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def section(fn):
    """Run one section; an exception becomes a FAIL row instead of aborting the whole script."""
    def wrapped(*a, **k):
        t0 = time.time()
        try:
            fn(*a, **k)
        except ModuleNotFoundError as e:
            if e.name in KIT_MODULES:  # official client not redistributed: an expected absence, not a failure
                record(fn.__name__, f"{fn.__name__}:kit", "section needs the official evaluator client", "SKIP",
                       note=KIT)
            else:
                record(fn.__name__, f"{fn.__name__}:error", "section raised", "FAIL", type(e).__name__, "", "",
                       str(e)[:160])
        except FileNotFoundError as e:
            where = _excluded_path(e.filename or "")
            if where:  # an input the public release deliberately leaves out
                record(fn.__name__, f"{fn.__name__}:excluded", "section needs an input not in the public release",
                       "SKIP", note=f"{GATED}: missing {where}")
            else:
                record(fn.__name__, f"{fn.__name__}:error", "section raised", "FAIL", type(e).__name__, "", "",
                       str(e)[:160])
        except Exception as e:  # noqa: BLE001 - we want to report, not crash
            import traceback
            record(fn.__name__, f"{fn.__name__}:error", "section raised", "FAIL", type(e).__name__, "", "",
                   str(e)[:160])
            traceback.print_exc()
        TIMES[fn.__name__] = time.time() - t0
    wrapped.__name__ = fn.__name__
    return wrapped


TIMES: dict[str, float] = {}


# ============================================================================================ ARCH: archives
@section
def ARCH():
    s = "ARCH"
    v13, v12 = DIST / "v1_3_named_facets_noguard.zip", DIST / "v1_2.zip"
    exact(s, "ARCH.sha_v1_3", "v1_3 archive SHA-256 (App. A)", sha(v13), SHA_V13, lit=SHA_V13)
    if INTERNAL:
        ARCH_variant(s)
    bad = []
    for line in (DIST / "SHA256SUMS").read_text().split("\n"):
        if line.strip():
            h, n = line.split()
            if sha(DIST / n) != h:
                bad.append(n)
    exact(s, "ARCH.sha256sums", "submissions/dist/SHA256SUMS matches every listed zip", bad, [])
    a, b = zipfile.ZipFile(v13), zipfile.ZipFile(v12)
    exact(s, "ARCH.files", "v1_3 archive has three files", sorted(a.namelist()),
          ["labeling.py", "model.py", "v1_2_params.json"])
    size_kb = v13.stat().st_size / 1000
    ok = 4.5 <= size_kb < 5.5
    record(s, "ARCH.size", "archive 'about 5 KB zipped'", "PASS" if ok else "FAIL", size_kb, 5.0, "[4.5,5.5)", "kB")
    exact(s, "ARCH.code_identical", "v1_3 model.py+labeling.py byte-identical to v1_2",
          [a.read(n) == b.read(n) for n in ("model.py", "labeling.py")], [True, True])
    pa, pb = json.loads(a.read("v1_2_params.json")), json.loads(b.read("v1_2_params.json"))
    diff = {k: [pb.get(k), pa[k]] for k in sorted(set(pa) | set(pb)) if pa.get(k) != pb.get(k)}
    exact(s, "ARCH.param_diff", "only facets true->'named' and guard 1->0 differ", diff,
          {"facets": [True, "named"], "guard": [1, 0]})
    num(s, "ARCH.G", "fitted G", pa["G"], "-0.58", lit="G=-0.58")
    num(s, "ARCH.s", "offset prior scale s", pa["prior_scale"], "0.78", lit="s=0.78")
    num(s, "ARCH.tau", "tau", pa["tau"], "1.13", lit="\\tau=1.13")
    num(s, "ARCH.sigma_beta", "sigma_beta", pa["sigma_beta"], "2.70", lit="2.70")
    close(s, "ARCH.centre", "prior centre equals G/2", pa["prior_center"], pa["G"] / 2, 1e-12)
    num(s, "ARCH.centre_print", "printed centre G/2", pa["prior_center"], "-0.29", lit="G/2=-0.29")
    exact(s, "ARCH.clip", "clip [0.005, 0.995]", pa["clip"], [0.005, 0.995], lit="[0.005, 0.995]")
    exact(s, "ARCH.rho0", "false-merge prior rho0 (v1_1 step-2 decision)", pa["rho0"], 0.05)
    try:  # the ladder module imports the official evaluator client (starting kit)
        from paiec.models import ladder as Lad
        exact(s, "ARCH.grid", "121-point mu grid", len(Lad.MU_GRID), 121, lit="121-point grid")
    except ModuleNotFoundError as e:
        if e.name not in KIT_MODULES:
            raise
        record(s, "ARCH.grid", "121-point mu grid", "SKIP", note=KIT)
    eq = J(PREREG / "exp_0044_tempering_replay_2026-10-14.json")["reference_equivalence"]["package_vs_engine"]
    worst = max(v["max_abs_delta_effective_all"] for v in eq.values())
    ok = 1.5e-15 <= worst < 2.5e-15
    record(s, "ARCH.eq_saved", "EXP-0044 saved archive-vs-engine max |dp| prints as 2e-15", "PASS" if ok else "FAIL",
           worst, 2e-15, "[1.5e-15,2.5e-15)", "live check: tests/test_v1_3_archive.py")


# ============================================================================================ T1: data table
@section
def T1():
    s = "T1"
    if not have_data():
        record(s, "T1", "Table 1 (recomputed from the dataset)", "SKIP", note=f"{GATED}: missing data/ (measurement-db bc8204d811)")
        return
    import pandas as pd
    rev = (DATA / "REVISION").read_text().strip()
    exact(s, "T1.revision", "data/REVISION starts with bc8204d811", rev[:10], "bc8204d811", lit="bc8204d811")
    paper = {  # subjects, items, responses, pairs>=80, mean pair accuracy (Table 1)
        "matharena": (104, 1755, 92944, 81, "0.61"), "multi_swebench": (82, 2126, 57975, 82, "0.22"),
        "real_webagents": (33, 233, 4449, 26, "0.34"), "researchcodebench": (31, 212, 6784, 31, "0.35"),
        "swe_rebench": (1, 6306, 67074, 1, "0.48")}
    for b, (ns, ni, nr, npair, acc) in paper.items():
        r = pd.read_parquet(DATA / b / "response.parquet", columns=["subject_id", "item_id", "response"])
        rb = r[r.response.isin([0.0, 1.0])]
        n = rb.groupby("subject_id").item_id.nunique()
        el = n[n >= 80].index
        exact(s, f"T1.{b}.subjects", f"{b} subjects", int(r.subject_id.nunique()), ns)
        exact(s, f"T1.{b}.items", f"{b} items", int(r.item_id.nunique()), ni)
        exact(s, f"T1.{b}.responses", f"{b} responses (all rows)", int(len(r)), nr)
        exact(s, f"T1.{b}.pairs80", f"{b} pairs with >=80 binary items", int(len(el)), npair)
        num(s, f"T1.{b}.acc", f"{b} mean accuracy over pairs >=80", float(rb.groupby("subject_id").response.mean()[el].mean()),
            acc, lit=None, note="definition: binary rows, pairs with >=80 distinct items")


# ============================================================================================ T2: representations
@section
def T2():
    s = "T2"
    rows = list(csv.DictReader(open(RUNS / "phase1_item_repr" / "item_repr_results.csv", encoding="utf-8")))

    def m(mode, rep):
        v = [float(r["spearman"]) for r in rows if r["mode"] == mode and r["rep"] == rep]
        return float(np.mean(v)), len(v)

    for lab, rep, within, unseen in [("length and format", "triv", "0.22", "-0.01"), ("TF-IDF", "tfidf", "0.36", "-0.08"),
                                     ("LLM rating (direct difficulty)", "jevd", "0.22", "0.23"), ("all combined", "triv+jev+tfidf", "0.41", "0.18")]:
        g, n = m("within", rep)
        num(s, f"T2.{rep}.within", f"{lab}: rho within a benchmark (mean of {n} folds)", g, within, lit=None)
        g, n = m("lobo", rep)
        num(s, f"T2.{rep}.unseen", f"{lab}: rho on unseen benchmark (mean of {n} folds)", g, unseen, lit=None)
    g, n = m("lobo_emb3", "emb")
    num(s, "T2.emb.unseen", f"embed vectors: rho unseen ({n}-benchmark subset)", g, "-0.08",
        note="saved CSV; live rerun: python -m paiec.analysis.item_repr (needs data/embed, data/jev_cache)")
    jd = [float(r["spearman"]) for r in rows if r["mode"] == "lobo" and r["rep"] == "jevd"]
    num(s, "T2.llm_range_lo", "LLM rating range on unseen folds, low end (sec. 6: 0.19-0.34)", min(jd), "0.19",
        lit="0.19--0.34")
    num(s, "T2.llm_range_hi", "LLM rating range, high end", max(jd), "0.34", lit="0.19--0.34")
    for e, key in (("all-MiniLM-L6-v2", "MiniLM"), ("all-mpnet-base-v2", "MPNet")):
        r = J(RUNS / "e2" / f"e2c_{e}.json")["retention_aggregate"]
        ok = r <= 0.035
        record(s, f"T2.distil.{key}", f"{key} distillation retains at most 3% of the LLM rank correlation",
               "PASS" if ok else "FAIL", r, "<=0.03", "rounds to <=3%", "experiments/runs/e2", "3\\%")


# ============================================================================================ T3: ladder
LADDER = [  # paper Table 2: B0 B1 B3 B7 B15 B31 ALC
    # official row on the same seed 0-9 episodes as the other rows (review round 1, #5); the earlier phase-2 row used
    # seeds 0-4 (0.250 0.356 0.240 0.207 0.194 0.189, ALC 0.2433) and is checked separately as T3.official_seeds0-4
    ("official", ["0.250", "0.360", "0.243", "0.209", "0.194", "0.189", "0.2451"]),
    ("v0", ["0.250", "0.231", "0.214", "0.201", "0.192", "0.188", "0.2114"]),
    ("v1", ["0.247", "0.229", "0.213", "0.200", "0.191", "0.187", "0.2098"]),
    ("v1_1", ["0.246", "0.229", "0.213", "0.200", "0.191", "0.187", "0.2097"]),
    ("v1_2", ["0.246", "0.229", "0.212", "0.200", "0.190", "0.184", "0.2089"]),
    ("v1_3", ["0.246", "0.229", "0.212", "0.199", "0.189", "0.183", "0.2088"]),
]


def phase2_l0(regime="R2"):
    reg = None
    for line in (RUNS / "phase2" / "phase2_table.txt").read_text(encoding="utf-8").splitlines():
        mm = re.match(r"### (R\d)", line)
        if mm:
            reg = mm.group(1)
        elif reg == regime and line.startswith("| L0 "):
            c = [x.strip() for x in line.strip("|").split("|")]
            return [float(x) for x in c[1:7]], float(c[7].split("±")[0])
    raise RuntimeError("L0 row not found")


BASELINE10 = RUNS / "review_round1" / "table3_baseline_seeds0-9.json"


def saved_ladder():
    e5, e4 = J(RUNS / "e5" / "real.json")["R2"], J(RUNS / "e4" / "real.json")["R2"]
    v13 = J(PREREG / "exp_0044_tempering_replay_2026-10-14.json")["results"]["R2"]["baseline"]
    rr = J(BASELINE10)["rows"]["official"]
    return {"official": (rr["curve"], rr["alc"]), "v0": (e5["v0"]["curve"], e5["v0"]["alc"]), "v1": (e5["v1"]["curve"], e5["v1"]["alc"]),
            "v1_1": (e5["v1_1"]["curve"], e5["v1_1"]["alc"]), "v1_2": (e4["both"]["curve"], e4["both"]["alc"]),
            "v1_3": (v13["curve"], v13["alc"])}


@section
def T3():
    s = "T3"
    sv = saved_ladder()
    for name, printed in LADDER:
        curve, alc = sv[name]
        for j, b in enumerate(BUDGETS):
            num(s, f"T3.{name}.B{b}", f"Table 2 {name} B{b} (saved)", curve[j], printed[j], lit=None)
        num(s, f"T3.{name}.ALC", f"Table 2 {name} ALC (saved)", alc, printed[6])
    v13 = J(PREREG / "exp_0044_tempering_replay_2026-10-14.json")["results"]
    num(s, "T3.v1_3.R2_exact", "v1_3 R2 ALC to 6 dp (REPRODUCE.md; paper prints 0.2088)", v13["R2"]["baseline"]["alc"], "0.208824",
        lit="")
    num(s, "T3.v1_3.R1_exact", "v1_3 R1 ALC to 6 dp (REPRODUCE.md; paper prints 0.2143)", v13["R1"]["baseline"]["alc"], "0.214294",
        lit="")
    num(s, "T3.v1_3.R1", "v1_3 R1 ALC (caption: 0.2143)", v13["R1"]["baseline"]["alc"], "0.2143")
    off, v0, v3 = sv["official"][1], sv["v0"][1], sv["v1_3"][1]
    num(s, "T3.shrink", "shrinkage gain official->v0, seeds 0-9 (x1e-3)", off - v0, "33.7", 1e3, tol=0.05,
        lit="33.7 of the total")
    num(s, "T3.total", "total gain official->v1_3, seeds 0-9 (x1e-3)", off - v3, "36.3", 1e3, tol=0.05,
        lit="36.3\\ee{3}")
    l0c, l0a = phase2_l0()
    num(s, "T3.official_seeds0-4", "official row, phase-2 seeds 0-4 ALC (superseded in Table 2)", l0a, "0.2433", lit="")
    b10 = J(BASELINE10)
    pd_ = b10["paired_deltas"]["official_minus_v1_3_x1e3"]
    num(s, "T3.official_paired", "official minus v1_3, paired on identical seed 0-9 episodes (x1e-3)", pd_["pt"], "36.3",
        lit="", note=f"pair-bootstrap 95% [{pd_['lo']:.1f}, {pd_['hi']:.1f}]")
    exact(s, "T3.baseline_file_checks", "baseline file: v0 and v1_3 on its episodes equal the saved rows",
          [b10["checks"]["v0_matches_e5_real_json"], b10["checks"]["v1_3_matches_exp0044_json"]], [0.0, 0.0])
    c5 = b10["checks"]["official_seeds0-4_curve_alc"]
    close(s, "T3.baseline_seeds0-4", "baseline script with seeds 0-4 reproduces the stored phase-2 row (4 dp)",
          max(abs(x - y) for x, y in zip(c5["curve"] + [c5["alc"]], l0c + [l0a])), 0.0, 5e-5)
    num(s, "T3.after", "gain after shrinkage v0->v1_3 (x1e-3, '2.6')", v0 - v3, "2.6", 1e3, lit="2.6\\ee{3}")
    # paper: v1_1 -> v1_2 (item-group) t-interval over the four fold means, printed [-1.9, +0.3] (Table 2, Limitations)
    tt = J(RR / "ladder_intervals.json")["R2"]["steps"]["v1_1->v1_2"]["t3"]
    num(s, "T3.fold_t_lo", "item-group step t-interval over 4 folds, low (x1e-3)", tt[0], "-1.9", lit="[-1.9, +0.2]")
    num(s, "T3.fold_t_hi", "item-group step t-interval, high (x1e-3)", tt[1], "0.2", lit="[-1.9, +0.2]")  # corrected 2026-10-22: +0.249 rounds to +0.2


@section
def T3_live():
    """Recompute the Table 2 R2 rows from code + data and compare with the saved result files (1e-12)."""
    s = "T3live"
    if not have_data():
        record(s, "T3live", "Table 2 recomputed live", "SKIP", note=f"{GATED}: missing data/ (measurement-db bc8204d811)")
        return
    from paiec.evaluation import episodes as E
    from paiec.models import ladder as Lad, v1_1 as V11, v1_2 as V12
    sv = saved_ladder()
    seeds10 = {"R2": list(range(10))}
    with contextlib.redirect_stdout(io.StringIO()):
        r11 = V11.evaluate({"v0": "v0", "v1": V11.V1_CFG, "v1_1": V11.BASE2}, regimes=("R2",), seeds=seeds10)["R2"]
        cf = {"v1_2": V12.cfg_of(facets="all", guard=1), "v1_3": V12.cfg_of(facets="all", guard=0)}
        r12, _ = V12.evaluate(cf, ("R2",), seeds10, tag="reproduce_all")
        s12 = V12.summarise(r12["R2"], list(cf), "v1_3")
    live = {k: (V11.mean_curve(r11, k).tolist(), V11.mean_alc(r11, k)) for k in ("v0", "v1", "v1_1")}
    live.update({k: (s12[k]["curve"], s12[k]["alc"]) for k in cf})
    res = {}
    for B in E.EVAL_FOLDS:
        ep = E.build(E.compact(B), "R2", list(range(10)))
        H = V11.H_fold(B)
        probs = {b: Lad.predict_budget(ep, H, "L0", b)[0] for b in E.BUDGETS}
        br, ec = ep.score(probs)
        res[B] = {"L0": dict(br=br, alc=br @ E.W, pair=ep.inst_pair.copy())}
    live["official"] = (V11.mean_curve(res, "L0").tolist(), V11.mean_alc(res, "L0"))
    for name in ("official", "v0", "v1", "v1_1", "v1_2", "v1_3"):
        lc, la = live[name]
        scurve, salc = sv[name]
        tol = 1e-12
        worst = max(abs(a - b) for a, b in zip(lc + [la], list(scurve) + [salc]))
        record(s, f"T3live.{name}", f"{name} R2 curve+ALC recomputed vs saved", "PASS" if worst <= tol else "FAIL",
               worst, 0.0, f"<={tol:.0e}", "max |live - saved| over 6 budgets + ALC")
        num(s, f"T3live.{name}.ALC", f"{name} live ALC vs paper", la, LADDER[[n for n, _ in LADDER].index(name)][1][6])


# ============================================================================================ PLAT: platform runs
@section
def PLAT_():
    s = "PLAT"
    if not requires(s, "PLAT", "platform run means, SDs, Welch test (Fig. 3)", "experiments/platform_submissions"):
        return
    mf = load_module(ROOT / "docs" / "publication" / "neurips" / "make_figures.py", "rep_mf_runs")
    runs = mf.platform_runs()
    by: dict[str, list] = {}
    for rid, (arch, alc) in runs.items():
        by.setdefault(arch, []).append((rid, alc))
    v13 = dict(by["v1_3_named_facets_noguard.zip"])
    exact(s, "PLAT.v1_3.ids", "v1_3 run ids", sorted(v13), [967620, 967635, 968785, 971666])
    for rid, printed in ((967620, "0.1663"), (967635, "0.1895"), (968785, "0.2156"), (971666, "0.1643")):
        num(s, f"PLAT.v1_3.{rid}", f"v1_3 run {rid} (FACTS; the paper prints only 967620)", v13[rid], printed,
            lit=printed if rid == 967620 else "")
    x = np.array(list(v13.values()))
    num(s, "PLAT.v1_3.mean", "v1_3 mean of 4 runs", x.mean(), "0.1839")
    num(s, "PLAT.v1_3.sd", "v1_3 run SD", x.std(ddof=1), "0.024", lit="SD 0.024")
    num(s, "PLAT.v1_3.min", "v1_3 range low (fig. 1 caption)", x.min(), "0.164", lit="0.164 to 0.216")
    num(s, "PLAT.v1_3.max", "v1_3 range high", x.max(), "0.216", lit="0.164 to 0.216")
    v11 = np.array([a for _, a in by["v1_1.zip"]])
    exact(s, "PLAT.v1_1.n", "v1_1 number of runs", len(v11), 5, lit="5, 0.1923")
    num(s, "PLAT.v1_1.mean", "v1_1 mean", v11.mean(), "0.1923")
    from scipy import stats
    se = v11.std(ddof=1) / math.sqrt(len(v11))
    t = stats.t.ppf(0.975, len(v11) - 1)
    num(s, "PLAT.v1_1.ci_lo", "v1_1 95% t-interval low", v11.mean() - t * se, "0.176", lit="[0.176, 0.208]")
    num(s, "PLAT.v1_1.ci_hi", "v1_1 95% t-interval high", v11.mean() + t * se, "0.208", lit="[0.176, 0.208]")
    num(s, "PLAT.v1_1.sd", "v1_1 repeat SD (sec. 5: 0.013 observed)", v11.std(ddof=1), "0.013")
    v1 = [a for _, a in by["v1.zip"]]
    num(s, "PLAT.v1", "v1 first submission scored 0.216", v1[0], "0.216", lit="which scored 0.216")
    pre = (PLAT / "2026-10-01_v1.md").read_text(encoding="utf-8")
    exact(s, "PLAT.v1_pred", "pre-submission record states the 0.212 prediction (2026-10-01_v1.md)",
          "prediction ALC 0.212 (R1)" in pre, True, lit="predicted 0.212",
          note="documentary: the record predates the run")
    if INTERNAL:
        PLAT_variant(s, by, x, stats)
    v18 = [a for r, a in by["v1_8_ext10.zip"]]
    num(s, "PLAT.v1_8", "v1_8 platform run 990888 (FACTS)", v18[0], "0.1951", lit="")
    # 'run-to-run SDs of repeated archives (0.010-0.024)': observed repeat SDs of archives with >=3 runs
    # (corrected 2026-10-19; the paper said 0.016-0.024, which no saved result supported)
    sds = {a: float(np.std([v for _, v in r], ddof=1)) for a, r in by.items() if len(r) >= 3}
    lo, hi = min(sds.values()), max(sds.values())
    ok = abs(lo - 0.010) <= 5e-4 and abs(hi - 0.024) <= 5e-4
    record(s, "PLAT.sd_range", "'run-to-run SDs of repeated archives (0.010-0.024)' (archives with >=3 runs)",
           "PASS" if ok else "FAIL", [lo, hi], [0.010, 0.024], "±5e-4",
           "observed: " + ", ".join(f"{k.replace('.zip', '')} {v:.4f}" for k, v in sorted(sds.items())),
           lit="0.010--0.024")
    # best-of-N for an archive with v1_1's mean and SD 0.013 (exact normal order statistics)
    for n, printed in ((5, "0.177"), (20, "0.168")):
        z = np.linspace(-9, 9, 200001)
        pdf = stats.norm.pdf(z)
        emin = float(np.trapezoid(z * n * pdf * stats.norm.sf(z) ** (n - 1), z))
        num(s, f"PLAT.best_of_{n}", f"expected best of {n} runs (mean v1_1, SD 0.013)", v11.mean() + 0.013 * emin,
            printed, lit="")  # removed from main.tex 2026-10-22; kept as a derivation check
    # Statements about other participants' public entries are deliberately not checked here (team rule).


# ============================================================================================ PROG: programme
@section
def PROG():
    s = "PROG"

    def one(p):
        h = sorted(glob.glob(str(PREREG / p)))
        assert len(h) == 1, (p, h)
        return J(h[0])
    e42 = one("exp_0042_*.json")["results"]["R2"]["lexical"]["d"]
    e46 = one("exp_0046_gpirt_2026*.json")["decisions"]["R2"]["gpirt"]
    e52 = one("exp_0052_*.json")
    e53 = one("exp_0053_*.json")
    e54 = one("exp_0054_*.json")
    e57 = one("exp_0057_*.json")
    rows = [("0042", e42["pt"], e42["lo"], e42["hi"], ("+1.68", "+1.08", "+2.30")),
            ("0046", e46["mean_delta_alc_x1e3"], *e46["pair_bootstrap_95ci_x1e3"], ("+6.88", "+4.80", "+8.94")),
            ("0052", *(e52["regimes"]["R2"]["brier"]["lambda_corrected"]["d"][k] for k in ("pt", "lo", "hi")),
             ("+0.74", "-0.28", "+1.85")),
            ("0053", *(e53["regimes"]["R2"]["brier"]["Tfit"]["d"][k] for k in ("pt", "lo", "hi")), ("+0.40", "+0.24", "+0.56")),
            ("0057", e57["delta_alc_x1e3"], *e57["pair_bootstrap_95ci_x1e3"], ("+0.52", "+0.22", "+0.80"))]
    for k, pt, lo, hi, pr in rows:
        for lab, g, p in zip(("pt", "lo", "hi"), (pt, lo, hi), pr):
            num(s, f"PROG.{k}.{lab}", f"EXP-{k} paired dALC x1e-3 {lab} (Fig. 4 / FACTS)", g, p, lit="")
    exact(s, "PROG.0057.negfolds", "v1_8: 1 of 4 folds negative (FACTS)", e57["negative_folds"], 1)
    ph = e52["regimes"]["R2"]["pooled_heldout"]
    num(s, "PROG.halflife_emp", "empirical half-life (labels)", ph["h_emp"], "5.27")
    num(s, "PROG.halflife_mod", "model half-life (labels)", ph["h_model"], "4.30")
    T = [f["T_nested"] for f in e53["regimes"]["R2"]["folds"].values()]
    num(s, "PROG.T_lo", "fitted temperature, lowest R2 fold", min(T), "0.945", lit="$T=0.945$--$1.05$")
    num(s, "PROG.T_hi", "fitted temperature, highest R2 fold", max(T), "1.05", lit="$T=0.945$--$1.05$")
    r = e54["regimes"]["R2"]["results"]
    num(s, "PROG.stress_centre", "prior centre +1 logit costs (x1e-3)", r["centre_plus1"]["d"]["pt"], "15.4", lit="")  # not printed since 2026-10-23
    num(s, "PROG.stress_halflife", "half-life x2 (x1e-3)", r["halflife_x2"]["d"]["pt"], "-0.26", lit="")  # not printed
    # label acquisition: largest gain of any non-random policy (v1, v1_1; R1, R2)
    st4 = J(RUNS / "phase3_v1_1" / "step4.json")
    gains = [abs(v["d_A1"]["pt"]) for R in st4.values() if isinstance(R, dict)
             for k, v in R.items() if isinstance(v, dict) and "d_A1" in v]
    num(s, "PROG.acq", f"largest |gain| of 4 non-random policies over {len(gains)} cells (x1e-3)", max(gains), "0.27",
        lit="0.27\\ee{3}")
    e5 = J(RUNS / "e5" / "real.json")["R2"]
    x1 = {k: v["d_v1_1"]["pt"] for k, v in e5.items() if k.startswith("X1")}
    e7 = J(RUNS / "e7" / "e7_primary_s10.json")["arms"]
    e1m = J(RUNS / "e1" / "e1_results.json")["R2"]["metadata_dALC"]["pt"]
    best = min(min(x1.values()), e7["subject"]["d"]["pt"], e1m)
    num(s, "PROG.prior_lines", "best gain of mixture prior / metadata / name prior (x1e-3, 'at most 1.0')", -best, "1.0",
        lit="", note=f"X1 min {min(x1.values()):+.3f}, E7 subject {e7['subject']['d']['pt']:+.3f}, "
        f"E1 metadata {e1m:+.3f}")
    ok = e7["hyper"]["d"]["lo"] > 0
    record(s, "PROG.e7_hyper", "refitting the benchmark level on the larger pool made results worse (CI > 0)",
           "PASS" if ok else "FAIL", [e7["hyper"]["d"]["pt"], e7["hyper"]["d"]["lo"]], "> 0", "", "x1e-3")
    # Laplace-versus-grid audit 'difference of up to 0.013': only a unit test measures a comparable quantity
    from scipy import integrate, stats
    from paiec.models import exp_0049_variants as XV, v1_1 as V
    wl = we = 0.0
    for mu in (-2.0, -0.5, 1.0):
        for S_, N_ in ((2, 5), (10, 20), (25, 31)):
            th, v, logL = XV.theta_laplace(np.array([S_], float), np.array([N_], float), np.array([1.15]), np.array([mu]))
            f = lambda z: np.exp(S_ * XV.log_sig(mu + z) + (N_ - S_) * XV.log_sig(-mu - z)) * stats.norm.pdf(z, scale=1.15)  # noqa: E731
            L = integrate.quad(f, -40, 40, limit=400, epsabs=1e-300)[0]
            Ex = integrate.quad(lambda z: f(z) * XV.sig(mu + z), -40, 40, limit=400, epsabs=1e-300)[0] / L
            El = float(XV.sig(mu + th[0, 0] + math.sqrt(v[0, 0]) * V.GH_X) @ V.GH_W)
            wl, we = max(wl, abs(logL[0, 0] - math.log(L))), max(we, abs(El - Ex))
    # paper (corrected 2026-10-19): Laplace vs exact grid 'differ by 0.012-0.018 in its unit test'
    ok = abs(we - 0.012) <= 1e-3 and abs(wl - 0.018) <= 1e-3
    record(s, "PROG.laplace", "Laplace-vs-grid difference '0.012-0.018 in its unit test'", "PASS" if ok else "FAIL",
           [we, wl], [0.012, 0.018], "±1e-3",
           f"Laplace vs dense quadrature: predictive {we:.4f}, log-marginal {wl:.4f}", "0.012--0.018")
    # The damping '+1.7e-3' claim was removed from the paper on 2026-10-19 (only a prose log line supported it).
    reg = list(csv.DictReader(open(ROOT / "experiments" / "registry.csv", encoding="utf-8")))
    exact(s, "PROG.registry", "registered experiments", len(reg), 60, lit="60 logged experiments")
    npre = sum(r["pre_registered"].strip().lower() == "true" for r in reg)
    exact(s, "PROG.prereg", "pre-registered", npre, 22, lit="22 pre-registered")
    exact(s, "PROG.post", "registered after the fact", len(reg) - npre, 38, lit="38 of the 60")


# ============================================================================================ SIM: simulator
@section
def SIM():
    s = "SIM"
    r = J(PREREG / "exp_0039r_simulator_repair_2026-10-15.json")
    o = J(PREREG / "exp_0039_simulator_fidelity_2026-10-15.json")
    loo = J(PREREG / "exp_0039r_loo_sensitivity_2026-10-16.json")
    exact(s, "SIM.n_bench", "R1-drawable benchmarks in the enlarged pool", r["n_included_r1_drawable"], 92, lit="92 benchmarks")
    exact(s, "SIM.runs", "platform runs checked", o["n_official_runs"], 11, lit="of 11 platform runs")
    exact(s, "SIM.outside", "runs outside their 90% interval", r["primary"]["n_runs_outside"], 0)
    t = r["primary"]["tail_count"]
    exact(s, "SIM.tail", "tail count / allowed range", [t["observed"], t["range90"]], [6, [1, 7]], lit="allowed 1--7")
    num(s, "SIM.envelope", "envelope test p", r["primary"]["envelope"]["p_global"], "0.544")
    single = [d for d in loo["drops"] if len(d["dropped"]) == 1]
    exact(s, "SIM.drop1", f"pass survives every single-benchmark drop ({len(single)} drops)",
          all(d["decision_row"].startswith("All three") for d in single), True)
    six = [d for d in loo["drops"] if len(d["dropped"]) == 6]
    exact(s, "SIM.drop6", "dropping the six largest contributors fails the run-level criterion",
          [d["criteria"]["runs_inside_pi90"] for d in six], [False])
    rs = r["primary"]["repeat_sd"]["v1_1"]
    num(s, "SIM.sd_pred", "predicted repeat SD", rs["pred_run_sd"], "0.030", lit="0.030")
    num(s, "SIM.sd_obs", "observed repeat SD (v1_1, 5 runs)", rs["obs_sd"], "0.013")
    exact(s, "SIM.n_pairs", "saved platform pair records", o["n_official_pair_records"], 34)
    num(s, "SIM.wrong_p0", "withdrawn p0 = 1 - 0.448722", 1 - 0.448722, "0.5513")


# ============================================================================================ Q4 + platform-feedback facts
def _extract(zp: Path, name: str) -> Path:
    d = TMP / "archives" / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    zipfile.ZipFile(zp).extractall(d)
    return d


FEEDBACK_ONLY = "requires formative feedback (not released)"


@section
def Q4():  # replaced by the internal version when scripts/internal_variant_checks.py is present
    """Feedback-inversion checks (p0 recovery, 34 saved pairs, ECE agreement, platform accuracy, floor and ceiling).
    They need our formative feedback and internal scripts, which are not released; the full section lives in
    scripts/internal_variant_checks.py and replaces this stub when that file is present."""
    record("Q4", "Q4", "feedback-inversion checks (p0, 34 saved pairs, 5.1e-6, median accuracy, floor, ceiling)", "SKIP",
           note=FEEDBACK_ONLY)


# ============================================================================================ HEAD: headroom
@section
def HEAD():
    s = "HEAD"
    e6 = J(RUNS / "e6" / "e6.json")
    A = e6["arms"]
    base = A["base"]["alc"]
    rhos = [str(r) for r in e6["config"]["rhos"]] + ["1.0"]
    pts = [(float(r), A[f"item({r})"]["d"]["pt"]) for r in rhos]
    x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    c = float(np.sum(x * x * y) / np.sum(x ** 4))
    num(s, "HEAD.c", "least-squares fit dALC = c rho^2 (x1e-3; figure 3a)", c, "-71", lit="$-71\\rs^2$")
    ratio = -y / x ** 2
    num(s, "HEAD.ratio_lo", "-dALC/rho^2, min over rho in [0.2, 1.0]", ratio.min(), "58", lit="58$--$74")
    num(s, "HEAD.ratio_hi", "-dALC/rho^2, max", ratio.max(), "74", lit="58$--$74")
    num(s, "HEAD.rho03", "value of rho=0.3 signal (x1e-3)", base - A["item(0.3)"]["alc"], "5.7", 1e3, lit="5.7\\ee{3}")
    num(s, "HEAD.rho05", "value of rho=0.5 signal (ALC)", base - A["item(0.5)"]["alc"], "0.018")
    raw = {r: A[f"item_raw({r})"]["alc"] - base for r in ("0.2", "0.3", "0.5", "0.7")}
    ok = raw["0.5"] > 0 and raw["0.7"] < 0
    record(s, "HEAD.raw", "unshrunk signal harmful below rho~0.6 (worse at 0.5, better at 0.7)", "PASS" if ok else "FAIL",
           [raw["0.5"], raw["0.7"]], "[>0, <0]", "", "dALC raw arm vs base", "")  # not printed since 2026-10-23
    num(s, "HEAD.oracle_item", "perfect item oracle ALC", A["item(1.0)"]["alc"], "0.1387")
    e1 = J(RUNS / "e1" / "e1_results.json")["R2"]
    num(s, "HEAD.b0_bound", "budget-0 Brier with offset, ability and item oracles", e1["mean"]["oracle_all_lpo|B0"], "0.1016")
    sh = e1["shares"]["B0"]["mean"]
    ok = 0.4 <= sh["item"] <= 0.6 and 0.4 <= sh["SUPP_lpo"]["item"] <= 0.6
    record(s, "HEAD.item_share", "item difficulty is 'about half' of the closable budget-0 gap", "PASS" if ok else "FAIL",
           [sh["item"], sh["SUPP_lpo"]["item"]], 0.5, "[0.4,0.6] stated", "E1 shares (in-sample, leave-pair-out)",
           "about half")
    rows = list(csv.DictReader(open(META / "m10_excess_map.csv", encoding="utf-8")))
    M = np.array([[float(r[f"B{b}"]) for b in (0, 1, 3, 7, 15, 31)] for r in rows])
    num(s, "HEAD.excess_b01", "share of label-removable excess at budgets 0-1 (%)", M[:, :2].sum() / M.sum(), "71", 100,
        lit="71\\%")
    num(s, "HEAD.excess_hard", "share on pairs with accuracy <= 0.15 (%)", sum(float(r["share"]) for r in rows[:2]), "56",
        100, lit="56\\%")
    exact(s, "HEAD.slots", "pair-slots in the excess map", sum(int(r["n_slots"]) for r in rows), 26, lit="")  # count no longer printed
    m3 = {int(r["n_pairs"]) for r in csv.DictReader(open(META / "m3_grid.csv", encoding="utf-8"))}
    exact(s, "HEAD.replay_pairs", "pairs in the m3 replay grid (loose 0.005 threshold; NOT the paper's 18 singleton "
          "pairs of the later runs, which REVIEW.counts checks)", sorted(m3), [17], lit="")
    if available("experiments/pattern_mining/meta_2026-10-16/m2_k.csv"):  # per-pair file: not in the public release
        resid = {}
        for r in csv.DictReader(open(META / "m2_k.csv", encoding="utf-8")):
            res = max(float(r[f"res{b}"]) for b in (1, 3, 7, 15, 31))
            if r["rep"] == "1" and res < 0.005:  # m3_platform_counterfactual.singleton_pairs() selection
                q = (r["subject"], r["bench"])
                resid[q] = min(res, resid.get(q, 1.0))  # exact if any archive's feedback recovers k exactly
        n_exact = sum(v < 1e-5 for v in resid.values())
        exact(s, "HEAD.replay_exact", "singleton pairs replayed exactly (k-recovery residual < 1e-5 in some run) / total",
              [n_exact, len(resid)], [14, 17], note="the other 3 have residuals 7e-4..4e-3 (approximate)")
    else:
        record(s, "HEAD.replay_exact", "singleton pairs replayed exactly", "SKIP",
               note=f"{GATED}: per-pair recovery table m2_k.csv is not published")
    exact(s, "HEAD.public_bench", "public benchmarks in the replay", J(META / "m4_public_summary.json")["n_bench"], 89,
          lit="89 public")
    # The 'after-success cell (5 pairs)' claim was removed from the paper on 2026-10-19 (no saved source).
    reg = J(RUNS / "phase3_v1_1" / "final.json")
    red = {R: 1 - reg[f"regret_{R}"]["v1_1"]["max"] / reg[f"regret_{R}"]["v1"]["max"] for R in ("R1", "R2")}
    ok = abs(red["R1"] - 0.29) <= 0.01 and abs(red["R2"] - 0.50) <= 0.01  # paper (corrected): '29% ... to 50%'
    record(s, "HEAD.regret", "G/2 centre cuts worst-case regret '29% (platform-like) to 50% (one subject)'",
           "PASS" if ok else "FAIL", [red["R1"], red["R2"]], [0.29, 0.50], "±0.01 each regime",
           "phase3_v1_1/final.json max regret v1 vs v1_1 (R1, R2)", "29\\%")


# ============================================================================================ FIG: figures
@section
def FIG():
    s = "FIG"
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        record(s, "FIG", "make_figures.py needs matplotlib", "SKIP", note="pip install matplotlib")
        return
    try:
        from fontTools import subset as _fs
        broken = not hasattr(_fs, "Options")
    except ImportError:
        broken = True
    if broken:
        import fontTools
        record(s, "FIG.env", "figures not regenerated: this interpreter's fontTools.subset is broken", "SKIP",
               getattr(fontTools, "__file__", "?"), "", "", "ENVIRONMENT, not a reproduction failure: matplotlib's "
               "PDF backend needs fontTools.subset (subset/__init__.py is missing here). Use a fresh venv "
               "(pip install -r requirements-lock.txt) and rerun; see REPRODUCE.md")
        print("WARNING: figures skipped -- broken fontTools in this interpreter; use a fresh venv", file=sys.stderr)
        return
    out = TMP / "figures"
    if out.exists():
        shutil.rmtree(out)
    mf = load_module(ROOT / "docs" / "publication" / "neurips" / "make_figures.py", "rep_mf")
    mf.OUT = out
    buf = io.StringIO()
    fns = ["fig_platform_runs", "fig_programme_forest", "fig_rho_value", "fig_excess_map", "fig_platform_replay",
           "fig_budget_curves", "fig_teaser"]
    fns = [f for f in fns if hasattr(mf, f)]
    for fn in fns:
        if fn == "fig_platform_runs" and not requires(s, f"FIG.{fn}", "fig_platform_runs.pdf (Fig. 3)",
                                                      "experiments/platform_submissions"):
            continue
        try:
            with contextlib.redirect_stdout(buf):
                getattr(mf, fn)()
            ok = True
            note = ""
        except Exception as e:  # noqa: BLE001
            ok, note = False, f"{type(e).__name__}: {str(e)[:120]}"
        pdf = out / (fn + ".pdf")
        record(s, f"FIG.{fn}", f"{fn}.pdf regenerated (its FACTS asserts pass)",
               "PASS" if ok and pdf.exists() and pdf.stat().st_size > 1000 else "FAIL", pdf.name if pdf.exists() else "",
               "", "", note)
    used = sorted(set(re.findall(r"\\fig(?:\[[^\]]*\])?\{([^}]+)\}", tex())))
    missing = [u for u in used if not (out / u).exists()
               and not (u == "fig_platform_runs.pdf" and not available("experiments/platform_submissions"))]
    exact(s, "FIG.referenced", f"every figure main.tex includes is regenerated ({len(used)})", missing, [])


# ============================================================================================ FULL tier
FULL_JOBS = {
    # job: (module or None, description, approx minutes)
    "ladder_r1": (None, "Table 2 rows in R1 (v0, v1, v1_1, v1_2, v1_3; 100 seeds) vs e5/e4/EXP-0044 saved", 8),
    "exp0044": ("paiec.analysis.evaluate_exp0044_tempering", "EXP-0044 v1_3 replay (R2+R1, 4 configs)", 3),
    "exp0057": ("paiec.analysis.exp_0057_v1_8_ext10", "EXP-0057 v1_8 vs v1_3", 8),
    "exp0042": ("paiec.analysis.exp_0042_lexical_residual", "EXP-0042 lexical smoothing", 33),
    "exp0046": ("paiec.analysis.exp_0046_gpirt", "EXP-0046 gp-IRT", 6),
    "exp0052": ("paiec.analysis.exp_0052_shrinkage", "EXP-0052 shrinkage", 12),
    "exp0053": ("paiec.analysis.exp_0053_temperature", "EXP-0053 temperature", 30),
    "exp0054": ("paiec.analysis.exp_0054_stress", "EXP-0054 stress", 5),
    "e6": ("paiec.analysis.e6_headroom", "E6 value of rho", 8),
    "e1": ("paiec.analysis.e1_b0_decomp", "E1 budget-0 oracle decomposition", 15),
    "review_ladder": ("paiec.analysis.review_round1:ladder", "review round 1 ladder intervals (R2 + R1)", 15),
    "review_floor": ("paiec.analysis.review_round1:floor", "review round 1 floor/ceiling incl. local R2", 2),
    "exp0058": ("paiec.analysis.exp_0058_oos_confirmation:run", "EXP-0058 out-of-sample confirmation (89 benchmarks)", 2),
}
# inputs a full job needs; the job SKIPs (with this reason) when any is absent
FULL_REQUIRES = {
    "exp0058": ["data/_main_54074453", "external/official_baselines/empirical_mean/model.py"],
}
# result file (relative to experiments/runs) and the keys compared, per job
FULL_COMPARE = {
    "exp0044": ("preregistration/exp_0044_tempering_replay_2026-10-14.json",
                ["results.R2.baseline", "results.R1.baseline", "decisions"]),
    "exp0057": ("preregistration/exp_0057_v1_8_ext10_2026-10-15.json",
                ["delta_alc_x1e3", "pair_bootstrap_95ci_x1e3", "negative_folds"]),
    "exp0042": ("preregistration/exp_0042_lexical_residual_2026-10-15.json", ["results.R2.lexical.d"]),
    "exp0046": ("preregistration/exp_0046_gpirt_2026-10-15.json",
                ["decisions.R2.gpirt.mean_delta_alc_x1e3", "decisions.R2.gpirt.pair_bootstrap_95ci_x1e3"]),
    "exp0052": ("preregistration/exp_0052_shrinkage_2026-10-15.json",
                ["regimes.R2.brier.lambda_corrected.d", "regimes.R2.pooled_heldout.h_emp",
                 "regimes.R2.pooled_heldout.h_model"]),
    "exp0053": ("preregistration/exp_0053_temperature_2026-10-15.json", ["regimes.R2.brier.Tfit.d", "regimes.R2.folds"]),
    "exp0054": ("preregistration/exp_0054_prior_stress_2026-10-15.json", ["regimes.R2.results"]),
    "e6": ("e6/e6.json", ["arms"]),
    "e1": ("e1/e1_results.json", ["R2.mean", "R2.shares"]),
    "review_ladder": ("review_round1/ladder_intervals.json", ["R2", "R1"]),
    "review_floor": ("review_round1/floor_ceiling.json", ["local_R2", "earlier_draws_v1_1_to_v1_3"]),
    "exp0058": ("EXP-0058/exp_0058_oos_*.json", ["stats", "mean_alc_equal_bench", "r2_pairs_total", "per_benchmark"]),
}


def _get(d, dotted):
    for k in dotted.split("."):
        d = d[k] if not isinstance(d, list) else d[int(k)]
    return d


def _flat(d, p=""):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from _flat(v, f"{p}.{k}" if p else str(k))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from _flat(v, f"{p}[{i}]")
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        yield p, float(d)


def worker(job: str, out: Path):
    """Executed inside run_slot.py. Recomputes into a scratch copy of experiments/runs, never the real one."""
    from paiec import paths
    scratch = TMP / "full" / f"runs_{job}"
    if scratch.exists():
        shutil.rmtree(scratch)
    for f in RUNS.rglob("*"):
        rel = f.relative_to(RUNS)
        if f.is_file() and (f.suffix in (".json", ".txt", ".csv") or (rel.parts[0] == "jev" and f.suffix == ".parquet")) \
                and rel.parts[0] not in (".slots", "e2"):
            (scratch / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, scratch / rel)
    paths.RUNS = scratch
    paths.DECISIONS_LOG = scratch / "decisions.jsonl"
    t0 = time.time()
    if job == "ladder_r1":
        from paiec.models import v1_1 as V11, v1_2 as V12
        sd = {"R1": list(range(100))}
        r11 = V11.evaluate({"v0": "v0", "v1": V11.V1_CFG, "v1_1": V11.BASE2}, regimes=("R1",), seeds=sd)["R1"]
        cf = {"v1_2": V12.cfg_of(facets="all", guard=1), "v1_3": V12.cfg_of(facets="all", guard=0)}
        r12, _ = V12.evaluate(cf, ("R1",), sd, tag="full")
        s12 = V12.summarise(r12["R1"], list(cf), "v1_3")
        res = {k: {"alc": V11.mean_alc(r11, k), "curve": V11.mean_curve(r11, k).tolist()} for k in ("v0", "v1", "v1_1")}
        res.update({k: {"alc": s12[k]["alc"], "curve": s12[k]["curve"]} for k in cf})
        out.write_text(json.dumps({"job": job, "seconds": time.time() - t0, "result": res}), encoding="utf-8")
        return
    mod_name, _, fn_name = FULL_JOBS[job][0].partition(":")
    import importlib
    mod = importlib.import_module(mod_name)
    if hasattr(mod, "OUT") and mod_name.endswith("review_round1"):
        mod.OUT = scratch / "review_round1"
        mod.OUT.mkdir(parents=True, exist_ok=True)
    fn = getattr(mod, fn_name) if fn_name else (getattr(mod, "run", None) or getattr(mod, "main"))
    with contextlib.redirect_stdout(io.StringIO()):
        fn()
    rel = FULL_COMPARE[job][0]
    hit = sorted(scratch.glob(rel), key=lambda q: q.stat().st_mtime)[-1]  # dated names: take the file just written
    out.write_text(json.dumps({"job": job, "seconds": time.time() - t0, "result": J(hit)}), encoding="utf-8")


def full_tier(jobs):
    s = "FULL"
    for job in jobs:
        if not requires(s, f"FULL.{job}", FULL_JOBS[job][1], *FULL_REQUIRES.get(job, [])):
            continue
        out = TMP / "full" / f"{job}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            out.unlink()
        cmd = [sys.executable, str(ROOT / "scripts" / "run_slot.py"), "--", sys.executable, str(Path(__file__).resolve()),
               "--worker", job, "--worker-out", str(out)]
        print(f"[full] {job}: {FULL_JOBS[job][1]} (~{FULL_JOBS[job][2]} min) ...", flush=True)
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
        TIMES[f"full:{job}"] = time.time() - t0
        if rc != 0 or not out.exists():
            record(s, f"FULL.{job}", FULL_JOBS[job][1], "FAIL", f"exit {rc}", "", "", "worker failed")
            continue
        new = J(out)["result"]
        if job == "ladder_r1":
            e5, e4 = J(RUNS / "e5" / "real.json")["R1"], J(RUNS / "e4" / "real.json")["R1"]
            v13 = J(PREREG / "exp_0044_tempering_replay_2026-10-14.json")["results"]["R1"]["baseline"]
            saved = {"v0": e5["v0"], "v1": e5["v1"], "v1_1": e5["v1_1"], "v1_2": e4["both"], "v1_3": v13}
            for k, sv in saved.items():
                worst = max(abs(a - b) for a, b in zip(new[k]["curve"] + [new[k]["alc"]], sv["curve"] + [sv["alc"]]))
                record(s, f"FULL.ladder_r1.{k}", f"{k} R1 (100 seeds) recomputed vs saved",
                       "PASS" if worst <= 1e-12 else "FAIL", worst, 0.0, "<=1e-12")
            num(s, "FULL.v1_3.R1", "v1_3 R1 ALC live vs paper", new["v1_3"]["alc"], "0.214294")
            continue
        rel, keys = FULL_COMPARE[job]
        saved = J(E58 if job == "exp0058" else RUNS / rel)
        for key in keys:
            a, b = dict(_flat(_get(new, key))), dict(_flat(_get(saved, key)))
            if not b and isinstance(_get(saved, key), (int, float)):
                a, b = {"": float(_get(new, key))}, {"": float(_get(saved, key))}
            common = set(a) & set(b)
            worst = max((abs(a[k] - b[k]) for k in common), default=float("nan"))
            ok = bool(common) and set(a) == set(b) and worst <= 1e-9
            record(s, f"FULL.{job}.{key}", f"{FULL_JOBS[job][1]}: {key} rerun vs saved", "PASS" if ok else "FAIL",
                   worst, 0.0, "<=1e-9", f"{len(common)} numbers compared")



# ============================================================================================ REBUILD: final archive
@section
def REBUILD():
    s = "REBUILD"
    bv = load_module(ROOT / "scripts" / "build_v1_3_archive.py", "rep_build_v13")
    r = bv.verify(bv.build())
    exact(s, "REBUILD.content", "v1_3 archive rebuilt from submissions/v1_2 + facets/guard: same member bytes",
          r["content_identical"], True)
    exact(s, "REBUILD.bytes", "rebuilt zip is byte-identical (SHA-256 a4b0e841...)", r["sha256"], SHA_V13,
          note="byte identity needs zlib 1.3.x deflate output; content identity is what scoring depends on")


# ============================================================================================ INV: no-data unit tests
@section
def INV():
    s = "INV"
    tests = [("tests/test_feedback_inversion.py", "not regression and not zz_print",
              "feedback-inversion package on synthetic data"),
             ("tests/test_v1_3_archive.py", "hash or contents", "v1_3 archive hash and contents")]
    for path, sel, what in tests:
        if not (ROOT / path).exists():
            record(s, f"INV.{Path(path).stem}", what, "SKIP", note=f"{path} not in this checkout")
            continue
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        t0 = time.time()
        p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", path, "-k", sel],
                           cwd=str(ROOT), capture_output=True, text=True, env=env, timeout=600)
        last = [ln for ln in p.stdout.splitlines() if ln.strip()][-1:] or [p.stderr[-120:]]
        ok = p.returncode == 0 and "passed" in last[0] and "failed" not in last[0]
        record(s, f"INV.{Path(path).stem}", f"{what} (pytest -k '{sel}')", "PASS" if ok else "FAIL", last[0], "all pass",
               "", f"{time.time() - t0:.0f} s")


# ============================================================================================ REVIEW: review round 1
RR = RUNS / "review_round1"


@section
def REVIEW():
    """Numbers added in review round 1 (experiments/runs/review_round1/, src/paiec/analysis/review_round1.py)."""
    s = "REVIEW"
    if not (RR / "pair_counts.json").exists():
        record(s, "REVIEW", "review-round-1 result files", "SKIP", note="experiments/runs/review_round1/ absent")
        return
    pc = J(RR / "pair_counts.json")
    pl = J(RR / "platform_baseline_replay.json") if available("experiments/runs/review_round1/platform_baseline_replay.json") \
        else None
    fc, li, b10 = J(RR / "floor_ceiling.json"), J(RR / "ladder_intervals.json"), J(BASELINE10)
    # cross-checks between independently computed files (no tolerance games: these are the same quantities)
    lad = li["R2"]["alc"]
    sv = saved_ladder()
    worst = max(abs(lad[a] - sv[b][1]) for a, b in (("baseline", "official"), ("v0", "v0"), ("v1", "v1"),
                                                    ("v1_1", "v1_1"), ("v1_2", "v1_2"), ("final", "v1_3")))
    close(s, "REVIEW.ladder_vs_saved", "ladder_intervals.json R2 ALCs == Table 2 sources (6 rows)", worst, 0.0, 1e-12)
    st = li["R2"]["steps"]
    num(s, "REVIEW.step_base_v0", "ladder step official->v0, R2 (x1e-3)", st["baseline->v0"]["pt"], "-33.70", lit="")
    num(s, "REVIEW.step_v0_final", "ladder step v0->final, R2 (x1e-3)", st["v0->final"]["pt"], "-2.62", lit="")
    num(s, "REVIEW.step_v12_final", "ladder step v1_2->final, R2 (x1e-3)", st["v1_2->final"]["pt"], "-0.09", lit="")
    num(s, "REVIEW.base_hier_lo", "official->v0 hierarchical CI low (x1e-3)", st["baseline->v0"]["hlo"], "-42.78", lit="")
    num(s, "REVIEW.base_hier_hi", "official->v0 hierarchical CI high (x1e-3)", st["baseline->v0"]["hhi"], "-23.41", lit="")
    close(s, "REVIEW.base_paired_eq", "official-v1_3 paired delta: ladder file == baseline file",
          st["baseline->v0"]["pt"] + st["v0->final"]["pt"],
          -b10["paired_deltas"]["official_minus_v1_3_x1e3"]["pt"], 1e-9)
    # pair counts (#11)
    # released copy: "later_saved_runs"; older saved files used a longer key with the same "later_three_plus" prefix
    lp = pc.get("later_saved_runs") or next(v for k, v in pc.items() if k.startswith("later_three_plus"))
    exact(s, "REVIEW.counts", "singleton pairs: later saved runs (exact/distinct), later three, all five",
          [[lp["exact_distinct"], lp["singleton_distinct"]],
           [pc["later_three"]["exact_distinct"], pc["later_three"]["singleton_distinct"]],
           [pc["all_five_runs"]["exact_distinct"], pc["all_five_runs"]["singleton_distinct"]]],
          [[14, 18], [12, 15], [18, 22]], lit="")
    exact(s, "REVIEW.loose", "pairs passing the loose 0.005 threshold (later saved runs)",
          lp["loose_0p005_distinct"], 17)
    # platform replay (#4): feedback-derived, keyed by hidden IDs, not released
    if pl is None:
        record(s, "REVIEW.replay", "platform replay of the official baseline (0.0322 [0.0140, 0.0515], 18/8 pairs)", "SKIP",
               note=FEEDBACK_ONLY)
        a = None
    else:
        a = pl["all_exact_singletons"]
    if a is not None:
        num(s, "REVIEW.replay_base", "replay, 18 exact singletons: official minus final ALC", a["diffs"]["baseline_minus_final"]["mean"],
            "0.0322", lit="")
        num(s, "REVIEW.replay_base_lo", "  pair-bootstrap 95% low", a["diffs"]["baseline_minus_final"]["ci95_pair"][0], "0.0140",
            lit="")
        num(s, "REVIEW.replay_base_hi", "  pair-bootstrap 95% high", a["diffs"]["baseline_minus_final"]["ci95_pair"][1], "0.0515",
            lit="")
        num(s, "REVIEW.replay_v0", "replay, 18 exact singletons: v0 minus final ALC", a["diffs"]["v0_minus_final"]["mean"],
            "0.0023", lit="")
        exact(s, "REVIEW.replay_n", "replay pair sets (all exact / strict)", [a["n_pairs"], pl["strict"]["n_pairs"]], [18, 8])
    # floor and ceiling (#7)
    e = fc["earlier_draws_v1_1_to_v1_3"]
    num(s, "REVIEW.floor_early", "floor, earlier draws (26 slots)", e["floor"], "0.1511", lit="")
    num(s, "REVIEW.floor_early_lo", "  benchmark-cluster 95% low", e["floor_ci95_benchmark_cluster"][0], "0.090", lit="")
    num(s, "REVIEW.floor_early_hi", "  benchmark-cluster 95% high", e["floor_ci95_benchmark_cluster"][1], "0.205", lit="")
    num(s, "REVIEW.ceiling_early", "ceiling, earlier draws", e["ceiling"], "-0.0218", lit="")
    if INTERNAL:
        REVIEW_variant(s, fc)
    if "local_R2" in fc:
        num(s, "REVIEW.floor_local", "local R2 pair-constant floor", fc["local_R2"]["mean"], "0.1797", lit="")
        num(s, "REVIEW.ceiling_local", "local R2 ceiling of the final model", fc["local_R2"]["final_model"]["ceiling"],
            "-0.0291", lit="")
        close(s, "REVIEW.local_final_alc", "local final ALC in floor file == v1_3 R2 0.208824",
              fc["local_R2"]["final_model"]["alc"], sv["v1_3"][1], 1e-12)
    # live rerun of the cheap parts on our own feedback (writes only to .tmp)
    if not requires(s, "REVIEW.rerun", "rerun counts/platform/floor from feedback",
                    "experiments/platform_submissions", VARIANT_DIR_REL,
                    "external/official_baselines/empirical_mean/model.py"):
        return
    import importlib
    rr = importlib.import_module("paiec.analysis.review_round1")
    rr.OUT = TMP / "review_round1_rerun"
    rr.OUT.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        new = {"pair_counts": rr.counts(), "platform_baseline_replay": rr.platform(), "floor_ceiling": rr.floor(local=False)}
    for name, obj in new.items():
        old = J(RR / f"{name}.json")
        if name == "pair_counts":  # key renamed in the code; older saved files carry the longer name
            old = {("later_saved_runs" if k.startswith("later_three_plus") else k): v for k, v in old.items()}
        if name == "floor_ceiling":
            old = {k: v for k, v in old.items() if k != "local_R2"}
            obj = {k: v for k, v in obj.items() if k in old}
        a_, b_ = dict(_flat(json.loads(json.dumps(obj, default=float)))), dict(_flat(old))
        common = set(a_) & set(b_)
        w = max((abs(a_[k] - b_[k]) for k in common), default=float("nan"))
        ok = bool(common) and set(a_) == set(b_) and w <= 1e-9
        record(s, f"REVIEW.rerun.{name}", f"review_round1.{name.split('_')[0]}() rerun == saved {name}.json",
               "PASS" if ok else "FAIL", w, 0.0, "<=1e-9", f"{len(common)} numbers")


# ============================================================================================ REVIEW2: review round 2
RR2 = RUNS / "review_round2"


@section
def REVIEW2():
    """Numbers added in review round 2 (experiments/runs/review_round2/, src/paiec/analysis/review_round2.py)."""
    s = "REVIEW2"
    if not (RR2 / "per_budget_final.json").exists():
        record(s, "REVIEW2", "review-round-2 result files", "SKIP", note="experiments/runs/review_round2/ absent")
        return
    pb, lv = J(RR2 / "per_budget_final.json"), J(RR2 / "local_vs_platform.json")
    f = pb["final_model_v1_3"]
    cols = ["B0", "B1", "B3", "B7", "B15", "B31", "ALC"]
    for c, m_, se in zip(cols, ["0.2369", "0.1903", "0.1613", "0.1445", "0.1463", "0.1412", "0.1663"],
                         ["0.0106", "0.0241", "0.0268", "0.0277", "0.0292", "0.0290", "0.0243"]):
        num(s, f"REVIEW2.967620.{c}", f"run 967620 {c} (per-pair feedback, 9 pairs)", f[c]["mean_run_967620"], m_, lit="")
        num(s, f"REVIEW2.se.{c}", f"within-draw SE over pairs, {c}", f[c]["within_draw_se"], se, lit="")
    sv = saved_ladder()
    worst = max(abs(f[c]["local_R2"] - v) for c, v in zip(cols, list(sv["v1_3"][0]) + [sv["v1_3"][1]]))
    close(s, "REVIEW2.local_eq", "per-budget file's local R2 column == EXP-0044 v1_3 curve", worst, 0.0, 1e-12)
    a = f["ALC"]
    num(s, "REVIEW2.alc_mean", "final model, 4 runs: mean", a["mean_over_runs"], "0.1839", lit="")
    num(s, "REVIEW2.alc_sd", "final model, 4 runs: one-draw SD", a["one_draw_sd"], "0.0240", lit="SD 0.0240")
    num(s, "REVIEW2.alc_se", "final model, 4 runs: SE of the mean", a["se_mean"], "0.0120", lit="SE 0.0120")
    num(s, "REVIEW2.alc_ci_lo", "final model, 4 runs: t 95% CI low", a["ci95_t"][0], "0.146", lit="[0.146, 0.222]")
    num(s, "REVIEW2.alc_ci_hi", "final model, 4 runs: t 95% CI high", a["ci95_t"][1], "0.222", lit="")
    exact(s, "REVIEW2.sign", "runs above the local value / sign-test p", [a["runs_above_local"], a["sign_test_p"]],
          [1, 0.625])
    from scipy import stats
    close(s, "REVIEW2.sign_recomp", "sign test recomputed (two-sided binomial, 1 of 4)",
          float(stats.binomtest(1, 4, 0.5).pvalue), a["sign_test_p"], 1e-12)
    T = lv["table"]
    worst = max(abs(T[k]["local_R2_alc"] - sv[k][1]) for k in ("v0", "v1", "v1_1", "v1_2", "v1_3"))
    close(s, "REVIEW2.local_col", "local-vs-platform table: local R2 column == Table 2 sources", worst, 0.0, 1e-12)
    for k, mean, lo, hi in (("v1_1", "0.1923", "0.176", "0.208"), ("v1_2", "0.1909", "0.170", "0.212"),
                            ("v1_3", "0.1839", "0.146", "0.222"), ("proto", "0.1588", "0.135", "0.183")):
        if k not in T:
            continue
        num(s, f"REVIEW2.plat.{k}", f"platform mean, {k} (n={T[k]['platform_n']})", T[k]["platform_mean"], mean, lit="")
        num(s, f"REVIEW2.plat.{k}.lo", f"  t 95% CI low", T[k]["platform_ci95_t"][0], lo, lit="")
        num(s, f"REVIEW2.plat.{k}.hi", f"  t 95% CI high", T[k]["platform_ci95_t"][1], hi, lit="")
    num(s, "REVIEW2.plat.v1", "platform v1, single run", T["v1"]["platform_mean"], "0.2157", lit="")
    num(s, "REVIEW2.plat.v1_2_n5", "platform v1_2 with the re-package (n=5)", T["v1_2"]["platform_with_repackage"]["mean"],
        "0.1886", lit="")
    for k, r18, r8 in (("v0", "0.1978", "0.1888"), ("v1", "0.1945", "0.1909"), ("v1_1", "0.1954", "0.1896"),
                       ("v1_2", "0.1954", "0.1896"), ("v1_3", "0.1954", "0.1896"), ("proto", "0.1854", "0.1817")):
        if k not in T:
            continue
        num(s, f"REVIEW2.replay.{k}", f"replay ALC, 18 exact pairs, {k}", T[k]["replay_alc_18"], r18, lit="")
        num(s, f"REVIEW2.replay8.{k}", f"replay ALC, strict 8 pairs, {k}", T[k]["replay_alc_strict8"], r8, lit="")
    r1_ok = available("experiments/runs/review_round1/platform_baseline_replay.json")
    if r1_ok:
        r1 = J(RR / "platform_baseline_replay.json")["all_exact_singletons"]["alc"]
        close(s, "REVIEW2.replay_eq_r1", "round-2 replay ALCs (v0, final) == round-1 replay file",
              max(abs(T["v0"]["replay_alc_18"] - r1["v0"]), abs(T["v1_3"]["replay_alc_18"] - r1["v1_3"])), 0.0, 1e-12)
    pr = lv["paired_replay"]
    for k, m_, lo, hi in (("v0->v1", "-0.0033", "-0.0092", "0.0028"), ("v1->v1_1", "0.0009", "-0.0020", "0.0039"),
                          ("v1_3->proto", "-0.0100", "-0.0155", "-0.0054")):
        if k not in pr:
            continue
        num(s, f"REVIEW2.paired.{k}", f"paired replay {k}, 18 pairs", pr[k]["mean_18"], m_, lit="")
        num(s, f"REVIEW2.paired.{k}.lo", "  95% CI low", pr[k]["ci95_18"][0], lo, lit="")
        num(s, f"REVIEW2.paired.{k}.hi", "  95% CI high", pr[k]["ci95_18"][1], hi, lit="")
    # strict-8 shrinkage-minus-final: round 1 (platform_baseline_replay.json) is the canonical source. Round 2 draws its
    # own bootstrap, so its interval differs by Monte-Carlo error only; the point estimate must be identical.
    if r1_ok:
        r1s = J(RR / "platform_baseline_replay.json")["strict"]["diffs"]["v0_minus_final"]
        r2s = pr["v0->v1_3"]
        close(s, "REVIEW2.strict8_point", "strict-8 v0 minus final: round 2 point == round 1 point (canonical)",
              -r2s["mean_strict8"], r1s["mean"], 1e-12)
        close(s, "REVIEW2.strict8_ci_mc", "strict-8 interval: round 2 vs round 1 endpoints differ only by bootstrap error",
              max(abs(-r2s["ci95_strict8"][1] - r1s["ci95_pair"][0]), abs(-r2s["ci95_strict8"][0] - r1s["ci95_pair"][1])),
              0.0, 5e-4)
        num(s, "REVIEW2.strict8_canon_lo", "canonical strict-8 interval low (round 1)", r1s["ci95_pair"][0], "-0.0052", lit="")
        num(s, "REVIEW2.strict8_canon_hi", "canonical strict-8 interval high (round 1)", r1s["ci95_pair"][1], "0.0029", lit="")
    else:
        record(s, "REVIEW2.strict8", "canonical strict-8 interval [-0.0052, +0.0029] (round-1 replay file)", "SKIP",
               note=FEEDBACK_ONLY)
    if "v1_3->proto" in pr:
        num(s, "REVIEW2.paired.proto8", "paired replay v1_3->proto, strict 8", pr["v1_3->proto"]["mean_strict8"], "-0.0079",
            lit="")
    exact(s, "REVIEW2.item_group_zero", "replay cannot see the item-group step (v1_1->v1_2->v1_3 exactly 0)",
          [pr["v1_1->v1_2"]["mean_18"], pr["v1_2->v1_3"]["mean_18"]], [0.0, 0.0])
    rk = lv["ranking"]
    exact(s, "REVIEW2.ranking", "platform-mean order == local order (Spearman 1.0, n=4)",
          [rk["platform_mean_order"] == rk["local_order"], rk["spearman_local_vs_platform"]], [True, 1.0])
    num(s, "REVIEW2.spread", "local ALC spread across v1..v1_3", rk["local_spread"], "0.0010", lit="")
    if not requires(s, "REVIEW2.rerun", "rerun review_round2 from feedback",
                    "experiments/platform_submissions", VARIANT_DIR_REL,
                    "external/official_baselines/empirical_mean/model.py"):
        return
    import importlib
    r2 = importlib.import_module("paiec.analysis.review_round2")
    r2.OUT = TMP / "review_round2_rerun"
    with contextlib.redirect_stdout(io.StringIO()):
        r2.main()
    for name in ("per_budget_final", "local_vs_platform"):
        a_ = dict(_flat(J(r2.OUT / f"{name}.json")))
        b_ = dict(_flat(J(RR2 / f"{name}.json")))
        common = set(a_) & set(b_)
        w = max((abs(a_[k] - b_[k]) for k in common), default=float("nan"))
        ok = bool(common) and set(a_) == set(b_) and w <= 1e-9
        record(s, f"REVIEW2.rerun.{name}", f"review_round2 rerun == saved {name}.json", "PASS" if ok else "FAIL", w, 0.0,
               "<=1e-9", f"{len(common)} numbers")


# ============================================================================================ EXP0058: out-of-sample check
E58 = RUNS / "EXP-0058" / "exp_0058_oos_2026-10-22.json"


@section
def EXP0058():
    """Pre-registered out-of-sample confirmation on 89 public benchmarks never used in development (paper sec. 5.1,
    Limitations). Saved file: experiments/runs/EXP-0058/exp_0058_oos_2026-10-22.json (the pre-registration named a
    different output path; see REPRODUCE.md). Live rerun: --full --jobs exp0058 (needs data/_main_54074453)."""
    s = "EXP0058"
    if not E58.exists():
        record(s, "EXP0058", "EXP-0058 result file", "FAIL", note=f"missing {E58.relative_to(ROOT)}")
        return
    d = J(E58)
    st = d["stats"]
    vo, fv = st["R2"]["all"]["v0-official"], st["R2"]["all"]["final-v0"]
    num(s, "EXP0058.shrink", "R2 shrinkage minus official (x1e-3)", vo["mean"], "-32.7", 1e3, lit="-32.7")
    num(s, "EXP0058.shrink_lo", "  benchmark-bootstrap 95% low", vo["ci95"][0], "-37.1", 1e3, lit="[-37.1, -28.2]")
    num(s, "EXP0058.shrink_hi", "  benchmark-bootstrap 95% high", vo["ci95"][1], "-28.2", 1e3, lit="[-37.1, -28.2]")
    exact(s, "EXP0058.shrink_wins", "shrinkage better on 82 of 89 benchmarks", [vo["n_neg"], vo["n_bench"]], [82, 89],
          lit="82 of 89")
    num(s, "EXP0058.final_r2", "R2 final minus shrinkage (x1e-3)", fv["mean"], "+0.2", 1e3, lit="+0.2")
    num(s, "EXP0058.final_r2_lo", "  95% low", fv["ci95"][0], "-0.9", 1e3, lit="[-0.9, +1.5]")
    num(s, "EXP0058.final_r2_hi", "  95% high", fv["ci95"][1], "+1.5", 1e3, lit="[-0.9, +1.5]")
    f1 = st["R1"]["all"]["final-v0"]
    num(s, "EXP0058.final_r1", "R1 (platform-like) final minus shrinkage (x1e-3)", f1["mean"], "-1.7", 1e3, lit="-1.7")
    num(s, "EXP0058.final_r1_lo", "  95% low", f1["ci95"][0], "-3.1", 1e3, lit="[-3.1, -0.2]")
    num(s, "EXP0058.final_r1_hi", "  95% high", f1["ci95"][1], "-0.2", 1e3, lit="[-3.1, -0.2]")
    ex = st["R1"]["excl_relatives"]["final-v0"]
    exact(s, "EXP0058.relatives", "relatives excluded (4) / benchmarks left", [len(d["relatives_excluded"]), ex["n_bench"]],
          [4, 85])
    num(s, "EXP0058.r1_excl_hi", "R1 without the 4 related benchmarks: upper CI end (x1e-3)", ex["ci95"][1], "+0.05", 1e3,
        lit="+0.05")
    h2, h1 = st["R2"]["acc_le_0_15"], st["R1"]["acc_le_0_15"]
    exact(s, "EXP0058.hard_n", "held-out benchmarks with accuracy <= 0.15", h2["final-v0"]["n_bench"], 5,
          lit="")  # not printed since 2026-10-23
    num(s, "EXP0058.hard_r2", "hard benchmarks, R2 final minus shrinkage (x1e-3)", h2["final-v0"]["mean"], "-7.1", 1e3,
        lit="-7.1")
    num(s, "EXP0058.hard_r1", "hard benchmarks, R1 final minus shrinkage (x1e-3)", h1["final-v0"]["mean"], "-13.6", 1e3,
        lit="-13.6")
    num(s, "EXP0058.hard_shrink", "hard benchmarks, shrinkage minus official, R2 (x1e-3)", h2["v0-official"]["mean"],
        "+9.9", 1e3, lit="+9.9", note=f"R2 value; the R1 value is {1e3 * h1['v0-official']['mean']:+.1f}")
    exact(s, "EXP0058.pairs", "R2 pairs / benchmarks", [d["r2_pairs_total"], d["n_bench"]], [2610, 89], lit="2{,}610")
    exact(s, "EXP0058.archives", "arms use the shipped archives (SHA-256 in the result file)",
          [d["package_checks"]["v0"]["sha256"] == sha(DIST / "v0.zip"),
           d["package_checks"]["final"]["sha256"] == SHA_V13], [True, True])
    close(s, "EXP0058.pkg_eq", "archive vs engine max |dp| recorded by the run", max(
        d["package_checks"][k]["max_abs_prob_delta"] for k in ("v0", "final", "official")), 0.0, 1e-9)
    prereg = ROOT / "docs" / "preregistrations" / "EXP-0058_oos_confirmation_preregistration.md"
    if prereg.exists():
        exact(s, "EXP0058.prereg_sha", "pre-registration file unchanged since the run (SHA-256 recorded in the result)",
              sha(prereg), d["prereg_sha256"])

Q4_STUB = Q4
# ============================================================================================ internal checks
INTERNAL_FILE = ROOT / "scripts" / "internal_variant_checks.py"
INTERNAL = False
if INTERNAL_FILE.exists():  # defines ARCH_variant, PLAT_variant and the full Q4 section; never released
    exec(compile(INTERNAL_FILE.read_text(encoding="utf-8"), str(INTERNAL_FILE), "exec"))
    INTERNAL = True


# ============================================================================================ main
SECTIONS = {"ARCH": ARCH, "T1": T1, "T2": T2, "T3": T3, "T3live": T3_live, "PLAT": PLAT_, "PROG": PROG, "SIM": SIM,
            "Q4": Q4, "HEAD": HEAD, "FIG": FIG, "REBUILD": REBUILD, "INV": INV, "REVIEW": REVIEW, "REVIEW2": REVIEW2, "EXP0058": EXP0058}
# --smoke: no gated data, no organiser-held feedback, < 2 min
SMOKE = ["ARCH", "REBUILD", "INV", "T2", "T3", "PROG", "SIM", "HEAD", "FIG", "REVIEW", "REVIEW2", "EXP0058"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true", help="saved results + cheap recomputation (default)")
    g.add_argument("--full", action="store_true", help="quick + rerun heavy experiments through run_slot.py")
    g.add_argument("--smoke", action="store_true", help="< 2 min, no gated data: archives, hashes, rebuild, "
                   "synthetic inversion tests, saved-result checks, figures")
    ap.add_argument("--only", default="", help="comma-separated sections (" + ",".join(SECTIONS) + ") or full jobs")
    ap.add_argument("--jobs", default="", help="--full: comma-separated subset of " + ",".join(FULL_JOBS))
    ap.add_argument("--strict", action="store_true", help="also fail on SKIP rows (inputs absent)")
    ap.add_argument("--simulate-public", action="store_true", help="pretend the non-public inputs are absent")
    ap.add_argument("--json", default=str(TMP / "last_run.json"), help="machine-readable results")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--worker-out", help=argparse.SUPPRESS)
    a = ap.parse_args()
    global SIMULATE_PUBLIC, INTERNAL, Q4
    SIMULATE_PUBLIC = a.simulate_public
    if SIMULATE_PUBLIC and INTERNAL:
        INTERNAL = False
        SECTIONS["Q4"] = Q4_STUB
    if a.worker:
        return worker(a.worker, Path(a.worker_out))
    if a.list:
        for k, f in SECTIONS.items():
            print(f"{k:7s} {f.__name__}")
        for k, v in FULL_JOBS.items():
            print(f"full:{k:10s} ~{v[2]:>3} min  {v[1]}")
        return 0
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    TMP.mkdir(parents=True, exist_ok=True)
    only = [x for x in a.only.split(",") if x] or (SMOKE if a.smoke else [])
    t0 = time.time()
    for k, f in SECTIONS.items():
        if not only or k in only:
            print(f"[{k}] ...", flush=True)
            f()
    if a.full:
        jobs = [j for j in (a.jobs.split(",") if a.jobs else FULL_JOBS)]
        full_tier(jobs)
    elapsed = time.time() - t0

    w = {k: max(len(str(r[k])) for r in ROWS + [dict(id="id", status="status", got="got", expected="expected", tol="tol")])
         for k in ("id", "status", "got", "expected", "tol")}
    w = {k: min(v, 34) for k, v in w.items()}
    print()
    print(f"{'status':6s}  {'id':{w['id']}s}  {'got':{w['got']}s}  {'expected':{w['expected']}s}  {'tol':{w['tol']}s}  what")
    for r in ROWS:
        print(f"{r['status']:6s}  {r['id'][:w['id']]:{w['id']}s}  {r['got'][:w['got']]:{w['got']}s}  "
              f"{r['expected'][:w['expected']]:{w['expected']}s}  {r['tol'][:w['tol']]:{w['tol']}s}  {r['what']}"
              + (f"  [{r['note']}]" if r["note"] and r["status"] != "PASS" else ""))
    counts = {k: sum(r["status"] == k for r in ROWS) for k in ("PASS", "FAIL", "NOSRC", "SKIP")}
    print()
    for x in WARN:
        print("WARN", x)
    print("section seconds: " + ", ".join(f"{k} {v:.1f}" for k, v in TIMES.items()))
    print(f"TOTAL {len(ROWS)} checks: " + ", ".join(f"{k} {v}" for k, v in counts.items()) + f"  ({elapsed:.0f} s)")
    bad = [r for r in ROWS if r["status"] in ("FAIL", "NOSRC") or (r["status"] == "SKIP" and a.strict)]
    skipped = [r for r in ROWS if r["status"] == "SKIP"]
    if skipped:
        print("SKIPPED (inputs not available here):")
        for r in skipped:
            print(f"  SKIP  {r['id']}: {r['what']} -- {r['note']}")
    if bad:
        print("NOT REPRODUCED:")
        for r in bad:
            print(f"  {r['status']:5s} {r['id']}: {r['what']}  got {r['got']} expected {r['expected']}"
                  + (f"  -- {r['note']}" if r["note"] else ""))
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(dict(rows=ROWS, warnings=WARN, counts=counts, seconds=elapsed, times=TIMES,
                                            tier="full" if a.full else "smoke" if a.smoke else "quick", root=str(ROOT),
                                            python=sys.version.split()[0]), indent=1), encoding="utf-8")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
