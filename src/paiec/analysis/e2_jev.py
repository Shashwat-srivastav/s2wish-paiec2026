"""E2 -- Is Jev's difficulty signal real, and can a local model reproduce it?

Pre-registration: docs/preregistrations/E1_E2_preregistration.md (E2a -> E2b -> E2c, fixed stop rules).
Report:           docs/research_notes/20_E2_JEV_MEMORISATION_DISTILLATION.md
Outputs:          experiments/runs/e2/*.json, *.log, *.parquet

Subcommands
  e2a   memorisation tests: (1) MathArena release-date split, (2) surface-perturbation re-scoring.
  e2b   pairwise (Bradley-Terry) Jev scoring              [only if e2a does not fire STOP]
  e2c   distil jev_difficulty into a local sentence encoder [only if e2a does not fire STOP]

Conventions
  * beta = per-benchmark Rasch difficulty (analysis.item_repr.rasch: L2-regularised JML on trial
    counts, response > 0.5 binarised), centred within benchmark; higher = harder.  Each benchmark's
    beta uses only its own responses.  Same target as report 13.
  * Jev scores are the pinned jev-1.13.0 `jev_difficulty` (0..1), computed from item content only
    (no benchmark name, no item_features) -- same state construction as report 13.
  * Bootstrap CIs resample duplicate-content groups (MathArena has exact duplicates across
    competitions), 2000 replicates, fixed seeds.
  * TYPESAFE_API_KEY is read from the environment by paiec.jev.item_features; never printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from paiec import paths

RUN = paths.RUNS / "e2"
JEV_IN = paths.RUNS / "jev" / "jev_in"
JEV_OUT = paths.RUNS / "jev"
BENCHES = ["matharena", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench"]
BENCH_NAME_RE = re.compile(r"matharena|multi[_\- ]?swe[_\- ]?bench|real[_\- ]?webagents?|researchcodebench|"
                           r"swe[_\- ]?rebench|swe[_\- ]?bench", re.I)
NBOOT = 2000


def log(*a):
    print(*a, flush=True)


# =====================================================================================
# Data
# =====================================================================================
_ITEMS: dict = {}


def bench_items(b: str) -> pd.DataFrame:
    """item_id, content, jev (cached jev_difficulty), beta (centred Rasch), grp (content group)."""
    if b in _ITEMS:
        return _ITEMS[b]
    from paiec.analysis.item_repr import rasch
    ji = pd.read_parquet(JEV_IN / f"{b}.parquet")                       # exactly what Jev saw
    jo = pd.read_parquet(JEV_OUT / f"jev_out_{b}.parquet", columns=["item_id", "jev_key", "jev_difficulty"])
    r = pd.read_parquet(paths.DATA / b / "response.parquet", columns=["subject_id", "item_id", "response"])
    r = r[r.item_id.isin(ji.item_id)].dropna(subset=["response"])
    r["response"] = (r.response > 0.5).astype(float)
    _, be, _ = rasch(r)
    del r
    it = ji.merge(jo, on="item_id", how="left")
    it = it[it.item_id.isin(be.index)].copy()
    it["beta"] = be.reindex(it.item_id).to_numpy()
    it["beta"] -= it.beta.mean()
    it["grp"] = pd.factorize(it.item_content)[0]
    it["bench"] = b
    if b == "matharena":
        f = pd.read_parquet(paths.DATA / b / "items.parquet", columns=["item_id", "item_features"])
        it = it.merge(f, on="item_id", how="left")
        it["comp"] = it.item_features.str.extract(r"competition=([^;]+)")[0]
    it = it.rename(columns={"item_content": "content", "jev_difficulty": "jev"}).reset_index(drop=True)
    _ITEMS[b] = it
    return it


# =====================================================================================
# Statistics helpers
# =====================================================================================
def rho(x, y) -> float:
    return float(spearmanr(x, y)[0])


def _group_index(grp):
    u, inv = np.unique(grp, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(u))]
    return members


def boot_rho(x, y, grp, nboot=NBOOT, seed=0):
    """Cluster bootstrap (resample content groups) of Spearman(x, y). Returns (pt, lo, hi, reps)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    members = _group_index(grp)
    rng = np.random.default_rng(seed)
    G = len(members)
    reps = np.empty(nboot)
    for k in range(nboot):
        pick = rng.integers(0, G, G)
        idx = np.concatenate([members[g] for g in pick])
        reps[k] = rho(x[idx], y[idx])
    return rho(x, y), float(np.nanpercentile(reps, 2.5)), float(np.nanpercentile(reps, 97.5)), reps


def ci(reps):
    return float(np.nanpercentile(reps, 2.5)), float(np.nanpercentile(reps, 97.5))


# =====================================================================================
# E2a.1 -- MathArena release-date split
# =====================================================================================
# competition -> (public release month, uncertain?, note). Uncertain rows are excluded in the
# pre-registered sensitivity run. "Release" = month the problem text first became public.
RELEASE = {
    "aime_2025":           ("2025-02", False, "AIME I 6 Feb / AIME II 12 Feb 2025"),
    "aime_2025_I":         ("2025-02", False, "AIME I, 6 Feb 2025"),
    "aime_2025_II":        ("2025-02", False, "AIME II, 12 Feb 2025"),
    "hmmt_feb_2025":       ("2025-02", False, "HMMT February, 15 Feb 2025"),
    "usamo_2025":          ("2025-03", False, "USAMO, 19-20 Mar 2025"),
    "kangaroo_2025_1_2":   ("2025-03", True,  "Math Kangaroo contest 20 Mar 2025; date the problem text was posted publicly is not documented"),
    "kangaroo_2025_3_4":   ("2025-03", True,  "as above"),
    "kangaroo_2025_5_6":   ("2025-03", True,  "as above"),
    "kangaroo_2025_7_8":   ("2025-03", True,  "as above"),
    "kangaroo_2025_9_10":  ("2025-03", True,  "as above"),
    "kangaroo_2025_11_12": ("2025-03", True,  "as above"),
    "smt_2025":            ("2025-04", False, "Stanford Math Tournament, Apr 2025"),
    "brumo_2025":          ("2025-04", False, "Brown University Math Olympiad, Apr 2025"),
    "cmimc_2025":          ("2025-04", True,  "CMIMC 2025, spring (month not verified); pre-2025-07 either way"),
    "imo_2025":            ("2025-07", False, "IMO, 15-16 Jul 2025"),
    "imc_2025":            ("2025-08", True,  "IMC (university), late Jul / early Aug 2025; post either way"),
    "apex_2025":           ("2025-08", True,  "MathArena Apex set (Aug 2025), compiled from 2025 competitions whose own dates vary"),
    "apex_shortlist":      ("2025-08", True,  "Apex shortlist; compiled from 2025 competitions; publication month not verified"),
    "miklos_2025":         ("2025-10", True,  "Miklos Schweitzer 2025, autumn (month not verified); post either way"),
    "hmmt_nov_2025":       ("2025-11", False, "HMMT November, Nov 2025"),
    "putnam_2025":         ("2025-12", False, "Putnam, 6 Dec 2025"),
    "arxivmath_1225":      ("2025-12", False, "arXiv-derived problems, Dec 2025"),
    "arxivmath_0126":      ("2026-01", False, "arXiv-derived problems, Jan 2026"),
    "arxivmath_0226":      ("2026-02", False, "arXiv-derived problems, Feb 2026"),
    "aime_2026":           ("2026-02", False, "AIME 2026, Feb 2026"),
    "aime_2026_I":         ("2026-02", False, "AIME I 2026, Feb 2026"),
    "hmmt_feb_2026":       ("2026-02", False, "HMMT February 2026"),
}
SPLIT = "2025-07"


def release_split(it: pd.DataFrame, exclude_uncertain: bool, seed: int):
    it = it.copy()
    if exclude_uncertain:
        it = it[~it.comp.map(lambda c: RELEASE[c][1])]
    it["month"] = it.comp.map(lambda c: RELEASE[c][0])
    # a duplicate-content group was public from its earliest competition on
    it["month"] = it.groupby("grp").month.transform("min")
    it["post"] = it.month >= SPLIT
    out = {"n_items": {}, "n_groups": {}, "rho": {}, "ci": {}}
    reps = {}
    for nm, m in (("pre", ~it.post), ("post", it.post), ("all", it.post | ~it.post)):
        s = it[m]
        pt, lo, hi, rp = boot_rho(s.jev, s.beta, s.grp, seed=seed + {"pre": 1, "post": 2, "all": 3}[nm])
        out["n_items"][nm], out["n_groups"][nm] = int(len(s)), int(s.grp.nunique())
        out["rho"][nm], out["ci"][nm] = pt, [lo, hi]
        reps[nm] = rp
    d = reps["pre"] - reps["post"]
    out["diff_pre_minus_post"] = out["rho"]["pre"] - out["rho"]["post"]
    out["diff_ci"] = list(ci(d))
    # diagnostic (not pre-registered): rho after removing competition means of both variables
    def wc(s):
        a = s.jev - s.groupby("comp").jev.transform("mean")
        b = s.beta - s.groupby("comp").beta.transform("mean")
        return rho(a, b)
    out["diag_within_competition_rho"] = {"pre": wc(it[~it.post]), "post": wc(it[it.post])}
    out["per_competition"] = {c: dict(n=int(len(s)), month=RELEASE[c][0], post=bool(s.post.iloc[0]),
                                      rho=rho(s.jev, s.beta) if s.jev.nunique() > 1 else None,
                                      mean_beta=float(s.beta.mean()), mean_jev=float(s.jev.mean()))
                              for c, s in it.groupby("comp")}
    return out


# =====================================================================================
# E2a.2 -- surface perturbations (deterministic rules, logged)
# =====================================================================================
FIRST_NAMES = ["Alex", "Alice", "Amy", "Andrew", "Anna", "Ben", "Bob", "Carl", "Carol", "Charlie", "Chris",
               "Claire", "Dan", "Daniel", "Dave", "David", "Derek", "Diana", "Emily", "Emma", "Eric", "Eve",
               "Frank", "Fred", "George", "Grace", "Hannah", "Harry", "Helen", "Henry", "Jack", "James",
               "Jane", "Jason", "Jerry", "Jessica", "Joe", "John", "Kate", "Kevin", "Laura", "Leo", "Linda",
               "Lisa", "Lucy", "Mark", "Mary", "Max", "Michael", "Mike", "Nancy", "Nick", "Olivia", "Oscar",
               "Paul", "Peter", "Rachel", "Robert", "Ryan", "Sam", "Sarah", "Sophia", "Steve", "Susan",
               "Ted", "Tim", "Tom", "Victor", "Will", "Zoe", "Barbara", "Thomas", "Smith", "Johnson",
               "Williams", "Brown", "Jones", "Miller", "Davis", "Wilson", "Taylor", "Anderson"]
REPL_NAMES = ["Imani", "Tariq", "Yusuf", "Ingrid", "Mateo", "Priya", "Kenji", "Astrid", "Dmitri", "Femi",
              "Lucia", "Omar", "Rhea", "Soren", "Thandi", "Viktor", "Wen", "Xavi", "Yara", "Zainab", "Arjun",
              "Bettina", "Cosmin", "Dalia", "Emeka", "Farida", "Goran", "Hiroko", "Ilse", "Jovan", "Kalani",
              "Leif", "Marisol", "Nadia", "Orrin", "Paloma", "Quentin", "Rafaela", "Stellan", "Tamsin",
              "Ulla", "Vesna", "Wiremu", "Ximena", "Yannick", "Zora", "Anouk", "Bjorn", "Catalina",
              "Dorian", "Elif", "Florin", "Greta", "Hamza", "Inés", "Jarrah", "Kofi", "Lorna", "Matthias",
              "Noor", "Odile", "Pavel", "Rosalind", "Saoirse", "Teodor", "Uriel", "Valka", "Wolfram",
              "Yasmin", "Zeno", "Okafor", "Lindqvist", "Haddad", "Moreau", "Nakamura", "Petrov",
              "Quispe", "Rasmussen", "Sorensen", "Takahashi", "Varga", "Whitlock"]
assert len(FIRST_NAMES) == len(REPL_NAMES) and len(set(REPL_NAMES) & set(FIRST_NAMES)) == 0
NAME_MAP = dict(zip(FIRST_NAMES, REPL_NAMES))
# clone-site names that appear as task prefixes in the web-agent pool (surface only)
SITE_MAP = {"Zilloft": "Homefinder", "UdriverX": "RideHub", "Udriver": "RideHub", "OpenDining": "TableNow",
            "GoCalendar": "DayPlanner", "Go Calendar": "DayPlanner", "GoMail": "PostBox", "Go Mail": "PostBox",
            "Staynb": "Roomly", "DashDish": "MealDrop", "NetworkIn": "LinkPro", "Omnizon": "ShopAll"}
WORDS = ["amber", "birch", "cedar", "delta", "ember", "fjord", "grove", "haven", "iris", "juniper", "kestrel",
         "lumen", "maple", "nimbus", "onyx", "prism", "quartz", "raven", "sierra", "tundra", "umber", "vale",
         "willow", "xenon", "yarrow", "zephyr", "basalt", "coral", "dune", "flint", "garnet", "heron",
         "indigo", "jasper", "kelp", "lichen", "marble", "nectar", "orchid", "pebble"]
ID_STOP = {"isinstance", "TypeError", "ValueError", "KeyError", "IndexError", "RuntimeError",
           "AttributeError", "ImportError", "NotImplementedError", "AssertionError", "StopIteration",
           "JavaScript", "TypeScript", "GitHub", "GitLab", "PyTorch", "NumPy", "README", "JSON", "macOS",
           "iOS", "OpenAI", "DeepMind", "LaTeX", "PostgreSQL", "MySQL", "SQLite", "YouTube", "WebSocket",
           "PowerShell", "VSCode", "IPython", "DataFrame", "UnicodeEncodeError", "UnicodeDecodeError",
           "FileNotFoundError", "ZeroDivisionError", "OverflowError", "MemoryError", "PermissionError",
           "OSError", "NameError", "SyntaxError", "IndentationError", "TimeoutError", "ConnectionError",
           "has_key", "self", "kwargs", "args", "iclr2025_conference", "newcommand", "usepackage"}
ID_RE = re.compile(r"(?<![\w\\])(?:[a-z][a-z0-9]*(?:_[a-z0-9]+)+|[a-z]+[A-Z][A-Za-z0-9]*|(?:[A-Z][a-z0-9]+){2,}[A-Za-z0-9]*)(?!\w)")
FENCE_RE = re.compile(r"```.*?```|`[^`\n]+`", re.S)
NUM_RE = re.compile(r"(?<![\w.,:#/\\$-])(\d{1,3})(st|nd|rd|th)?(?![\w]|[.:]\d|,\d)")
MATH_RE = re.compile(r"\$\$.*?\$\$|\$[^$]+\$|\\\(.*?\\\)|\\\[.*?\\\]", re.S)
PROTECT_CMD = re.compile(r"\\(?:mathbb|mathcal|mathrm|mathfrak|mathsf|mathbf|text|textbf|textit|operatorname|begin|end|label|tag)\{[^{}]*\}")
UPPER_PERM = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "PQRSTUVWXYZABCDEFGHIJKLMNO"))   # fixed shift by 15


def _ordinal(n):
    return "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def _syn_ident(tok, used):
    parts = re.split(r"_|(?<=[a-z0-9])(?=[A-Z])", tok)
    k = max(2, min(len([p for p in parts if p]), 3))
    h = int(hashlib.sha1(tok.encode()).hexdigest(), 16)
    for probe in range(200):
        ws = [WORDS[(h // (41 ** j) + probe * (j + 1)) % len(WORDS)] for j in range(k)]
        if "_" in tok:
            new = "_".join(ws)
        elif tok[0].isupper():
            new = "".join(w.capitalize() for w in ws)
        else:
            new = ws[0] + "".join(w.capitalize() for w in ws[1:])
        if new not in used:
            used.add(new)
            return new
    return tok + "_x"


def _sub_outside(text, pattern, fn):
    """apply fn to every segment of text NOT matched by pattern (code spans / math)."""
    out, last = [], 0
    for m in pattern.finditer(text):
        out.append(fn(text[last:m.start()])); out.append(m.group(0)); last = m.end()
    out.append(fn(text[last:]))
    return "".join(out)


def r_names(text, log_):
    n = 0
    def f(m):
        nonlocal n
        n += 1
        return NAME_MAP[m.group(0)]
    pat = re.compile(r"\b(" + "|".join(sorted(map(re.escape, NAME_MAP), key=len, reverse=True)) + r")\b")
    t = pat.sub(f, text)
    for k, v in SITE_MAP.items():
        c = t.count(k)
        if c:
            t = t.replace(k, v); n += c
    if n:
        log_.append(f"names/sites:{n}")
    return t


def r_identifiers(text, log_):
    found = []
    for m in ID_RE.finditer(text):
        tok = m.group(0)
        if tok in ID_STOP or tok.startswith("__") or len(tok) < 5 or tok in found:
            continue
        found.append(tok)
    if not found:
        return text
    used = set(found)
    mp = {tok: _syn_ident(tok, used) for tok in found}
    pat = re.compile(r"(?<![\w\\])(" + "|".join(sorted(map(re.escape, mp), key=len, reverse=True)) + r")(?!\w)")
    t = pat.sub(lambda m: mp[m.group(1)], text)
    log_.append(f"identifiers:{len(mp)}")
    return t


def r_numbers(text, log_):
    """n -> n+1 for standalone 1-3 digit integers in prose (not in code spans, not versions/times/
    decimals/grouped numbers/issue refs). Only for benchmarks whose answer does not depend on them."""
    n = 0
    def seg(s):
        nonlocal n
        def f(m):
            nonlocal n
            v = int(m.group(1)) + 1
            n += 1
            return str(v) + (_ordinal(v) if m.group(2) else "")
        return NUM_RE.sub(f, s)
    t = _sub_outside(text, FENCE_RE, seg)
    if n:
        log_.append(f"numbers:{n}")
    return t


def r_math_letters(text, log_):
    """Consistent relabelling of all-uppercase letter runs inside math mode (points, sets, matrices),
    with \\mathbb{..}, \\text{..} etc. protected. Labels only; no constant changes."""
    n = 0
    def mseg(ms):
        nonlocal n
        prot = []
        def keep(m):
            prot.append(m.group(0)); return f"\x00{len(prot)-1}\x00"
        s = PROTECT_CMD.sub(keep, ms)
        def run(m):
            nonlocal n
            n += 1
            return "".join(UPPER_PERM[c] for c in m.group(0))
        s = re.sub(r"(?<![\\A-Za-z])[A-Z]+(?![A-Za-z])", run, s)
        return re.sub(r"\x00(\d+)\x00", lambda m: prot[int(m.group(1))], s)
    out, last = [], 0
    for m in MATH_RE.finditer(text):
        out.append(text[last:m.start()]); out.append(mseg(m.group(0))); last = m.end()
    out.append(text[last:])
    if n:
        log_.append(f"math_labels:{n}")
    return "".join(out)


def r_move_preamble(text, log_):
    """Reorder independent clauses: move a leading instruction paragraph to the end."""
    paras = re.split(r"\n\s*\n", text.strip(), maxsplit=1)
    if len(paras) == 2 and len(paras[0]) <= 400 and re.search(r"reason step by step|final answer|Please|You are", paras[0]):
        log_.append("reorder:preamble_to_end")
        return paras[1].rstrip() + "\n\n" + paras[0].strip()
    return text


def r_move_title(text, log_):
    """Issue-style text: move the title line (independent summary) after the body."""
    lines = text.strip().split("\n", 1)
    if len(lines) == 2 and 0 < len(lines[0]) <= 200 and len(lines[1].strip()) > 0:
        log_.append("reorder:title_to_end")
        return lines[1].strip() + "\n\n" + lines[0].strip()
    return text


def r_reverse_list(text, log_):
    """Reverse short coordinated lists 'a, b, c(,) and d' (order-free attribute lists)."""
    pat = re.compile(r"\b((?:[\w$'-]+(?: [\w$'-]+){0,2}, ){2,}[\w$'-]+(?: [\w$'-]+){0,2},? and [\w$'-]+(?: [\w$'-]+){0,2})\b")
    n = 0
    def f(m):
        nonlocal n
        parts = re.split(r",? and |, ", m.group(1))
        if len(parts) < 3:
            return m.group(1)
        n += 1
        parts = parts[::-1]
        return ", ".join(parts[:-1]) + ", and " + parts[-1]
    t = pat.sub(f, text)
    if n:
        log_.append(f"reorder:list_reversed:{n}")
    return t


VERB_MAP = [("Show me", "Display"), ("Find me", "Locate for me"), ("Find", "Locate"), ("Search for", "Look up"),
            ("Book me", "Reserve me"), ("Book", "Reserve"), ("Schedule", "Arrange"), ("Go to", "Navigate to"),
            ("Open", "Bring up"), ("Look for", "Hunt for"), ("Get me", "Fetch me"), ("Tell me", "Report"),
            ("Check", "Verify"), ("Create", "Set up"), ("Send", "Dispatch"), ("Write", "Compose"),
            ("Add", "Insert"), ("Remove", "Delete"), ("Buy", "Purchase"), ("Order", "Purchase")]


def r_verbs(text, log_):
    """Sentence-initial instruction verbs -> fixed synonyms (same requested action)."""
    n = 0
    for a, b in VERB_MAP:
        pat = re.compile(r"(^|(?<=[.!?:]\s)|(?<=\n))" + re.escape(a) + r"\b")
        text, k = pat.subn(lambda m: m.group(1) + b, text)
        n += k
    if n:
        log_.append(f"verbs:{n}")
    return text


RULES = {   # applied in this order; benchmark identity selects rules locally and is never sent
    "matharena":         [r_names, r_math_letters, r_move_preamble],          # constants determine the answer: unchanged
    "multi_swebench":    [r_identifiers, r_names, r_numbers, r_move_title],
    "swe_rebench":       [r_identifiers, r_names, r_numbers, r_move_title],
    "researchcodebench": [r_identifiers, r_names, r_move_preamble],           # paper numbers left intact
    "real_webagents":    [r_names, r_numbers, r_reverse_list, r_verbs],
}


def perturb(text: str, bench: str):
    log_ = []
    t = text
    for f in RULES[bench]:
        t = f(t, log_)
    return t, log_


def scrub_bench_names(text):
    return BENCH_NAME_RE.sub("the task set", text)


def sample_perturb_items(n_per=60, seed=0):
    rows = []
    for b in BENCHES:
        it = bench_items(b)
        g = it.drop_duplicates("grp")            # one item per distinct content
        g = g[g.jev.notna()]
        pick = g.sample(n=min(n_per, len(g)), random_state=seed)
        rows.append(pick[["item_id", "bench", "content", "jev", "beta", "grp"]])
    return pd.concat(rows, ignore_index=True)


def jev_score(texts: pd.DataFrame, cache_name: str):
    """texts: item_id, item_content. Same state construction as report 13 (content only)."""
    from paiec.jev import item_features as JF
    for t in texts.item_content:
        assert not BENCH_NAME_RE.search(t), "benchmark name in a Jev state"
    cache = paths.JEV_CACHE / cache_name
    f = JF.compute_features(texts[["item_id", "item_content"]], cache_path=cache, include_benchmark=False,
                            workers=8, verbose=True)
    tok = f.drop_duplicates("jev_key")["jev_input_tokens"].fillna(0).sum()
    return f[["item_id", "jev_difficulty", "jev_model", "jev_input_tokens"]], int(tok)


def mean_within(df, col):
    return float(np.mean([rho(s[col], s.beta) for _, s in df.groupby("bench")]))


def boot_perturb(df, nboot=NBOOT, seed=0):
    rng = np.random.default_rng(seed)
    by = {b: s.reset_index(drop=True) for b, s in df.groupby("bench")}
    reps = {k: np.empty(nboot) for k in ("orig", "pert", "retest", "ratio", "absd")}
    for k in range(nboot):
        o, p, r, ad = [], [], [], []
        for b, s in by.items():
            i = rng.integers(0, len(s), len(s))
            ss = s.iloc[i]
            o.append(rho(ss.jev, ss.beta)); p.append(rho(ss.jev_pert, ss.beta)); r.append(rho(ss.jev_retest, ss.beta))
            ad.append(np.abs(ss.jev_pert - ss.jev).mean())
        reps["orig"][k], reps["pert"][k], reps["retest"][k] = np.mean(o), np.mean(p), np.mean(r)
        reps["ratio"][k] = np.mean(p) / np.mean(o) if np.mean(o) != 0 else np.nan
        reps["absd"][k] = np.mean(ad)
    return {k: list(ci(v)) for k, v in reps.items()}


def e2a(args):
    RUN.mkdir(parents=True, exist_ok=True)
    out = {"prereg": "docs/preregistrations/E1_E2_preregistration.md#E2a", "split": SPLIT}
    t0 = time.time()
    # ---------------------------------------------------------------- test 1
    ma = bench_items("matharena")
    out["release_map"] = {c: dict(month=m, uncertain=u, note=n) for c, (m, u, n) in RELEASE.items()}
    missing = sorted(set(ma.comp) - set(RELEASE))
    assert not missing, missing
    out["test1_main"] = release_split(ma, exclude_uncertain=False, seed=100)
    out["test1_sens_excl_uncertain"] = release_split(ma, exclude_uncertain=True, seed=200)
    for k in ("test1_main", "test1_sens_excl_uncertain"):
        x = out[k]
        log(f"[{k}] pre rho={x['rho']['pre']:.3f} {np.round(x['ci']['pre'],3)} n={x['n_items']['pre']}/{x['n_groups']['pre']}g | "
            f"post rho={x['rho']['post']:.3f} {np.round(x['ci']['post'],3)} n={x['n_items']['post']}/{x['n_groups']['post']}g | "
            f"pre-post={x['diff_pre_minus_post']:+.3f} {np.round(x['diff_ci'],3)} | within-comp {x['diag_within_competition_rho']}")
    # ---------------------------------------------------------------- test 2
    S = sample_perturb_items(seed=0)
    pert, logs = [], []
    for t, b in zip(S.content, S.bench):
        p, lg = perturb(t, b)
        pert.append(scrub_bench_names(p)); logs.append(";".join(lg))
    S["content_pert"] = pert
    S["rules"] = logs
    S["changed"] = S.content_pert != S.content
    log(f"[perturb] {len(S)} items, changed {S.changed.mean():.3f}; by bench "
        f"{S.groupby('bench').changed.mean().round(2).to_dict()}")
    retest, tok_r = jev_score(S[["item_id"]].assign(item_content=S.content), "jev_v1_e2a_retest.parquet")
    pj, tok_p = jev_score(S[["item_id"]].assign(item_content=S.content_pert), "jev_v1_e2a_perturbed.parquet")
    S["jev_retest"] = S.item_id.map(dict(zip(retest.item_id, retest.jev_difficulty)))
    S["jev_pert"] = S.item_id.map(dict(zip(pj.item_id, pj.jev_difficulty)))
    S["jev_model"] = S.item_id.map(dict(zip(pj.item_id, pj.jev_model)))
    assert S[["jev_retest", "jev_pert"]].notna().all().all(), "missing Jev scores"
    S.drop(columns=["content", "content_pert"]).to_parquet(RUN / "e2a_perturb_scores.parquet", index=False)
    S[["item_id", "bench", "rules", "changed"]].to_csv(RUN / "e2a_perturb_rules_log.csv", index=False)
    # 5 before/after examples (one per benchmark; first 500 chars of a changed region)
    ex = []
    for b in BENCHES:
        s = S[(S.bench == b) & S.changed].head(1)
        for r in s.itertuples():
            ex.append(dict(bench=b, item_id=r.item_id, rules=r.rules, before=r.content[:600], after=r.content_pert[:600],
                           before_tail=r.content[-300:], after_tail=r.content_pert[-300:],
                           jev=r.jev, jev_retest=r.jev_retest, jev_pert=r.jev_pert))
    json.dump(ex, open(RUN / "e2a_perturb_examples.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    res = {}
    for nm, D in (("all", S), ("changed_only", S[S.changed])):
        r = dict(n=int(len(D)),
                 mean_abs_delta=float(np.abs(D.jev_pert - D.jev).mean()),
                 mean_abs_delta_retest=float(np.abs(D.jev_retest - D.jev).mean()),
                 frac_identical_pert=float((D.jev_pert == D.jev).mean()),
                 frac_identical_retest=float((D.jev_retest == D.jev).mean()),
                 mean_signed_delta=float((D.jev_pert - D.jev).mean()),
                 rho_orig=mean_within(D, "jev"), rho_pert=mean_within(D, "jev_pert"),
                 rho_retest=mean_within(D, "jev_retest"),
                 rho_orig_pert=float(np.mean([rho(s.jev, s.jev_pert) for _, s in D.groupby("bench")])),
                 per_bench={b: dict(n=int(len(s)), rho_orig=rho(s.jev, s.beta), rho_pert=rho(s.jev_pert, s.beta),
                                    rho_retest=rho(s.jev_retest, s.beta),
                                    mean_abs_delta=float(np.abs(s.jev_pert - s.jev).mean()),
                                    agree_orig_pert=rho(s.jev, s.jev_pert))
                            for b, s in D.groupby("bench")})
        r["ratio"] = r["rho_pert"] / r["rho_orig"]
        r["ci"] = boot_perturb(D, seed=300 if nm == "all" else 400)
        res[nm] = r
        log(f"[perturb {nm}] n={r['n']} |d|={r['mean_abs_delta']:.3f} (retest {r['mean_abs_delta_retest']:.3f}) "
            f"rho orig={r['rho_orig']:.3f} pert={r['rho_pert']:.3f} retest={r['rho_retest']:.3f} ratio={r['ratio']:.2f} {np.round(r['ci']['ratio'],2)}")
        for b, v in r["per_bench"].items():
            log(f"     {b:18s} " + " ".join(f"{k}={v[k]:.3f}" if isinstance(v[k], float) else f"{k}={v[k]}" for k in v))
    out["test2"] = res
    out["jev_cost"] = dict(tokens_retest=tok_r, tokens_perturbed=tok_p,
                           usd=(tok_r + tok_p) * 0.042 / 1e6, model=str(S.jev_model.iloc[0]))
    # ---------------------------------------------------------------- decision (mechanical)
    m, p = out["test1_main"], res["all"]
    post_lo, post_hi = m["ci"]["post"]
    cond_go = (m["rho"]["post"] >= 0.15 and post_lo > 0) and (p["ratio"] >= 0.8)
    cond_stop = (post_lo <= 0 <= post_hi) or (m["diff_pre_minus_post"] >= 0.15 and m["diff_ci"][0] > 0) or (p["ratio"] < 0.6)
    row = "STOP" if cond_stop else ("PROCEED" if cond_go else "PARTIAL")
    s = out["test1_sens_excl_uncertain"]
    s_lo, s_hi = s["ci"]["post"]
    s_go = (s["rho"]["post"] >= 0.15 and s_lo > 0) and (p["ratio"] >= 0.8)
    s_stop = (s_lo <= 0 <= s_hi) or (s["diff_pre_minus_post"] >= 0.15 and s["diff_ci"][0] > 0) or (p["ratio"] < 0.6)
    out["decision"] = dict(row=row, cond_go=bool(cond_go), cond_stop=bool(cond_stop),
                           sensitivity_row="STOP" if s_stop else ("PROCEED" if s_go else "PARTIAL"))
    log(f"[E2a decision] {out['decision']}  ({time.time()-t0:.0f}s)")
    json.dump(out, open(RUN / "e2a.json", "w", encoding="utf-8"), indent=1, default=float)
    return out


# =====================================================================================
# E2b -- pairwise (Bradley-Terry) Jev scoring
# =====================================================================================
PAIR_Q = {"harder": {
    "type": "choice",
    "instructions": "Two tasks are given: `task_a.text` and `task_b.text`. Which of the two tasks would be "
                    "harder for a current frontier AI system (a top large language model, given whatever tools "
                    "the task provides) to complete fully correctly?",
    "criteria": {"task_a": "The task in `task_a.text` is the harder one: a frontier model is less likely to complete it fully correctly.",
                 "task_b": "The task in `task_b.text` is the harder one: a frontier model is less likely to complete it fully correctly."},
}}
PAIR_CACHE = paths.JEV_CACHE / "jev_e2b_pairs.parquet"
N_PARTNERS = 8


def call_questions(state, questions, model, max_retries=8, timeout=120.0):
    from paiec.jev import item_features as JF
    import random as _r
    import requests
    body = {"state": state, "model": model, "questions": questions}
    delay, err = 1.0, ""
    for _ in range(max_retries):
        try:
            r = JF._session().post(JF.API_URL, json=body, timeout=timeout)
        except requests.RequestException as e:
            err = f"conn:{type(e).__name__}"
        else:
            if r.status_code == 200:
                return r.json()
            if r.status_code in (401, 403, 422):
                raise RuntimeError(f"TypeSafe {r.status_code}: {r.text[:300]}")
            err = f"http{r.status_code}"
        time.sleep(delay * (1 + 0.25 * _r.random()))
        delay = min(delay * 2, 60.0)
    raise RuntimeError(f"TypeSafe failed ({err})")


def pair_design(seed=0):
    """Per benchmark: unique content states; each draws 8 random partners; unordered pairs deduped;
    A/B position assigned by a seeded coin (position bias balanced)."""
    rows = []
    for b in BENCHES:
        it = bench_items(b)
        g = it.drop_duplicates("grp")[["grp", "content"]].reset_index(drop=True)
        n = len(g)
        rng = np.random.default_rng(seed + BENCHES.index(b))
        seen = set()
        for i in range(n):
            cand = rng.choice(np.delete(np.arange(n), i), size=min(N_PARTNERS, n - 1), replace=False)
            for j in cand:
                key = (min(i, int(j)), max(i, int(j)))
                if key in seen:
                    continue
                seen.add(key)
                a, c = (i, int(j)) if rng.random() < 0.5 else (int(j), i)
                rows.append((b, int(g.grp[a]), int(g.grp[c])))
    return pd.DataFrame(rows, columns=["bench", "grp_a", "grp_b"])


def fit_bt(ga, gb, pa, n, lam=0.1, iters=200):
    """Soft-outcome Bradley-Terry: P(a harder) = sig(s_a - s_b); L2 ridge lam; Newton on diagonal."""
    s = np.zeros(n)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(s[ga] - s[gb])))
        r = pa - p
        g = np.bincount(ga, r, n) - np.bincount(gb, r, n) - lam * s
        w = p * (1 - p)
        h = np.bincount(ga, w, n) + np.bincount(gb, w, n) + lam
        step = g / h
        s += step
        if np.abs(step).max() < 1e-8:
            break
    return s - s.mean()


def e2b(args):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from paiec.jev import item_features as JF
    RUN.mkdir(parents=True, exist_ok=True)
    P = pair_design()
    log(f"[e2b] {len(P)} pairs; by bench {P.bench.value_counts().to_dict()}")
    content = {b: dict(zip(bench_items(b).grp, bench_items(b).content)) for b in BENCHES}

    def state_of(r):
        ta = JF.truncate(content[r.bench][r.grp_a], 8000)
        tb = JF.truncate(content[r.bench][r.grp_b], 8000)
        assert not BENCH_NAME_RE.search(ta) and not BENCH_NAME_RE.search(tb)
        return {"task_a": {"text": ta}, "task_b": {"text": tb}}

    def key_of(st):
        blob = json.dumps({"m": JF.DEFAULT_MODEL, "q": PAIR_Q, "s": st}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]

    cache = pd.read_parquet(PAIR_CACHE) if PAIR_CACHE.exists() else pd.DataFrame(columns=["key", "p_a", "tokens", "model"])
    done = set(cache.key)
    keys, todo = [], []
    for r in P.itertuples():
        st = state_of(r)
        k = key_of(st)
        keys.append(k)
        if k not in done:
            todo.append((k, r))
    P["key"] = keys
    log(f"[e2b] {len(todo)} to query")
    new, t0 = [], time.time()

    def work(k, r):
        resp = call_questions(state_of(r), PAIR_Q, JF.DEFAULT_MODEL)
        a = resp["answers"]["harder"]
        return dict(key=k, p_a=float(a["probabilities"]["task_a"]), tokens=int(resp.get("usage", {}).get("input_tokens", 0)),
                    model=resp.get("model", ""))

    fails = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(work, k, r): k for k, r in todo}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                new.append(f.result())
            except Exception as e:
                fails += 1
                log(f"[e2b] failed: {str(e)[:120]}")
            if len(new) >= 1000 or i == len(futs):
                cache = pd.concat([cache, pd.DataFrame(new)], ignore_index=True).drop_duplicates("key", keep="last")
                cache.to_parquet(PAIR_CACHE.with_suffix(".tmp.parquet"), index=False)
                import os
                os.replace(PAIR_CACHE.with_suffix(".tmp.parquet"), PAIR_CACHE)
                new = []
                log(f"[e2b] {i}/{len(futs)} ({time.time()-t0:.0f}s)")
    P = P.merge(cache, on="key", how="left")
    P.drop(columns=[]).to_parquet(RUN / "e2b_pairs.parquet", index=False)
    out = {"n_pairs": int(len(P)), "n_missing": int(P.p_a.isna().sum()), "fails": fails,
           "tokens": int(P.tokens.fillna(0).sum()), "model": str(P.model.dropna().iloc[0])}
    out["usd"] = out["tokens"] * 0.042 / 1e6
    out["position_bias_mean_p_a"] = float(P.p_a.mean())
    res = {}
    for b in BENCHES:
        it = bench_items(b)
        q = P[(P.bench == b) & P.p_a.notna()]
        ug = np.unique(it.grp)
        gi = {g: k for k, g in enumerate(ug)}
        s = fit_bt(q.grp_a.map(gi).to_numpy(), q.grp_b.map(gi).to_numpy(), q.p_a.to_numpy(), len(ug))
        it = it.assign(bt=it.grp.map(dict(zip(ug, s))))
        pt_p, lo_p, hi_p, rp = boot_rho(it.jev, it.beta, it.grp, seed=500)
        pt_b, lo_b, hi_b, rb = boot_rho(it.bt, it.beta, it.grp, seed=500)   # same seed -> paired resamples
        d = rb - rp
        # pairwise agreement with pointwise: fraction of comparisons where the higher jev was judged harder
        ja = q.grp_a.map(dict(zip(it.grp, it.jev))).to_numpy(); jb = q.grp_b.map(dict(zip(it.grp, it.jev))).to_numpy()
        ne = ja != jb
        res[b] = dict(n=int(len(it)), n_pairs=int(len(q)), rho_pointwise=pt_p, ci_pointwise=[lo_p, hi_p],
                      rho_bt=pt_b, ci_bt=[lo_b, hi_b], diff=pt_b - pt_p, diff_ci=list(ci(d)),
                      rho_bt_vs_pointwise=rho(it.bt, it.jev),
                      pair_agree_with_pointwise=float(((q.p_a.to_numpy() > 0.5) == (ja > jb))[ne].mean()))
        it[["item_id", "bt"]].to_parquet(RUN / f"e2b_bt_{b}.parquet", index=False)
        log(f"[e2b] {b:18s} pointwise {pt_p:.3f} [{lo_p:.3f},{hi_p:.3f}]  BT {pt_b:.3f} [{lo_b:.3f},{hi_b:.3f}]  "
            f"diff {pt_b-pt_p:+.3f} {np.round(ci(d),3)}  rho(bt,jev)={res[b]['rho_bt_vs_pointwise']:.3f}")
    out["per_bench"] = res
    wins = sum(v["diff"] >= 0.05 for v in res.values())
    out["n_bench_bt_better_by_0.05"] = int(wins)
    out["decision"] = "USE_BT" if wins >= 3 else "KEEP_POINTWISE"
    log(f"[e2b decision] {out['decision']} ({wins}/5); tokens {out['tokens']:,} ~ ${out['usd']:.2f}; P(a) mean {out['position_bias_mean_p_a']:.3f}")
    json.dump(out, open(RUN / "e2b.json", "w", encoding="utf-8"), indent=1, default=float)
    return out


# =====================================================================================
# E2c -- distil jev_difficulty into a local frozen encoder + ridge head
# =====================================================================================
HF_CACHE = paths.ROOT / ".hf_cache"
STUDENT = "sentence-transformers/all-MiniLM-L6-v2"
FALLBACK = "sentence-transformers/all-mpnet-base-v2"
CHUNK_CHARS = 1000          # head chunk + tail chunk, each <= 256 word pieces; fixed before any result
MAX_TOK = 256
ALPHAS_C = [0.1, 1.0, 10.0, 100.0, 1000.0]
# retention rule fixed before results: aggregate (mean student rho / mean Jev rho over 5 LOBO folds)
# >= 0.70 AND >= 3 of 5 folds individually >= 0.70 (prereg says "on that fold"; mirrors the 3-of-4 ALC rule)
RETAIN = 0.70


def _hf_env():
    import os
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(HF_CACHE)
    os.environ["HF_HUB_CACHE"] = str(HF_CACHE / "hub")
    os.environ["XDG_CACHE_HOME"] = str(HF_CACHE / "xdg")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


class Encoder:
    """Frozen encoder; exactly the computation the packaged model.py performs at runtime."""

    def __init__(self, repo=STUDENT, threads=2):
        _hf_env()
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(threads)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(repo, cache_dir=str(HF_CACHE / "hub"), local_files_only=True)
        self.model = AutoModel.from_pretrained(repo, cache_dir=str(HF_CACHE / "hub"), local_files_only=True).eval()

    @staticmethod
    def chunks(text):
        t = (text or "").strip()
        return [t[:CHUNK_CHARS], t[-CHUNK_CHARS:] if len(t) > CHUNK_CHARS else t[:CHUNK_CHARS]]

    def encode_chunks(self, texts, bs=32):
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(texts), bs):
                b = self.tok(texts[i:i + bs], padding=True, truncation=True, max_length=MAX_TOK, return_tensors="pt")
                h = self.model(**b).last_hidden_state
                m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
                e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
                e = self.torch.nn.functional.normalize(e, p=2, dim=1)
                out.append(e.numpy().astype(np.float32))
        return np.concatenate(out) if out else np.zeros((0, self.model.config.hidden_size), np.float32)

    def embed(self, texts, bs=32):
        ch = [c for t in texts for c in self.chunks(t)]
        e = self.encode_chunks(ch, bs)
        return e.reshape(len(texts), -1)            # [head | tail]


def _tag(repo):
    return repo.split("/")[-1]


def embed_all(repo):
    enc = None
    out = {}
    for b in BENCHES:
        f = RUN / f"e2c_emb_{_tag(repo)}_{b}.npz"
        if f.exists():
            z = np.load(f)
            out[b] = (z["grp"], z["X"])
            continue
        enc = enc or Encoder(repo)
        it = bench_items(b).drop_duplicates("grp")
        t0 = time.time()
        X = enc.embed(it.content.tolist())
        log(f"[e2c] embedded {b}: {len(it)} contents in {time.time()-t0:.0f}s")
        np.savez(f, grp=it.grp.to_numpy(), X=X)
        out[b] = (it.grp.to_numpy(), X)
    return out


def _zb(v):
    return (v - v.mean()) / (v.std() + 1e-9)


def fit_head(Xs, ys, alpha=None, groups=None):
    """Ridge (sklearn) with benchmark-balanced weights; alpha by inner leave-one-benchmark-out."""
    from sklearn.linear_model import Ridge
    X = np.vstack(Xs); y = np.concatenate(ys)
    w = np.concatenate([np.full(len(v), 1.0 / len(v)) for v in ys])
    g = np.concatenate([np.full(len(v), k) for k, v in enumerate(ys)])
    if alpha is None:
        best = None
        for a in ALPHAS_C:
            err = 0.0
            for k in range(len(ys)):
                tr, te = g != k, g == k
                p = Ridge(alpha=a).fit(X[tr], y[tr], sample_weight=w[tr]).predict(X[te])
                err += float(np.mean((_zb(p) - y[te]) ** 2)) if p.std() > 0 else 2.0  # ranking-relevant error
            if best is None or err < best[1]:
                best = (a, err)
        alpha = best[0]
    return Ridge(alpha=alpha).fit(X, y, sample_weight=w), alpha


def lobo_student(emb):
    """For each held-out benchmark H: head on the other 4 (alpha by inner LOBO); held-out preds; plus
    inner-LOBO preds on each training benchmark (head refit without it, same alpha) for the slope and
    the deployable standardisation statistics."""
    tgt = {}
    for b in BENCHES:
        it = bench_items(b).drop_duplicates("grp").set_index("grp")
        tgt[b] = _zb(it.loc[emb[b][0], "jev"].to_numpy(float))
    res = {}
    for H in BENCHES:
        tr = [b for b in BENCHES if b != H]
        m, a = fit_head([emb[b][1] for b in tr], [tgt[b] for b in tr])
        inner = {}
        for b in tr:
            rest = [x for x in tr if x != b]
            mi, _ = fit_head([emb[x][1] for x in rest], [tgt[x] for x in rest], alpha=a)
            inner[b] = mi.predict(emb[b][1])
        res[H] = dict(alpha=a, pred=m.predict(emb[H][1]), inner=inner, coef=m.coef_, intercept=float(m.intercept_))
        log(f"[e2c] head H={H:18s} alpha={a}")
    return res, tgt


def item_level_prior(ep, H, q, li, lk, ly, rho0, mu, sr):
    """v1_1 item posterior (momd false-merge guard) with an item-difficulty prior:
    logit p = c_pair + mu_key + sr*x.  mu == 0, sr == sigma_beta reproduces v1_1 exactly."""
    from paiec.models import v1_1 as V
    sig, log_sig, GH_X, GH_W, LOG_GH_W = V.sig, V.log_sig, V.GH_X, V.GH_W, V.LOG_GH_W
    from paiec.models import ladder as Lad
    sb = H["sigma_beta"]
    c = Lad._invert_c(q, sb)
    K = len(ep.D["key_jev"])
    g_t = ep.inst_group[ep.t_inst] * K + ep.t_key
    prior_t = sig(c[ep.t_inst][:, None] + mu[ep.t_key][:, None] + sr * GH_X[None, :]) @ GH_W
    out = prior_t.copy()
    if not len(li):
        return out
    g_l = ep.inst_group[li] * K + lk
    uk, inv = np.unique(g_l, return_inverse=True)
    e = c[li][:, None] + mu[lk][:, None] + sr * GH_X[None, :]
    contrib = np.where(ly[:, None] > 0.5, log_sig(e), log_sig(-e))
    acc = np.zeros((len(uk), len(GH_X)))
    np.add.at(acc, inv, contrib)
    pos = np.minimum(np.searchsorted(uk, g_t), len(uk) - 1)
    has = uk[pos] == g_t
    if has.any():
        a = acc[pos[has]] + LOG_GH_W[None, :]
        wgt = np.exp(a - a.max(1, keepdims=True)); wgt /= wgt.sum(1, keepdims=True)
        p_same = (wgt * sig(c[ep.t_inst[has]][:, None] + mu[ep.t_key[has]][:, None] + sr * GH_X[None, :])).sum(1)
        if rho0 > 0:
            qq = np.clip(sig(e) @ GH_W, 1e-12, 1 - 1e-12)
            lind = np.zeros(len(uk))
            np.add.at(lind, inv, np.where(ly > 0.5, np.log(qq), np.log1p(-qq)))
            aall = acc + LOG_GH_W[None, :]
            amx = aall.max(1)
            lr = np.log(np.exp(aall - amx[:, None]).sum(1)) + amx - lind
            rg = V.mom_rho(li, lk, ep.inst_group[li], rho0, y=ly, qi=q)
            rho_k = np.array([rg[int(g)] for g in uk // K])
            ps = sig(np.log1p(-rho_k[pos[has]]) - np.log(rho_k[pos[has]]) + lr[pos[has]])
            p_same = ps * p_same + (1 - ps) * prior_t[has]
        out[has] = p_same
    return out


def predict_prior(ep, H, b, mu, sr):
    from paiec.models import v1_1 as V
    cfg = V.BASE2
    li, lk, ly = ep.labels(b)
    q, _ = V.pooled_pair(ep, H, li, ly, cfg)
    return np.clip(item_level_prior(ep, H, q, li, lk, ly, cfg["rho"], mu, sr), 0.005, 0.995)


def key_scores(D, score_by_item):
    """per runtime key: mean score of its item_ids (identical content -> identical score)."""
    K = len(D["key_jev"])
    s = np.full(K, np.nan)
    v = pd.Series(D["item_ids"]).map(score_by_item).to_numpy(float)
    ok = np.isfinite(v)
    acc = np.bincount(D["item_key"][ok], v[ok], K); cnt = np.bincount(D["item_key"][ok], minlength=K)
    s[cnt > 0] = acc[cnt > 0] / cnt[cnt > 0]
    return s


def easiness_slope(b, score_by_item):
    """slope of centred Rasch easiness (ladder.rasch) on within-benchmark z(score), over scored keys."""
    from paiec.models import ladder as Lad
    from paiec.evaluation import episodes as EP
    D = EP.compact(b)
    _, be, _ = Lad.rasch(D)
    s = key_scores(D, score_by_item)
    ok = np.isfinite(s)
    z = _zb(s[ok]); y = be[ok] - be[ok].mean()
    return float(np.dot(y, z) / np.dot(z, z))


def e2c(args):
    repo = getattr(args, "encoder", None) or STUDENT
    RUN.mkdir(parents=True, exist_ok=True)
    tag = _tag(repo)
    out = {"encoder": repo, "chunk_chars": CHUNK_CHARS, "max_tokens": MAX_TOK, "retain_rule": RETAIN}
    t0 = time.time()
    emb = embed_all(repo)
    st, tgt = lobo_student(emb)
    # ---------------------------------------------------------------- 1-2: agreement and rho retention
    rows = {}
    item_scores = {}
    for H in BENCHES:
        it = bench_items(H)
        gmap = dict(zip(emb[H][0], st[H]["pred"]))
        it = it.assign(stu=it.grp.map(gmap))
        item_scores[H] = it
        pj, lj, hj, rj = boot_rho(it.jev, it.beta, it.grp, seed=600)
        ps, ls, hs, rs = boot_rho(it.stu, it.beta, it.grp, seed=600)
        pa, la, ha, _ = boot_rho(it.stu, it.jev, it.grp, seed=601)
        rows[H] = dict(n=int(len(it)), alpha=st[H]["alpha"], rho_jev_beta=pj, ci_jev=[lj, hj], rho_stu_beta=ps, ci_stu=[ls, hs],
                       rho_stu_jev=pa, ci_stu_jev=[la, ha], retention=ps / pj if pj else float("nan"),
                       retention_ci=list(ci(rs / rj)))
        log(f"[e2c] {H:18s} rho(jev,beta)={pj:.3f} rho(stu,beta)={ps:.3f} [{ls:.3f},{hs:.3f}] rho(stu,jev)={pa:.3f} "
            f"retention={ps/pj:.2f}")
    agg = float(np.mean([r["rho_stu_beta"] for r in rows.values()]) / np.mean([r["rho_jev_beta"] for r in rows.values()]))
    nfold = int(sum(r["retention"] >= RETAIN for r in rows.values()))
    out["rho"] = rows
    out["retention_aggregate"] = agg
    out["retention_folds_ge"] = nfold
    out["retains"] = bool(agg >= RETAIN and nfold >= 3)
    log(f"[e2c] retention aggregate={agg:.3f}, folds>=0.70: {nfold}/5 -> retains={out['retains']}")
    # ---------------------------------------------------------------- 3: end-to-end (v1_1 + prior)
    from paiec.models import v1_1 as V
    from paiec.evaluation import episodes as EP
    e2e = {}
    for R, seeds in (("R2", list(range(5))), ("R1", list(range(100)))):
        res = {}
        for B in EP.EVAL_FOLDS:
            H = V.H_fold(B)
            tr = [b for b in BENCHES if b != B]
            inner_scores = {b: dict(zip(bench_items(b).item_id,
                                        bench_items(b).grp.map(dict(zip(emb[b][0], st[B]["inner"][b]))))) for b in tr}
            gam_s = float(np.mean([easiness_slope(b, inner_scores[b]) for b in tr]))
            # deployable standardisation stats: benchmark-balanced mean, mean within-benchmark sd (inner-LOBO preds)
            m_tr = float(np.mean([st[B]["inner"][b].mean() for b in tr]))
            s_tr = float(np.mean([st[B]["inner"][b].std() for b in tr]))
            D = EP.compact(B)
            sc = key_scores(D, dict(zip(item_scores[B].item_id, item_scores[B].stu)))
            okk = np.isfinite(sc)
            zA = np.zeros(len(sc)); zA[okk] = _zb(sc[okk])
            zB = np.zeros(len(sc)); zB[okk] = (sc[okk] - m_tr) / s_tr
            jz = D["key_jev"]; okj = np.isfinite(jz)
            zJ = np.zeros(len(jz)); zJ[okj] = _zb(jz[okj])
            sb = H["sigma_beta"]
            sr_s = math.sqrt(max(sb ** 2 - gam_s ** 2, 0.25))
            variants = {"v1_1": (np.zeros(len(sc)), sb), "v1_1+jev": (H["gamma"] * zJ, H["sigma_r"]),
                        "v1_1+stuA": (gam_s * zA, sr_s), "v1_1+stuB": (gam_s * zB, sr_s)}
            ep = EP.build(D, R, seeds)
            rr = {}
            for nm, (mu, sr) in variants.items():
                probs = {b: predict_prior(ep, H, b, mu, sr) for b in EP.BUDGETS}
                br, ec = ep.score(probs)
                rr[nm] = dict(br=br, ec=ec, alc=br @ EP.W, pair=ep.inst_pair.copy())
            # exactness check: zero prior == packaged v1_1 engine
            if R == "R2":
                ref = V.run_cfgs(ep, H, {"x": V.BASE2})["x"]["alc"]
                e2e.setdefault("max_abs_diff_vs_v1_1", 0.0); e2e["max_abs_diff_vs_v1_1"] = max(e2e["max_abs_diff_vs_v1_1"], float(np.abs(ref - rr["v1_1"]["alc"]).max())); assert e2e["max_abs_diff_vs_v1_1"] < 1e-9, "prior engine != v1_1 at mu=0"
            res[B] = rr
            e2e.setdefault("fold_params", {}).setdefault(R, {})[B] = dict(gamma_student=gam_s, gamma_jev=H["gamma"],
                                                                         sr_student=sr_s, m_tr=m_tr, s_tr=s_tr,
                                                                         heldout_mean_zB=float(zB[okk].mean()),
                                                                         heldout_sd_zB=float(zB[okk].std()))
            log(f"[e2c] {R} {B} done ({time.time()-t0:.0f}s) gamma_s={gam_s:.3f} gamma_j={H['gamma']:.3f} "
                f"zB mean={zB[okk].mean():+.2f} sd={zB[okk].std():.2f}")
        o = {}
        for nm in ("v1_1", "v1_1+jev", "v1_1+stuA", "v1_1+stuB"):
            o[nm] = dict(alc=V.mean_alc(res, nm), curve=V.mean_curve(res, nm).tolist())
            if nm != "v1_1":
                d = V.delta(res, nm, "v1_1")
                d["budget"] = [V.delta(res, nm, "v1_1", budget=j) for j in range(6)]
                pf = list(d["per_fold"].values())
                d["threshold"] = bool(d["pt"] <= -1.0 and d["hi"] < 0 and sum(x < 0 for x in pf) >= 3 and max(pf) <= 1.0)
                o[nm]["d_v1_1"] = d
                log(f"[e2c] {R} {nm:10s} ALC={o[nm]['alc']:.4f} d={V.fmt(d)} threshold={d['threshold']}")
        e2e[R] = o
    out["e2e"] = e2e
    # ---------------------------------------------------------------- 4: runtime
    enc = Encoder(repo, threads=1)
    sample = bench_items("multi_swebench").content.head(40).tolist() + bench_items("researchcodebench").content.head(10).tolist()
    enc.embed(sample[:2], bs=1)
    t1 = time.perf_counter()
    for s in sample:
        enc.embed([s], bs=1)
    ms = 1000 * (time.perf_counter() - t1) / len(sample)
    import os
    snap = next((HF_CACHE / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots").iterdir())
    size_mb = sum(os.path.getsize(p) for p in snap.rglob("*") if p.is_file()) / 1e6
    out["runtime"] = dict(ms_per_item_1thread=ms, model_files_mb=size_mb,
                          est_hours_1000_items_x6_budgets_no_cache=ms * 1000 * 6 / 3.6e6)
    log(f"[e2c] runtime {ms:.1f} ms/item (1 thread, batch 1); model files {size_mb:.0f} MB")
    runtime_ok = ms < 50
    # ---------------------------------------------------------------- decision (mechanical)
    dA = e2e["R2"]["v1_1+stuB"]["d_v1_1"]   # the deployable variant is what would ship
    if out["retains"] and dA["threshold"] and runtime_ok:
        row = "SHIP"
    elif out["retains"]:
        row = "RESEARCH_ONLY"
    else:
        row = "FALLBACK" if repo == STUDENT else "STOP"
    out["decision"] = dict(row=row, runtime_ok=runtime_ok,
                           prereg_variant_A_threshold_R2=e2e["R2"]["v1_1+stuA"]["d_v1_1"]["threshold"],
                           deployable_variant_B_threshold_R2=dA["threshold"])
    log(f"[e2c decision] {out['decision']} ({time.time()-t0:.0f}s)")
    # head for packaging (fit on ALL public benchmarks), saved only as a research artefact here
    m_all, a_all = fit_head([emb[b][1] for b in BENCHES], [tgt[b] for b in BENCHES])
    np.savez(RUN / f"e2c_head_{tag}_all5.npz", coef=m_all.coef_, intercept=m_all.intercept_, alpha=a_all)
    json.dump(out, open(RUN / f"e2c_{tag}.json", "w", encoding="utf-8"), indent=1, default=float)
    return out


# =====================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["e2a", "e2b", "e2c"])
    ap.add_argument("--encoder", default=None, help="HF repo of the frozen student encoder (default MiniLM)")
    a = ap.parse_args()
    globals()[a.cmd](a)


if __name__ == "__main__":
    main()
