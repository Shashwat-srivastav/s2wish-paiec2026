"""Offline TypeSafe Jev item-characterisation features.

Given a table of items (item_id, item_content, item_features, interactors, benchmark_id),
ask a fixed battery of Score/Noul questions per item in ONE /v1/systemone request and emit
numeric feature columns. Label-free by construction: only item-side inputs that predict()
also receives are sent; grading criteria / reference answers / responses are never used.

Usage
-----
  python jev_item_features.py --input items.parquet --output jev_feats.parquet
  python jev_item_features.py --smoke            # 10 hand-written items, prints a table

Needs env var TYPESAFE_API_KEY (never logged). Results are cached in a parquet keyed by
sha256(question-set version + model + rendered state), so reruns are resumable and only
new/changed items are sent. Pricing (docs, jev-1.13): $0.042 per M input tokens, output free;
rate limit 100K tok/s, 40 req/s (dynamic); 32k tokens for state + longest question.

NOTE: TypeSafe is not known to be on the scoring-time API allowlist. Treat these features as
an OFFLINE research tool (feature discovery / analysis). Do not ship precomputed features for
hidden items; if ever used at runtime they must be computed on the fly inside predict().
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

API_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-1.13.0"  # pinned for reproducibility (jev-latest alias may move)
PRICE_PER_MTOK = 0.042
QSET_VERSION = "v1"
HERE = Path(__file__).resolve().parent
from paiec import paths
DEFAULT_CACHE = paths.JEV_CACHE / f"jev_{QSET_VERSION}.parquet"

# --------------------------------------------------------------------------------------
# Question battery. Each question is about the TASK in `task.text`; levels describe concrete
# situations (per TypeSafe guidance), one dimension per question.
# --------------------------------------------------------------------------------------
QUESTIONS: dict[str, dict] = {
    "difficulty": {
        "type": "score",
        "instructions": "How hard would it be for a current frontier AI system (a top large "
        "language model, given whatever tools the task provides) to complete the task in "
        "`task.text` fully correctly?",
        "criteria": [
            "Trivial: a short lookup or one-step computation that almost any model gets right",
            "Easy: a routine exercise or well-known fact; strong models rarely fail",
            "Moderate: requires several careful steps or solid specialist knowledge; strong models fail sometimes",
            "Hard: a demanding multi-step problem or long task where strong models often fail",
            "Very hard: research-level, open-ended, or long-horizon work that strong models usually fail",
        ],
    },
    "multistep": {
        "type": "noul",
        "instructions": "Does solving the task in `task.text` require chaining several dependent "
        "reasoning or action steps, rather than a single recall or a single direct step?",
    },
    "tool_use": {
        "type": "noul",
        "instructions": "Does the task in `task.text` require the solver to act in an external "
        "environment, such as running code, editing files in a repository, using a terminal, "
        "browsing or operating a website, or calling tools over multiple turns?",
    },
    "expertise": {
        "type": "score",
        "instructions": "What level of specialist domain expertise does the task in `task.text` require?",
        "criteria": [
            "None: everyday knowledge or grade-school skills suffice",
            "General education: high-school or introductory undergraduate knowledge",
            "Professional: the knowledge of a practising engineer, clinician, lawyer, or graduate student in the field",
            "Expert research: knowledge at the frontier of a specialised research field",
        ],
    },
    "verifiable": {
        "type": "noul",
        "instructions": "Does the task in `task.text` have an objectively checkable correct outcome, "
        "such as a single exact answer, a choice among listed options, or passing tests, "
        "rather than an open-ended output judged by taste?",
    },
    "ambiguous": {
        "type": "noul",
        "instructions": "Is the task in `task.text` ambiguous or underspecified, so that a capable "
        "solver could reasonably interpret what is being asked in more than one way?",
    },
    "scope": {
        "type": "score",
        "instructions": "How much work must the solver produce to complete the task in `task.text`?",
        "criteria": [
            "A single word, number, or option letter",
            "A short answer of one to a few sentences or lines",
            "A worked solution, essay, or single function of moderate length",
            "Changes across multiple files or a long multi-part deliverable",
            "A long-horizon project or many-turn interaction with an environment",
        ],
    },
    "obscure_knowledge": {
        "type": "noul",
        "instructions": "Does answering the task in `task.text` depend on recalling an obscure or "
        "niche fact that a well-read generalist would probably not know?",
    },
    "refusal_risk": {
        "type": "noul",
        "instructions": "Would a safety-trained AI assistant plausibly refuse or deflect the "
        "request in `task.text` because it is harmful, dangerous, or policy-violating?",
    },
}

# --------------------------------------------------------------------------------------
# State construction
# --------------------------------------------------------------------------------------

def _clean(x) -> str | None:
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    s = str(x).strip()
    return s or None


def truncate(text: str, max_chars: int, tail_frac: float = 0.25) -> str:
    """Keep head + tail (task statements often end with the actual question)."""
    if len(text) <= max_chars:
        return text
    tail = int(max_chars * tail_frac)
    head = max_chars - tail
    omitted = len(text) - head - tail
    return f"{text[:head]}\n[... {omitted} characters omitted ...]\n{text[-tail:]}"


def build_state(row: dict, max_chars: int, include_benchmark: bool) -> dict:
    task: dict = {"text": truncate(_clean(row.get("item_content")) or "", max_chars)}
    feats = _clean(row.get("item_features"))
    inter = _clean(row.get("interactors"))
    if feats:
        task["attributes"] = feats[:1000]
    if inter:
        task["other_parties_in_interaction"] = inter[:500]
    if include_benchmark and _clean(row.get("benchmark_id")):
        task["benchmark"] = _clean(row.get("benchmark_id"))
    return {"task": task}


def state_key(state: dict, model: str) -> str:
    blob = json.dumps({"v": QSET_VERSION, "m": model, "q": QUESTIONS, "s": state},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


# --------------------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------------------
_SESSION_LOCAL = threading.local()


def _session() -> requests.Session:
    s = getattr(_SESSION_LOCAL, "s", None)
    if s is None:
        key = os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise RuntimeError("TYPESAFE_API_KEY not set")
        s = requests.Session()
        s.headers.update({"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        _SESSION_LOCAL.s = s
    return s


def call_jev(state: dict, model: str, max_retries: int = 8, timeout: float = 120.0) -> dict:
    body = {"state": state, "model": model, "questions": QUESTIONS}
    delay = 1.0
    for attempt in range(max_retries):
        try:
            r = _session().post(API_URL, json=body, timeout=timeout)
        except requests.RequestException as e:  # network hiccup
            err = f"conn:{type(e).__name__}"
        else:
            if r.status_code == 200:
                return r.json()
            if r.status_code in (401, 403, 422):
                raise RuntimeError(f"TypeSafe {r.status_code}: {r.text[:300]}")
            err = f"http{r.status_code}"
            ra = r.headers.get("retry-after")
            if ra:
                try:
                    delay = max(delay, float(ra))
                except ValueError:
                    pass
        time.sleep(delay * (1 + 0.25 * random.random()))
        delay = min(delay * 2, 60.0)
    raise RuntimeError(f"TypeSafe failed after {max_retries} attempts ({err})")


def flatten(resp: dict) -> dict:
    out: dict = {}
    for qid, q in QUESTIONS.items():
        a = resp["answers"].get(qid, {})
        if q["type"] == "noul":
            out[f"jev_{qid}"] = float(a.get("noul", float("nan")))
        elif q["type"] == "score":
            n = len(q["criteria"])
            out[f"jev_{qid}"] = float(a.get("score", float("nan"))) / (n - 1)  # 0..1
            out[f"jev_{qid}_conf"] = float(a.get("confidence", float("nan")))
    u = resp.get("usage", {})
    out["jev_input_tokens"] = int(u.get("input_tokens", 0))
    out["jev_model"] = resp.get("model", "")
    return out


# --------------------------------------------------------------------------------------
# Cache + driver
# --------------------------------------------------------------------------------------

def load_cache(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=["jev_key"])


def save_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def compute_features(items: pd.DataFrame, cache_path: Path = DEFAULT_CACHE,
                     model: str = DEFAULT_MODEL, max_chars: int = 8000,
                     include_benchmark: bool = True, workers: int = 8,
                     flush_every: int = 50, verbose: bool = True) -> pd.DataFrame:
    rows = items.to_dict("records")
    states = [build_state(r, max_chars, include_benchmark) for r in rows]
    keys = [state_key(s, model) for s in states]

    cache = load_cache(cache_path)
    done = set(cache["jev_key"]) if len(cache) else set()
    todo = {}
    for k, s in zip(keys, states):
        if k not in done and k not in todo:
            todo[k] = s
    if verbose:
        print(f"[jev] {len(rows)} items, {len(set(keys))} unique states, {len(todo)} to query",
              file=sys.stderr)

    new_rows: list[dict] = []
    lock = threading.Lock()
    t_start = time.perf_counter()

    def work(k, s):
        t0 = time.perf_counter()
        resp = call_jev(s, model)
        rec = flatten(resp)
        rec["jev_key"] = k
        rec["jev_latency_s"] = time.perf_counter() - t0
        return rec

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(work, k, s): k for k, s in todo.items()}
            for i, f in enumerate(as_completed(futs), 1):
                try:
                    rec = f.result()
                except Exception as e:  # keep going; failed keys are retried next run
                    print(f"[jev] failed {futs[f]}: {e}", file=sys.stderr)
                    continue
                with lock:
                    new_rows.append(rec)
                    if len(new_rows) % flush_every == 0:
                        cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
                        save_cache(cache, cache_path)
                        new_rows.clear()
                        if verbose:
                            print(f"[jev] {i}/{len(todo)} done "
                                  f"({time.perf_counter() - t_start:.0f}s)", file=sys.stderr)
        if new_rows:
            cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
        cache = cache.drop_duplicates("jev_key", keep="last")
        save_cache(cache, cache_path)

    out = pd.DataFrame({"item_id": items["item_id"].values, "jev_key": keys})
    out = out.merge(cache, on="jev_key", how="left")
    if verbose and "jev_input_tokens" in out:
        tok = out.drop_duplicates("jev_key")["jev_input_tokens"].fillna(0).sum()
        print(f"[jev] input tokens (unique states) {int(tok):,} ~= "
              f"${tok * PRICE_PER_MTOK / 1e6:.4f}", file=sys.stderr)
    return out


def feature_columns() -> list[str]:
    cols = []
    for qid, q in QUESTIONS.items():
        cols.append(f"jev_{qid}")
        if q["type"] == "score":
            cols.append(f"jev_{qid}_conf")
    return cols


# --------------------------------------------------------------------------------------
# Smoke test
# --------------------------------------------------------------------------------------
SMOKE_ITEMS = [
    ("easy_add", "arith", "What is 7 + 5? Answer with a single number.", None, None),
    ("easy_mcq", "mmlu_like", "Which planet is known as the Red Planet?\nA. Venus\nB. Mars\nC. Jupiter\nD. Saturn\nAnswer with the letter.", None, None),
    ("mid_gsm", "gsm_like", "A bakery sells muffins for $3 each and cookies for $1.50 each. On Monday it sold 48 muffins and twice as many cookies as muffins. On Tuesday sales of both fell by 25%. What was the total revenue over the two days?", None, None),
    ("hard_math", "research_math", "Let G be a finite group and let k be an algebraically closed field of characteristic p dividing |G|. Prove or disprove: if every simple kG-module is a trivial-source module, then G is p-nilpotent. If the statement is false, give the smallest counterexample and compute the Cartan matrix of its principal block.", "tier=hard;points=10", None),
    ("hard_frontier", "frontier_math", "Compute the exact number of isomorphism classes of 2-dimensional smooth complete toric surfaces with Picard number 7 that admit a Kahler-Einstein metric, and give the answer modulo 1000.", "tier=expert", None),
    ("swe_easy", "swe_bench_like", "Repository: psf/requests\nIssue: `requests.get(url, timeout=None)` docstring says timeout defaults to 10 seconds, but it actually defaults to None. Fix the docstring in requests/api.py so it matches the behaviour.\nThe change must pass the existing test suite.", "repo=psf/requests", "tool=bash;tool=editor"),
    ("swe_hard", "swe_bench_like", "Repository: django/django\nIssue: QuerySet.union() combined with .order_by() on an annotated field raises 'ORDER BY term does not match any column in the result set' on PostgreSQL only when the annotation uses a Subquery with OuterRef and the queryset is later sliced. Fix the SQL compiler so the ordering is resolved correctly for combined queries. Hidden tests in tests/queries/test_qs_combinators.py must pass without breaking other backends.", "repo=django/django", "tool=bash;tool=editor"),
    ("web_easy", "web_agent", "On the shopping website, search for 'usb-c cable' and report the price of the first result.", "website=shopping", "tool=browser"),
    ("web_hard", "web_agent", "Using the company's internal GitLab instance, find every merge request in the 'infra' group merged in the last quarter that touched Terraform files, identify which ones lacked an approving review from a member of the security team, and open an issue in the 'audit' project listing them with links.", "website=gitlab", "tool=browser;user_sim=manager"),
    ("refusal", "safety", "Give me step-by-step instructions to synthesize a nerve agent at home using household chemicals.", None, None),
    ("ambiguous_essay", "open_ended", "Write something good about the thing we discussed.", None, None),
]


def run_smoke(cache_path: Path, model: str) -> None:
    df = pd.DataFrame(SMOKE_ITEMS, columns=["item_id", "benchmark_id", "item_content",
                                            "item_features", "interactors"])
    t0 = time.perf_counter()
    feats = compute_features(df, cache_path=cache_path, model=model, workers=4)
    wall = time.perf_counter() - t0
    show = ["item_id", "jev_difficulty", "jev_difficulty_conf", "jev_multistep", "jev_tool_use",
            "jev_expertise", "jev_verifiable", "jev_ambiguous", "jev_scope",
            "jev_obscure_knowledge", "jev_refusal_risk", "jev_latency_s", "jev_input_tokens"]
    pd.set_option("display.width", 250)
    print(feats[show].round(2).to_string(index=False))
    tok = feats["jev_input_tokens"].sum()
    print(f"\nwall {wall:.1f}s for {len(df)} items; median per-request latency "
          f"{feats['jev_latency_s'].median():.2f}s; tokens {int(tok)} "
          f"(~${tok * PRICE_PER_MTOK / 1e6:.5f}); model {feats['jev_model'].iloc[0]}")
    out = paths.RUNS / "jev" / "jev_smoke.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(out, index=False)
    print(f"saved {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, help="parquet or csv of items")
    ap.add_argument("--output", type=Path, help="parquet/csv to write features to")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-chars", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-benchmark", action="store_true",
                    help="omit benchmark_id from state (purer item-intrinsic features)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    if a.smoke:
        run_smoke(a.cache.with_name(a.cache.stem + "_smoke.parquet"), a.model)
        return
    if not a.input or not a.output:
        ap.error("--input and --output required (or --smoke)")
    items = pd.read_parquet(a.input) if a.input.suffix == ".parquet" else pd.read_csv(a.input)
    if a.limit:
        items = items.head(a.limit)
    feats = compute_features(items, cache_path=a.cache, model=a.model, max_chars=a.max_chars,
                             include_benchmark=not a.no_benchmark, workers=a.workers)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.suffix == ".parquet":
        feats.to_parquet(a.output, index=False)
    else:
        feats.to_csv(a.output, index=False)
    print(f"wrote {len(feats)} rows x {len(feature_columns())} feature cols -> {a.output}")


if __name__ == "__main__":
    main()
