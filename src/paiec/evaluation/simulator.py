"""Local replica of the PAIEC streaming evaluator (protocol ``streaming_alc_v1``).

What is official and what is ours
---------------------------------
The organizers ship only the *client* half of the streaming evaluator
(``external/starting_kit/tools/streaming_ingestion.py::run_streaming`` plus
``prediction_workers.py``). The *coordinator* (eligibility, the persistent
50/50 split, candidate order, which labels are revealed, scoring) runs on the
organizer service and is not shipped. Therefore:

* Client semantics are reused from the official code wherever possible:
  ``AcquisitionHook`` (signature dispatch, deep copies, native-bool check),
  ``prediction_input_key`` / ``group_prediction_inputs`` (one predict() call
  per distinct input per checkpoint), and the smoke test's probability
  validator / input builder. The default random policy (no labeling.py) is a
  verbatim copy of the 3 lines inside ``run_streaming``.
* Coordinator semantics are implemented once, in ``_engine``, from
  starting_kit/README.md. ``LocalCoordinator`` exposes that engine over the
  exact wire protocol so the *unmodified official* ``run_streaming`` can be
  driven locally (``run_official_client``). ``evaluate`` drives the same engine
  in-process (fast path). test_simulator.py proves both give identical scores.

Coordinator assumptions (README-derived; server code not available)
--------------------------------------------------------------------
1. A subject-benchmark pair is eligible iff it has >= 80 DISTINCT items
   (response runs do not count). Ineligible pairs: ``ineligible="skip"``
   (default, not scored) or ``"eval_all"`` (all rows scored, no acquisition).
2. Persistent random 50/50 split over distinct items; odd count -> extra item
   to acquisition; all response runs of an item stay in its pool. The real
   assignment is a server secret; we use a sha256-seeded shuffle.
3. Acquisition candidates = distinct acquisition items in a fixed random
   order. Accepting reveals ONE recorded response of the item
   (``reveal="first"`` data-order row by default, or ``"hash"``).
4. Context = {subject_id, benchmark_id, labels_remaining = 31 - acquired,
   labels_acquired, max_labels=31, items_remaining (incl. current)}.
5. One trajectory, checkpoints 0,1,3,7,15,31. Budget 0 is evaluated with
   labeled=[]. Before checkpoint b each pair streams candidates until it holds
   min(b,31) labels or its stream is exhausted (rejections are permanent; no
   forced acceptance; shortfall carries to later caps). All pairs finish a
   checkpoint before any pair gets more labels. Pairs are advanced one after
   another in sorted (subject_id, benchmark_id) order (true interleaving is
   unknown; it only matters for predictors/hooks that use OTHER pairs' labels).
6. ``labeled`` = all evidence purchased so far across all pairs, in purchase
   order, as [[subject, item], y]; frozen per checkpoint for evaluation.
7. Every checkpoint scores all stored response rows of every pair's
   evaluation pool; Brier per pair per budget; ALC weights 0.1/0.2x4/0.1;
   overall = unweighted mean over pairs. ECE: 10 fixed bins, count-weighted.
8. Legacy one-argument hooks: every acquisition candidate is scored once with
   no labels; top-n (ties -> stream order) per pair revealed at each cap.
9. Optional draw limit (~1000 distinct items, complete groups, prefer
   under-represented benchmarks/subjects, random ties, skip groups that do
   not fit): ``draw_limit=None`` disables it (default).
"""
from __future__ import annotations

import contextlib
import copy
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import statistics
import sys
import tempfile
import time
from collections import defaultdict, deque
from pathlib import Path

from paiec.evaluation.metric import ALC_WEIGHTS, BUDGETS, brier_alc, brier_score, ece

from paiec.paths import KIT_TOOLS, ROOT as AIMS_ROOT  # official client lives in external/starting_kit
if str(KIT_TOOLS) not in sys.path:
    sys.path.insert(0, str(KIT_TOOLS))

import streaming_ingestion as _official_ingestion  # noqa: E402
from streaming_ingestion import AcquisitionHook  # noqa: E402
from prediction_workers import group_prediction_inputs, prediction_input_key  # noqa: E402
from run_smoke_test import (  # noqa: E402
    ITEM_FIELDS, SUBJECT_FIELDS, _assert_probability, _build_input,
)

PROTOCOL = "streaming_alc_v1"
MAX_LABELS = 31
MIN_ELIGIBLE_ITEMS = 80


def validate_score(value):
    """Same validator the official smoke test passes to the ingestion code."""
    return _assert_probability(value, "predict()")


def _worker_score(value):
    """Check done inside an official prediction worker before the parent re-validates."""
    score = float(value)
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError(f"predict() must return a float in [0, 1]: got {score}")
    return score


def _random_query(context, current_input):
    """Verbatim copy of the default (no labeling.py) policy in run_streaming."""
    key = json.dumps([context, current_input], sort_keys=True, separators=(",", ":"))
    uniform = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2 ** 64
    return uniform < min(1, context["labels_remaining"] / context["items_remaining"])


# ---------------------------------------------------------------------------
# Rows -> pairs
# ---------------------------------------------------------------------------
def _s(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return str(v)


def row_input(row: dict) -> list:
    return [{k: _s(row.get(k)) for k in SUBJECT_FIELDS}, {k: _s(row.get(k)) for k in ITEM_FIELDS}]


def _subject_id(row: dict) -> str:
    sid = row.get("subject_id")
    if sid:
        return str(sid)
    key = json.dumps([row_input(row)[0]], sort_keys=True)
    return "sim_" + hashlib.sha256(key.encode()).hexdigest()[:12]


def _hseed(*parts) -> int:
    return int(hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()[:16], 16)


def build_pairs(rows: list[dict]) -> dict[tuple, list[dict]]:
    """Group response rows by (subject_id, benchmark_id)."""
    pairs = defaultdict(list)
    for r in rows:
        pairs[(_subject_id(r), _s(r.get("benchmark_id")))].append(r)
    return dict(pairs)


def split_pair(rows: list[dict], seed: int = 0):
    """(acquisition_item_ids, eval_item_ids) or None if < 80 distinct items."""
    item_ids = sorted({str(r["item_id"]) for r in rows})
    if len(item_ids) < MIN_ELIGIBLE_ITEMS:
        return None
    sid, bid = _subject_id(rows[0]), _s(rows[0].get("benchmark_id"))
    shuffled = item_ids[:]
    random.Random(_hseed("split", sid, bid, seed)).shuffle(shuffled)
    cut = (len(shuffled) + 1) // 2  # odd -> extra item to acquisition
    return set(shuffled[:cut]), set(shuffled[cut:])


class PairData:
    __slots__ = ("subject_id", "benchmark_id", "acq", "eval_rows", "n_items", "eligible")

    def __init__(self, subject_id, benchmark_id, acq, eval_rows, n_items, eligible):
        self.subject_id, self.benchmark_id = subject_id, benchmark_id
        self.acq, self.eval_rows = acq, eval_rows
        self.n_items, self.eligible = n_items, eligible

    @property
    def key(self):
        return (self.subject_id, self.benchmark_id)


def _rec(r):
    return {"item_id": str(r["item_id"]), "input": row_input(r), "y": int(r["y"])}


def prepare_pairs(rows, seed=0, ineligible="skip", reveal="first", draw_limit=None):
    if ineligible not in ("skip", "eval_all"):
        raise ValueError("ineligible must be 'skip' or 'eval_all'")
    groups = build_pairs(rows)
    if draw_limit is not None:
        keep = draw_groups(groups, draw_limit, seed, ineligible)
        groups = {k: groups[k] for k in keep}
    out = []
    for (sid, bid), prow in sorted(groups.items()):
        n_items = len({str(r["item_id"]) for r in prow})
        split = split_pair(prow, seed=seed)
        if split is None:
            if ineligible == "skip":
                continue
            out.append(PairData(sid, bid, [], [_rec(r) for r in prow], n_items, False))
            continue
        acq_ids, eval_ids = split
        by_item = defaultdict(list)
        eval_rows = []
        for r in prow:
            if str(r["item_id"]) in acq_ids:
                by_item[str(r["item_id"])].append(r)
            else:
                eval_rows.append(_rec(r))
        order = sorted(acq_ids)
        random.Random(_hseed("order", sid, bid, seed)).shuffle(order)
        acq = []
        for iid in order:
            runs = by_item[iid]
            if reveal == "first":
                chosen = runs[0]
            elif reveal == "hash":
                chosen = runs[_hseed("reveal", sid, bid, iid, seed) % len(runs)]
            else:
                raise ValueError("reveal must be 'first' or 'hash'")
            acq.append(_rec(chosen))
        out.append(PairData(sid, bid, acq, eval_rows, n_items, True))
    return out


def draw_groups(groups, limit=1000, seed=0, ineligible="skip"):
    """Approximate the platform draw: complete (subject, benchmark) groups under a
    distinct-item limit, preferring under-represented benchmarks then subjects,
    random tie-break, skipping groups that no longer fit."""
    sizes = {k: len({str(r["item_id"]) for r in v}) for k, v in groups.items()}
    if ineligible == "skip":
        sizes = {k: n for k, n in sizes.items() if n >= MIN_ELIGIBLE_ITEMS}
    rng = random.Random(_hseed("draw", seed))
    tiebreak = {k: rng.random() for k in sorted(sizes)}
    bench_n, subj_n = defaultdict(int), defaultdict(int)
    remaining, chosen, pool = limit, [], set(sizes)
    while pool:
        k = min(pool, key=lambda g: (bench_n[g[1]], subj_n[g[0]], tiebreak[g]))
        pool.discard(k)
        if sizes[k] > remaining:
            continue
        chosen.append(k)
        remaining -= sizes[k]
        bench_n[k[1]] += 1
        subj_n[k[0]] += 1
    return chosen


# ---------------------------------------------------------------------------
# Coordinator engine (single source of truth for server-side semantics)
# ---------------------------------------------------------------------------
def _new_result(pairs):
    return {"pairs": pairs, "targets": [], "spans": [], "probabilities": {},
            "labels": {}, "acquired_items": {}, "mode": None,
            "acquisition_calls": 0, "rank_calls": 0}


def _engine(pairs, res, mode="stream", budgets=BUDGETS, max_labels=MAX_LABELS):
    """Generator: yields coordinator events, receives client payloads."""
    labeled = []
    for p in pairs:
        start = len(res["targets"])
        res["targets"].extend({"input": r["input"]} for r in p.eval_rows)
        res["spans"].append((start, len(res["targets"])))
    targets = res["targets"]
    state = [{"idx": 0, "acquired": 0, "items": []} for _ in pairs]
    ranking = None
    for b in budgets:
        if b > 0:
            cap = min(b, max_labels)
            if mode == "legacy_rank":
                if ranking is None:
                    ranking = []
                    for p in pairs:
                        scores = []
                        for c in p.acq:
                            resp = yield {"type": "rank", "input": c["input"], "labeled": []}
                            res["rank_calls"] += 1
                            score = resp["score"]
                            if isinstance(score, bool) or not isinstance(score, (int, float)) \
                                    or not math.isfinite(score):
                                raise ValueError("rank score must be finite numeric")
                            scores.append(float(score))
                        ranking.append(sorted(range(len(scores)), key=lambda i: (-scores[i], i)))
                for p, st, rank in zip(pairs, state, ranking):
                    while st["acquired"] < min(cap, len(p.acq)):
                        c = p.acq[rank[st["acquired"]]]
                        labeled.append([c["input"], c["y"]])
                        st["items"].append(c["item_id"])
                        st["acquired"] += 1
            else:
                for p, st in zip(pairs, state):
                    while st["acquired"] < cap and st["idx"] < len(p.acq):
                        c = p.acq[st["idx"]]
                        context = {
                            "subject_id": p.subject_id, "benchmark_id": p.benchmark_id,
                            "labels_remaining": max_labels - st["acquired"],
                            "labels_acquired": st["acquired"], "max_labels": max_labels,
                            "items_remaining": len(p.acq) - st["idx"],
                        }
                        resp = yield {"type": "acquire", "input": c["input"],
                                      "context": context, "labeled": labeled}
                        res["acquisition_calls"] += 1
                        st["idx"] += 1  # rejected candidates are skipped permanently
                        query = resp["query"]
                        if type(query) is not bool:
                            raise ValueError("acquire reply must be a bool")
                        if query:
                            labeled.append([c["input"], c["y"]])
                            st["items"].append(c["item_id"])
                            st["acquired"] += 1
        res["labels"][b] = [st["acquired"] for st in state]
        res["acquired_items"][b] = [list(st["items"]) for st in state]
        resp = yield {"type": "evaluate", "budget": b, "targets": targets, "labeled": labeled}
        probs = resp["probabilities"]
        if len(probs) != len(targets):
            raise ValueError("probabilities length mismatch")
        res["probabilities"][b] = [validate_score(v) for v in probs]
        if b == budgets[0] and resp.get("acquisition_mode") == "legacy_rank":
            mode = "legacy_rank"
    res["mode"] = mode


def _wire(obj):
    return json.loads(json.dumps(obj, allow_nan=False))


class LocalCoordinator:
    """Speaks streaming_alc_v1 so the unmodified official run_streaming can run."""

    def __init__(self, pairs, budgets=BUDGETS, max_labels=MAX_LABELS, send_configure=True):
        self.pairs, self.budgets, self.max_labels = pairs, budgets, max_labels
        self.send_configure = send_configure
        self.res = _new_result(pairs)
        self._gen, self._eid, self._pending, self._stage = None, 0, None, "start"
        self.mode = "stream"

    def _emit(self, event):
        self._eid += 1
        self._pending = self._eid
        return _wire(dict(event, protocol=PROTOCOL, event_id=self._eid))

    def _start(self):
        self._stage = "running"
        self._gen = _engine(self.pairs, self.res, self.mode, self.budgets, self.max_labels)
        return self._emit(next(self._gen))

    def exchange(self, payload):
        payload = _wire(payload)
        if self._stage == "start":
            if payload != {}:
                raise ValueError("first exchange must be empty")
            if self.send_configure:
                self._stage = "configuring"
                return self._emit({"type": "configure"})
            return self._start()
        if payload.get("event_id") != self._pending:
            raise ValueError("event_id mismatch")
        if self._stage == "configuring":
            self.mode = payload["acquisition_mode"]
            return self._start()
        if self._stage == "done":
            raise ValueError("protocol already finished")
        body = {k: v for k, v in payload.items() if k != "event_id"}
        try:
            event = self._gen.send(body)
        except StopIteration:
            self._stage = "done"
            return self._emit({"type": "finished", "predictions_csv": self.predictions_csv()})
        return self._emit(event)

    def predictions_csv(self):
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(["model_id", "item_id", "budget", "predicted_probability"])
        for b in self.budgets:
            probs = self.res["probabilities"][b]
            for p, (s, e) in zip(self.pairs, self.res["spans"]):
                for r, v in zip(p.eval_rows, probs[s:e]):
                    w.writerow([p.subject_id, r["item_id"], b, repr(v)])
        return buf.getvalue()


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------
def _labeled_copier(labeled, isolation="call"):
    """Workers pass copy.deepcopy(labeled) per call. labeled is pure JSON data, so a
    json.loads of one serialization is an equivalent (and ~10x faster) fresh copy.
    isolation="checkpoint" shares one copy per checkpoint (exact unless predict()
    mutates its arguments; much faster for big runs)."""
    text = json.dumps(labeled, allow_nan=False)
    if isolation == "checkpoint":
        shared = json.loads(text)
        return lambda: shared
    if isolation != "call":
        raise ValueError("isolation must be 'call' or 'checkpoint'")
    return lambda: json.loads(text)


def _predict_checkpoint(predict, inputs, labeled, isolation="call"):
    """One call per distinct input (official grouping), worker-style copies."""
    unique, groups = group_prediction_inputs(iter(inputs), len(inputs), float("inf"))
    fresh = _labeled_copier(labeled, isolation)
    out = [None] * len(inputs)
    for i, inp in enumerate(unique):
        score = validate_score(_worker_score(predict(_wire(inp), fresh())))
        for j in groups[i]:
            out[j] = score
    return out, len(unique)


def _drive_direct(gen, predict, hook, isolation="call"):
    """In-process client mirroring run_streaming's event handling."""
    calls = {"prediction_calls": 0, "acquisition_prediction_calls": 0}
    event = next(gen)
    while True:
        kind, labeled = event["type"], event["labeled"]
        if kind == "rank":
            if hook.mode != "legacy_rank":
                raise ValueError("Coordinator acquisition mode does not match the submission hook")
            reply = {"score": hook(event["input"])}
        elif kind == "acquire":
            if hook.mode == "legacy_rank":
                raise ValueError("Coordinator acquisition mode does not match the submission hook")
            inp, ctx = event["input"], event["context"]
            if hook.mode == "random":
                query = _random_query(ctx, inp)
            else:
                probability = None
                if hook.needs_prediction:
                    probability = validate_score(predict(copy.deepcopy(inp), copy.deepcopy(labeled)))
                    calls["acquisition_prediction_calls"] += 1
                query = hook(inp, probability, labeled, ctx)
            reply = {"query": query}
        elif kind == "evaluate":
            probs, n = _predict_checkpoint(predict, [t["input"] for t in event["targets"]], labeled,
                                           isolation)
            calls["prediction_calls"] += n
            reply = {"probabilities": probs}
            if event["budget"] == 0 and hook.mode == "legacy_rank":
                reply["acquisition_mode"] = "legacy_rank"
        else:
            raise ValueError(f"unknown event {kind}")
        try:
            event = gen.send(reply)
        except StopIteration:
            return calls


def load_model_module(model_path, name=None):
    """Import model.py like a prediction worker (model dir first on sys.path)."""
    model_path = Path(model_path).resolve()
    if str(model_path.parent) not in sys.path:
        sys.path.insert(0, str(model_path.parent))
    name = name or f"paiec_sim_model_{_hseed(str(model_path), time.perf_counter_ns()) % 10**12}"
    spec = importlib.util.spec_from_file_location(name, model_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _inprocess_prediction_stream(model_path, inputs, labeled, *, concurrency, deadline,
                                 per_call_timeout=0, worker_env=None):
    """Windows stand-in for prediction_workers.parallel_predictions (which needs
    pass_fds/os.killpg): fresh model import per checkpoint, JSON transport,
    one call per unique input, worker-side float/range check."""
    model = load_model_module(model_path)
    fresh = _labeled_copier(labeled, "call")
    for index, item in enumerate(inputs):
        if time.monotonic() >= deadline:
            raise TimeoutError("Participant code exceeded the time limit")
        yield index, _worker_score(model.predict(_wire(item), fresh()))


def run_official_client(pairs, model_path, acquisition_function=None, predict=None,
                        send_configure=True, use_official_workers=None, concurrency=1,
                        ingestion_module=None):
    """Run the UNMODIFIED official run_streaming against LocalCoordinator.

    ``ingestion_module`` lets the older paiec_baseline client (no "configure"
    event; pass send_configure=False) be driven too. Returns the scored result
    (same shape as the direct path)."""
    if predict is None:
        predict = load_model_module(model_path).predict
    if use_official_workers is None:
        use_official_workers = os.name == "posix"
    if use_official_workers:
        from prediction_workers import parallel_predictions as stream_fn
    else:
        stream_fn = _inprocess_prediction_stream
    coord = LocalCoordinator(pairs, send_configure=send_configure)
    with tempfile.TemporaryDirectory(prefix="paiec_sim_") as tmp:
        cfg = Path(tmp) / "config.json"
        cfg.write_text(json.dumps({"protocol": PROTOCOL}))
        stats = (ingestion_module or _official_ingestion).run_streaming(
            config_path=cfg, predict=predict, acquisition_function=acquisition_function,
            model_path=str(model_path), output_dir=Path(tmp) / "out", concurrency=concurrency,
            deadline=time.monotonic() + 6 * 3600, per_call_timeout=0, acquisition_timeout=0,
            call_window=lambda timeout, name: contextlib.nullcontext(),
            validate_score=validate_score, prediction_stream=stream_fn,
            group_inputs=group_prediction_inputs, log=lambda msg: None,
            exchange=coord.exchange)
    coord.res["client_stats"] = stats
    return score_result(coord.res)


def run_direct(pairs, predict, acquisition_function=None, isolation="call"):
    hook = AcquisitionHook(acquisition_function)
    res = _new_result(pairs)
    mode = "legacy_rank" if hook.mode == "legacy_rank" else "stream"
    res["client_stats"] = _drive_direct(_engine(pairs, res, mode), predict, hook, isolation)
    return score_result(res)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_result(res, budgets=BUDGETS):
    per_pair = {}
    for k, (p, (s, e)) in enumerate(zip(res["pairs"], res["spans"])):
        ys = [r["y"] for r in p.eval_rows]
        brier = {b: brier_score(ys, res["probabilities"][b][s:e]) for b in budgets}
        eces = {b: ece(ys, res["probabilities"][b][s:e]) for b in budgets}
        per_pair[p.key] = {
            "brier": brier, "ece": eces, "alc": brier_alc(brier),
            "ece_alc": float(sum(ALC_WEIGHTS[b] * eces[b] for b in budgets)),
            "labels": {b: res["labels"][b][k] for b in budgets},
            "acquired_items": {b: res["acquired_items"][b][k] for b in budgets},
            "n_unique_pairs": len({r["item_id"] for r in p.eval_rows}),
            "n_eval_rows": len(p.eval_rows), "n_acq_items": len(p.acq),
            "n_items": p.n_items, "eligible": p.eligible,
        }
    alcs = [v["alc"] for v in per_pair.values()]
    return {
        "overall_alc": statistics.mean(alcs) if alcs else float("nan"),
        "overall_ece_alc": statistics.mean(v["ece_alc"] for v in per_pair.values()) if alcs else float("nan"),
        "per_pair": per_pair, "probabilities": res["probabilities"],
        "client_stats": res.get("client_stats"), "mode": res.get("mode"),
    }


def evaluate(rows, predict, acquisition_function=None, seeds=(0,), ineligible="skip",
             reveal="first", draw_limit=None, isolation="checkpoint"):
    """Fast path. Returns overall ALC averaged over split/order seeds."""
    per_pair_alc = defaultdict(list)
    overall, runs = [], []
    n_skipped = len(build_pairs(rows))
    for seed in seeds:
        pairs = prepare_pairs(rows, seed, ineligible, reveal, draw_limit)
        if seed == seeds[0]:
            n_skipped -= len(pairs)
        out = run_direct(pairs, predict, acquisition_function, isolation)
        runs.append(out)
        overall.append(out["overall_alc"])
        for key, v in out["per_pair"].items():
            per_pair_alc[key].append(v["alc"])
    return {
        "overall_alc_mean": statistics.mean(overall) if overall else float("nan"),
        "overall_alc_by_seed": overall, "per_pair_alc": dict(per_pair_alc),
        "n_eligible_pairs": len(per_pair_alc), "n_skipped_pairs": n_skipped, "runs": runs,
    }


def run_pair(rows, predict, acquisition_function=None, seed=0):
    """Single pair in isolation: {budget: [(y, p), ...]} or None if ineligible."""
    pairs = prepare_pairs(rows, seed)
    if not pairs:
        return None
    out = run_direct(pairs[:1], predict, acquisition_function)
    ys = [r["y"] for r in pairs[0].eval_rows]
    return {b: list(zip(ys, out["probabilities"][b])) for b in BUDGETS}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_sample_data_dir(root) -> list[dict]:
    """Load starting_kit sample_data format: test/{models,test_items,test_pairs}.csv
    + ref/ground_truth.csv. Repeated rows in test_pairs = repeated trials, matched
    to ground_truth rows with the same (model_id, item_id) in order."""
    root = Path(root)
    items = {r["item_id"]: r for r in _read_csv(root / "test" / "test_items.csv")}
    models = {r["model_id"]: r for r in _read_csv(root / "test" / "models.csv")}
    truth = defaultdict(deque)
    for r in _read_csv(root / "ref" / "ground_truth.csv"):
        truth[(r["model_id"], r["item_id"])].append(int(r["label"]))
    rows = []
    for pair in _read_csv(root / "test" / "test_pairs.csv"):
        subject, item = _build_input(pair, items, models)
        row = {**subject, **item, "item_id": pair["item_id"], "subject_id": pair["model_id"],
               "y": truth[(pair["model_id"], pair["item_id"])].popleft()}
        rows.append(row)
    return rows


def load_benchmark_dir(dir_path: str) -> list[dict]:
    """Join subjects+items+response parquet tables (measurement-db v3 layout)."""
    import pandas as pd

    subjects = pd.read_parquet(os.path.join(dir_path, "subjects.parquet"))
    items = pd.read_parquet(os.path.join(dir_path, "items.parquet"))
    resp_path = os.path.join(dir_path, "response.parquet")
    if not os.path.exists(resp_path):
        resp_path = os.path.join(dir_path, "responses.parquet")
    response = pd.read_parquet(resp_path).dropna(subset=["response"])
    merged = response.merge(subjects, on="subject_id", how="left", suffixes=("", "_subj"))
    merged = merged.merge(items, on="item_id", how="left", suffixes=("", "_item"))
    rows = []
    for r in merged.itertuples(index=False):
        row = {f: _s(getattr(r, f, "")).strip() for f in SUBJECT_FIELDS}
        row["item_content"] = _s(getattr(r, "content", ""))
        row["item_features"] = _s(getattr(r, "item_features", "")).strip()
        row["interactors"] = _s(getattr(r, "interactors", "")).strip()
        row["benchmark_id"] = _s(getattr(r, "benchmark_id", "")).strip()
        row["item_id"] = getattr(r, "item_id")
        row["subject_id"] = getattr(r, "subject_id")
        row["y"] = int(getattr(r, "response"))
        rows.append(row)
    return rows


def load_payload(data_dir: str, benchmark_dirs: list[str] | None = None) -> list[dict]:
    dirs = benchmark_dirs or sorted(
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))
    rows = []
    for d in dirs:
        rows.extend(load_benchmark_dir(os.path.join(data_dir, d)))
    return rows


if __name__ == "__main__":
    import unittest
    sys.exit(0 if unittest.main(module="test_simulator", exit=False).result.wasSuccessful() else 1)
