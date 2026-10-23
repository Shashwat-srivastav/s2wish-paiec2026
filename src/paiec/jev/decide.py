"""Jev Decision Layer (orchestrator_merged.md §4 PI / prompt.md §10 Expert 20).

Jev is used as a fast, calibrated second opinion on *research decisions* — never as a
feature inside the submission. Each decision record (question, options, evidence, numbers,
the rules that apply) is sent as state; Jev answers a fixed battery that mirrors the
canonical gate sequence (orchestrator_merged.md §18–§19):

  verdict            Choice  ADVANCE / REPLICATE / MODIFY / DEFER / ELIMINATE
  evidence_adequate  Noul    is the evidence sufficient under the stated rules?
  within_noise       Noul    could the claimed gain be noise (CI, folds, seeds)?
  simpler_suffices   Noul    Gate K: does a simpler alternative match it?
  low_budget_harm    Noul    Gate F: does it hurt budgets 1/3 while helping 31?
  leakage_risk       Noul    could the gain come from leakage / hidden information?
  favoured_position  Choice  which persona position the evidence supports (if any)

The PI (the orchestrator) still makes the call. Both Jev's answers and the PI decision are
appended to experiments/decisions.jsonl; disagreements are flagged for a second look.

Usage
-----
  python jev_decide.py --record decision.json [--pi-decision ELIMINATE]
  python jev_decide.py --batch decisions.json        # list of records
Record keys: id, question, options (dict name->description, optional), evidence (str|list),
numbers (dict), rules (list[str]), positions (dict persona->position, optional).
Needs env var TYPESAFE_API_KEY (never logged).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from paiec.jev.item_features import API_URL, DEFAULT_MODEL, _session  # shared HTTP session/retry

from paiec.paths import DECISIONS_LOG as LOG

VERDICTS = {
    "ADVANCE": "The evidence clears the relevant gates: keep the component/method and move on",
    "REPLICATE": "Promising but rests on too few seeds, folds or pairs; rerun before deciding",
    "MODIFY": "A specific identified flaw must be repaired, then re-tested",
    "DEFER": "Cannot be used or judged yet because of an external constraint or missing data",
    "ELIMINATE": "Fails a gate (no robust gain, noise-level gain, leakage, or a simpler method matches)",
}

RULES_PREAMBLE = (
    "Decision rules of this research lab: the objective is Brier ALC (lower is better) under "
    "benchmark-level cold start; a component is kept only if its gain is real and robust under "
    "the realistic evaluation regime; negligible gains are removed; gains that are real but "
    "unstable are replicated; when a simpler method is statistically indistinguishable it is "
    "preferred; low-budget regressions are a strong warning; one leaderboard result is weak evidence."
)


def build_questions(record: dict) -> dict:
    q = {
        "verdict": {
            "type": "choice",
            "instructions": "Given `decision.question`, the evidence in `decision.evidence`, the "
            "numbers in `decision.numbers` and the rules in `decision.rules`, which decision "
            "should the lab take?",
            "criteria": record.get("options") or VERDICTS,
        },
        "evidence_adequate": {
            "type": "noul",
            "instructions": "Is the evidence in `decision.evidence` and `decision.numbers` "
            "sufficient, under `decision.rules`, to justify a decision on `decision.question` now "
            "rather than collecting more evidence first?",
        },
        "within_noise": {
            "type": "noul",
            "instructions": "Could the improvement claimed in `decision.evidence` plausibly be "
            "explained by noise — a confidence interval that touches zero, too few folds, seeds or "
            "benchmarks, or a difference far smaller than its spread?",
        },
        "simpler_suffices": {
            "type": "noul",
            "instructions": "According to `decision.evidence`, does a simpler alternative achieve "
            "essentially the same result as the more complex option being considered?",
        },
        "low_budget_harm": {
            "type": "noul",
            "instructions": "Does `decision.evidence` show the option making predictions worse at "
            "the small label budgets (0, 1 or 3 labels) even if it helps at larger budgets?",
        },
        "leakage_risk": {
            "type": "noul",
            "instructions": "Does `decision.evidence` give a concrete reason to suspect the gain "
            "comes from information that would not be available at real evaluation time (leakage, "
            "test data, hidden fields, an over-optimistic evaluation regime)?",
        },
    }
    pos = record.get("positions")
    if pos:
        crit = {k: v for k, v in pos.items()}
        crit["neither"] = "The evidence does not favour any stated position over the others"
        q["favoured_position"] = {
            "type": "choice",
            "instructions": "Which expert position in `decision.positions` is better supported "
            "by `decision.evidence` and `decision.numbers`?",
            "criteria": crit,
        }
    return q


def ask(record: dict, model: str = DEFAULT_MODEL) -> dict:
    state = {"decision": {
        "question": record["question"],
        "evidence": record.get("evidence", ""),
        "numbers": record.get("numbers", {}),
        "rules": [RULES_PREAMBLE] + list(record.get("rules", [])),
        **({"positions": record["positions"]} if record.get("positions") else {}),
    }}
    body = {"state": state, "model": model, "questions": build_questions(record)}
    r = _session().post(API_URL, json=body, timeout=120)
    r.raise_for_status()
    return r.json()


def summarise(resp: dict) -> dict:
    out = {}
    for qid, a in resp.get("answers", {}).items():
        if a.get("type") == "choice":
            out[qid] = {"choice": a.get("choice"), "confidence": round(a.get("confidence", 0), 3),
                        "probabilities": {k: round(v, 3) for k, v in (a.get("probabilities") or {}).items()}}
        else:
            out[qid] = round(float(a.get("noul", float("nan"))), 3)
    return out


def decide(record: dict, pi_decision: str | None = None, model: str = DEFAULT_MODEL) -> dict:
    resp = ask(record, model)
    jev = summarise(resp)
    entry = {
        "time": dt.datetime.now().isoformat(timespec="seconds"),
        "id": record.get("id"),
        "question": record["question"],
        "jev_model": resp.get("model"),
        "jev": jev,
        "pi_decision": pi_decision,
        "agree": (pi_decision is None) or (jev["verdict"]["choice"] == pi_decision),
        "input_tokens": resp.get("usage", {}).get("input_tokens"),
    }
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--record", type=Path)
    g.add_argument("--batch", type=Path)
    ap.add_argument("--pi-decision", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    a = ap.parse_args()
    recs = [json.loads(a.record.read_text(encoding="utf-8"))] if a.record else \
        json.loads(a.batch.read_text(encoding="utf-8"))
    for rec in recs:
        e = decide(rec, rec.get("pi_decision", a.pi_decision), a.model)
        v = e["jev"]["verdict"]
        flag = "" if e["agree"] else "  <-- DISAGREES WITH PI"
        print(f"[{e['id']}] Jev: {v['choice']} ({v['confidence']:.2f}) | PI: {e['pi_decision']}{flag}")
        for k in ("evidence_adequate", "within_noise", "simpler_suffices", "low_budget_harm", "leakage_risk"):
            print(f"    {k:18s} {e['jev'][k]:.2f}")
        if "favoured_position" in e["jev"]:
            fp = e["jev"]["favoured_position"]
            print(f"    favoured_position  {fp['choice']} ({fp['confidence']:.2f})")


if __name__ == "__main__":
    sys.exit(main())
