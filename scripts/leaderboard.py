"""Fetch the public Codabench leaderboard, save a dated snapshot, and show where we stand.

Run this before every submission (see docs/research_notes/22_LEADERBOARD.md):

    python scripts/leaderboard.py                 # fetch, save snapshot, print table + changes since last snapshot
    python scripts/leaderboard.py --score 0.1761  # also show where a given ALC would rank
    python scripts/leaderboard.py --offline       # re-print the newest saved snapshot (no network)

Data source: the Codabench REST API (no login needed), competition 17828, phase 29785.
"""
import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = ROOT / "experiments" / "leaderboard_snapshots"
COMPETITION = 17828
PHASE = 29785
API = "https://www.codabench.org/api"
OUR_USERS = {"slash"}            # our Codabench account(s), matched case-insensitively
BASELINE_USERS = {"AIMS Foundations"}


def get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "paiec-leaderboard/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch() -> dict:
    comp = get(f"{API}/competitions/{COMPETITION}/")
    board = get(f"{API}/phases/{PHASE}/get_leaderboard/")
    phase = next((p for p in comp.get("phases", []) if p.get("id") == PHASE), {})
    rows = []
    for s in board.get("submissions", []):
        sc = {x["column_key"]: float(x["score"]) for x in s.get("scores", [])}
        rows.append({"owner": s.get("owner"), "brier_alc": sc.get("brier"), "ece": sc.get("calibration_ece"),
                     "submitted": (s.get("created_when") or "")[:19], "submission_id": s.get("id")})
    rows.sort(key=lambda r: (r["brier_alc"] is None, r["brier_alc"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return {
        "fetched_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "competition": COMPETITION, "phase": PHASE,
        "participants": comp.get("participants_count"), "submissions_total": comp.get("submissions_count"),
        "phase_end": phase.get("end"), "max_per_day": phase.get("max_submissions_per_day"),
        "rows": rows,
    }


def previous(exclude: Path | None = None) -> dict | None:
    snaps = sorted(p for p in SNAPSHOTS.glob("*.json") if p != exclude)
    return json.loads(snaps[-1].read_text(encoding="utf-8")) if snaps else None


def where(score: float, rows: list) -> int:
    return 1 + sum(1 for r in rows if r["brier_alc"] is not None and r["brier_alc"] < score)


def show(snap: dict, prev: dict | None, score: float | None) -> None:
    rows = snap["rows"]
    print(f"Codabench {snap['competition']} leaderboard  fetched {snap['fetched_utc']}")
    print(f"participants {snap['participants']}  total submissions {snap['submissions_total']}  "
          f"on board {len(rows)}  phase ends {snap['phase_end']}  max/day {snap['max_per_day']}\n")
    old = {r["owner"]: r for r in (prev or {}).get("rows", [])}
    print(f"{'rank':>4}  {'Brier ALC':>9}  {'ECE':>6}  {'owner':22} {'submitted':10}  change")
    for r in rows:
        tag = " <== US" if (r["owner"] or "").lower() in OUR_USERS else (" (official baseline)" if r["owner"] in BASELINE_USERS else "")
        o = old.get(r["owner"])
        if prev is None:
            ch = ""
        elif o is None:
            ch = "NEW"
        elif o["brier_alc"] != r["brier_alc"]:
            ch = f"{r['brier_alc'] - o['brier_alc']:+.4f} (was #{o['rank']})"
        else:
            ch = "" if o["rank"] == r["rank"] else f"rank {o['rank']}->{r['rank']}"
        ece = f"{r['ece']:.3f}" if r["ece"] is not None else "  -  "
        print(f"{r['rank']:>4}  {r['brier_alc']:9.6f}  {ece:>6}  {r['owner'][:22]:22} {r['submitted'][:10]}  {ch}{tag}")
    if prev:
        gone = set(old) - {r["owner"] for r in rows}
        if gone:
            print("\nleft the board since last snapshot:", ", ".join(sorted(gone)))
        print(f"(changes relative to snapshot {prev['fetched_utc']})")
    if not any((r["owner"] or "").lower() in OUR_USERS for r in rows):
        print(f"\nWARNING: none of {sorted(OUR_USERS)} is on the board - choose a submission to show on Codabench.")
    if score is not None:
        print(f"\nA Brier ALC of {score:.4f} would rank #{where(score, rows)} of {len(rows) + 1}.")
    base = next((r for r in rows if r["owner"] in BASELINE_USERS), None)
    if base:
        print(f"official baseline: {base['brier_alc']:.4f} (rank {base['rank']}); top: {rows[0]['brier_alc']:.4f} ({rows[0]['owner']})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", type=float, help="show where this Brier ALC would rank")
    ap.add_argument("--offline", action="store_true", help="print newest saved snapshot without fetching")
    a = ap.parse_args()
    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    if a.offline:
        snaps = sorted(SNAPSHOTS.glob("*.json"))
        if not snaps:
            print("no saved snapshots")
            return 1
        snap = json.loads(snaps[-1].read_text(encoding="utf-8"))
        show(snap, previous(exclude=snaps[-1]), a.score)
        return 0
    try:
        snap = fetch()
    except Exception as e:  # network or API change
        print(f"fetch failed: {e}\nre-run with --offline to see the last saved snapshot")
        return 1
    prev = previous()
    out = SNAPSHOTS / f"{snap['fetched_utc'].replace(':', '').replace('-', '')}.json"
    out.write_text(json.dumps(snap, indent=1), encoding="utf-8")
    show(snap, prev, a.score)
    print(f"\nsnapshot saved: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
