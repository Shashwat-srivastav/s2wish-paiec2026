"""Log repeat Codabench submissions of one fixed zip and summarise them against the top-10 cutoff.

Every run is kept (not just the best), so the batch doubles as repeat-run data for EXP-0039/0040.

    python scripts/log_runs.py add 0.1712 --id 971234                # default version v1_3_named_facets_noguard
    python scripts/log_runs.py add 0.1688 --id 971240 --ece 0.12
    python scripts/log_runs.py summary                                # n, mean, CI, best, chance per draw vs cutoff

The cutoff is the 10th-best non-us score in the newest snapshot from scripts/leaderboard.py
(run that first to refresh it). Codabench's submissions API needs a login, so scores are entered by hand.
"""
import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "experiments" / "platform_submissions"
SNAPSHOTS = ROOT / "experiments" / "leaderboard_snapshots"
OUR_USERS = {"slash"}
FROZEN = {"v1_3_named_facets_noguard": "a4b0e841d4a344ff207bed2c855342212e95f7d550195b667446ea4e92685b86"}


def runs_file(version: str) -> Path:
    return RUNS_DIR / f"repeats_{version}.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(version: str) -> dict:
    f = runs_file(version)
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    zip_path = ROOT / "submissions" / "dist" / f"{version}.zip"
    if not zip_path.exists():
        sys.exit(f"no zip at {zip_path}")
    h = sha256(zip_path)
    if version in FROZEN and h != FROZEN[version]:
        sys.exit(f"{zip_path.name} hash {h[:12]} != frozen {FROZEN[version][:12]}; not logging against a changed zip")
    return {"version": version, "zip": zip_path.relative_to(ROOT).as_posix(), "sha256": h,
            "note": "repeat submissions of one unchanged zip; every run logged, best-of-N is not model quality",
            "runs": []}


def top10_cutoff() -> tuple[float, str] | None:
    snaps = sorted(SNAPSHOTS.glob("*.json"))
    if not snaps:
        return None
    snap = json.loads(snaps[-1].read_text(encoding="utf-8"))
    others = sorted(r["brier_alc"] for r in snap["rows"]
                    if r["brier_alc"] is not None and (r["owner"] or "").lower() not in OUR_USERS)
    return (others[9], snap["fetched_utc"]) if len(others) >= 10 else None


def summarise(d: dict) -> dict:
    x = [r["alc"] for r in d["runs"]]
    n = len(x)
    out = {"n_runs": n}
    if n == 0:
        return out
    mean = sum(x) / n
    out.update(mean=round(mean, 6), best=min(x))
    if n >= 2:
        sd = math.sqrt(sum((v - mean) ** 2 for v in x) / (n - 1))
        half = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
        out.update(sd=round(sd, 6), se=round(sd / math.sqrt(n), 6), ci95_t=[round(mean - half, 6), round(mean + half, 6)])
    return out


def cmd_add(a):
    d = load(a.version)
    h = sha256(ROOT / d["zip"])
    if h != d["sha256"]:
        sys.exit(f"{d['zip']} changed since the first logged run ({h[:12]} != {d['sha256'][:12]}); not logging")
    if a.id is not None and any(r.get("id") == a.id for r in d["runs"]):
        sys.exit(f"submission {a.id} already logged")
    run = {"id": a.id, "alc": a.alc, "logged_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    if a.ece is not None:
        run["ece"] = a.ece
    d["runs"].append(run)
    d.update(summarise(d))
    runs_file(a.version).write_text(json.dumps(d, indent=1) + "\n", encoding="utf-8")
    print(f"logged run {len(d['runs'])}: {a.alc:.6f}  ->  {runs_file(a.version).relative_to(ROOT)}")
    cmd_summary(a)


def cmd_summary(a):
    d = load(a.version)
    s = summarise(d)
    n = s["n_runs"]
    print(f"{a.version}  sha {d['sha256'][:12]}  runs {n}")
    if n == 0:
        return
    print(f"  mean {s['mean']:.4f}  best {s['best']:.4f}", end="")
    if n >= 2:
        print(f"  sd {s['sd']:.4f}  95% CI [{s['ci95_t'][0]:.4f}, {s['ci95_t'][1]:.4f}]")
    else:
        print()
    cut = top10_cutoff()
    if not cut:
        print("  no leaderboard snapshot; run scripts/leaderboard.py for the top-10 cutoff")
        return
    c, when = cut
    hits = sum(r["alc"] < c for r in d["runs"])
    print(f"  top-10 cutoff {c:.4f} (snapshot {when}): {hits}/{n} runs below it", end="")
    if n >= 3:
        # normal fit to this zip's own runs; a rough guide, not a guarantee (draws are not exactly normal)
        p = stats.norm.cdf(c, s["mean"], s["sd"])
        print(f"; est. chance per draw {p:.1%}", end="")
        if 0 < p < 1:
            print(f"; draws for 50%/80% chance {math.ceil(math.log(0.5) / math.log(1 - p))}"
                  f"/{math.ceil(math.log(0.2) / math.log(1 - p))}", end="")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", default="v1_3_named_facets_noguard", help="zip name in submissions/dist without .zip")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("add", help="log one run")
    p.add_argument("alc", type=float)
    p.add_argument("--id", type=int, help="Codabench submission id")
    p.add_argument("--ece", type=float)
    p.set_defaults(fn=cmd_add)
    sub.add_parser("summary", help="summarise logged runs").set_defaults(fn=cmd_summary)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
