"""One-screen progress check for running experiments.

    python scripts/status.py            # snapshot
    python scripts/status.py --watch    # refresh every 30 s (Ctrl+C to stop)
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "runs"
KEYS = ("decision", "verdict", "fired", "action", "stop", "ship")


def age(p: Path) -> str:
    m = (time.time() - p.stat().st_mtime) / 60
    return f"{m:5.0f} min ago" if m < 600 else datetime.fromtimestamp(p.stat().st_mtime).strftime("%m-%d %H:%M")


def find_decisions(obj, path="", out=None):
    out = [] if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else k
            if any(t in k.lower() for t in KEYS) and not isinstance(v, (dict, list)):
                out.append((p, v))
            find_decisions(v, p, out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:50]):
            find_decisions(v, f"{path}[{i}]", out)
    return out


def snapshot():
    print(f"=== status {datetime.now():%Y-%m-%d %H:%M:%S} ===\n")
    # 1. running python processes
    try:
        if sys.platform == "win32":
            r = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV", "/NH"],
                               capture_output=True, text=True).stdout.strip().splitlines()
            r = [l for l in r if l.startswith('"python')]
        else:
            r = subprocess.run(["pgrep", "-af", "python"], capture_output=True, text=True).stdout.strip().splitlines()
        print(f"python processes running: {len(r)}")
        for l in r[:6]:
            print("   ", l.replace('"', "")[:110])
    except Exception as e:
        print("process check failed:", e)
    # 2. experiment run folders, newest first
    print("\nexperiment output folders (newest first):")
    dirs = sorted([d for d in RUNS.iterdir() if d.is_dir()], key=lambda d: max((f.stat().st_mtime for f in d.rglob("*") if f.is_file()), default=0), reverse=True)
    for d in dirs[:4]:
        files = sorted([f for f in d.rglob("*") if f.is_file()], key=lambda f: f.stat().st_mtime, reverse=True)
        print(f"  {d.name}/  ({len(files)} files)")
        for f in files[:5]:
            print(f"      {age(f)}  {f.relative_to(d)}  ({f.stat().st_size/1024:.0f} KB)")
    # 3. tail of newest log
    logs = sorted(RUNS.rglob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)
    if logs:
        print(f"\nlast lines of newest log: {logs[0].relative_to(ROOT)}  ({age(logs[0])})")
        for line in logs[0].read_text(encoding="utf-8", errors="replace").splitlines()[-8:]:
            print("   ", line[:140])
    # 4. decisions recorded in result json files of the newest run
    if dirs:
        print(f"\ndecision fields found in {dirs[0].name}/*.json:")
        n = 0
        for j in sorted(dirs[0].glob("*.json")):
            try:
                for k, v in find_decisions(json.loads(j.read_text(encoding="utf-8")))[:6]:
                    print(f"    {j.name}: {k} = {str(v)[:90]}")
                    n += 1
            except Exception:
                pass
        if not n:
            print("    (none yet)")
    # 5. reports and registry
    reps = sorted((ROOT / "docs" / "research_notes").glob("*.md"), key=lambda f: f.stat().st_mtime, reverse=True)[:3]
    print("\nnewest reports:")
    for r in reps:
        print(f"    {age(r)}  {r.name}")
    reg = (ROOT / "experiments" / "registry.csv").read_text(encoding="utf-8").splitlines()
    print(f"\nregistry: {len(reg)-1} experiments; last: {reg[-1][:120]}")
    jc = ROOT / "data" / "jev_cache"
    if jc.exists():
        print("jev cache:", ", ".join(f"{f.name} {f.stat().st_size/1024:.0f} KB ({age(f).strip()})" for f in jc.glob("*.parquet")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    a = ap.parse_args()
    if not a.watch:
        snapshot()
    else:
        try:
            while True:
                print("\033[2J\033[H", end="")
                snapshot()
                time.sleep(30)
        except KeyboardInterrupt:
            pass
