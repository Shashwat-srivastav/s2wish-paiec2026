"""Insert or replace experiments/registry.csv rows from CSV lines on stdin (one row per line, header order).

    python scripts/registry_upsert.py < rows.csv

Each row must have exactly the header's field count. A row whose experiment_id already exists replaces that row in
place; a new ID is inserted after its numeric predecessor (or appended).
"""
import csv
import sys
from pathlib import Path

REGISTRY = Path(__file__).resolve().parents[1] / "experiments" / "registry.csv"


def main():
    rows = list(csv.reader(REGISTRY.open(encoding="utf-8", newline="")))
    header = rows[0]
    new = [r for r in csv.reader(sys.stdin) if r]
    for r in new:
        if len(r) != len(header):
            sys.exit(f"{r[0]}: {len(r)} fields, header has {len(header)}; nothing written")
    for r in new:
        ids = [i for i, x in enumerate(rows) if x and x[0] == r[0]]
        if ids:
            rows[ids[0]] = r
            print(f"replaced {r[0]}")
        else:
            prefix = r[0].rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            pos = max((i for i, x in enumerate(rows) if x and x[0].startswith(prefix)), default=len(rows) - 1)
            rows.insert(pos + 1, r)
            print(f"inserted {r[0]}")
    with REGISTRY.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)


if __name__ == "__main__":
    main()
