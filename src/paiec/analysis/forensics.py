"""Data-forensics battery — runs on a prepared payload (see prepare_data.py).

Usage (after HF login + approval):
  python tools/prepare_data.py --out payload --no-embeddings   # in paiec_baseline
  python src/paiec/analysis/forensics.py --data payload/data --out docs/research_notes/forensics/

Outputs: summary.json + CSVs + PNGs (matplotlib optional; CSVs always).
No Codabench/private-test access; training-pool only.
"""
from __future__ import annotations
import argparse, json, os, hashlib
from collections import Counter
import pandas as pd
import numpy as np

def sha16(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]

def audit_benchmark(d: str, tables: dict) -> dict:
    rep = {"dir": d}
    for t, df in tables.items():
        rep[f"{t}.rows"] = int(len(df))
        rep[f"{t}.cols"] = list(df.columns)
        nulls = {c: int(df[c].isna().sum()) for c in df.columns}
        rep[f"{t}.nulls"] = nulls
    if "response" in tables:
        r = tables["response"]
        col = "response" if "response" in r.columns else None
        if col:
            vals = r[col].dropna()
            rep["response.n_nonnull"] = int(len(vals))
            rep["response.mean"] = float(vals.mean()) if len(vals) else None
            rep["response.value_counts"] = {str(k): int(v) for k, v in vals.value_counts().head(20).items()}
            rep["response.n_subjects"] = int(r["subject_id"].nunique()) if "subject_id" in r.columns else None
            rep["response.n_items"] = int(r["item_id"].nunique()) if "item_id" in r.columns else None
    if "subjects" in tables:
        s = tables["subjects"]
        for c in ["provider", "harness", "reasoning_effort", "normalized_name"]:
            if c in s.columns:
                rep[f"subjects.{c}.top"] = [{str(k): int(v)} for k, v in s[c].fillna("(null)").value_counts().head(10).items()]
        if "release_date" in s.columns:
            rep["subjects.release_date.min"] = str(s["release_date"].dropna().min()) if s["release_date"].notna().any() else None
            rep["subjects.release_date.max"] = str(s["release_date"].dropna().max()) if s["release_date"].notna().any() else None
    if "items" in tables:
        it = tables["items"]
        if "content" in it.columns:
            lens = it["content"].fillna("").str.len()
            rep["items.len.p50"] = float(lens.median()); rep["items.len.p90"] = float(lens.quantile(0.9))
            rep["items.len.max"] = int(lens.max())
    return rep

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dirs = sorted([d for d in os.listdir(args.data) if os.path.isdir(os.path.join(args.data, d))])
    full = {}
    for d in dirs:
        tables = {}
        for t in ["benchmarks", "subjects", "items", "response", "traces"]:
            p = os.path.join(args.data, d, f"{t}.parquet")
            if os.path.exists(p):
                tables[t] = pd.read_parquet(p)
        rep = audit_benchmark(d, tables)
        full[d] = rep
        print(f"[{d}] " + json.dumps({k: v for k, v in rep.items() if k.endswith('.rows') or k in ('response.mean',)}, indent=0))
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(full, f, indent=2, default=str)
    print(f"Wrote {args.out}/summary.json for {len(dirs)} benchmark dirs.")

if __name__ == "__main__":
    main()
