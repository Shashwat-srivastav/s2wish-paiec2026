"""Leakage-check stubs — run after gated payload is available.

Each check takes training/eval long-form frames (dicts with subject fields,
benchmark_id, item content/features, y) and returns a report dict.
No external deps beyond stdlib + numpy (embeddings optional, passed in).
"""
from __future__ import annotations
import hashlib
from collections import Counter

def content_hash(content: str) -> str:
    norm = (content or "").strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]

def check_exact_dupes(train_contents, test_contents):
    tr = Counter(content_hash(c) for c in train_contents)
    hits = sum(1 for c in test_contents if content_hash(c) in tr)
    return {"n_test": len(test_contents), "exact_dupes": hits, "rate": hits / max(1, len(test_contents))}

def check_benchmark_overlap(train_benches, test_benches):
    return {"overlap": sorted(set(train_benches) & set(test_benches)),
            "must_be_empty": len(set(train_benches) & set(test_benches)) == 0}

def check_subject_overlap(train_subj_keys, test_subj_keys):
    te = set(test_subj_keys)
    return {"n_test_subjects": len(te),
            "seen_in_train": len(te & set(train_subj_keys))}

def check_benchmark_token_in_text(bench_names, contents):
    """Flag if benchmark-identifying tokens appear in item text/features."""
    flags = 0
    lowered = [b.lower() for b in bench_names if b]
    for c in contents:
        cl = (c or "").lower()
        if any(b in cl for b in lowered if len(b) > 3):
            flags += 1
    return {"flagged": flags, "n": len(contents)}

def check_temporal_order(train_dates, test_dates):
    """Return max train date vs min test date (ISO strings compare lexicographically)."""
    tr = sorted([d for d in train_dates if d]); te = sorted([d for d in test_dates if d])
    return {"max_train": tr[-1] if tr else None, "min_test": te[0] if te else None,
            "forward_valid": (not tr or not te) or (tr[-1] <= te[0])}

def make_splits(rows, by="benchmark", seed=0):
    """Deterministic group-holdout splitter. rows: list of dicts with keys for
    by in {benchmark, subject, family, provider, harness, temporal, random, item}."""
    import random
    rng = random.Random(seed)
    if by == "random":
        idx = list(range(len(rows))); rng.shuffle(idx)
        cut = int(0.8 * len(idx))
        return idx[:cut], idx[cut:]
    def key(r):
        if by == "benchmark": return r.get("benchmark_id", "")
        if by == "subject": return r.get("subject_key", "")
        if by == "family": return (r.get("normalized_name", "").split("-")[0])
        if by == "provider": return r.get("provider", "")
        if by == "harness": return r.get("harness", "")
        if by == "item": return r.get("item_id", "")
        if by == "temporal": return (r.get("release_date", "") or "9999")[:7]
        raise ValueError(by)
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault(key(r), []).append(i)
    names = sorted(groups)
    if by == "temporal":
        names_sorted = names  # chronological; hold out latest 20%
        cut = int(0.8 * len(names_sorted))
        test_names = set(names_sorted[cut:])
    else:
        rng.shuffle(names)
        cut = max(1, int(0.2 * len(names)))
        test_names = set(names[:cut])
    tr = [i for g, idxs in groups.items() if g not in test_names for i in idxs]
    te = [i for g, idxs in groups.items() if g in test_names for i in idxs]
    return tr, te
