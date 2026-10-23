from pathlib import Path
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
src_zip = ROOT / "submissions" / "dist" / "v1_3_named_facets_noguard.zip"
out_zip = ROOT / "submissions" / "dist" / "v1_4_kernel_experimental.zip"

kernel = r'''

# Experimental bounded within-benchmark content kernel.  It only uses fields
# present in the public submission contract and scans the already revealed
# labels once per prediction; it has no network, filesystem, or model calls.
_KWORD_RE = re.compile(r"[a-z0-9]{2,}")
_KTOKENS = {}

def _ktokens(value):
    text = _str(value).lower()
    got = _KTOKENS.get(text)
    if got is None:
        got = frozenset(_KWORD_RE.findall(text))
        _KTOKENS[text] = got
    return got

def _ksim(a, b):
    if not a or not b:
        return 0.0
    common = len(a & b)
    if common < 2:
        return 0.0
    return common / math.sqrt(len(a) * len(b))

def _kernel_adjust(subject, item, labeled, base):
    target_bench = _str(item.get("benchmark_id", ""))
    target_tokens = _ktokens(item.get("item_content", ""))
    if not target_bench or not target_tokens or not labeled:
        return base
    rows = []
    totals = {}
    for entry in labeled:
        try:
            (os_, oi), y = entry
            if not isinstance(os_, dict) or not isinstance(oi, dict):
                continue
            if _str(oi.get("benchmark_id", "")) != target_bench:
                continue
            y = float(y)
            if y not in (0.0, 1.0):
                continue
            st = _subject_key(os_)
            totals.setdefault(st, [0.0, 0.0])
            totals[st][0] += y
            totals[st][1] += 1.0
            toks = _ktokens(oi.get("item_content", ""))
            sim = _ksim(target_tokens, toks)
            if sim >= 0.20 and _item_key(oi) != _item_key(item):
                rows.append((sim, y, st))
        except Exception:
            continue
    if not rows:
        return base
    weights = []
    residuals = []
    tau = float(P.get("tau", 1.13))
    for sim, y, st in rows:
        s, n = totals[st]
        q = (s + 0.5 * tau) / (n + tau)
        weights.append(sim * sim)
        residuals.append(y - q)
    wsum = sum(weights)
    if wsum <= 0.0:
        return base
    residual = sum(w * r for w, r in zip(weights, residuals)) / wsum
    # Small shrinkage is intentional: lexical similarity is a weak signal.
    gain = 0.12 * (1.0 - math.exp(-wsum))
    lo, hi = P.get("clip", [0.005, 0.995])
    return min(max(float(base) + gain * residual, float(lo)), float(hi))
'''

with zipfile.ZipFile(src_zip, "r") as zin:
    files = {name: zin.read(name) for name in zin.namelist()}

model = files["model.py"].decode("utf-8")
model = model.replace("import os\n", "import os\nimport re\n", 1)
model = model.replace("\ndef _finish(p):\n", kernel + "\ndef _finish(p):\n", 1)
old = "return _finish(_core(subject, item, labeled))"
new = "return _finish(_kernel_adjust(subject, item, labeled, _core(subject, item, labeled)))"
if old not in model:
    raise RuntimeError("predict insertion point not found")
model = model.replace(old, new, 1)
files["model.py"] = model.encode("utf-8")

meta = json.loads(files["v1_2_params.json"].decode("utf-8"))
meta["_source"] = "v1_3 named facets/no-guard plus bounded lexical kernel; experimental"
files["v1_2_params.json"] = (json.dumps(meta, indent=1) + "\n").encode("utf-8")

out_zip.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zout:
    for name, data in sorted(files.items()):
        zout.writestr(name, data)
print(out_zip)
