from pathlib import Path
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
src_zip = ROOT / "submissions" / "dist" / "v1.zip"
out_zip = ROOT / "submissions" / "dist" / "v1_5_hybrid_experimental.zip"

kernel = r'''

# Conservative within-benchmark content residual, layered on the trained
# metadata/IRT predictor.  It uses only current-budget labels and runtime
# fields; no hidden pool or external service is consulted.
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
    if common < 4:
        return 0.0
    return common / math.sqrt(len(a) * len(b))

def _kernel_adjust(subject, item, labeled, base):
    bench = _str(item.get("benchmark_id", ""))
    target = _ktokens(item.get("item_content", ""))
    if not bench or not target or not labeled:
        return base
    rows = []
    totals = {}
    for entry in labeled:
        try:
            (os_, oi), y = entry
            if not isinstance(os_, dict) or not isinstance(oi, dict):
                continue
            if _str(oi.get("benchmark_id", "")) != bench:
                continue
            y = float(y)
            if y not in (0.0, 1.0):
                continue
            sk = _subject_key(os_)
            totals.setdefault(sk, [0.0, 0.0])
            totals[sk][0] += y
            totals[sk][1] += 1.0
            sim = _ksim(target, _ktokens(oi.get("item_content", "")))
            if sim >= 0.45 and _item_key(oi) != _item_key(item):
                rows.append((sim, y, sk))
        except Exception:
            continue
    if len(rows) < 2:
        return base
    tau = float(P.get("tau", 1.13))
    weights, residuals = [], []
    for sim, y, sk in rows:
        s, n = totals[sk]
        q = (s + 0.5 * tau) / (n + tau)
        weights.append(sim * sim)
        residuals.append(y - q)
    total = sum(weights)
    if total <= 0.0:
        return base
    residual = sum(w * r for w, r in zip(weights, residuals)) / total
    gain = 0.04 * (1.0 - math.exp(-total))
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

params = json.loads(files["v1_params.json"].decode("utf-8"))
params["_source"] = "v1 metadata/IRT predictor plus conservative within-benchmark lexical residual; experimental"
files["v1_params.json"] = (json.dumps(params, indent=1) + "\n").encode("utf-8")

with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zout:
    for name, data in sorted(files.items()):
        zout.writestr(name, data)
print(out_zip)
