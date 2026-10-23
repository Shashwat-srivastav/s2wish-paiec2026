from pathlib import Path
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
src_zip = ROOT / "submissions" / "dist" / "v1_3_named_facets_noguard.zip"
out_zip = ROOT / "submissions" / "dist" / "v1_7_adaptive_facets.zip"

robust = r'''

def _adaptive_core(subject, item, labeled):
    """Cap only unusually large sparse-evidence moves.

    Unlike global damping, ordinary small updates pass through unchanged. The
    cap relaxes with evidence and is inactive at budget 0 and effectively
    inactive by budget 31.
    """
    labels = labeled or []
    n = len(labels)
    p = float(_core(subject, item, labels))
    if n <= 0:
        return p
    p0 = float(_core(subject, item, []))
    caps = {1: 0.08, 2: 0.10, 3: 0.13, 4: 0.15, 5: 0.17,
            6: 0.19, 7: 0.21, 8: 0.23, 9: 0.25, 10: 0.27,
            11: 0.29, 12: 0.31, 13: 0.33, 14: 0.35, 15: 0.37}
    cap = caps.get(n, 0.50)
    delta = p - p0
    if abs(delta) <= cap:
        return p
    return p0 + math.copysign(cap, delta)
'''

with zipfile.ZipFile(src_zip, "r") as zin:
    files = {name: zin.read(name) for name in zin.namelist()}
model = files["model.py"].decode("utf-8")
model = model.replace("\ndef _finish(p):\n", robust + "\ndef _finish(p):\n", 1)
old = "return _finish(_core(subject, item, labeled))"
new = "return _finish(_adaptive_core(subject, item, labeled))"
if old not in model:
    raise RuntimeError("predict insertion point not found")
model = model.replace(old, new, 1)
files["model.py"] = model.encode("utf-8")
params = json.loads(files["v1_2_params.json"].decode("utf-8"))
params["_source"] = "v1_3 named facets/no-guard plus adaptive sparse-evidence jump cap; experimental"
files["v1_2_params.json"] = (json.dumps(params, indent=1) + "\n").encode("utf-8")
with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zout:
    for name, data in sorted(files.items()):
        zout.writestr(name, data)
print(out_zip)
