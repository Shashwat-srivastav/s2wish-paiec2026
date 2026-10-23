from pathlib import Path
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
src_zip = ROOT / "submissions" / "dist" / "v1_3_named_facets_noguard.zip"
out_zip = ROOT / "submissions" / "dist" / "v1_6_damped_facets.zip"

damp = r'''

def _damped_core(subject, item, labeled):
    """Shrink sparse evidence updates toward the cold-start posterior.

    The ALC weights budgets 1 and 3 heavily; one revealed Bernoulli outcome is
    too noisy to justify the full nonlinear item update.  The schedule reaches
    the original predictor by budget 31 and leaves budget 0 bit-identical.
    """
    labels = labeled or []
    n = len(labels)
    p = float(_core(subject, item, labels))
    if n <= 0:
        return p
    p0 = float(_core(subject, item, []))
    alpha = {1: 0.45, 2: 0.58, 3: 0.70, 4: 0.76, 5: 0.80,
             6: 0.83, 7: 0.86, 8: 0.88, 9: 0.90, 10: 0.92,
             11: 0.93, 12: 0.94, 13: 0.95, 14: 0.96, 15: 0.965}.get(n, 1.0)
    return p0 + alpha * (p - p0)
'''

with zipfile.ZipFile(src_zip, "r") as zin:
    files = {name: zin.read(name) for name in zin.namelist()}
model = files["model.py"].decode("utf-8")
model = model.replace("\ndef _finish(p):\n", damp + "\ndef _finish(p):\n", 1)
old = "return _finish(_core(subject, item, labeled))"
new = "return _finish(_damped_core(subject, item, labeled))"
if old not in model:
    raise RuntimeError("predict insertion point not found")
model = model.replace(old, new, 1)
files["model.py"] = model.encode("utf-8")
params = json.loads(files["v1_2_params.json"].decode("utf-8"))
params["_source"] = "v1_3 named facets/no-guard plus sparse-label posterior damping; experimental"
files["v1_2_params.json"] = (json.dumps(params, indent=1) + "\n").encode("utf-8")
with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zout:
    for name, data in sorted(files.items()):
        zout.writestr(name, data)
print(out_zip)
