import importlib.util
import tempfile
import zipfile
from pathlib import Path

from paiec.evaluation import simulator as S


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


root = Path(__file__).resolve().parents[1]
tmp = Path(tempfile.mkdtemp(prefix="paiec_damp_check_"))
v13 = tmp / "v13"
v16 = tmp / "v16"
v13.mkdir()
v16.mkdir()
zipfile.ZipFile(root / "submissions/dist/v1_3_named_facets_noguard.zip").extractall(v13)
zipfile.ZipFile(root / "submissions/dist/v1_7_adaptive_facets.zip").extractall(v16)
m13 = load("v13_unique", v13 / "model.py")
m16 = load("v16_unique", v16 / "model.py")
rows = [r for r in S.load_benchmark_dir(str(root / "data/real_webagents")) if r["y"] in (0, 1)]
r0, r1 = rows[0], rows[1]
sf = ("normalized_name", "provider", "release_date", "access_date", "harness",
      "harness_version", "reasoning_effort", "subject_features_extra")
ifields = ("item_content", "item_features", "interactors", "benchmark_id")
subject0 = {k: r0[k] for k in sf}
subject1 = {k: r1[k] for k in sf}
item0 = {k: r0[k] for k in ifields}
item1 = {k: r1[k] for k in ifields}
inp = [subject0, item0]
lab = [[subject1, item1], r1["y"]]
print("B0", m13.predict(inp, []), m16.predict(inp, []))
print("B1", m13.predict(inp, [lab]), m16.predict(inp, [lab]))
