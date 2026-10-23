import importlib.util
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from paiec.evaluation import simulator as S

ROOT = Path(__file__).resolve().parents[1]
TMP = Path(tempfile.mkdtemp(prefix="paiec_compare_"))


def unpack(name, archive):
    d = TMP / name
    d.mkdir()
    zipfile.ZipFile(ROOT / archive).extractall(d)
    return d


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v13 = unpack("v13", "submissions/dist/v1_3_named_facets_noguard.zip")
v17 = unpack("v17", "submissions/dist/v1_7_adaptive_facets.zip")
mods = {
    "v13": (load("v13_model", v13 / "model.py"), load("v13_label", v13 / "labeling.py")),
    "v17": (load("v17_model", v17 / "model.py"), load("v17_label", v17 / "labeling.py")),
}

for bench in ("matharena", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench"):
    rows = [r for r in S.load_benchmark_dir(str(ROOT / "data" / bench)) if r["y"] in (0, 1)]
    pairs = S.prepare_pairs(rows, seed=0)
    # Keep this a fast falsification screen: two prepared pairs, identical trajectories.
    pairs = pairs[:2]
    results = {}
    for name, (model, label) in mods.items():
        out = S.run_direct(pairs, model.predict, label.acquisition_function, isolation="call")
        vals = [v["alc"] for v in out["per_pair"].values()]
        results[name] = float(np.mean(vals))
    print(bench, results, "delta_v17_minus_v13", results["v17"] - results["v13"], flush=True)
