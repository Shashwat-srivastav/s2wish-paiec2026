"""The shipped candidate archive submissions/dist/v1_3_named_facets_noguard.zip == the research engine.

1. Hash: the archive is byte-for-byte the one quoted in the report (SHA-256 a4b0e841...).
2. Contents: model.py and labeling.py are byte-identical to v1_2.zip; the parameters differ only in
   facets true -> "named" and guard 1 -> 0.
3. Probability equivalence: the archive, imported from a scratch extraction and driven through the simulator's
   official-client semantics (run_direct, accept-all acquisition), matches the vectorised engine
   src/paiec/models/v1_2.py with cfg_of(facets="all", guard=0) on real episodes at every budget: max |dp| < 1e-9.
   The packaged model tests "facets" for truthiness, so "named" means all item_features fields; the test also
   checks that facets actually change predictions (non-vacuous) and that the engine's literal "named" mode does
   NOT match (it would be a different model).

Needs the gated public pool under data/ for part 3 (skipped otherwise).
Run: python -m pytest -q tests/test_v1_3_archive.py   (~15-40 s)
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

from paiec import paths

try:  # the engine imports the organisers' evaluator client (external/starting_kit); not redistributed here
    from paiec.evaluation import episodes as E
    _KIT_ERROR = None
except ImportError as exc:
    E, _KIT_ERROR = None, exc
needs_kit = pytest.mark.skipif(E is None, reason="requires the organisers' starting kit in external/starting_kit "
                               "(download from https://www.codabench.org/competitions/17828/)")

ARCHIVE = paths.DIST / "v1_3_named_facets_noguard.zip"
V1_2 = paths.DIST / "v1_2.zip"
SHA256 = "a4b0e841d4a344ff207bed2c855342212e95f7d550195b667446ea4e92685b86"
TOL = 1e-9
CASES = [("real_webagents", 0, 6), ("matharena", 3, 4)]


def test_v1_3_archive_hash():
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == SHA256


def test_v1_3_archive_contents_vs_v1_2():
    a, b = zipfile.ZipFile(ARCHIVE), zipfile.ZipFile(V1_2)
    assert sorted(a.namelist()) == ["labeling.py", "model.py", "v1_2_params.json"]
    for name in ("model.py", "labeling.py"):
        assert a.read(name) == b.read(name), name
    pa, pb = json.loads(a.read("v1_2_params.json")), json.loads(b.read("v1_2_params.json"))
    diff = {k: (pb.get(k), pa.get(k)) for k in set(pa) | set(pb) if pa.get(k) != pb.get(k)}
    assert diff == {"facets": (True, "named"), "guard": (1, 0)}


@pytest.fixture(scope="module")
def package():
    """Extract the archive under the repo's .tmp/ (never the system temp) and import it."""
    from paiec.evaluation import simulator as sim
    base = paths.ROOT / ".tmp"
    base.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="test_v1_3_", dir=base) as tmp:
        zipfile.ZipFile(ARCHIVE).extractall(tmp)
        model = sim.load_model_module(Path(tmp) / "model.py")
        label = sim.load_model_module(Path(tmp) / "labeling.py")
        params = json.loads((Path(tmp) / "v1_2_params.json").read_text(encoding="utf-8"))
        yield model, label, params


def _run(bench, seed, n_pairs, package):
    from paiec.evaluation import simulator as sim
    from paiec.models import v1_2 as V
    model, label, params = package
    D = E.compact(bench)
    n = min(n_pairs, len(D["pair_sid"]))
    sids = {str(s) for s in D["pair_sid"][:n]}
    rows = [r for r in sim.load_benchmark_dir(str(E.DATA / bench)) if str(r["subject_id"]) in sids and r["y"] in (0, 1)]
    pairs = sim.prepare_pairs(rows, seed=seed)
    assert [p.subject_id for p in pairs] == [str(s) for s in D["pair_sid"][:n]], "episode order"
    out = sim.run_direct(pairs, model.predict, label.acquisition_function, isolation="call")
    ep = E.Episodes(D, list(range(n)), [seed] * n, [0] * n)
    y_sim = np.concatenate([[r["y"] for r in p.eval_rows] for p in pairs])
    assert np.array_equal(y_sim, ep.r_y), "row alignment between simulator and engine"
    H = {k: params[k] for k in ("G", "tau", "sigma_beta")}
    common = dict(guard=params["guard"], prior=("t", 3, params["prior_center"], params["prior_scale"]),
                  rho=params["rho0"])
    cfg = V.cfg_of(facets="all", **common)
    cfg_named = V.cfg_of(facets="named", **common)
    fe, fe_named = V.fe_for(bench, D, cfg), V.fe_for(bench, D, cfg_named)
    worst = worst_named = 0.0
    n_facet = 0
    for b in E.BUDGETS:
        p_pkg = np.asarray(out["probabilities"][b], float)
        p_eng = V.predict(ep, H, b, cfg, fe)[ep.r_t]
        assert p_pkg.shape == p_eng.shape
        worst = max(worst, float(np.abs(p_pkg - p_eng).max()))
        worst_named = max(worst_named, float(np.abs(p_pkg - V.predict(ep, H, b, cfg_named, fe_named)[ep.r_t]).max()))
        n_facet += int((np.abs(p_eng - V.predict(ep, H, b, dict(cfg, facets=False))[ep.r_t]) > 1e-6).sum())
    for k, p in enumerate(pairs):  # accept-all reveals exactly the engine's nested prefix
        assert out["per_pair"][p.key]["labels"][31] == min(31, len(ep.acq[k]))
    return worst, worst_named, n_facet


@needs_kit
@pytest.mark.parametrize("bench,seed,n_pairs", CASES)
def test_v1_3_archive_matches_engine(bench, seed, n_pairs, package):
    if not (E.DATA / bench / "response.parquet").exists():
        pytest.skip(f"gated data for {bench} not present")
    assert package[2]["facets"] == "named" and package[2]["guard"] == 0
    worst, worst_named, n_facet = _run(bench, seed, n_pairs, package)
    assert n_facet > 0, "facets never changed a prediction: the test would be vacuous"
    assert worst < TOL, f"{bench}: max |dp| = {worst:.3e}"
    if bench == "matharena":  # the engine's literal 'named' subset is a different model here (EXP-0044: 0.052)
        assert worst_named > 1e-3, worst_named
