"""Shipped submission v1_1 == research engine (src/paiec/models/v1_1.py, config BASE2).

1. Probability equivalence: submissions/v1_1/model.py (+ labeling.py), imported from its
   path and driven through the simulator's official-client semantics (run_direct), vs
   the vectorised engine on the same real episode (split/order seed, pooling group),
   compared row by row at every budget: max |dp| < 1e-9.
2. Parameter provenance: rebuilding v1_1_params.json from the engine
   (paiec.build.v1_1_params) reproduces the shipped file.

3. v1_2 (submissions/v1_2) == engine src/paiec/models/v1_2.py cfg_of(facets="all", guard=1)
   on real episodes (facets exercised) and on synthetic 5->1 key merges (guard exercised);
   v1_2 with item_features stripped and guard off == v1_1 package; params rebuild.

Needs the gated public pool under data/ (or the episode cache); skipped otherwise.
Run: python -m pytest -q tests/test_submission_equivalence.py   (~10-30 s)
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from paiec import paths
from paiec.evaluation import episodes as E

SUB = paths.SUBMISSIONS / "v1_1"
PARAMS = SUB / "v1_1_params.json"
TOL = 1e-9


def _have_data(bench: str) -> bool:
    return (E.DATA / bench / "response.parquet").exists()


def _have_stats() -> bool:
    return all(_have_data(b) or (E.CACHE / f"{b}.npz").exists() for b in E.BENCHES)


def _shipped_engine_H_cfg():
    """Engine hyperparameters/config exactly as shipped (prior from the params file)."""
    from paiec.models import v1_1 as V11
    prm = json.loads(PARAMS.read_text(encoding="utf-8"))
    H = {"G": prm["G"], "tau": prm["tau"], "sigma_beta": prm["sigma_beta"]}
    cfg = dict(V11.BASE2, prior=("t", 3, float(prm["prior_center"]), float(prm["prior_scale"])),
               rho=float(prm["rho0"]))
    assert cfg["rho_pool"] == "momd" and cfg["kappa"] == 1.0 and cfg["w_od"] == 1.0
    assert cfg["tau_scale"] == 1.0 and cfg["item"] is True
    assert prm["clip"] == [0.005, 0.995]
    return H, cfg


def _run_both(bench: str, seed: int, n_pairs: int):
    from paiec.evaluation import simulator as sim
    from paiec.models import v1_1 as V11
    import inspect

    H, cfg = _shipped_engine_H_cfg()
    prm = json.loads(PARAMS.read_text(encoding="utf-8"))
    assert inspect.signature(V11.mom_rho).parameters["k0"].default == prm["rho_k0"]

    D = E.compact(bench)
    n_pairs = min(n_pairs, len(D["pair_sid"]))
    sids = {str(s) for s in D["pair_sid"][:n_pairs]}
    rows = [r for r in sim.load_benchmark_dir(str(E.DATA / bench))
            if str(r["subject_id"]) in sids and r["y"] in (0, 1)]
    pairs = sim.prepare_pairs(rows, seed=seed)
    assert [p.subject_id for p in pairs] == [str(s) for s in D["pair_sid"][:n_pairs]]

    mod = sim.load_model_module(SUB / "model.py")
    lab = sim.load_model_module(SUB / "labeling.py")
    out = sim.run_direct(pairs, mod.predict, lab.acquisition_function, isolation="call")

    ep = E.Episodes(D, list(range(n_pairs)), [seed] * n_pairs, [0] * n_pairs)
    worst = 0.0
    for b in E.BUDGETS:
        p_eng = V11.predict(ep, H, b, cfg)[ep.r_t]          # one value per eval row
        p_sim = np.asarray(out["probabilities"][b], float)
        assert p_sim.shape == p_eng.shape, (b, p_sim.shape, p_eng.shape)
        # same eval rows in the same order (labels must agree row by row)
        y_sim = np.concatenate([[r["y"] for r in p.eval_rows] for p in pairs])
        assert np.array_equal(y_sim, ep.r_y), "row alignment between simulator and engine"
        worst = max(worst, float(np.abs(p_sim - p_eng).max()))
    # the acquisition (accept-all) must reveal exactly the engine's nested prefix
    for k, p in enumerate(pairs):
        assert out["per_pair"][p.key]["labels"][31] == min(31, len(ep.acq[k]))
    return worst


@pytest.mark.parametrize("bench,seed,n_pairs", [("real_webagents", 0, 6), ("matharena", 3, 4)])
def test_v1_1_model_matches_engine(bench, seed, n_pairs):
    if not _have_data(bench):
        pytest.skip(f"gated data for {bench} not present")
    worst = _run_both(bench, seed, n_pairs)
    assert worst < TOL, f"{bench}: max |dp| = {worst:.3e}"


def test_v1_1_params_rebuild_matches_shipped():
    if not _have_stats():
        pytest.skip("gated data / episode cache not present")
    from paiec.build.v1_1_params import build_params
    shipped = json.loads(PARAMS.read_text(encoding="utf-8"))
    rebuilt = build_params()
    assert set(rebuilt) == set(shipped)
    for k, v in shipped.items():
        if isinstance(v, float):
            assert math.isclose(rebuilt[k], v, rel_tol=0, abs_tol=1e-12), (k, rebuilt[k], v)
        else:
            assert rebuilt[k] == v, (k, rebuilt[k], v)


# ---------------------------------------------------------------------------
# v1_2 = engine src/paiec/models/v1_2.py, cfg_of(facets="all", guard=1)
# ---------------------------------------------------------------------------
SUB12 = paths.SUBMISSIONS / "v1_2"
PARAMS12 = SUB12 / "v1_2_params.json"


def _pairs_for(bench, seed, n_pairs, strip_features=False, D=None, row_fn=None):
    from paiec.evaluation import simulator as sim
    D = E.compact(bench) if D is None else D
    n_pairs = min(n_pairs, len(D["pair_sid"]))
    sids = {str(s) for s in D["pair_sid"][:n_pairs]}
    rows = [r for r in sim.load_benchmark_dir(str(E.DATA / bench))
            if str(r["subject_id"]) in sids and r["y"] in (0, 1)]
    if strip_features:
        for r in rows:
            r["item_features"] = ""
    if row_fn is not None:
        for r in rows:
            row_fn(r)
    pairs = sim.prepare_pairs(rows, seed=seed)
    assert [p.subject_id for p in pairs] == [str(s) for s in D["pair_sid"][:n_pairs]]
    return D, n_pairs, pairs


def _run_package(sub, pairs, params=None):
    from paiec.evaluation import simulator as sim
    mod = sim.load_model_module(sub / "model.py")
    if params:
        mod.P.update(params)
    lab = sim.load_model_module(sub / "labeling.py")
    return sim.run_direct(pairs, mod.predict, lab.acquisition_function, isolation="call")


def _merged(bench, m):
    """Synthetic false merges (redteam.rekey random m->1) visible to BOTH sides: the package
    sees each merged key as one item dict (content = new key id, features = the engine's
    representative = first item_id of the key, as in v1_2.key_features)."""
    import pandas as pd
    from paiec.analysis.redteam import rekey
    from paiec.models import v1_2 as V12
    D, K = rekey(bench, "random", m)
    it = pd.read_parquet(E.DATA / bench / "items.parquet", columns=["item_id", "item_features"])
    f = dict(zip(it.item_id, it.item_features.map(E._s)))
    rep, kid = {}, {}
    for iid, k in zip(D["item_ids"], D["item_key"]):
        rep.setdefault(int(k), f.get(iid) or "")
        kid[str(iid)] = int(k)
    kf = {k: {a.strip(): b.strip() for a, b in (kv.split("=", 1) for kv in v.split(";") if "=" in kv)}
          for k, v in rep.items()}     # same parse as v1_2.key_features

    def row_fn(r):
        k = kid[str(r["item_id"])]
        r["item_content"] = f"merged-{k}"
        r["item_features"] = rep[k]
    return D, V12.encode(kf, K, "all"), row_fn


def _run_v1_2_vs_engine(bench, seed, n_pairs, merge=None):
    from paiec.models import v1_2 as V12
    prm = json.loads(PARAMS12.read_text(encoding="utf-8"))
    assert prm["facets"] is True and prm["guard"] == 1
    assert tuple(prm["facet_pis"]) == V12.FACET_PIS and prm["guard_rho"] == V12.GUARD_RHO
    assert np.allclose(np.log(prm["facet_prior_w"]), V12.FACET_PW, rtol=0, atol=1e-15)
    H = {"G": prm["G"], "tau": prm["tau"], "sigma_beta": prm["sigma_beta"]}
    cfg = V12.cfg_of(facets="all", guard=1,
                     prior=("t", 3, float(prm["prior_center"]), float(prm["prior_scale"])),
                     rho=float(prm["rho0"]))
    assert cfg["rho_pool"] == "momd" and cfg["kappa"] == 1.0 and cfg["item"] is True

    if merge:
        D0, fe, row_fn = _merged(bench, merge)
    else:
        D0, fe, row_fn = None, None, None
    D, n_pairs, pairs = _pairs_for(bench, seed, n_pairs, D=D0, row_fn=row_fn)
    out = _run_package(SUB12, pairs)
    ep = E.Episodes(D, list(range(n_pairs)), [seed] * n_pairs, [0] * n_pairs)
    if fe is None:
        fe = V12.fe_for(bench, D, cfg)
    y_sim = np.concatenate([[r["y"] for r in p.eval_rows] for p in pairs])
    assert np.array_equal(y_sim, ep.r_y), "row alignment between simulator and engine"
    worst, n_facet, n_guard = 0.0, 0, 0
    for b in E.BUDGETS:
        p_eng = V12.predict(ep, H, b, cfg, fe)[ep.r_t]
        p_v11 = V12.predict(ep, H, b, dict(cfg, facets=False, guard=0))[ep.r_t]
        p_ng = V12.predict(ep, H, b, dict(cfg, guard=0), fe)[ep.r_t]
        n_facet += int((np.abs(p_ng - p_v11) > 1e-6).sum())
        n_guard += int((np.abs(p_eng - p_ng) > 1e-6).sum())
        p_sim = np.asarray(out["probabilities"][b], float)
        assert p_sim.shape == p_eng.shape, (b, p_sim.shape, p_eng.shape)
        worst = max(worst, float(np.abs(p_sim - p_eng).max()))
    return worst, n_facet, n_guard


@pytest.mark.parametrize("bench,seed,n_pairs", [("real_webagents", 0, 6), ("matharena", 3, 4)])
def test_v1_2_model_matches_engine(bench, seed, n_pairs):
    if not _have_data(bench):
        pytest.skip(f"gated data for {bench} not present")
    worst, n_facet, _ = _run_v1_2_vs_engine(bench, seed, n_pairs)
    assert n_facet > 0, "facets never changed a prediction: the test would be vacuous"
    assert worst < TOL, f"{bench}: max |dp| = {worst:.3e}"


@pytest.mark.parametrize("bench,seed,n_pairs", [("real_webagents", 0, 6), ("matharena", 3, 4)])
def test_v1_2_guard_on_merged_keys_matches_engine(bench, seed, n_pairs):
    """The real episodes above never trigger the key guard; synthetic 5->1 merges do."""
    if not _have_data(bench):
        pytest.skip(f"gated data for {bench} not present")
    worst, _, n_guard = _run_v1_2_vs_engine(bench, seed, n_pairs, merge=5)
    assert n_guard > 0, "guard never changed a prediction: the test would be vacuous"
    assert worst < TOL, f"{bench} merged: max |dp| = {worst:.3e}"


@pytest.mark.parametrize("bench,seed,n_pairs", [("real_webagents", 0, 6), ("matharena", 3, 4)])
def test_v1_2_without_features_equals_v1_1(bench, seed, n_pairs):
    """item_features stripped (facets inert) and guard off -> v1_2 package == v1_1 package;
    at B0 the full v1_2 equals v1_1 exactly on the real inputs."""
    if not _have_data(bench):
        pytest.skip(f"gated data for {bench} not present")
    _, _, pairs = _pairs_for(bench, seed, n_pairs, strip_features=True)
    a = _run_package(SUB12, pairs, params={"guard": 0})
    b = _run_package(SUB, pairs)
    for bud in E.BUDGETS:
        d = np.abs(np.asarray(a["probabilities"][bud]) - np.asarray(b["probabilities"][bud])).max()
        assert d == 0.0, (bud, d)
    _, _, pairs = _pairs_for(bench, seed, n_pairs)
    a = _run_package(SUB12, pairs)
    b = _run_package(SUB, pairs)
    assert np.array_equal(np.asarray(a["probabilities"][0]), np.asarray(b["probabilities"][0]))


def test_v1_2_params_rebuild_matches_shipped():
    if not _have_stats():
        pytest.skip("gated data / episode cache not present")
    from paiec.build.v1_2_params import build_params
    shipped = json.loads(PARAMS12.read_text(encoding="utf-8"))
    rebuilt = build_params()
    assert set(rebuilt) == set(shipped)
    for k, v in shipped.items():
        if isinstance(v, float):
            assert math.isclose(rebuilt[k], v, rel_tol=0, abs_tol=1e-12), (k, rebuilt[k], v)
        else:
            assert rebuilt[k] == v, (k, rebuilt[k], v)
