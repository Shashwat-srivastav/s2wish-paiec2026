"""Pin the maths of paiec.feedback: synthetic round trips, the reflection pitfall, edge cases, and a regression on
our own saved formative feedback (FACTS 2026-10-16).

Run:  python -m pytest -q tests/test_feedback_inversion.py
Archives are extracted under .tmp/test_feedback_inversion (never to the system temp dir).
"""
from __future__ import annotations

import importlib.util
import itertools
import zipfile
from pathlib import Path

import numpy as np
import pytest

from paiec.evaluation import metric as official
from paiec.feedback import (BUDGETS, NotIdentifiable, accuracy_from_brier, alc, benchmark_multiplicity, brier,
                            brier_constant, budget0_delta, constancy_residual, ece, feedback_only_solutions,
                            floor_from_feedback, half_ulp, invert_pair, is_nested, load_feedback, pair_constant_floor,
                            pair_level_bounds, recover_k_path, reflect, replay, replay_curve)

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / ".tmp" / "test_feedback_inversion"
R6 = lambda x: round(float(x), 6)  # the platform logs report 6 decimals  # noqa: E731


# ----------------------------------------------------------------------------------------------------------------
# Synthetic: simulate pairs, score them the way the platform does, round, invert
# ----------------------------------------------------------------------------------------------------------------
def _sim_pair(rng, n_eval, a_true=None):
    a = rng.beta(0.6, 1.0) if a_true is None else a_true
    y_eval = (rng.random(n_eval) < a).astype(float)
    y_acq = (rng.random(31) < a).astype(float)
    return y_eval, y_acq


def _rule_beta(alpha, beta):
    return lambda k, b: (k + alpha) / (b + alpha + beta)


def test_scoring_matches_official_metric():
    rng = np.random.default_rng(1)
    for _ in range(200):
        y = (rng.random(rng.integers(5, 200)) < rng.random()).astype(float)
        p = rng.random(y.size)
        p[: y.size // 4] = rng.choice([0.1, 0.3, 0.6, 0.7, 1.0], y.size // 4)  # bin edges
        assert abs(brier(y, p) - official.brier_score(y, p)) < 1e-15
        assert abs(ece(y, p) - official.ece(y, p)) < 1e-12


def test_constant_prediction_identities_and_bin_independence():
    rng = np.random.default_rng(2)
    for q in [0.0, 0.1, 0.3, 0.448722, 0.6, 0.7, 0.995, 1.0]:
        y = (rng.random(97) < 0.37).astype(float)
        a = y.mean()
        assert abs(brier(y, q) - brier_constant(a, q)) < 1e-15
        assert abs(ece(y, q) - abs(q - a)) < 1e-15
        for nb in (5, 10, 20):  # all rows share one bin: ECE does not depend on the binning
            assert abs(ece(y, q, nb) - abs(q - a)) < 1e-15


def test_roundtrip_accuracy_from_rounded_feedback():
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(5000):
        p0 = rng.uniform(0.02, 0.98)
        if abs(1 - 2 * p0) < 0.02:
            continue
        y, _ = _sim_pair(rng, int(rng.integers(20, 400)))
        inv = invert_pair(R6(brier(y, p0)), R6(ece(y, p0)), p0)
        err = abs(inv.a_raw - y.mean())
        assert err <= inv.tol - half_ulp() + 1e-15  # A3 bound on identity (1)
        assert inv.consistent
        worst = max(worst, err / (inv.tol - half_ulp()))
    assert worst <= 1.0


def test_unrounded_feedback_is_exact():
    rng = np.random.default_rng(4)
    for _ in range(500):
        p0 = rng.uniform(0.05, 0.45)
        y, _ = _sim_pair(rng, 80)
        inv = invert_pair(brier(y, p0), ece(y, p0), p0)
        assert abs(inv.a_raw - y.mean()) < 1e-12 and inv.disagreement < 1e-12


def test_reflection_pitfall():
    """(a, p0) and (1-a, 1-p0) give identical (B0, ECE0); the ECE cross-check cannot detect a reflected p0."""
    rng = np.random.default_rng(5)
    for _ in range(500):
        p0 = rng.uniform(0.05, 0.45)
        y, _ = _sim_pair(rng, 120)
        a = y.mean()
        b0, e0 = brier(y, p0), ece(y, p0)
        a_m, p_m = reflect(a, p0)
        assert abs(brier_constant(a_m, p_m) - b0) < 1e-12 and abs(abs(a_m - p_m) - e0) < 1e-12
        wrong = invert_pair(b0, e0, 1 - p0)
        assert abs(wrong.a_raw - (1 - a)) < 1e-9 and wrong.consistent  # passes the check, yet wrong
        sols = feedback_only_solutions(b0, e0)
        assert any(abs(s[0] - a) < 1e-7 and abs(s[1] - p0) < 1e-7 for s in sols)
        assert any(abs(s[0] - (1 - a)) < 1e-7 and abs(s[1] - (1 - p0)) < 1e-7 for s in sols)
        assert abs(floor_from_feedback(b0, e0) - a * (1 - a)) < 1e-12  # floor is reflection-invariant


def test_p0_half_not_identifiable():
    for a in (0.0, 0.2, 0.7, 1.0):
        assert abs(brier_constant(a, 0.5) - 0.25) < 1e-15  # B0 = 1/4 whatever a is
    with pytest.raises(NotIdentifiable):
        accuracy_from_brier(0.25, 0.5)
    with pytest.raises(NotIdentifiable):
        invert_pair(0.25, 0.1, 0.5)
    # Near 1/2 the A3 error bound half_ulp/|1-2p0| blows up
    assert invert_pair(0.25, 0.1, 0.4999).tol > 2e-3


@pytest.mark.parametrize("a_true", [0.0, 1.0])
def test_accuracy_at_boundaries(a_true):
    for p0 in (0.05, 0.3, 0.448722, 0.7):
        y = np.full(64, a_true)
        inv = invert_pair(R6(brier(y, p0)), R6(ece(y, p0)), p0)
        assert inv.a == pytest.approx(a_true, abs=inv.tol) and 0.0 <= inv.a <= 1.0
        assert inv.floor == pytest.approx(0.0, abs=2 * inv.tol)


def test_roundtrip_k_recovery_and_exact_replay():
    rng = np.random.default_rng(6)
    n_exact = n_total = 0
    for _ in range(1500):
        rule = _rule_beta(rng.uniform(0.3, 3), rng.uniform(0.3, 3))
        p0 = rule(0, 0)
        if abs(1 - 2 * p0) < 0.05:
            continue
        y, y_acq = _sim_pair(rng, int(rng.integers(40, 300)))
        k_true = {b: int(y_acq[:b].sum()) for b in BUDGETS}
        B = {b: R6(brier(y, rule(k_true[b], b))) for b in BUDGETS}
        E = {b: R6(ece(y, rule(k_true[b], b))) for b in BUDGETS}
        inv = invert_pair(B[0], E[0], p0)
        path = recover_k_path(inv.a, E, rule, tol=inv.k_tol)
        n_total += 1
        for b, s in path.steps.items():  # the true k always lies within tolerance
            r_true = abs(abs(rule(k_true[b], b) - inv.a) - E[b])
            assert r_true <= inv.k_tol
            if s.unique:
                assert s.k == k_true[b]
            assert constancy_residual(B[b], E[b], inv.a) <= 4 * inv.tol
        if path.exact:
            n_exact += 1
            assert path.ks == k_true and path.nested
            # self-replay reproduces the reported curve; an alternative rule matches direct simulation
            self_curve = replay_curve(inv.a, path.ks, rule)
            assert max(abs(self_curve[b] - B[b]) for b in BUDGETS) <= 2 * inv.tol
            alt = _rule_beta(1.0, 2.5)
            rep = replay(inv.a, path.ks, alt, observed=B)
            direct = {b: brier(y, alt(k_true[b], b)) for b in BUDGETS}
            assert max(abs(rep.curve[b] - direct[b]) for b in BUDGETS) <= 2 * inv.tol
            assert rep.delta_alc == pytest.approx(alc(direct) - alc(B), abs=4 * inv.tol)
    assert n_exact / n_total > 0.95


def test_item_varying_prediction_is_detected():
    """A4 violated (prediction varies across scored rows): the constancy residual exposes it."""
    rng = np.random.default_rng(7)
    y, _ = _sim_pair(rng, 200, a_true=0.4)
    p = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.05, y.size), 0.01, 0.99)  # informative item-level prediction
    assert constancy_residual(brier(y, p), ece(y, p), y.mean()) > 1e-3


def test_nesting():
    assert is_nested({0: 0, 1: 1, 3: 2, 7: 5, 15: 5, 31: 20})
    assert not is_nested({0: 0, 1: 0, 3: 3, 7: 0})  # decreasing
    assert not is_nested({0: 0, 1: 1, 3: 4})        # k > b


def test_bounds_floor_and_ceiling():
    rng = np.random.default_rng(8)
    a = rng.beta(0.6, 1.0, 50)
    q = rng.random((50, 6))
    curves = brier_constant(a[:, None], q)
    bd = pair_level_bounds(a, curves)
    assert bd.floor == pytest.approx(pair_constant_floor(a).mean())
    assert bd.ceiling_gain == pytest.approx(bd.floor - bd.observed_alc)
    assert all(v >= 0 for v in bd.excess.values())  # pair-level curves never go below the floor
    oracle = brier_constant(a[:, None], np.repeat(a[:, None], 6, 1))
    assert pair_level_bounds(a, oracle).observed_alc == pytest.approx(bd.floor)  # attained by q = a
    for _ in range(100):
        qq = rng.random((50, 6))
        assert pair_level_bounds(a, brier_constant(a[:, None], qq)).observed_alc >= bd.floor - 1e-15


# ----------------------------------------------------------------------------------------------------------------
# Regression on our own saved feedback
# ----------------------------------------------------------------------------------------------------------------
FB = ROOT / "experiments" / "platform_submissions"
RUNS = {  # version: (feedback file, archive)
    "v1": ("2026-10-01_v1.json", "v1.zip"),
    "v1_1": ("2026-10-01_v1_1.json", "v1_1.zip"),
    "v1_2": ("2026-10-02_v1_2.json", "v1_2.zip"),
    "v1_3": ("2026-10-07_v1_3_named_facets_noguard.json", "v1_3_named_facets_noguard.zip"),
}
try:  # private locator of the explored variant's saved files (not released)
    from paiec._variant import DIR as VARIANT_DIR
except ImportError:
    VARIANT_DIR = ROOT / "explored_variant_feedback"
PROTO_FB = next(iter(sorted(VARIANT_DIR.glob("*_platform_feedback.json"))),
                VARIANT_DIR / "platform_feedback.json")  # explored variant (not released)
PROTO_DIR = next(iter(sorted(p.parent for p in VARIANT_DIR.glob("*/model.py"))),
                 VARIANT_DIR / "explored_variant")  # explored variant's folder (not released)
SUBJ = {"normalized_name": "X", "provider": "Y", "release_date": "2026-01-01", "access_date": "2026-02-01",
        "harness": "h", "reasoning_effort": "low", "harness_version": "1", "subject_features_extra": ""}
_need = [FB / f for f, _ in RUNS.values()] + [ROOT / "submissions" / "dist" / z for _, z in RUNS.values()] + \
        [PROTO_FB, PROTO_DIR / "model.py"]
real = pytest.mark.skipif(not all(p.exists() for p in _need), reason="saved feedback / archives not present")


def _load_module(name, model_py):
    spec = importlib.util.spec_from_file_location(f"fbinv_{name}", model_py)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def models():
    out = {}
    for v, (_, z) in RUNS.items():
        d = TMP / v
        zipfile.ZipFile(ROOT / "submissions" / "dist" / z).extractall(d)
        out[v] = _load_module(v, d / "model.py")
    out["proto"] = _load_module("proto", PROTO_DIR / "model.py")
    return out


def _item(bench, content="q"):
    return {"item_content": content, "item_features": "", "interactors": "", "benchmark_id": bench}


def _p(m, bench, k=0, b=0):
    lab = [((SUBJ, _item(bench, f"z{i}")), 1.0 if i < k else 0.0) for i in range(b)]
    return float(m.predict([SUBJ, _item(bench)], lab))


def _p0_constant(m, benches):
    """Budget-0 prediction from the archive, checked constant over items and subjects (A1)."""
    s2 = dict(SUBJ, normalized_name="Other", provider="Other")
    vals = {float(m.predict([s, _item(b, c)], [])) for s in (SUBJ, s2) for b in benches for c in ("q", "Prove x.")}
    assert max(vals) - min(vals) < 1e-12
    return vals.pop()


@pytest.fixture(scope="module")
def saved(models):
    rows = []
    for v, (f, _) in RUNS.items():
        pairs = load_feedback(FB / f)
        p0 = _p0_constant(models[v], sorted({p.benchmark_id for p in pairs}))
        mult = benchmark_multiplicity(pairs)
        for p in pairs:
            rows.append(dict(ver=v, pair=p, p0=p0, rep=mult[p.benchmark_id],
                             inv=invert_pair(p.brier[0], p.ece[0], p0)))
    return rows


@pytest.fixture(scope="module")
def proto(models):
    pairs = load_feedback(PROTO_FB)
    mult = benchmark_multiplicity(pairs)
    rows = []
    for p in pairs:
        pb = _p(models["proto"], p.benchmark_id)  # one constant per anonymous benchmark ID at budget 0
        rows.append(dict(ver="proto", pair=p, p0=pb, rep=mult[p.benchmark_id], inv=invert_pair(p.brier[0], p.ece[0], pb)))
    return rows


@real
def test_regression_accuracy_recovery_34_pairs(saved):
    assert len(saved) == 34
    assert {round(r["p0"], 6) for r in saved} == {0.400099, 0.448722}
    dis = max(r["inv"].disagreement for r in saved)
    assert dis <= 6e-6 and all(r["inv"].consistent for r in saved)
    assert max(abs(brier_constant(r["inv"].a_raw, r["p0"]) - r["pair"].brier[0]) for r in saved) < 1e-15
    # the 6-decimal p0 printed in our reports gives the same accuracies to 1e-5
    assert max(abs(invert_pair(r["pair"].brier[0], r["pair"].ece[0], round(r["p0"], 6)).a_raw - r["inv"].a_raw)
               for r in saved) < 1e-5


@real
def test_regression_reflection_error(saved):
    """Our 2026-10-15 error: p0 = 0.5513 (= 1 - 0.448722) flips every accuracy yet passes the ECE0 check."""
    for r in saved:
        wrong = invert_pair(r["pair"].brier[0], r["pair"].ece[0], 1 - r["p0"])
        assert wrong.consistent and abs(wrong.a_raw - (1 - r["inv"].a_raw)) < 1e-9
        # B0 - ECE0^2 needs no p0; its A3 error is half_ulp (1 + 2 ECE0), that of a(1-a) is |1 - 2a| tol_a
        e0, a_ = r["pair"].ece[0], r["inv"].a_raw
        bound = half_ulp() * (1 + 2 * e0) + abs(1 - 2 * a_) * (r["inv"].tol - half_ulp())
        assert abs(floor_from_feedback(r["pair"].brier[0], e0) - a_ * (1 - a_)) <= bound
    # corrected facts (34 pair-slots): median accuracy 0.27, 14.7% at <= 0.05; the reflected reading said 0.73, 0%
    a = np.array([r["inv"].a for r in saved])
    assert round(float(np.median(a)), 2) == 0.27 and round(float(np.mean(a <= 0.05)), 3) == 0.147
    assert round(float(np.median(1 - a)), 2) == 0.73 and np.mean(1 - a <= 0.05) == 0


@real
def test_regression_floor_and_ceiling_v1_1_to_v1_3(saved):
    use = [r for r in saved if r["ver"] != "v1"]
    a = np.array([r["inv"].a for r in use])
    curves = np.array([[r["pair"].brier[b] for b in BUDGETS] for r in use])
    bd = pair_level_bounds(a, curves)
    assert bd.n_pairs == 26
    assert round(bd.floor, 3) == 0.151 and round(bd.floor, 4) == 0.1511
    assert round(bd.ceiling_gain, 4) == -0.0218
    assert round(bd.observed_alc, 3) == 0.173
    assert [round(bd.excess[b], 3) for b in BUDGETS] == [0.081, 0.037, 0.021, 0.007, 0.004, 0.0]


@real
def test_regression_variant_budget0_counterfactual(saved, proto, models):
    assert len(proto) == 9
    assert max(r["inv"].disagreement for r in proto) <= 1e-5 and all(r["inv"].consistent for r in proto)
    p0_v13 = next(r["p0"] for r in saved if r["ver"] == "v1_3")
    a = np.array([r["inv"].a_raw for r in proto])
    q_proto = np.array([r["p0"] for r in proto])
    d = budget0_delta(a, q_proto, p0_v13)  # explored variant minus v1_3, exact per pair
    assert np.allclose(brier_constant(a, q_proto), [r["pair"].brier[0] for r in proto], atol=1e-15)
    # exact value -0.00575 (= -0.0057500 to 7 d.p.); an earlier note quoted -0.0058 after rounding each pair to 4 d.p.
    assert round(0.1 * d.mean(), 5) == -0.00575
    _K_REPORT["b0_cf_all9"] = 0.1 * d.mean()
    seen = {(r["pair"].subject_id, r["pair"].benchmark_id) for r in saved}
    oos = np.array([(r["pair"].subject_id, r["pair"].benchmark_id) not in seen for r in proto])
    assert oos.sum() == 7
    assert round(0.1 * d[oos].mean(), 4) == -0.0049
    _K_REPORT["b0_cf_oos7"] = 0.1 * d[oos].mean()
    bd = pair_level_bounds(np.clip(a, 0, 1), np.array([[r["pair"].brier[b] for b in BUDGETS] for r in proto]))
    assert round(bd.floor, 3) == 0.137
    assert round(bd.observed_alc, 6) == pytest.approx(0.150767, abs=1e-6)


@real
def test_regression_k_recovery_singletons(saved, proto, models):
    """Exact k paths on singleton pairs; nested; identical across runs; self-replay reproduces the feedback."""
    exact, paths, vers = {}, {}, {}
    for r in saved + proto:
        if r["rep"] != 1:
            continue
        m, bench, inv = models[r["ver"]], r["pair"].benchmark_id, r["inv"]
        path = recover_k_path(inv.a, r["pair"].ece, lambda k, b: _p(m, bench, k, b), tol=inv.k_tol)
        key = (r["pair"].subject_id, bench)
        paths.setdefault(key, []).append((r["ver"], path))
        if path.exact:
            assert path.nested
            curve = replay_curve(inv.a, path.ks, lambda k, b: _p(m, bench, k, b))
            dev = max(abs(curve[b] - r["pair"].brier[b]) for b in BUDGETS)
            assert dev <= 2 * inv.tol
            _K_REPORT["max_self_replay_dev"] = max(_K_REPORT.get("max_self_replay_dev", 0.0), dev)
            exact.setdefault(key, []).append(path.ks)
            vers.setdefault(key, set()).add(r["ver"])
    for ks in exact.values():
        assert all(k == ks[0] for k in ks)  # same pair, same platform label order -> same k path
    _K_REPORT.update(n_exact=len(exact), n_singleton=len(paths),
                     n_rows_exact=sum(len(v) for v in exact.values()),
                     residuals={f"{v}:{k[0][-6:]}": p.max_residual for k, lst in paths.items() for v, p in lst})
    # 18 distinct exact pairs (21 run-rows) with all five runs; 14 without v1 (the m2 set). Every non-exact
    # singleton has residual >= 5e-4, two orders above the largest exact residual (< 5e-6).
    assert len(exact) == 18 and sum(1 for k in vers if vers[k] - {"v1"}) == 14
    inexact = [p.max_residual for lst in paths.values() for _, p in lst if not p.exact]
    exact_res = [p.max_residual for lst in paths.values() for _, p in lst if p.exact]
    assert max(exact_res) < 5e-6 and min(inexact) > 5e-4


_K_REPORT: dict = {}


def test_zz_print_k_report():
    if _K_REPORT:
        print("\nregression report:", {k: v for k, v in _K_REPORT.items() if k != "residuals"})
        for k, v in sorted(_K_REPORT.get("residuals", {}).items(), key=lambda kv: kv[1]):
            print(f"  {k:16s} max residual {v:.2e}")


@real
def test_regression_variant_full_curve_replay(models):
    """Full-curve (budgets 0-31) exact counterfactual, explored variant vs final model, on that run's exact
    singleton pairs. Same acquisition (labeling.py identical), so the recovered k path applies to both (A5)."""
    pairs = load_feedback(PROTO_FB)
    mult = benchmark_multiplicity(pairs)
    d = {}
    for p in pairs:
        if mult[p.benchmark_id] != 1:
            continue
        rp = lambda k, b, bb=p.benchmark_id: _p(models["proto"], bb, k, b)  # noqa: E731
        inv = invert_pair(p.brier[0], p.ece[0], rp(0, 0))
        path = recover_k_path(inv.a, p.ece, rp, tol=inv.k_tol)
        if path.exact:
            rep = replay(inv.a, path.ks, lambda k, b, bb=p.benchmark_id: _p(models["v1_3"], bb, k, b), observed=p.brier)
            d[p.subject_id] = -rep.delta_alc  # explored variant minus final model
    assert len(d) == 3
    assert round(float(np.mean(list(d.values()))), 4) == -0.0059
    _K_REPORT["full_curve_cf_3pairs"] = float(np.mean(list(d.values())))
    _K_REPORT["full_curve_cf_per_pair"] = {k[-6:]: round(v, 5) for k, v in d.items()}
