"""EXP-0049 Stage 0: frozen candidate manifest, metrics-only resource pilot, Stage 1-3 resource plan.

No step here computes, persists or inspects ALC / Brier / ECE.  ``ep.score`` is never called.

    python -m paiec.analysis.exp_0049_stage0 --step manifest       # seconds, no data fitting
    python scripts/run_slot.py -- python -m paiec.analysis.exp_0049_stage0 --step pilot
    python -m paiec.analysis.exp_0049_stage0 --step plan           # reads manifest + pilot JSON
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import time

import numpy as np

from paiec import paths
from paiec.analysis.profile_factorial_resources import _peak_working_set_mb
from paiec.evaluation import episodes as E
from paiec.models import exp_0049_variants as XV
from paiec.models import v1_1 as V
from paiec.models import v1_2 as V2
from paiec.models import x1_prior as X

OUT = paths.RUNS / "preregistration"
MANIFEST = OUT / "exp_0049_manifest_2026-10-15.json"
PILOT = OUT / "exp_0049_resource_pilot_2026-10-15.json"
PLAN = OUT / "exp_0049_resource_plan_2026-10-15.json"
POLICIES = ("A1", "A2", "A3", "A5", "A8")
REGIMES = ("R1", "R2", "R3")

# ---------------------------------------------------------------------------
# Registered levels
# ---------------------------------------------------------------------------
REF_PRIOR = ("t", 3, "half", "learned")
OFFSETS = [  # registration §3 numbering 1..11
    ("O01", "t3(G,sB)", ("t", 3, "G", "sB")),
    ("O02", "t3(G/2,s_half) [reference]", REF_PRIOR),
    ("O03", "t1(G/2,s_half) (= Cauchy)", ("t", 1, "half", "learned")),
    ("O04", "X1-bma", X.CANDS["X1-bma"]["prior"]),
    ("O05", "X1-ext10", X.CANDS["X1-ext10"]["prior"]),
    ("O06", "X1-ext20", X.CANDS["X1-ext20"]["prior"]),
    ("O07", "X1-bma-ext10", X.CANDS["X1-bma-ext10"]["prior"]),
    ("O08", "X1-Gext10", X.CANDS["X1-Gext10"]["prior"]),
    ("O09", "Gumbel(G, sB*sqrt(6)/pi)", ("gumbel", "G", "sB")),
    ("O10", "reg. horseshoe(G/2, s_half, HalfCauchy, slab 3)", ("horseshoe", "half", "learned", 3.0)),
    ("O11", "spike-slab 0.5@G/2 + 0.5 t3(G/2, 2 s_half)", ("spike", "half", "learned", 0.5)),
]
ALIASES = {"Cauchy(G/2,s_half)": (("cauchy", "half", "learned"), "O03")}
FACETS = (("off", False), ("all", "all"), ("named", "named"))
GUARDS = (("off", 0), ("on", 1))
RHOS = (0.0, 0.05, 0.1, 0.2)
PAIRS = ("normal", "t3")
WODS = (0.7, 0.85, 1.0, 1.2)
TAUS = (0.8, 1.0, 1.2)
INTEGS = ("grid", "laplace")
P1 = dict(pair="normal", w_od=0.7, tau_scale=0.8)
P2 = dict(pair="t3", w_od=1.2, tau_scale=1.2)


def canonical(prior):
    """Alias resolution: Cauchy == t1 (same distribution; not a separate level)."""
    if prior[0] == "cauchy":
        return ("t", 1, prior[1], prior[2])
    return prior


def _jsonable(x):
    if isinstance(x, (tuple, list)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    return x


def encode(cfg):
    c = dict(cfg)
    c["prior"] = canonical(c["prior"])
    return json.dumps(_jsonable(c), sort_keys=True, separators=(",", ":"))


def build_manifest():
    ref = XV.REFERENCE
    rows = []

    def add(rid, block, label, **kw):
        cfg = dict(ref)
        cfg.update(kw)
        rows.append(dict(id=rid, block=block, label=label, cfg=cfg))

    add("M00", "reference", "frozen v1_3 = v1_2.cfg_of(facets='all', guard=0) + normal pair + grid")
    for oid, lab, pr in OFFSETS:
        if pr != REF_PRIOR:
            add(oid, "offset", lab, prior=pr)
    for fn, fv in FACETS:
        for gn, gv in GUARDS:
            for r in RHOS:
                add(f"T-{fn}-{gn}-{r:g}", "item", f"facets {fn}; guard {gn}; rho0={r:g}", facets=fv, guard=gv, rho=r)
    for pe in PAIRS:
        for w in WODS:
            for ts in TAUS:
                add(f"P-{pe}-w{w:g}-t{ts:g}", "pair", f"{pe} pair; w_od={w:g}; tau_scale={ts:g}",
                    pair=pe, w_od=w, tau_scale=ts)
    for ig in INTEGS:
        add(f"G-{ig}", "integration", f"{ig} integration", integ=ig)
    # deduplicate main effects against the reference (registered: 'deduplicating the reference setting')
    seen, main, dropped = {}, [], []
    for r in rows:
        e = encode(r["cfg"])
        if e in seen:
            dropped.append(dict(id=r["id"], duplicate_of=seen[e]))
            continue
        seen[e] = r["id"]
        main.append(r)
    inter = []
    tab = [
        ("I01", dict(prior=("t", 3, "G", "sB"), **P1)), ("I02", dict(prior=("t", 3, "G", "sB"), **P2)),
        ("I03", dict(prior=("t", 1, "half", "learned"), **P1)), ("I04", dict(prior=("t", 1, "half", "learned"), **P2)),
        ("I05", dict(prior=("gumbel", "G", "sB"), **P1)), ("I06", dict(prior=("gumbel", "G", "sB"), **P2)),
        ("I07", dict(prior=("spike", "half", "learned", 0.5), **P1)),
        ("I08", dict(prior=("spike", "half", "learned", 0.5), **P2)),
        ("I09", dict(facets=False, guard=0, rho=0.0, **P1)), ("I10", dict(facets=False, guard=0, rho=0.0, **P2)),
        ("I11", dict(facets="all", guard=1, rho=0.1, **P1)), ("I12", dict(facets="all", guard=1, rho=0.1, **P2)),
        ("I13", dict(facets="named", guard=1, rho=0.2, **P1)), ("I14", dict(facets="named", guard=1, rho=0.2, **P2)),
        ("I15", dict(facets=False, guard=1, rho=0.05, **P1)), ("I16", dict(facets=False, guard=1, rho=0.05, **P2)),
    ]
    for rid, kw in tab:
        cfg = dict(ref)
        cfg.update(kw)
        inter.append(dict(id=rid, block="interaction", label=json.dumps(_jsonable(kw)), cfg=cfg))
    allrows = main + inter
    for k, r in enumerate(allrows):
        r["order"] = k
        r["encoding"] = encode(r["cfg"])
        r["sha1"] = hashlib.sha1(r["encoding"].encode()).hexdigest()[:12]
        r["cfg"] = _jsonable(r["cfg"])
    return allrows, dropped


def functional_aliases(rows):
    """Settings that are distinct encodings but identical predictors by code inspection:
    v1_2.item_level applies the guard only inside the rho>0 branch, so guard on == guard off at rho0=0
    (unit-tested on synthetic data).  Recorded for compute reuse; rows are NOT removed (registered levels)."""
    out = []
    by = {}
    for r in rows:
        c = dict(r["cfg"])
        if c["rho"] == 0.0:
            c["guard"] = 0
        by.setdefault(json.dumps(c, sort_keys=True), []).append(r["id"])
    for ids in by.values():
        if len(ids) > 1:
            out.append(ids)
    return out


def validate(rows, dropped):
    enc = [r["encoding"] for r in rows]
    main = [r for r in rows if r["block"] != "interaction"]
    inter = [r for r in rows if r["block"] == "interaction"]
    ref_enc = encode(XV.REFERENCE)
    alias_ok = encode(dict(XV.REFERENCE, prior=ALIASES["Cauchy(G/2,s_half)"][0])) == \
        next(r["encoding"] for r in rows if r["id"] == "O03")
    nblocks = {b: sum(r["block"] == b for r in main) for b in ("reference", "offset", "item", "pair", "integration")}
    checks = dict(
        n_main=len(main), n_interaction=len(inter), n_total=len(rows),
        unique_encodings=len(set(enc)) == len(enc),
        unique_ids=len({r["id"] for r in rows}) == len(rows),
        interactions_unique_and_not_main=len({r["encoding"] for r in inter}) == 16
        and not ({r["encoding"] for r in inter} & {r["encoding"] for r in main}),
        reference_first=rows[0]["encoding"] == ref_enc,
        main_effect_block_sizes=nblocks,
        reference_duplicates_dropped=[d["id"] for d in dropped],
        t1_equals_cauchy_alias=alias_ok,
        main_one_block_from_reference=all(
            sum(1 for blk in _blocks_changed(r["cfg"])) <= 1 for r in main),
        interactions_two_blocks_from_reference=all(len(_blocks_changed(r["cfg"])) == 2 for r in inter),
    )
    assert checks["n_main"] == 58 and checks["n_interaction"] == 16, checks
    assert checks["unique_encodings"] and checks["unique_ids"] and checks["interactions_unique_and_not_main"]
    assert checks["reference_first"] and checks["t1_equals_cauchy_alias"]
    assert checks["main_one_block_from_reference"] and checks["interactions_two_blocks_from_reference"]
    return checks


BLOCK_KEYS = dict(offset=("prior",), item=("facets", "guard", "rho"), pair=("pair", "w_od", "tau_scale"),
                  integration=("integ",))


def _blocks_changed(cfg):
    ref = _jsonable(XV.REFERENCE)
    cfg = _jsonable(cfg)
    return [b for b, ks in BLOCK_KEYS.items() if any(cfg[k] != ref[k] for k in ks)]


def manifest():
    rows, dropped = build_manifest()
    checks = validate(rows, dropped)
    out = dict(
        experiment_id="EXP-0049", stage="0", frozen="2026-10-15",
        quality_metrics_recorded=False,
        registration=["docs/preregistrations/2026-10-14_comprehensive_research_preregistration.md#3",
                      "docs/preregistrations/2026-10-14_staged_factorial_amendment.md"],
        reference=dict(archive="submissions/dist/v1_3_named_facets_noguard.zip", engine_cfg=_jsonable(XV.REFERENCE),
                       note="packaged facets='named' is truthy = all facets (EXP-0044 addendum); "
                            "manifest level 'named' is the engine's genuine named-only mode"),
        candidate_table=rows,
        aliases={k: dict(spec=_jsonable(v[0]), equals=v[1]) for k, v in ALIASES.items()},
        functional_aliases=functional_aliases(rows),
        validation=checks,
        frozen_numerics=dict(
            mu_grid="linspace(-7, 5, 121)", item_and_normal_pair_quadrature="Gauss-Hermite 20 (probabilists')",
            t3_pair_quadrature=f"Gauss-Chebyshev-U {XV.T3_NODES} nodes, theta = sqrt(3) x / sqrt(1-x^2)",
            horseshoe=f"trapezoid over u=log(lambda) in [{XV.HS_U_LO},{XV.HS_U_HI}], du={XV.HS_DU}",
            spike="one extra mu node at exactly G/2 carrying mass 0.5",
            laplace=f"Newton tol {XV.LAPLACE_TOL}, max {XV.LAPLACE_MAXIT} it, step clip {XV.LAPLACE_STEP}; "
                    "mu: 3-point parabola at grid argmax, moment-matched fallback, Gaussian discretised on the grid",
            laplace_scope="normal pair effect and grid priors only (the only registered Laplace rows)",
            prediction_clip=[0.005, 0.995]),
        design=dict(
            policies=list(POLICIES), policy_tie_order=list(POLICIES), budgets=list(E.BUDGETS),
            alc_weights=E.W.tolist(), outer_folds=list(E.EVAL_FOLDS), training_only=["swe_rebench"],
            inner_folds={o: [b for b in E.EVAL_FOLDS if b != o] for o in E.EVAL_FOLDS},
            inner_hyperparameters="ladder.fit_hyper(BENCHES minus {outer, inner}) (exp_0049_variants.nested_H)",
            stage1=dict(seeds=dict(R1=[0], R2=[0], R3=[0]), cells=74 * 5 * 4 * 3 * 3,
                        comparator="frozen v1_3 (M00) under A1 on identical inner-fold episodes",
                        pass_rule="pooled inner dALC <= -0.0005, >= 2/3 inner-fold means < 0, no inner-fold mean > +0.0005",
                        advance="baseline + at most 8 passing configs ranked by best passing policy, then table order"),
            stage2=dict(seeds=dict(R1=list(range(10)), R2=list(range(10)), R3=list(range(10))), cells=9 * 5 * 4 * 3 * 30,
                        select="lowest pooled inner dALC among passing pairs; ties: table order, then A1/A2/A3/A5/A8"),
            stage3=dict(seeds=dict(R1=list(range(100)), R2=list(range(10)), R3=list(range(10))), cells=2 * 1 * 4 * 120,
                        adoption="mean dALC <= -0.001, pair-bootstrap 95% CI upper < 0, >= 3/4 fold means < 0, "
                                 "no fold delta > +0.0005; selection required in >= 3/4 inner loops"),
            bootstrap="v1_1.boot: 2000 resamples, numpy default_rng(0), pair-level within folds (and hierarchical "
                      "fold+pair as secondary); 4-fold t interval with t_{0.975,3}=3.182446305284263",
        ),
        output_schema=dict(
            per_cell=["stage", "outer", "inner", "regime", "seed", "row_id", "policy", "n_pairs",
                      "pair_alc[float32 n_pairs]", "pair_brier_by_budget[float32 n_pairs x 6]", "seconds",
                      "failed", "error"],
            container="one .npz per (stage, outer, inner, regime) + JSON summary; per-instance rows not stored"),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"manifest: {MANIFEST} rows={len(rows)} checks={checks}")
    return out


# ---------------------------------------------------------------------------
# Metrics-only resource pilot (EXP-0048 locked-protocol style)
# ---------------------------------------------------------------------------
PILOT_ROWS = ("M00", "O09", "O10", "O11", "P-t3-w1-t1", "G-laplace", "T-named-on-0.2")


def _structure(D, regime, seeds):
    ep = E.build(D, regime, seeds)
    _, gs = np.unique(ep.inst_group, return_counts=True)
    return dict(n_inst=int(ep.n), n_pairs=int(len(np.unique(ep.inst_pair))), n_groups=int(len(gs)),
                sum_group_sq=int((gs.astype(np.int64) ** 2).sum()), n_targets=int(len(ep.t_key)),
                n_eval_rows=int(len(ep.r_y)))


def pilot():
    rows = {r["id"]: r for r in json.loads(MANIFEST.read_text(encoding="utf-8"))["candidate_table"]}
    cfgs = {}
    for rid in PILOT_ROWS:
        c = dict(rows[rid]["cfg"])
        c["prior"] = tuple(c["prior"]) if c["prior"][0] != "x1" else ("x1", tuple(tuple(x) for x in c["prior"][1]))
        cfgs[rid] = c
    t_start = time.perf_counter()
    setup = {}
    cells, failures, peak_after = [], [], {}
    data, hyper = {}, {}
    t0 = time.perf_counter()
    for B in E.EVAL_FOLDS:
        data[B] = E.compact(B)
        hyper[B] = V.H_fold(B)
    setup["load_and_lobo_hyper"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    for o in E.EVAL_FOLDS:
        for i in E.EVAL_FOLDS:
            if i != o:
                XV.nested_H(o, i)
    setup["nested_hyper_12_fits"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    for B in E.EVAL_FOLDS:
        for rid, c in cfgs.items():
            XV.offset_prior(hyper[B], c["prior"])
    setup["offset_prior_eval_all_pilot_rows_4_folds"] = time.perf_counter() - t0
    fe_cache, ep_build = {}, {}
    for R in REGIMES:
        for B in E.EVAL_FOLDS:
            t0 = time.perf_counter()
            ep0 = E.build(data[B], R, [0])
            ep_build[f"{R}|{B}"] = time.perf_counter() - t0
            for rid, c in cfgs.items():
                key = (B, c["facets"])
                if key not in fe_cache:
                    t1 = time.perf_counter()
                    fe_cache[key] = V2.fe_for(B, data[B], c)
                    setup[f"facet_encode|{B}|{c['facets']}"] = time.perf_counter() - t1
                fe = fe_cache[key]
                for pol in POLICIES:
                    ts = time.perf_counter()
                    try:
                        ep = copy.copy(ep0)
                        tp = time.perf_counter()
                        XV.simulate_policy(ep, hyper[B], c, pol)
                        t_pol = time.perf_counter() - tp
                        preds = {b: XV.predict(ep, hyper[B], b, c, fe) for b in E.BUDGETS}
                        ok = all(np.isfinite(p).all() for p in preds.values())
                        del preds
                        err = None if ok else "non-finite prediction"
                    except Exception as ex:  # preserve failures, never substitute
                        t_pol, ok, err = float("nan"), False, repr(ex)
                    sec = time.perf_counter() - ts
                    cells.append(dict(regime=R, benchmark=B, row=rid, policy=pol, seconds=sec,
                                      policy_replay_seconds=t_pol, ok=ok))
                    if not ok:
                        failures.append(dict(regime=R, benchmark=B, row=rid, policy=pol, error=err))
            peak_after[f"{R}|{B}"] = _peak_working_set_mb()
            print(f"[pilot] {R} {B} done {time.perf_counter()-t_start:.0f}s peak={peak_after[f'{R}|{B}']:.1f}MiB",
                  flush=True)
    t0 = time.perf_counter()
    struct = {}
    for B in E.EVAL_FOLDS:
        D = data[B]
        r1 = []
        for s in range(100):
            pk, _ = E.r1_draw(D, s)
            r1.append(len(pk))
        struct[B] = dict(R1_seed0=_structure(D, "R1", [0]), R2_seed0=_structure(D, "R2", [0]),
                         R3_seed0=_structure(D, "R3", [0]), R1_pairs_seeds_0_99=r1,
                         n_pairs_total=int(len(D["pair_sid"])))
    setup["structure_counts"] = time.perf_counter() - t0
    secs = np.array([c["seconds"] for c in cells])
    by_row = {}
    for rid in cfgs:
        s = np.array([c["seconds"] for c in cells if c["row"] == rid])
        by_row[rid] = dict(n=len(s), sum=float(s.sum()), mean=float(s.mean()), median=float(np.median(s)),
                           p95=float(np.percentile(s, 95)), max=float(s.max()))
    report = dict(
        experiment_id="EXP-0049", stage="0 metrics-only resource pilot", preregistered=True,
        quality_metrics_recorded=False, score_function_called=False,
        protocol="pre-registration §8 EXP-0048 locked style: seed 0 per regime, 4 LOBO folds, 5 policies, 6 budgets; "
                 "cell = policy replay + 6 budget predictions; data load / episode build / hyper fits timed separately",
        engine="paiec.models.exp_0049_variants", manifest=str(MANIFEST.relative_to(paths.ROOT)),
        rows={rid: dict(label=rows[rid]["label"], cfg=rows[rid]["cfg"]) for rid in cfgs},
        regimes={R: [0] for R in REGIMES}, folds=list(E.EVAL_FOLDS), policies=list(POLICIES), budgets=list(E.BUDGETS),
        machine=dict(platform=platform.platform(), cpu_count=__import__("os").cpu_count(),
                     blas_threads=__import__("os").environ.get("OMP_NUM_THREADS")),
        setup_seconds=setup, episode_build_seconds=ep_build,
        cell_seconds=dict(count=len(cells), sum=float(secs.sum()), mean=float(secs.mean()),
                          median=float(np.median(secs)), p95=float(np.percentile(secs, 95)), max=float(secs.max())),
        by_row=by_row, failures=failures, structure=struct,
        total_seconds=time.perf_counter() - t_start, peak_working_set_mb=_peak_working_set_mb(),
        peak_working_set_mb_after_block=peak_after, cells=cells,
        limitations=[
            "Timing only; predictions are discarded and ep.score (Brier/ECE) is never called, so scoring time is excluded.",
            "One seed per regime; R1 cost for other seeds is extrapolated from structural draw sizes.",
            "Run under the shared 2-slot gate with 3 BLAS threads; another agent's job may share the CPU.",
            "Inner-fold cells use nested hyperparameters; their per-cell cost is assumed equal to the LOBO cell of the same benchmark.",
        ],
    )
    for _ in range(3):
        PILOT.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        report["output_bytes"] = PILOT.stat().st_size
    print(f"pilot: {PILOT} cells={len(cells)} total={report['total_seconds']:.1f}s bytes={report['output_bytes']}")
    return report


# ---------------------------------------------------------------------------
# Resource plan
# ---------------------------------------------------------------------------
def _cost_model(cells, struct, rid, pol):
    """Per (row, policy): fit seconds = a * n_inst + b * sum(group size^2) over the 12 (regime, bench) cells
    (non-negative least squares).  Used only to extrapolate R1 seeds other than 0."""
    from scipy.optimize import nnls
    A, y = [], []
    for c in cells:
        if c["row"] == rid and c["policy"] == pol:
            s = struct[c["benchmark"]][f"{c['regime']}_seed0"]
            A.append([s["n_inst"], s["sum_group_sq"]]); y.append(c["seconds"])
    coef, _ = nnls(np.array(A, float), np.array(y, float))
    return coef


def _row_class(cfg):
    """Pilot cost class(es) whose measured cost ratio applies to a manifest row."""
    cls = []
    if cfg["integ"] == "laplace":
        cls.append("G-laplace")
    if cfg["pair"] == "t3":
        cls.append("P-t3-w1-t1")
    k = cfg["prior"][0]
    if k in ("gumbel", "horseshoe", "spike"):
        cls.append({"gumbel": "O09", "horseshoe": "O10", "spike": "O11"}[k])
    if cfg["facets"] == "named" or cfg["guard"] == 1:
        cls.append("T-named-on-0.2")
    return cls


def plan(allowance_h=70.0):
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    pil = json.loads(PILOT.read_text(encoding="utf-8"))
    cells, struct = pil["cells"], pil["structure"]
    cost = {}   # (row, policy, regime, bench) -> seconds (seed 0)
    for c in cells:
        cost[(c["row"], c["policy"], c["regime"], c["benchmark"])] = c["seconds"]
    pil_rows = list(pil["rows"])
    ref_sum = {pol: sum(cost[("M00", pol, R, B)] for R in REGIMES for B in E.EVAL_FOLDS) for pol in POLICIES}

    def row_cost(cfg, pol, R, B):
        """Seed-0 seconds: reference cell x product of measured class ratios (ratio of 12-cell sums per policy)."""
        base = cost[("M00", pol, R, B)]
        cls = _row_class(cfg)
        if len(cls) == 1:
            return cost[(cls[0], pol, R, B)]
        x = base
        for k in cls:
            x *= cost[(k, pol, R, B)] / base
        return x

    # R1 seed scaling per (row class key, policy, bench) via the a*n + b*sum(g^2) model
    def r1_factor(cls_row, pol, B, seeds):
        a, b = _cost_model(cells, struct, cls_row, pol)
        k0 = struct[B]["R1_seed0"]["n_inst"]
        ks = np.array(struct[B]["R1_pairs_seeds_0_99"])[seeds]
        f0 = a * k0 + b * k0 * k0
        if f0 <= 0:
            return float(len(seeds))
        return float((a * ks + b * ks * ks).sum() / f0)

    def seeds_cost(cfg, pol, B, nseeds):
        """Seconds for one (config, policy, benchmark) over nseeds dict {R: n} seeds."""
        cls = _row_class(cfg) or ["M00"]
        tot = 0.0
        for R, n in nseeds.items():
            c0 = row_cost(cfg, pol, R, B)
            if R == "R1":
                tot += c0 * r1_factor(cls[0], pol, B, list(range(n)))
            else:
                tot += c0 * n
        return tot

    def inst_count(B, nseeds):
        s = struct[B]
        n = 0
        for R, k in nseeds.items():
            if R == "R1":
                n += int(np.sum(s["R1_pairs_seeds_0_99"][:k]))
            else:
                n += s[f"{R}_seed0"]["n_inst"] * k
        return n

    rows = man["candidate_table"]
    # Stage 1: every row x policy x regime(seed 0) x (outer, inner): inner bench b appears for 3 outers
    s1 = {}
    for r in rows:
        s1[r["id"]] = sum(3 * row_cost(r["cfg"], pol, R, B)
                          for pol in POLICIES for R in REGIMES for B in E.EVAL_FOLDS)
    s1_total = sum(s1.values())
    n_alias_saved = sum(len(g) - 1 for g in man["functional_aliases"])
    alias_ids = {i for g in man["functional_aliases"] for i in g[1:]}
    s1_reuse = s1_total - sum(s1[i] for i in alias_ids)
    s1_cells = len(rows) * 5 * 4 * 3 * 3
    # Stage 2: baseline + 8 configs, 10 seeds per regime, 3 inner per outer -> 3 x each bench
    ten = dict(R1=10, R2=10, R3=10)
    per_cfg2 = {r["id"]: sum(3 * seeds_cost(r["cfg"], pol, B, ten) for pol in POLICIES for B in E.EVAL_FOLDS)
                for r in rows}
    ranked = sorted((v, k) for k, v in per_cfg2.items() if k != "M00")
    s2_worst = per_cfg2["M00"] + sum(v for v, _ in ranked[-8:])
    s2_typical = per_cfg2["M00"] + 8 * float(np.median([v for v, _ in ranked]))
    # Stage 3: outer bench, R1 100 / R2 10 / R3 10; ref under A1 + selected (worst case) row-policy
    full = dict(R1=100, R2=10, R3=10)
    ref_cfg = rows[0]["cfg"]
    s3_ref = sum(seeds_cost(ref_cfg, "A1", B, full) for B in E.EVAL_FOLDS)
    sel = max(((sum(seeds_cost(r["cfg"], pol, B, full) for B in E.EVAL_FOLDS), r["id"], pol)
               for r in rows for pol in POLICIES))
    s3_worst = s3_ref + sel[0]
    s3_typical = 2 * s3_ref
    # setup overheads (per process): nested hyper fits + data load (measured)
    setup = sum(v for k, v in pil["setup_seconds"].items() if k != "structure_counts")
    # output: per cell pair_alc + 6 pair-brier float32 per pair
    pairs_bytes = lambda B, ns: 7 * 4 * inst_count(B, ns)  # noqa: E731 (instances ~ pairs per seed)
    one = dict(R1=1, R2=1, R3=1)
    out1 = len(rows) * 5 * 3 * sum(pairs_bytes(B, one) for B in E.EVAL_FOLDS)
    out2 = 9 * 5 * 3 * sum(pairs_bytes(B, ten) for B in E.EVAL_FOLDS)
    out3 = 2 * sum(pairs_bytes(B, full) for B in E.EVAL_FOLDS)
    h = lambda s: s / 3600.0  # noqa: E731
    stages = dict(
        stage1=dict(cells=s1_cells, serial_h=h(s1_total), serial_h_with_alias_reuse=h(s1_reuse),
                    two_slot_h=h(s1_total) / 2, alias_rows_reusable=sorted(alias_ids),
                    output_mb=out1 / 2**20,
                    most_expensive_rows_h=sorted(((h(v), k) for k, v in s1.items()), reverse=True)[:8]),
        stage2=dict(cells=9 * 5 * 4 * 3 * 30, serial_h_worst=h(s2_worst), serial_h_typical=h(s2_typical),
                    two_slot_h_worst=h(s2_worst) / 2, output_mb=out2 / 2**20,
                    worst_case_rows=[k for _, k in ranked[-8:]]),
        stage3=dict(cells=2 * 1 * 4 * 120, serial_h_worst=h(s3_worst), serial_h_typical=h(s3_typical),
                    two_slot_h_worst=h(s3_worst) / 2, worst_selected=dict(row=sel[1], policy=sel[2]),
                    output_mb=out3 / 2**20),
    )
    tot_worst = h(s1_total) + h(s2_worst) + h(s3_worst)
    tot_typ = h(s1_total) + h(s2_typical) + h(s3_typical)
    rep = dict(
        experiment_id="EXP-0049", stage="0 resource plan (not an authorization)",
        quality_metrics_recorded=False, pilot=str(PILOT.relative_to(paths.ROOT)),
        method=[
            "Seed-0 cell seconds measured per (pilot row, policy, regime, benchmark).",
            "Unmeasured rows: reference cell x product of measured per-cell class ratios (Laplace, t3 pair, "
            "Gumbel/horseshoe/spike prior, named-facet/guard item); delegated t/x1 priors and other item/pair "
            "levels priced at the reference cell (same code path and array sizes).",
            "Inner-fold cell of benchmark b priced as the LOBO cell of b; each b is an inner fold for 3 outers.",
            "R2/R3 seeds scale linearly (one group per pair / per seed). R1 seeds scale by a*k + b*k^2 fitted "
            "(NNLS) on the 12 seed-0 cells per row-policy, with k = drawn pairs per R1 seed 0..99.",
            "Scoring (ep.score) and result writing are excluded (not timed in a metrics-only pilot).",
            "Two-slot time assumes perfect split over both machine-wide slots with no contention.",
        ],
        per_process_setup_s=setup,
        stages=stages,
        total=dict(cells=30480, serial_h_worst=tot_worst, serial_h_typical=tot_typ,
                   two_slot_h_worst=tot_worst / 2, two_slot_h_typical=tot_typ / 2,
                   allowance_h=allowance_h, within_allowance_serial_worst=tot_worst <= allowance_h,
                   within_allowance_two_slot_worst=tot_worst / 2 <= allowance_h),
        memory=dict(pilot_peak_working_set_mb=pil["peak_working_set_mb"],
                    two_slot_peak_mb=2 * pil["peak_working_set_mb"], contract_limit_mb_per_process=1024),
        output=dict(total_mb=(out1 + out2 + out3) / 2**20, contract_limit_mb=50,
                    pilot_report_bytes=pil.get("output_bytes")),
    )
    PLAN.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("stages", "total", "memory", "output")}, indent=1))
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="manifest")
    for s in ap.parse_args().step.split(","):
        globals()[s]()
