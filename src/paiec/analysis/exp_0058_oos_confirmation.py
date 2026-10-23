"""EXP-0058: out-of-sample confirmation on 89 held-out public benchmarks.

Pre-registration (locked, commit bb51964, SHA-256 57c98c9f...129ec9):
docs/preregistrations/EXP-0058_oos_confirmation_preregistration.md

Arms (frozen, no refit): official empirical mean (L0), v0 Beta-binomial m=0.5 k=2 clip .02/.98 (L1),
final = v1_3 engine v1_2.cfg_of(facets="all", guard=0) with the shipped archive parameters (no-facet path:
the enlarged release has no items table). Accept-all acquisition, identical episodes for all arms.
R2 primary (seeds 0-4; pairs with 80-1000 distinct items; <= 200 per benchmark by sha256(subject_id) order),
R1 secondary (EXP-0039R draws, seeds 0-19). Benchmark-level bootstrap (10,000, default_rng(58)).

Run: python scripts/run_slot.py -- python -m paiec.analysis.exp_0058_oos_confirmation
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from paiec import paths
from paiec.analysis import exp_0039_simulator_fidelity as F
from paiec.analysis import exp_0039r_simulator_repair as R
from paiec.evaluation import episodes as E, simulator as S
from paiec.models import v1_2 as V2

PREREG = paths.ROOT / "docs" / "preregistrations" / "EXP-0058_oos_confirmation_preregistration.md"
PREREG_SHA = "57c98c9fba1a4477942facba0095456c53aa86478fbb7285dcf5b2ba51129ec9"
OUT_DIR = paths.RUNS / "EXP-0058"
DIST = paths.DIST
V0_ZIP, V0_SHA = DIST / "v0.zip", "57ad9d986c101cb44d0cfb02913ec1d1aa3de99f257d42f508e745fa8f77dde9"
V13_ZIP, V13_SHA = DIST / "v1_3_named_facets_noguard.zip", "a4b0e841d4a344ff207bed2c855342212e95f7d550195b667446ea4e92685b86"
EMP_MODEL = paths.ROOT / "external" / "official_baselines" / "empirical_mean" / "model.py"
CFG = V2.cfg_of(facets="all", guard=0)
R2_SEEDS, R1_SEEDS = list(range(5)), list(range(20))
MAX_PAIRS, MAX_ITEMS = 200, 1000
N_BOOT, BOOT_SEED, MDE = 10000, 58, 0.001
RELATIVES = ["swe_smith", "swebench_live", "swepolybench", "visualwebarena"]
ARMS = ("official", "v0", "final")
CONTRASTS = (("final", "v0"), ("v0", "official"))
BENCHES = """afrieval agentic_review_perturbation algotune alignment_faking appworld averimatec babilong beavertails benger
bird_sql braveguard bridging_gap cellverse critic_discernment_game crow devbench editbench edu_circuit_hw egoschema
elicitation_game engdesign enginemt_qa exploitgym faithcot fewshot_ttt_bbh financebench finegrain_t2i hallusionbench
harmbench healthadminbench helm_bbq helm_thaiexam hle igakuqa119 ikp jailbreakbench kernelbench legal_rag_bench
llmail_inject llms_lean_formalization mathconstraint mathvista_mini medhelm_ehr_sql medhelm_head_qa medhelm_med_mcqa
medhelm_med_qa medhelm_medbullets medhelm_medcalc_bench medhelm_medec medhelm_medhallu medhelm_pubmed_qa
medhelm_race_based_med mj_bench mmbench_v11 monkey_power_laws morphkv mtbbench naep_llm_students naturebench nester
nyu_ctf_bench openbiorq os_harm perfcodebench planbench proofaug pxplore realtimeqa reliancescope revolve risebench
robust_reasoning_benchmark safeagentbench sib200 software_world_models sustainable_food swe_smith swebench_live
swepolybench tau2_bench taubench truthfulqa_mc tulu_human_eval tumlu visual_riddles visualwebarena vl_rewardbench
workarena workbench_revisited""".split()
assert len(BENCHES) == 89


def log(*a):
    print(*a, flush=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Arms (vectorised)
# ---------------------------------------------------------------------------
def counts(ep, b):
    li, _, ly = ep.labels(b)
    Sx = np.bincount(li, ly, ep.n) if len(li) else np.zeros(ep.n)
    Nx = np.bincount(li, minlength=ep.n).astype(float) if len(li) else np.zeros(ep.n)
    return Sx, Nx


def probs(arm, ep, H):
    out = {}
    for b in E.BUDGETS:
        if arm == "final":
            out[b] = V2.predict(ep, H, b, CFG, None)
            continue
        Sx, Nx = counts(ep, b)
        if arm == "official":
            q = np.where(Nx > 0, Sx / np.maximum(Nx, 1), 0.5)
        else:
            q = np.clip((2.0 * 0.5 + Sx) / (2.0 + Nx), 0.02, 0.98)
        out[b] = q[ep.t_inst]
    return out


def score_all(ep, H):
    return {arm: ep.score(probs(arm, ep, H)) for arm in ARMS}     # (n, 6) Brier per arm


# ---------------------------------------------------------------------------
# Asserts: shipped packages == engines (no-facet path, item_id keys)
# ---------------------------------------------------------------------------
def check_packages(bench="real_webagents", seed=0, n=4):
    D = E.compact(bench)
    sids = [str(s) for s in D["pair_sid"][:n]]
    rows = []
    for r in S.load_benchmark_dir(str(E.DATA / bench)):
        if str(r["subject_id"]) in set(sids) and r["y"] in (0, 1):
            r = dict(r, item_content=str(r["item_id"]), item_features="")   # item_id keys, no facets
            rows.append(r)
    pairs = S.prepare_pairs(rows, seed=seed)
    assert [p.subject_id for p in pairs] == sids
    Dk = dict(D); Dk["item_key"] = np.arange(len(D["item_ids"]), dtype=np.int32)
    Dk["key_jev"] = np.zeros(len(D["item_ids"]))
    ep = E.Episodes(Dk, list(range(n)), [seed] * n, [0] * n)
    assert np.array_equal(np.concatenate([[r["y"] for r in p.eval_rows] for p in pairs]), ep.r_y)
    TMP = paths.ROOT / ".tmp"; TMP.mkdir(exist_ok=True)
    out = {}
    for arm, zp, zsha in (("v0", V0_ZIP, V0_SHA), ("final", V13_ZIP, V13_SHA), ("official", None, None)):
        tmp = tempfile.mkdtemp(prefix=f"exp0058_{arm}_", dir=TMP)
        try:
            if zp is not None:
                assert sha(zp) == zsha, f"{arm} archive hash mismatch"
                with zipfile.ZipFile(zp) as z:
                    z.extractall(tmp)
                model = S.load_model_module(Path(tmp) / "model.py")
                label = S.load_model_module(Path(tmp) / "labeling.py")
            else:
                shutil.copy(EMP_MODEL, Path(tmp) / "model.py")
                model = S.load_model_module(Path(tmp) / "model.py")
                label = None
            # official baseline ships no labeling.py: use the same accept-all hook as v0 to keep labels paired
            acq = label.acquisition_function if label is not None else _accept_all
            pkg = S.run_direct(pairs, model.predict, acq, isolation="call")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        H = F._engine_H(json.loads(zipfile.ZipFile(V13_ZIP).read("v1_2_params.json")))
        eng = probs(arm, ep, H)
        d = max(float(np.abs(np.asarray(pkg["probabilities"][b], float) - eng[b][ep.r_t]).max()) for b in E.BUDGETS)
        out[arm] = dict(max_abs_prob_delta=d, sha256=sha(zp) if zp else sha(EMP_MODEL))
        log(f"  [check] {arm}: max|dp|={d:.2e}")
        assert d < 1e-9, (arm, d)
    return out


def _accept_all(input, *, context=None):
    if context is None:
        return 0.0
    return bool(int(context.get("labels_remaining", 0)) > 0)


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------
def r2_episode(D):
    sizes = np.bincount(D["cell_pair"], minlength=len(D["pair_sid"]))
    ok = [p for p in range(len(sizes)) if E.MIN_ITEMS <= sizes[p] <= MAX_ITEMS]
    ok.sort(key=lambda p: hashlib.sha256(str(D["pair_sid"][p]).encode()).hexdigest())
    ok = ok[:MAX_PAIRS]
    ip, isd, ig, g = [], [], [], 0
    for s in R2_SEEDS:
        for p in ok:
            ip.append(p); isd.append(s); ig.append(g); g += 1
    return R.CellEpisodes(D, ip, isd, ig), len(ok)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def summarise(per_bench, names):
    """per_bench[B] = dict(d_alc={contrast: float}, d_budget={contrast: [6]}). Benchmark bootstrap, rng(58)."""
    out = {}
    idx = np.random.default_rng(BOOT_SEED).integers(0, len(names), (N_BOOT, len(names)))
    for a, b in CONTRASTS:
        key = f"{a}-{b}"
        xs = np.array([per_bench[B]["d_alc"][key] for B in names])
        bm = xs[idx].mean(1)
        lo, hi = np.percentile(bm, [2.5, 97.5])
        neg = int((xs < 0).sum()); nz = int((xs != 0).sum())
        p = float(binomtest(neg, nz, 0.5).pvalue) if nz else 1.0
        per_budget = []
        for j in range(6):
            y = np.array([per_bench[B]["d_budget"][key][j] for B in names])
            ybm = y[idx].mean(1)
            per_budget.append(dict(budget=E.BUDGETS[j], mean=float(y.mean()),
                                   ci95=np.percentile(ybm, [2.5, 97.5]).tolist(), n_neg=int((y < 0).sum())))
        out[key] = dict(n_bench=len(xs), mean=float(xs.mean()), ci95=[float(lo), float(hi)], n_neg=neg,
                        n_nonzero=nz, sign_p=p, row=row(lo, hi), per_budget=per_budget,
                        median=float(np.median(xs)), sd=float(xs.std(ddof=1)))
    return out


def row(lo, hi):
    if hi < 0:
        return "upper < 0: refinement confirmed out of sample"
    if lo > 0:
        return "lower > 0: refinement reversed out of sample"
    if lo >= -MDE and hi <= MDE:
        return "includes 0, within +-0.001: equivalent within the MDE"
    return "includes 0, extends beyond +-0.001: indistinguishable, bounded by the CI"


def run():
    t0 = time.perf_counter()
    assert sha(PREREG) == PREREG_SHA, "pre-registration file changed"
    checks = check_packages()
    H = F._engine_H(json.loads(zipfile.ZipFile(V13_ZIP).read("v1_2_params.json")))
    res = {"R2": {}, "R1": {}}
    meta = {}
    for k, B in enumerate(BENCHES):
        D = R.load_cache(B)
        meta[B] = dict(pooled_acc=float(D["cell_s"].sum() / D["cell_n"].sum()))
        for reg in ("R2", "R1"):
            if reg == "R2":
                ep, npairs = r2_episode(D)
                meta[B]["r2_pairs"] = npairs
            else:
                ep = R.build(D, R1_SEEDS)
                meta[B]["r1_instances"] = ep.n
            br = score_all(ep, H)
            alc = {a: br[a] @ E.W for a in ARMS}
            rec = dict(n_inst=int(ep.n), alc={a: float(alc[a].mean()) for a in ARMS},
                       budget={a: br[a].mean(0).tolist() for a in ARMS}, d_alc={}, d_budget={})
            for a, b in CONTRASTS:
                rec["d_alc"][f"{a}-{b}"] = float((alc[a] - alc[b]).mean())
                rec["d_budget"][f"{a}-{b}"] = (br[a] - br[b]).mean(0).tolist()
            res[reg][B] = rec
            del ep
        log(f"{k:2d} {B:30s} R2 n={res['R2'][B]['n_inst']:4d} f-v0={res['R2'][B]['d_alc']['final-v0']:+.4f} "
            f"v0-off={res['R2'][B]['d_alc']['v0-official']:+.4f} | R1 n={res['R1'][B]['n_inst']} "
            f"{time.perf_counter()-t0:5.0f}s rss {R.peak_mb():.0f}MB")
    low_acc = [B for B in BENCHES if meta[B]["pooled_acc"] <= 0.15]
    no_rel = [B for B in BENCHES if B not in RELATIVES]
    stats = {}
    for reg in ("R2", "R1"):
        stats[reg] = dict(all=summarise(res[reg], BENCHES),
                          acc_le_0_15=summarise(res[reg], low_acc),
                          excl_relatives=summarise(res[reg], no_rel))
    mean_alc = {reg: {a: float(np.mean([res[reg][B]["alc"][a] for B in BENCHES])) for a in ARMS} for reg in res}
    out = dict(experiment_id="EXP-0058", preregistration=str(PREREG.relative_to(paths.ROOT)), prereg_sha256=PREREG_SHA,
               prereg_commit="bb51964", data="data/_main_54074453 (rev 54074453da)", n_bench=len(BENCHES),
               arms=dict(official="external/official_baselines/empirical_mean/model.py (L0)",
                         v0=f"v0.zip {V0_SHA}", final=f"v1_3_named_facets_noguard.zip {V13_SHA}, no-facet path"),
               regimes=dict(R2=dict(seeds=R2_SEEDS, max_pairs=MAX_PAIRS, max_items=MAX_ITEMS), R1=dict(seeds=R1_SEEDS)),
               bootstrap=dict(n=N_BOOT, seed=BOOT_SEED, unit="benchmark"), package_checks=checks,
               low_acc_benchmarks=low_acc, relatives_excluded=RELATIVES,
               r2_pairs_total=int(sum(meta[B]["r2_pairs"] for B in BENCHES)),
               r1_instances_total=int(sum(meta[B]["r1_instances"] for B in BENCHES)),
               mean_alc_equal_bench=mean_alc, stats=stats, per_benchmark=res, benchmark_meta=meta,
               seconds=time.perf_counter() - t0, peak_rss_mb=R.peak_mb())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    f = OUT_DIR / f"exp_0058_oos_{time.strftime('%Y-%m-%d')}.json"
    f.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for reg in ("R2", "R1"):
        for key, s in stats[reg]["all"].items():
            log(f"[{reg}] {key}: mean {s['mean']:+.5f} CI [{s['ci95'][0]:+.5f},{s['ci95'][1]:+.5f}] "
                f"neg {s['n_neg']}/{s['n_nonzero']} p={s['sign_p']:.2g} -> {s['row']}")
    log(f"mean ALC {mean_alc}; {out['seconds']:.0f}s peak {out['peak_rss_mb']:.0f}MB -> {f}")
    return out


if __name__ == "__main__":
    run()
