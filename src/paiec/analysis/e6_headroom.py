"""E6 — headroom of a degraded label-free item-difficulty oracle on the shipped model.

Pre-registration: docs/preregistrations/E6_headroom_preregistration.md.
Run: ``python -m paiec.analysis.e6_headroom``  ->  experiments/runs/e6/{e6.json,e6.log}

SCOPE (coordinator change during the run, recorded as a deviation): arms pair_oracle, both and
floor are NOT re-measured here (report 19 / experiments/runs/e1 already hold them); the only new
measurement is item_oracle(rho), rho in {0.2, 0.3, 0.5, 0.7}. rho = 1.0 is run once as a harness
validation against E1's leave-pair-out ``oracle_item_lpo`` (B0/B1, seeds 0-4).

Arms (same episodes for every arm; R2, 4 LOBO folds, seeds 0-9, budgets 0/1/3/7/15/31):
  base            v1_2 ``cfg_of(facets="named", guard=0)`` (the shipped v1_3 config)
  item(rho)       PRIMARY. offset = rho * sd * z_noisy   (scale shrunk by rho for the noise)
  item_raw(rho)   SUPPLEMENTARY (original pre-registered scaling): offset = sd * z_noisy
  item(1.0)       validation only

Item oracle construction (per target pair p, never using p's own responses):
  beta   = E1's leave-pair-out Rasch easiness ``oracle_stats(B)["beta_lpo"][p]`` (fit on all OTHER
           subjects of the held-out benchmark), restricted to pool = keys those subjects answered.
  noise  = one N(0,1) draw per item key per episode seed (one "scorer" per seed, shared by pairs).
  mix    = r * z + sqrt(1 - r^2) * eps, z = standardised beta; r found by bisection so that the
           within-benchmark Spearman(mix, beta) over the pool equals the target rho.
  offset = shrink * sd(beta) * standardise(mix); keys outside the pool get offset 0.
Injection (E1's oracle_item machinery + the model's own label updates on top):
  q       = the model's pair-level mean (v1_1.pooled_pair, labels at the budget)
  c       : mean_{k in pool} sig(c + offset_k) = q        (E1.invert_c_emp)
  p_item  = sig(c + offset_key)
  p       = sig(logit(p_item) + [logit(p_model) - logit(q)])
  where p_model = v1_2.predict (item level + named facets). The bracket is the model's own
  label-driven item/facet update; it is exactly 0 at budget 0. This is a documented post-hoc logit
  composition: the pair-level likelihood does not know the offsets, and facet updates may double
  count difficulty the oracle already carries.
"""
from __future__ import annotations

import json
import time

import numpy as np
from scipy.stats import rankdata

from paiec import paths
from paiec.analysis import e1_b0_decomp as E1
from paiec.evaluation import episodes as E
from paiec.models import v1_1 as Base
from paiec.models import v1_2 as V

OUT = paths.RUNS / "e6"
REGIME = "R2"
SEEDS = list(range(10))
RHOS = (0.2, 0.3, 0.5, 0.7)
NOISE_SEED = 20261008
CFG = V.cfg_of(facets="named", guard=0)
SANITY_REF, SANITY_TOL = 0.208722, 5e-4          # docs/research_notes/OPTIMIZATION_LOG.md baseline
sig = V.sig

ARMS = ["base"] + [f"item({r})" for r in RHOS] + [f"item_raw({r})" for r in RHOS] + ["item(1.0)"]
SPEC = {f"item({r})": (r, r) for r in RHOS}       # name -> (target spearman, shrink)
SPEC.update({f"item_raw({r})": (r, 1.0) for r in RHOS})
SPEC["item(1.0)"] = (1.0, 1.0)


def log(*a):
    print(*a, flush=True)
    with open(OUT / "e6.log", "a", encoding="utf-8") as f:
        print(*a, file=f)


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def spearman(a, b_rank):
    ra = rankdata(a)
    ra = ra - ra.mean()
    return float(ra @ b_rank / np.sqrt((ra @ ra) * (b_rank @ b_rank)))


def noisy_z(z, eps, rho):
    """Standardised mix of z and eps whose Spearman with z is rho (bisection on the weight)."""
    if rho >= 1.0:
        return z.copy(), 1.0
    br = rankdata(z)
    br = br - br.mean()
    lo, hi = 0.0, 1.0
    for _ in range(40):
        r = 0.5 * (lo + hi)
        if spearman(r * z + np.sqrt(1 - r * r) * eps, br) < rho:
            lo = r
        else:
            hi = r
    r = 0.5 * (lo + hi)
    m = r * z + np.sqrt(1 - r * r) * eps
    m = (m - m.mean()) / m.std()
    return m, spearman(m, br)


def offsets(D, O, ep, rho, shrink):
    """Per-instance (offset over all K keys, pool index array, achieved spearman, pearson)."""
    K = len(D["key_jev"])
    rk = D["item_key"][D["row_item"]]
    pools = {}
    out, ach, pear = [], [], []
    for p, s in zip(ep.inst_pair, ep.inst_seed):
        p, s = int(p), int(s)
        if p not in pools:
            pools[p] = np.unique(rk[D["row_pair"] != p])
        pool = pools[p]
        b = O["beta_lpo"][p][pool]
        sd = b.std()
        z = (b - b.mean()) / sd
        eps = np.random.default_rng([NOISE_SEED, s]).standard_normal(K)[pool]
        m, a = noisy_z(z, eps, rho)
        o = np.zeros(K)
        o[pool] = shrink * sd * m + (b.mean() if rho >= 1.0 else 0.0)
        out.append((o, pool))
        ach.append(a)
        pear.append(float(np.corrcoef(m, z)[0, 1]))
    return out, float(np.mean(ach)), float(np.mean(pear))


def predict_item(ep, H, b, fe, off, p_model, q):
    c = np.empty(ep.n)
    for n, (o, pool) in enumerate(off):
        c[n] = E1.invert_c_emp(q[n:n + 1], o[pool])[0]
    o_t = np.empty(len(ep.t_key))
    bounds = np.searchsorted(ep.t_inst, np.arange(ep.n + 1))
    for n, (o, _) in enumerate(off):
        o_t[bounds[n]:bounds[n + 1]] = o[ep.t_key[bounds[n]:bounds[n + 1]]]
    upd = logit(p_model) - logit(q[ep.t_inst])
    return np.clip(sig(c[ep.t_inst] + o_t + upd), 0.005, 0.995)


def run_fold(B):
    D = E.compact(B)
    H = Base.H_fold(B)
    O = E1.oracle_stats(B)
    ep = E.build(D, REGIME, SEEDS)
    fe = V.fe_for(B, D, CFG)
    offs, diag = {}, {}
    for nm, (rho, shrink) in SPEC.items():
        offs[nm], a, pr = offsets(D, O, ep, rho, shrink)
        diag[nm] = dict(spearman=a, pearson=pr)
    # share of scored targets whose key no other subject answered (offset 0)
    bounds = np.searchsorted(ep.t_inst, np.arange(ep.n + 1))
    miss = np.mean([(~np.isin(ep.t_key[bounds[n]:bounds[n + 1]], offs["item(1.0)"][n][1])).mean()
                    for n in range(ep.n)])
    probs = {nm: {} for nm in ARMS}
    zero = [(np.zeros(len(D["key_jev"])), o[1]) for o in offs["item(1.0)"]]
    ident = 0.0
    for b in E.BUDGETS:
        li, lk, ly = ep.labels(b)
        pm = V.predict(ep, H, b, CFG, fe, None)
        q, _ = Base.pooled_pair(ep, H, li, ly, CFG)
        probs["base"][b] = pm
        # harness identity: zero offset must return the base model
        ident = max(ident, float(np.abs(predict_item(ep, H, b, fe, zero, pm, q) - pm).max()))
        for nm in SPEC:
            probs[nm][b] = predict_item(ep, H, b, fe, offs[nm], pm, q)
    res = {}
    for nm in ARMS:
        br, ec = ep.score(probs[nm])
        res[nm] = dict(br=br, ec=ec, alc=br @ E.W, pair=ep.inst_pair.copy(), seed=ep.inst_seed.copy())
    return res, dict(diag=diag, target_keys_without_other_subject=float(miss), zero_offset_maxdiff=ident,
                     n_pairs=int(len(D["pair_sid"])), n_inst=int(ep.n))


def fold_mean(x, pair):
    return float(Base.pair_mean(x, pair).mean())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "e6.log").write_text("", encoding="utf-8")
    log(f"E6 start regime={REGIME} seeds={SEEDS} rhos={RHOS} cfg={CFG}")
    t0 = time.time()
    res, info = {}, {}
    for B in E.EVAL_FOLDS:
        res[B], info[B] = run_fold(B)
        log(f"  [{B}] {time.time() - t0:.0f}s {json.dumps(info[B])}")
        log("     ALC " + " ".join(f"{nm}={fold_mean(res[B][nm]['alc'], res[B][nm]['pair']):.4f}" for nm in ARMS))
    out = dict(config=dict(regime=REGIME, seeds=SEEDS, rhos=RHOS, cfg=CFG, noise_seed=NOISE_SEED), info=info, arms={})
    base = Base.mean_alc(res, "base")
    out["sanity"] = dict(base_alc=base, ref=SANITY_REF, ok=bool(abs(base - SANITY_REF) < SANITY_TOL))
    log(f"SANITY base ALC={base:.6f} ref={SANITY_REF} ok={out['sanity']['ok']}")
    if not out["sanity"]["ok"]:
        json.dump(out, open(OUT / "e6.json", "w", encoding="utf-8"), indent=1, default=str)
        raise SystemExit("base arm does not reproduce the reference; stopping")
    for nm in ARMS:
        a = dict(alc=Base.mean_alc(res, nm), curve=Base.mean_curve(res, nm).tolist(),
                 per_fold={B: fold_mean(res[B][nm]["alc"], res[B][nm]["pair"]) for B in res})
        if nm != "base":
            a["d"] = Base.delta(res, nm, "base")            # x1e-3; lo/hi = bootstrap by pair within fold
            a["d_budget"] = [Base.delta(res, nm, "base", budget=j) for j in range(6)]
        out["arms"][nm] = a
        log(f"{nm:14s} ALC={a['alc']:.5f} curve=" + " ".join(f"{x:.4f}" for x in a["curve"])
            + (f"  dALC(x1e-3) {Base.fmt(a['d'])}" if "d" in a else ""))
    # validation against E1 oracle_item_lpo (R2, seeds 0-4, B0/B1; E1 base = v1_1 without item/facet updates)
    e1 = json.load(open(paths.RUNS / "e1" / "e1_results.json", encoding="utf-8"))["R2"]
    val = {}
    for j, b in ((0, 0), (1, 1)):
        mine, ref, basev, e1b = {}, {}, {}, {}
        for B in res:
            r = res[B]["item(1.0)"]
            m = r["seed"] < 5
            mine[B] = fold_mean(r["br"][m, j], r["pair"][m])
            basev[B] = fold_mean(res[B]["base"]["br"][m, j], r["pair"][m])
            ref[B] = e1["per_fold"][B][f"oracle_item_lpo|B{b}"]
            e1b[B] = e1["per_fold"][B][f"v1_1|B{b}"]
        val[f"B{b}"] = dict(mine=mine, e1=ref, mine_mean=float(np.mean(list(mine.values()))),
                            e1_mean=float(np.mean(list(ref.values()))), base_seeds04=basev, e1_v1_1=e1b)
        log(f"VALIDATION B{b} seeds0-4: item(1.0)={val[f'B{b}']['mine_mean']:.5f} E1 oracle_item_lpo={val[f'B{b}']['e1_mean']:.5f} "
            + " ".join(f"{B[:5]}={mine[B]:.4f}/{ref[B]:.4f}" for B in res))
    out["validation"] = val
    out["e1_quoted"] = {k: e1["mean"][k] for k in e1["mean"]
                        if isinstance(e1["mean"][k], float) and (k.endswith("|B0") or k.endswith("|B1") or k.endswith("ALC"))}
    d03, d07 = out["arms"]["item(0.3)"]["d"]["pt"] / 1e3, out["arms"]["item(0.7)"]["d"]["pt"] / 1e3
    fired = []
    if d03 <= -0.02:
        fired.append("item_oracle(0.3) <= -0.02: item route is the priority (E7)")
    if -0.02 < d03 <= -0.005 and d07 <= -0.03:
        fired.append("item(0.3) in (-0.02,-0.005] and item(0.7) <= -0.03: strong scorer needed, test quality offline first")
    if d03 > -0.005:
        fired.append("item(0.3) > -0.005 (item half of the stop row; pair half not re-measured)")
    out["decision"] = dict(d_item03=d03, d_item07=d07, fired=fired)
    log("DECISION " + json.dumps(out["decision"]))
    json.dump(out, open(OUT / "e6.json", "w", encoding="utf-8"), indent=1, default=str)
    log(f"done {time.time() - t0:.0f}s")
    return out


if __name__ == "__main__":
    main()
