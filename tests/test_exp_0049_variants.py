"""EXP-0049 Stage 0 unit tests: deterministic synthetic inputs only; no ALC/Brier is computed."""
import copy
import math

import numpy as np
import pytest
from scipy import integrate, stats

from paiec.models import exp_0049_variants as XV
from paiec.models import v1_1 as V
from paiec.models import v1_2 as V2
from paiec.models import x1_prior as X

MU = XV.MU
H_SYN = dict(G=-0.9, sB=0.7, tau=1.15, sigma_beta=2.6, _means=np.array([-1.6, -0.4, 0.3, -1.9]),
             train=["a", "b", "c", "d"])
S_HALF = max(0.5, math.sqrt(np.mean((H_SYN["_means"] - H_SYN["G"] / 2) ** 2)))

OFFSETS = [("t", 3, "G", "sB"), ("t", 3, "half", "learned"), ("t", 1, "half", "learned"),
           *(X.CANDS[k]["prior"] for k in ("X1-bma", "X1-ext10", "X1-ext20", "X1-bma-ext10", "X1-Gext10")),
           ("gumbel", "G", "sB"), ("horseshoe", "half", "learned", 3.0), ("spike", "half", "learned", 0.5),
           ("cauchy", "half", "learned")]


def _p(lp):
    w = np.exp(lp - lp.max())
    return w / w.sum()


# ---------------------------------------------------------------- offset priors
@pytest.mark.parametrize("prior", OFFSETS, ids=str)
def test_offset_prior_proper_on_grid(prior):
    mu, lp = XV.offset_prior(H_SYN, prior)
    assert np.isfinite(lp).all() and len(mu) == len(lp)
    w = _p(lp)
    assert abs(w.sum() - 1) < 1e-12 and (w >= 0).all()   # Gumbel left tail underflows in exp only
    if prior[0] in ("cauchy", "gumbel", "horseshoe", "spike"):
        assert abs(np.logaddexp.reduce(lp)) < 1e-12          # new kinds returned normalised


def test_gumbel_matches_scipy():
    _, lp = XV.offset_prior(H_SYN, ("gumbel", "G", "sB"))
    ref = stats.gumbel_r.pdf(MU, loc=H_SYN["G"], scale=H_SYN["sB"] * math.sqrt(6) / math.pi)
    assert np.abs(_p(lp) - ref / ref.sum()).max() < 1e-13
    # sd of the registered Gumbel equals sB
    assert abs(stats.gumbel_r.std(scale=H_SYN["sB"] * math.sqrt(6) / math.pi) - H_SYN["sB"]) < 1e-12


def test_horseshoe_density_matches_adaptive_quadrature_and_integrates_to_one():
    tau, c = S_HALF, 3.0

    def dens(d):
        f = lambda lam: (stats.norm.pdf(d, scale=math.sqrt((tau * c * lam) ** 2 / (c * c + (tau * lam) ** 2)))
                         * 2 / (math.pi * (1 + lam * lam)))
        return integrate.quad(f, 0, np.inf, limit=400, epsabs=1e-13, epsrel=1e-11)[0]

    for d in (0.03, 0.27, 0.9, 2.4, 5.5):
        assert abs(XV.horseshoe_density(d, tau, c)[0] / dens(d) - 1) < 1e-6
    tot = integrate.quad(lambda d: XV.horseshoe_density(d, tau, c)[0], -60, 60, points=[0], limit=400)[0]
    assert abs(tot - 1) < 2e-3                               # tails beyond +-60 ~ slab N(0,3^2)-free; mass ~1


def test_horseshoe_grid_is_normalised_density():
    mu, lp = XV.offset_prior(H_SYN, ("horseshoe", "half", "learned", 3.0))
    d = XV.horseshoe_density(mu - H_SYN["G"] / 2, S_HALF, 3.0)
    assert np.abs(_p(lp) - d / d.sum()).max() < 1e-13
    assert np.array_equal(mu, MU)                            # support unchanged


def test_spike_and_slab_masses_and_support():
    mu, lp = XV.offset_prior(H_SYN, ("spike", "half", "learned", 0.5))
    w = np.exp(lp)
    assert len(mu) == len(MU) + 1 and mu[-1] == H_SYN["G"] / 2 and np.array_equal(mu[:-1], MU)
    assert abs(w[-1] - 0.5) < 1e-15 and abs(w[:-1].sum() - 0.5) < 1e-12
    slab = stats.t.pdf(MU, 3, loc=H_SYN["G"] / 2, scale=2 * S_HALF)
    assert np.abs(w[:-1] / w[:-1].sum() - slab / slab.sum()).max() < 1e-13


def test_t1_equals_cauchy_alias():
    _, a = XV.offset_prior(H_SYN, ("t", 1, "half", "learned"))
    _, b = XV.offset_prior(H_SYN, ("cauchy", "half", "learned"))
    assert np.abs(_p(a) - _p(b)).max() < 1e-14
    ref = stats.cauchy.pdf(MU, loc=H_SYN["G"] / 2, scale=S_HALF)
    assert np.abs(_p(b) - ref / ref.sum()).max() < 1e-13


def test_x1_mixture_delegation_unchanged():
    for k in ("X1-bma", "X1-ext10", "X1-ext20", "X1-bma-ext10", "X1-Gext10"):
        pr = X.CANDS[k]["prior"]
        assert np.array_equal(XV.offset_prior(H_SYN, pr)[1], X.x1_log_prior(H_SYN, pr))


# ---------------------------------------------------------------- t3 pair quadrature
BATTERY = [(mu, S, N) for mu in (-3, -1, 0, 1.5) for S, N in ((0, 0), (3, 5), (20, 31), (0, 31), (31, 31))]


def _quad_err(th, w, pdf, s):
    sg = lambda z: 1 / (1 + np.exp(-z))
    err = 0.0
    for mu, S, N in BATTERY:
        f = lambda t: sg(mu + s * t) ** S * (1 - sg(mu + s * t)) ** (N - S)
        L = integrate.quad(lambda t: f(t) * pdf(t), -np.inf, np.inf, epsabs=1e-14, limit=500)[0]
        Ex = integrate.quad(lambda t: f(t) * pdf(t) * sg(mu + s * t), -np.inf, np.inf, epsabs=1e-14, limit=500)[0] / L
        ft = f(th)
        err = max(err, abs((w * ft * sg(mu + s * th)).sum() / (w * ft).sum() - Ex))
    return err


def test_t3_pair_quadrature_normalised_and_at_least_as_accurate_as_reference_gh():
    th, lw = XV.pair_quad("t3")
    w = np.exp(lw)
    assert abs(w.sum() - 1) < 1e-13 and abs((w * th).sum()) < 1e-12        # centred
    s = 1.15 * 1.2
    with np.errstate(over="ignore"):
        e_t3 = _quad_err(th, w, lambda t: stats.t.pdf(t, 3), s)
        e_gh = _quad_err(V.GH_X, V.GH_W, stats.norm.pdf, s)
    assert e_t3 <= e_gh and e_t3 < 5e-3


# ---------------------------------------------------------------- Laplace
def test_theta_laplace_exact_without_labels_and_converged():
    n = 5
    S, N = np.zeros(n), np.zeros(n)
    th, v, logL = XV.theta_laplace(S, N, np.full(n, 1.3), MU)
    assert np.abs(th).max() == 0 and np.allclose(v, 1.69) and np.abs(logL).max() < 1e-12
    rng = np.random.default_rng(0)
    N = rng.integers(0, 32, 40).astype(float) * 0.85
    S = np.floor(N * rng.random(40))
    t = np.full(40, 1.15)
    th, v, logL = XV.theta_laplace(S, N, t, MU)
    s = XV.sig(MU[None, :] + th)
    grad = S[:, None] - N[:, None] * s - th / (t * t)[:, None]
    assert np.abs(grad).max() < 1e-8


def test_theta_laplace_vs_brute_force():
    """Laplace log-marginal and predictive vs dense quadrature for moderate counts."""
    t = 1.15
    worst_L, worst_E = 0.0, 0.0
    for mu in (-2.0, -0.5, 1.0):
        for S, N in ((2, 5), (10, 20), (25, 31)):
            th, v, logL = XV.theta_laplace(np.array([S], float), np.array([N], float), np.array([t]), np.array([mu]))
            f = lambda x: np.exp(S * XV.log_sig(mu + x) + (N - S) * XV.log_sig(-mu - x)) * stats.norm.pdf(x, scale=t)
            L = integrate.quad(f, -40, 40, limit=400, epsabs=1e-300)[0]
            Ex = integrate.quad(lambda x: f(x) * XV.sig(mu + x), -40, 40, limit=400, epsabs=1e-300)[0] / L
            El = float(XV.sig(mu + th[0, 0] + math.sqrt(v[0, 0]) * V.GH_X) @ V.GH_W)
            worst_L = max(worst_L, abs(logL[0, 0] - math.log(L)))
            worst_E = max(worst_E, abs(El - Ex))
    assert worst_L < 0.05 and worst_E < 0.02            # Laplace error on skewed logistic posteriors (measured ~0.012)


class _SynEp:
    """Minimal Episodes stand-in for pooled_pair (only n, inst_group are used)."""
    def __init__(self, groups):
        self.inst_group = np.asarray(groups)
        self.n = len(groups)


def _syn_labels(n, per, p, seed):
    rng = np.random.default_rng(seed)
    li = np.repeat(np.arange(n), per)
    ly = (rng.random(len(li)) < np.repeat(p, per)).astype(float)
    return li, ly


def test_laplace_agrees_with_exact_grid_on_synthetic_data():
    rng = np.random.default_rng(1)
    n = 30
    ep = _SynEp(np.zeros(n, int))                       # one pooled group (R3-like)
    p_true = XV.sig(-0.8 + 1.15 * rng.standard_normal(n))
    cfg = dict(XV.REFERENCE)
    for per, tol in ((31, 1e-2), (7, 2e-2)):            # measured 0.0081 / 0.0128 (GH20 grid vs dense: 0.0044 / 7e-6)
        li, ly = _syn_labels(n, per, p_true, 2)
        qg, eg = XV.pooled_pair(ep, H_SYN, li, ly, cfg)
        ql, el = XV.pooled_pair(ep, H_SYN, li, ly, dict(cfg, integ="laplace"))
        assert np.array_equal(eg, el)
        assert np.abs(qg - ql).max() < tol, (per, np.abs(qg - ql).max())
    # separate groups (R2-like), rich data
    ep2 = _SynEp(np.arange(n))
    li, ly = _syn_labels(n, 31, p_true, 3)
    qg, _ = XV.pooled_pair(ep2, H_SYN, li, ly, cfg)
    ql, _ = XV.pooled_pair(ep2, H_SYN, li, ly, dict(cfg, integ="laplace"))
    assert np.abs(qg - ql).max() < 2e-2


def test_laplace_mu_weights_recover_gaussian():
    m, sd = -1.234, 0.37
    lp = -0.5 * ((MU - m) / sd) ** 2 + 3.0
    w, nfb = XV.laplace_mu_weights(lp[None, :])
    ref = _p(lp)
    assert nfb == 0 and np.abs(w[0] - ref).max() < 1e-10


def test_t1_cauchy_alias_identical_predictions():
    rng = np.random.default_rng(4)
    ep = _SynEp(np.zeros(12, int))
    li, ly = _syn_labels(12, 3, XV.sig(rng.standard_normal(12)), 5)
    a, _ = XV.pooled_pair(ep, H_SYN, li, ly, dict(XV.REFERENCE, prior=("t", 1, "half", "learned")))
    b, _ = XV.pooled_pair(ep, H_SYN, li, ly, dict(XV.REFERENCE, prior=("cauchy", "half", "learned")))
    assert np.abs(a - b).max() < 1e-13


# ---------------------------------------------------------------- reference reproduction (synthetic outcomes)
@pytest.fixture(scope="module")
def syn_episode():
    from paiec.analysis.redteam import synth_D
    from paiec.evaluation import episodes as E
    B = "matharena"
    D = synth_D(B, 0.5, 7)                                   # synthetic outcomes on real runtime structure
    H = V.H_fold(B)
    pairs = list(range(min(8, len(D["pair_sid"]))))
    eps = {"pooled": E.Episodes(D, pairs, [0] * len(pairs), [0] * len(pairs)),
           "single": E.Episodes(D, pairs, [1] * len(pairs), list(range(len(pairs))))}
    return B, D, H, eps


def test_reference_reproduces_existing_engine_exactly(syn_episode):
    from paiec.evaluation import episodes as E
    B, D, H, eps = syn_episode
    ref = XV.REFERENCE
    fe = V2.fe_for(B, D, ref)
    for name, ep0 in eps.items():
        for pol in V.POLICIES:
            a, b = copy.copy(ep0), copy.copy(ep0)
            sa = V.simulate_policy(a, H, ref, pol)
            sb = XV.simulate_policy(b, H, ref, pol)
            assert sa == sb
            assert all(np.array_equal(x, y) for x, y in zip(a.acq, b.acq)), (name, pol)
            for bud in E.BUDGETS:
                pa = V2.predict(a, H, bud, ref, fe)
                pb = XV.predict(b, H, bud, ref, fe)
                assert np.array_equal(pa, pb), (name, pol, bud)


def test_delegated_priors_reproduce_engine(syn_episode):
    B, D, H, eps = syn_episode
    ep = eps["pooled"]
    for k in ("X1-bma", "X1-Gext10"):
        cfg = dict(XV.REFERENCE, prior=X.CANDS[k]["prior"])
        X.install()
        try:
            a = V.pooled_pair(ep, H, *ep.labels(7)[::2], cfg)[0]
        finally:
            V.log_prior = X._ORIG
        b = XV.pooled_pair(ep, H, *ep.labels(7)[::2], cfg)[0]
        assert np.array_equal(a, b)


def test_guard_inert_at_rho0(syn_episode):
    """Functional alias recorded in the manifest: guard on == guard off when rho0 = 0."""
    from paiec.evaluation import episodes as E
    B, D, H, eps = syn_episode
    for fac in (False, "all"):
        c0 = dict(XV.REFERENCE, facets=fac, guard=0, rho=0.0)
        c1 = dict(c0, guard=1)
        fe = V2.fe_for(B, D, c0)
        for bud in E.BUDGETS:
            assert np.array_equal(XV.predict(eps["pooled"], H, bud, c0, fe), XV.predict(eps["pooled"], H, bud, c1, fe))


NEW_VARIANTS = [dict(prior=("gumbel", "G", "sB")), dict(prior=("horseshoe", "half", "learned", 3.0)),
                dict(prior=("spike", "half", "learned", 0.5)), dict(pair="t3", w_od=1.2, tau_scale=1.2),
                dict(prior=("spike", "half", "learned", 0.5), pair="t3", w_od=1.2, tau_scale=1.2),
                dict(integ="laplace")]


@pytest.mark.parametrize("kw", NEW_VARIANTS, ids=str)
def test_new_variants_run_all_policies(syn_episode, kw):
    from paiec.evaluation import episodes as E
    B, D, H, eps = syn_episode
    cfg = dict(XV.REFERENCE, **kw)
    fe = V2.fe_for(B, D, cfg)
    for pol in V.POLICIES:
        ep = copy.copy(eps["pooled"])
        st = XV.simulate_policy(ep, H, cfg, pol)
        assert st["seen"] >= st["skipped"]
        for bud in E.BUDGETS:
            p = XV.predict(ep, H, bud, cfg, fe)
            assert np.isfinite(p).all() and p.min() >= 0.005 and p.max() <= 0.995


def test_laplace_group_state_matches_vectorised(syn_episode):
    """GroupX(laplace) after adding all B7 labels gives the same pair means as the vectorised path."""
    B, D, H, eps = syn_episode
    ep = eps["pooled"]
    cfg = dict(XV.REFERENCE, integ="laplace", w_od=0.85)
    li, lk, ly = ep.labels(7)
    G = XV.GroupX(H, cfg, ep.n)
    for i, k, y in zip(li, lk, ly):
        G.S[i] += y; G.N[i] += 1
    G.refresh()
    q, _ = XV.pooled_pair(ep, H, li, ly, cfg)
    assert np.abs(G.q - q).max() < 1e-9
