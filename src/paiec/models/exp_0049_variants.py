"""EXP-0049 Stage 0: not-yet-implemented Bayesian component variants (new file; no existing code edited).

Registration: docs/preregistrations/2026-10-14_comprehensive_research_preregistration.md §3 and
docs/preregistrations/2026-10-14_staged_factorial_amendment.md §0-§1.

A configuration is the v1_2 config dict (``v1_2.cfg_of``) plus two switches:

  pair   "normal" (reference: theta ~ N(0, (tau_scale tau)^2), 20-node Gauss-Hermite)
         | "t3"   (centred Student-t3 scale mixture, same scale tau_scale*tau; T3_NODES-node
                   Gauss-Chebyshev-U rule on theta = sqrt(3) x / sqrt(1-x^2), exact change of variable)
  integ  "grid"   (reference: exact mu grid x quadrature over theta)
         | "laplace" (nested Laplace: theta per pair and mu node by Newton mode + curvature,
                   Gaussian theta posterior integrated by 20-node GH; mu posterior replaced by the
                   Gaussian at the (parabolically refined) grid mode, discretised on the same mu grid)

New offset-prior kinds (``prior`` key), all normalised on the existing 121-point mu grid:

  ("cauchy", centre, scale)            alias of ("t", 1, centre, scale); equality is unit-tested
  ("gumbel", "G", "sB")                type-I (max) Gumbel, location G, scale sB*sqrt(6)/pi
  ("horseshoe", "half", "learned", c)  regularised horseshoe at G/2, global scale s_half, local
                                       lambda ~ HalfCauchy(0,1), slab scale c (=3); deterministic
                                       trapezoid quadrature over u = log(lambda) in [U_LO, U_HI]
  ("spike", "half", "learned", w)      mass w at exactly G/2 (one extra mu node appended to the grid,
                                       so the support is grid + {G/2}, not a moved spike) and mass
                                       1-w on t3(G/2, 2*s_half) normalised on the grid

Existing kinds ("t", "mix", "x1") are delegated to v1_1.log_prior / x1_prior.x1_log_prior unchanged,
so the reference setting reproduces the v1_2 engine bit-for-bit (unit-tested).

Acquisition: ``simulate_policy`` is v1_1.simulate_policy with the group state generalised
(``GroupX``) to the offset prior / pair effect / integration of the configuration; the decision rules
are v1_1._decide unchanged. Facets/guard do not enter acquisition (as in the existing engine).
"""
from __future__ import annotations

import math

import numpy as np

from paiec.evaluation import episodes as E
from paiec.models import ladder as Lad
from paiec.models import v1_1 as V
from paiec.models import v1_2 as V2
from paiec.models import x1_prior as X

GH_X, GH_W, LOG_GH_W = V.GH_X, V.GH_W, V.LOG_GH_W
MU = V.MU
DMU = float(MU[1] - MU[0])
sig, log_sig = V.sig, V.log_sig

# Frozen numerical settings (Stage 0 freeze)
T3_NODES = 40                 # t3 pair-effect quadrature nodes (error <= reference GH20 on the test battery)
HS_U_LO, HS_U_HI, HS_DU = -40.0, 25.0, 0.02   # horseshoe log-lambda trapezoid
LAPLACE_TOL, LAPLACE_MAXIT, LAPLACE_STEP = 1e-10, 100, 5.0

REFERENCE = dict(V2.cfg_of(facets="all", guard=0), pair="normal", integ="grid")


def cfg_of(**kw):
    c = dict(REFERENCE)
    c.update(kw)
    return c


# ---------------------------------------------------------------------------
# Offset priors
# ---------------------------------------------------------------------------
def _norm(lp):
    m = lp.max()
    return lp - (np.log(np.exp(lp - m).sum()) + m)


def cauchy_logpdf_grid(H, centre="half", scale="learned", mu=MU):
    c = X._centre(H, centre)
    s = X._scale(H, scale, c)
    z = (mu - c) / s
    return _norm(-np.log1p(z * z))


def gumbel_logpdf_grid(H, mu=MU):
    loc = float(H["G"])
    beta = float(H["sB"]) * math.sqrt(6.0) / math.pi
    z = (mu - loc) / beta
    return _norm(-z - np.exp(-z))


_HS_U = np.arange(HS_U_LO, HS_U_HI + HS_DU / 2, HS_DU)


def horseshoe_density(d, tau, slab):
    """Regularised-horseshoe marginal density at offsets d from the centre (unnormalised constant
    included: this is the proper density).  sd^2(l) = tau^2 c^2 l^2 / (c^2 + tau^2 l^2), l ~ HalfCauchy."""
    d = np.atleast_1d(np.asarray(d, float))
    lam = np.exp(_HS_U)
    var = (tau * slab * lam) ** 2 / (slab ** 2 + (tau * lam) ** 2)
    jac = 2.0 / math.pi * lam / (1.0 + lam * lam)                  # HalfCauchy pdf x dlambda/du
    f = np.exp(-0.5 * d[:, None] ** 2 / var[None, :]) / np.sqrt(2 * math.pi * var[None, :]) * jac[None, :]
    w = np.full(len(_HS_U), HS_DU)
    w[0] = w[-1] = HS_DU / 2
    return f @ w


def horseshoe_logpdf_grid(H, centre="half", scale="learned", slab=3.0, mu=MU):
    c = X._centre(H, centre)
    tau = X._scale(H, scale, c)
    return _norm(np.log(horseshoe_density(mu - c, tau, float(slab))))


def offset_prior(H, prior):
    """-> (mu nodes, log prior weights).  Delegated kinds return exactly what the existing engine
    uses (constant irrelevant); new kinds return weights normalised on the nodes."""
    kind = prior[0]
    if kind in ("t", "mix"):
        return MU, V.log_prior(H, prior)
    if kind == "x1":
        return MU, X.x1_log_prior(H, prior)
    if kind == "cauchy":
        return MU, cauchy_logpdf_grid(H, prior[1], prior[2])
    if kind == "gumbel":
        return MU, gumbel_logpdf_grid(H)
    if kind == "horseshoe":
        return MU, horseshoe_logpdf_grid(H, prior[1], prior[2], prior[3])
    if kind == "spike":
        _, centre, scale, w = prior
        c = X._centre(H, centre)
        s = X._scale(H, scale, c)
        slab = _norm(V._t_logpdf_norm(MU, c, 2.0 * s, 3))
        return np.append(MU, c), np.append(math.log1p(-w) + slab, math.log(w))
    raise ValueError(prior)


# ---------------------------------------------------------------------------
# Pair-effect quadrature
# ---------------------------------------------------------------------------
def t3_nodes(n=T3_NODES):
    """Unit-scale Student-t3 rule: theta = sqrt(3) x/sqrt(1-x^2) maps t3 to the weight sqrt(1-x^2)
    on (-1, 1) (Gauss-Chebyshev second kind).  Returns (nodes, log weights) with weights summing to 1."""
    i = np.arange(1, n + 1)
    x = np.cos(i * math.pi / (n + 1))
    w = np.sin(i * math.pi / (n + 1)) ** 2
    return math.sqrt(3.0) * x / np.sqrt(1.0 - x * x), np.log(w / w.sum())


_T3 = t3_nodes()


def pair_quad(pair):
    if pair == "normal":
        return GH_X, LOG_GH_W
    if pair == "t3":
        return _T3
    raise ValueError(pair)


# ---------------------------------------------------------------------------
# Laplace integration
# ---------------------------------------------------------------------------
def theta_laplace(S, N, t, mu, th0=None):
    """Mode/curvature of g(th) = S log sig(mu+th) + (N-S) log sig(-mu-th) - th^2/(2 t^2) for every
    (pair, mu node).  Returns (theta_hat, posterior variance, Laplace log marginal), each n x M."""
    S, N = S[:, None], N[:, None]
    t2 = (t * t)[:, None]
    th = np.zeros((S.shape[0], len(mu))) if th0 is None else th0.copy()
    for _ in range(LAPLACE_MAXIT):
        s = sig(mu[None, :] + th)
        g = S - N * s - th / t2
        h = N * s * (1 - s) + 1.0 / t2
        step = np.clip(g / h, -LAPLACE_STEP, LAPLACE_STEP)
        th += step
        if np.abs(step).max() < LAPLACE_TOL:
            break
    eta = mu[None, :] + th
    s = sig(eta)
    h = N * s * (1 - s) + 1.0 / t2
    logL = S * log_sig(eta) + (N - S) * log_sig(-eta) - th * th / (2 * t2) - 0.5 * np.log(t2 * h)
    return th, 1.0 / h, logL


def laplace_mu_weights(lp, mu=MU):
    """Rows of lp (groups x M, log posterior on the grid) -> Gaussian-at-mode weights on the same grid.
    Mode/curvature by a three-point parabola at the grid argmax; if the argmax is at the boundary or the
    parabola is not concave, the curvature is moment-matched to the grid posterior (counted)."""
    lp = np.atleast_2d(lp)
    G, M = lp.shape
    dmu = float(mu[1] - mu[0])
    k = lp.argmax(1)
    kk = np.clip(k, 1, M - 2)
    a, b, c = lp[np.arange(G), kk - 1], lp[np.arange(G), kk], lp[np.arange(G), kk + 1]
    den = a - 2 * b + c
    ok = (k == kk) & (den < -1e-12)
    m = np.where(ok, mu[kk] + dmu * (a - c) / (2 * np.where(ok, den, -1.0)), mu[k])
    prec = np.where(ok, -den / dmu ** 2, 0.0)
    if (~ok).any():
        p = np.exp(lp[~ok] - lp[~ok].max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        mm = p @ mu
        vv = np.maximum(p @ mu ** 2 - mm ** 2, 1e-12)
        m[~ok], prec[~ok] = mm, 1.0 / vv
    lw = -0.5 * prec[:, None] * (mu[None, :] - m[:, None]) ** 2
    w = np.exp(lw - lw.max(1, keepdims=True))
    return w / w.sum(1, keepdims=True), int((~ok).sum())


# ---------------------------------------------------------------------------
# Vectorised pooled pair level (mirror of v1_1.pooled_pair with generalised nodes)
# ---------------------------------------------------------------------------
def pooled_pair(ep, H, li, ly, cfg):
    n = ep.n
    w = cfg["w_od"]
    S = (np.bincount(li, ly, n) if len(li) else np.zeros(n)) * w
    N = (np.bincount(li, minlength=n).astype(float) if len(li) else np.zeros(n)) * w
    t = np.full(n, H["tau"] * cfg["tau_scale"])
    mu, lpr = offset_prior(H, cfg["prior"])
    groups, ginv = np.unique(ep.inst_group, return_inverse=True)
    if cfg.get("integ", "grid") == "laplace":
        if cfg.get("pair", "normal") != "normal" or len(mu) != len(MU):
            raise NotImplementedError("Laplace is registered only with the normal pair effect on the mu grid")
        th, v, logL = theta_laplace(S, N, t, mu)
        Eq = sig(mu[None, :, None] + th[:, :, None] + np.sqrt(v)[:, :, None] * GH_X[None, None, :]) @ GH_W
        lp = np.zeros((len(groups), len(mu)))
        np.add.at(lp, ginv, logL)
        lp += lpr[None, :]
        post, _ = laplace_mu_weights(lp, mu)
    else:
        tx, tlw = pair_quad(cfg.get("pair", "normal"))
        eta = mu[None, :, None] + t[:, None, None] * tx[None, None, :]
        ll = S[:, None, None] * log_sig(eta) + (N - S)[:, None, None] * log_sig(-eta) + tlw
        mx = ll.max(2, keepdims=True)
        wt = np.exp(ll - mx)
        Ln = wt.sum(2)
        logL = np.log(Ln) + mx[:, :, 0]
        Eq = (wt * sig(eta)).sum(2) / Ln
        lp = np.zeros((len(groups), len(mu)))
        np.add.at(lp, ginv, logL)
        lp += lpr[None, :]
        lp -= lp.max(1, keepdims=True)
        post = np.exp(lp); post /= post.sum(1, keepdims=True)
    q = (post[ginv] * Eq).sum(1)
    if len(li):
        gN = np.bincount(ginv, np.bincount(li, minlength=n).astype(float), len(groups))
    else:
        gN = np.zeros(len(groups))
    empty = gN[ginv] == 0
    return q, empty


def predict(ep, H, b, cfg, fe=None):
    """Mirror of v1_2.predict with the generalised pooled pair level."""
    li, lk, ly = ep.labels(b)
    q, empty = pooled_pair(ep, H, li, ly, cfg)
    if cfg.get("item", True):
        p = V2.item_level(ep, H, q, li, lk, ly, cfg["rho"], cfg.get("rho_pool", False), cfg.get("guard", 0))
    else:
        p = q[ep.t_inst]
    k = cfg["kappa"]
    if k != 1.0:
        e = empty[ep.t_inst]
        p = np.where(e, 0.5 + k * (p - 0.5), p)
    if cfg.get("facets") and fe is not None:
        p = V2.facet_adjust(ep, H, q, p, li, lk, ly, fe)
    return np.clip(p, 0.005, 0.995)


# ---------------------------------------------------------------------------
# Streaming acquisition: generalised group state
# ---------------------------------------------------------------------------
class GroupX(V._Group):
    """v1_1._Group with configurable offset prior, pair-effect quadrature and integration.
    For the reference setting every array equals v1_1._Group's (same operations, same order)."""

    def __init__(self, H, cfg, n):
        self.H, self.cfg, self.n = H, cfg, n
        self.sb = H["sigma_beta"]
        self.mu, self.lprior = offset_prior(H, cfg["prior"])
        self.laplace = cfg.get("integ", "grid") == "laplace"
        self.t = H["tau"] * cfg["tau_scale"]
        self.S = np.zeros(n); self.N = np.zeros(n)
        self.obs = {}
        if self.laplace:
            if cfg.get("pair", "normal") != "normal" or len(self.mu) != len(MU):
                raise NotImplementedError("Laplace is registered only with the normal pair effect on the mu grid")
            self.th = None
        else:
            tx, self.tlw = pair_quad(cfg.get("pair", "normal"))
            self.eta = self.mu[:, None] + self.t * tx[None, :]
            self.lsp, self.lsn, self.seta = log_sig(self.eta), log_sig(-self.eta), sig(self.eta)
            self.cnode = Lad._invert_c(self.seta.ravel(), self.sb).reshape(self.eta.shape)
        self.refresh()

    def refresh(self):
        if not self.laplace:
            w = self.cfg["w_od"]
            S, N = self.S * w, self.N * w
            ll = S[:, None, None] * self.lsp + (N - S)[:, None, None] * self.lsn + self.tlw
            mx = ll.max(2, keepdims=True)
            self.wt = np.exp(ll - mx)
            Ln = self.wt.sum(2)
            self.logL = np.log(Ln) + mx[:, :, 0]
            self.Ln = Ln
            self.Eq = (self.wt * self.seta).sum(2) / Ln
            lp = self.logL.sum(0) + self.lprior
            lp -= lp.max()
            post = np.exp(lp)
            self.post = post / post.sum()
        else:
            w = self.cfg["w_od"]
            S, N = self.S * w, self.N * w
            self.th, v, self.logL = theta_laplace(S, N, np.full(self.n, self.t), self.mu, self.th)
            self.eta_i = self.mu[None, :, None] + self.th[:, :, None] + np.sqrt(v)[:, :, None] * GH_X[None, None, :]
            self.seta_i = sig(self.eta_i)
            self.Eq = self.seta_i @ GH_W
            self.post = laplace_mu_weights((self.logL.sum(0) + self.lprior)[None, :], self.mu)[0][0]
            self._cnode_i = {}
        self.q = self.Eq @ self.post
        self.c = Lad._invert_c(self.q, self.sb)
        self._unseen = {}

    def ebr(self, i, a, ps):
        if not self.laplace:
            return super().ebr(i, a, ps)
        # Laplace belief state: theta | mu ~ N(th, v) on GH nodes; one-step update of that state.
        if a is None and ("e", i) in self._unseen:
            return self._unseen[("e", i)]
        seta = self.seta_i[i]
        l1 = seta
        if a is not None:
            if i not in self._cnode_i:
                self._cnode_i[i] = Lad._invert_c(seta.ravel(), self.sb).reshape(seta.shape)
            known = sig(self._cnode_i[i][:, :, None] + self.sb * GH_X[None, None, :]) @ a
            l1 = ps * known + (1 - ps) * seta
        qs, py = [], []
        for ly in (l1, 1 - l1):
            r = ly @ GH_W                                    # P(y | mu) under the Laplace theta posterior
            Eq2 = (ly * seta) @ GH_W / r
            py.append(float(self.post @ r))
            post2 = self.post * r
            post2 /= post2.sum()
            qs.append(float(post2 @ Eq2))
        v = py[0] * py[1] * (qs[0] - qs[1]) ** 2
        if a is None:
            self._unseen[("e", i)] = v
        return v


def simulate_policy(ep, H, cfg, policy, max_labels=31):
    """v1_1.simulate_policy with GroupX (identical protocol and decision rules)."""
    orig = ep.acq
    new = [[] for _ in range(ep.n)]
    stats = dict(skipped=0, seen=0, forced=0)
    for g in np.unique(ep.inst_group):
        members = np.flatnonzero(ep.inst_group == g)
        G = GroupX(H, cfg, len(members))
        idx = [0] * len(members)
        counts = [[0, 0, 0] for _ in members]
        for b in E.BUDGETS[1:]:
            cap = min(b, max_labels)
            for li_, n in enumerate(members):
                acq = orig[n]
                while len(new[n]) < cap and idx[li_] < len(acq):
                    key, y = int(acq[idx[li_], 0]), int(acq[idx[li_], 1])
                    items_rem = len(acq) - idx[li_]
                    lab_rem = max_labels - len(new[n])
                    idx[li_] += 1
                    stats["seen"] += 1
                    if policy != "A1" and items_rem <= lab_rem:
                        ok = True
                        stats["forced"] += 1
                    else:
                        ok = V._decide(policy, G, li_, key, counts)
                    if ok:
                        if policy == "A5":
                            p, a, _ = G.pred(li_, key)
                            if a is not None:
                                counts[li_][min(int(p * 3), 2)] += 1
                        new[n].append((key, y))
                        G.add(li_, key, y)
                    else:
                        stats["skipped"] += 1
    ep.acq = [np.array(x, np.int64).reshape(-1, 2) for x in new]
    stats["mean_labels31"] = float(np.mean([len(x) for x in new]))
    return stats


def nested_H(outer, inner):
    """Hyperparameters for an inner fold: fit on all public benchmarks except outer and inner."""
    H = Lad.fit_hyper([b for b in E.BENCHES if b not in (outer, inner)])
    H["_means"] = V.train_means(H)
    return H
