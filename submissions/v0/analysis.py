"""Monte Carlo + closed form: empirical mean vs Beta-Binomial shrinkage.

Simulation of one subject-benchmark pair (vectorised over many pairs):
  * pair accuracy level q0 ~ Beta(a, b)
  * N=80 items; item success prob p_i = sigmoid(logit(q0) + tau * z_i), z~N(0,1)
    (heterogeneous items; the pair's true accuracy is q = mean_i p_i)
  * random 40/40 split into acquisition / evaluation pools
  * acquisition policy picks up to 31 items (nested prefixes); one 0/1 outcome
    per acquired item and per evaluation item
  * predictors are constant within a pair: empirical mean (0.5 at n=0, the
    official baseline) and Beta-Binomial (k*m + S)/(k + n), clipped [.02,.98]
Brier at budgets 0,1,3,7,15,31 and ALC = .1,.2,.2,.2,.2,.1 weighting.

Closed form (infinite pool, random sampling, labels iid Bernoulli(q)):
  Brier(p_hat) = E[q(1-q)] + E[(p_hat - q)^2]
  empirical mean, n>=1:  E[q(1-q)] * (1 + 1/n)
  Beta-Binomial:         E[q(1-q)] + (k^2 E[(m-q)^2] + n E[q(1-q)]) / (k+n)^2
  minimised at k* = E[q(1-q)] / E[(m-q)^2]  (= a+b when m = a/(a+b)).

Usage: python analysis.py [--pairs 20000] [--seed 0]
"""

from __future__ import annotations

import argparse

import numpy as np

BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)
N_ITEMS, N_ACQ = 80, 40


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def simulate(a, b, tau, pairs, rng, policy="random", noise_sd=1.0):
    q0 = rng.beta(a, b, size=pairs)
    logits = np.log(q0 / (1 - q0) + 1e-12)[:, None] + tau * rng.standard_normal((pairs, N_ITEMS))
    p = sigmoid(logits)
    perm = rng.permuted(np.tile(np.arange(N_ITEMS), (pairs, 1)), axis=1)
    p = np.take_along_axis(p, perm, axis=1)          # random split + stream order
    acq_p, ev_p = p[:, :N_ACQ], p[:, N_ACQ:]
    acq_y = (rng.random(acq_p.shape) < acq_p).astype(float)
    ev_y = (rng.random(ev_p.shape) < ev_p).astype(float)
    if policy == "random":
        order = np.broadcast_to(np.arange(N_ACQ), (pairs, N_ACQ))
    else:
        # Item-aware uncertainty sampling (e.g. BLE-style acquisition predictor
        # with noisy per-item estimates): stream through candidates, take those
        # with |p_tilde - 0.5| <= 0.15 first-come, force-fill at the end.
        p_tilde = sigmoid(np.log(acq_p / (1 - acq_p)) + noise_sd * rng.standard_normal(acq_p.shape))
        keep = np.abs(p_tilde - 0.5) <= 0.15
        # streaming with reserve ~ "accepted uncertain ones in order, then rest
        # at the tail of the stream"; ordering key: uncertain first by position.
        key = np.where(keep, 0, 1) * N_ACQ + np.arange(N_ACQ)
        order = np.argsort(key, axis=1)
    ys = np.take_along_axis(acq_y, order, axis=1)
    return ys, ev_y


def brier_curves(ys, ev_y, predictors):
    out = {}
    cums = np.concatenate([np.zeros((ys.shape[0], 1)), np.cumsum(ys, axis=1)], axis=1)
    ev_mean = ev_y.mean(axis=1)
    ev_sq = ev_mean  # y^2 = y
    for name, fn in predictors.items():
        row = []
        for n in BUDGETS:
            ph = fn(cums[:, n], n)
            # mean_j (ph - y_j)^2 = ph^2 - 2 ph ybar + ybar   (constant ph per pair)
            row.append(float(np.mean(ph * ph - 2 * ph * ev_mean + ev_sq)))
        out[name] = row + [float(np.dot(WEIGHTS, row))]
    return out


def make_predictors(m_oracle, k_oracle):
    clip = lambda x: np.clip(x, 0.02, 0.98)
    preds = {
        "emp_mean": lambda s, n: np.full_like(s, 0.5) if n == 0 else s / n,
    }
    for k in (1.0, 2.0, 4.0, 8.0):
        preds[f"BB m=.5 k={k:g}"] = (lambda kk: lambda s, n: clip((kk * 0.5 + s) / (kk + n)))(k)
    preds[f"BB oracle m={m_oracle:.2f} k={k_oracle:.1f}"] = lambda s, n: clip((k_oracle * m_oracle + s) / (k_oracle + n))
    return preds


def closed_form(a, b, m, k):
    mu = a / (a + b)
    var = a * b / ((a + b) ** 2 * (a + b + 1))
    eqq = mu * (1 - mu) - var
    emq = (m - mu) ** 2 + var
    rows = {}
    # empirical mean predicts 0.5 at n=0, raw mean S/n afterwards
    rows["emp_mean"] = [eqq * (1 + 1 / n) if n else eqq + (0.5 - mu) ** 2 + var for n in BUDGETS]
    rows[f"BB k={k:g}"] = [eqq + (k * k * emq + n * eqq) / (k + n) ** 2 for n in BUDGETS]
    for r in rows.values():
        r.append(float(np.dot(WEIGHTS, r)))
    return rows


def fmt(name, row):
    return f"| {name:<24} | " + " | ".join(f"{v:.4f}" for v in row) + " |"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=float, default=1.5, help="item logit spread")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    header = "| predictor                | B0     | B1     | B3     | B7     | B15    | B31    | ALC    |"
    sep = "|" + "---|" * 8
    scenarios = [(1, 1), (0.5, 0.5), (2, 2), (4, 2)]
    for a, b in scenarios:
        mu = a / (a + b)
        print(f"\n### q0 ~ Beta({a},{b}), tau={args.tau}, random sampling (MC, {args.pairs} pairs)\n")
        print(header); print(sep)
        ys, ev = simulate(a, b, args.tau, args.pairs, rng)
        for name, row in brier_curves(ys, ev, make_predictors(mu, a + b)).items():
            print(fmt(name, row))
        print(f"\nClosed form (m=0.5, iid labels, no clipping; q ~ Beta({a},{b}) directly):\n")
        print(header); print(sep)
        for k in (2.0,):
            for name, row in closed_form(a, b, 0.5, k).items():
                print(fmt(name, row))
    a, b = 1, 1
    print(f"\n### Acquisition bias: q0 ~ Beta(1,1), tau={args.tau}, item-aware uncertainty sampling\n")
    print(header); print(sep)
    ys, ev = simulate(a, b, args.tau, args.pairs, rng, policy="uncertainty")
    for name, row in brier_curves(ys, ev, make_predictors(0.5, 2.0)).items():
        if name in ("emp_mean", "BB m=.5 k=2"):
            print(fmt(name + " (unc.)", row))


if __name__ == "__main__":
    main()
