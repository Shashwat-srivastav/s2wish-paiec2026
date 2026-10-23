"""Stage-1 statistics over internal_catalogue.csv (no plots). Prints the numbers that the plots and findings use.

Conventions (verified in src/paiec/models/v1_1.py::delta/boot): pooled pt = equal-weight mean of the four fold
means; pair bootstrap resamples pairs within each fold; 'hier' also resamples folds. Units x1e-3 ALC.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FOLDS = ["multi_swebench", "real_webagents", "researchcodebench", "matharena"]
FS = ["f_" + f for f in FOLDS]
NP = {"R2": np.array([82, 26, 31, 81]), "R1": np.array([53, 26, 31, 78])}  # E1 n_pairs per fold
BUD = [0, 1, 3, 7, 15, 31]
W = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])


def load():
    d = pd.read_csv(HERE / "internal_catalogue.csv")
    d["label"] = d["source"].str.replace("phase3/", "P3/") + " | " + d["arm"].astype(str)
    return d


def primary(d):
    """One record per intervention: R2 if present else R1 (real LOBO only, no R3)."""
    d = d[d.regime.isin(["R2", "R1"])].copy()
    d["key"] = d["source"] + "|" + d["arm"].astype(str)
    d["rk"] = d.regime.map({"R2": 0, "R1": 1})
    return d.sort_values("rk").groupby("key", as_index=False).first()


def heterogeneity(r):
    """Cochran Q of the 4 fold deltas against within-fold pair noise implied by the pair-bootstrap CI."""
    if pd.isna(r.lo) or r.regime not in NP or any(pd.isna(r[c]) for c in FS):
        return pd.Series(dict(se_pair=np.nan, se_fold=np.nan, Q=np.nan, I2=np.nan, sigma=np.nan))
    n = NP[r.regime]
    se = (r.hi - r.lo) / 3.92
    sigma2 = 16 * se ** 2 / np.sum(1 / n)
    df = np.array([r[c] for c in FS])
    wts = n / sigma2
    m = np.sum(wts * df) / np.sum(wts)
    Q = float(np.sum(wts * (df - m) ** 2))
    I2 = max(0.0, (Q - 3) / Q) if Q > 0 else 0.0
    se_fold = float(np.std(df, ddof=1) / 2)  # t3-style SE of the equal-weight mean
    return pd.Series(dict(se_pair=se, se_fold=se_fold, Q=Q, I2=I2, sigma=np.sqrt(sigma2)))


if __name__ == "__main__":
    d = load()
    p = primary(d)
    p = pd.concat([p, p.apply(heterogeneity, axis=1)], axis=1)
    dep = p[p.kind.isin(["deployable", "control"])]
    pd.set_option("display.width", 250, "display.max_rows", 300)
    print(dep[["label", "family", "regime", "pt", "lo", "hi", "se_pair", "se_fold", "Q", "I2"] + FS].round(3).to_string())
    from scipy.stats import chi2
    dep = dep.dropna(subset=["Q"])
    print("\nN deployable+control with Q:", len(dep), " median se_fold/se_pair:",
          np.median(dep.se_fold / dep.se_pair).round(2), " frac Q p<0.05:",
          np.mean(chi2.sf(dep.Q, 3) < 0.05).round(2), " median I2:", dep.I2.median().round(2))
    # fold correlation across deployable primary interventions (R2 only)
    r2 = p[(p.kind == "deployable") & (p.regime == "R2")].dropna(subset=FS)
    X = r2[FS].to_numpy()
    print("\nR2 deployable n=", len(r2))
    print(pd.DataFrame(np.corrcoef(X.T), index=FOLDS, columns=FOLDS).round(2))
    from scipy.stats import spearmanr
    print("spearman", pd.DataFrame(spearmanr(X).correlation, index=FOLDS, columns=FOLDS).round(2))
    U, S, Vt = np.linalg.svd(X - 0, full_matrices=False)
    print("SVD var share", (S ** 2 / np.sum(S ** 2)).round(3), "\nPC1", Vt[0].round(2), "\nPC2", Vt[1].round(2))
    Xc = X - X.mean(1, keepdims=True)  # remove common (all-fold) shift
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    print("after removing per-intervention mean: var share", (S ** 2 / np.sum(S ** 2)).round(3), "PC1", Vt[0].round(2))
    print("share of total SS that is common shift:", (np.sum(X.mean(1) ** 2) * 4 / np.sum(X ** 2)).round(3))
    # worst fold
    wf = r2[FS].idxmax(axis=1).str[2:]
    print("\nworst-fold counts (R2 deployable):", wf.value_counts().to_dict())
    bf = r2[FS].idxmin(axis=1).str[2:]
    print("best-fold counts:", bf.value_counts().to_dict())
