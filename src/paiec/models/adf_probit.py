"""Online probit assumed-density filtering for benchmark-local latents.

This module is a small research primitive for the proposed v1_4 model.  The
state is Gaussian and each Bernoulli label is represented by a probit factor
with unit observation noise.  Conditioning is performed with the standard
rank-one ADF moment update, so labels can be absorbed sequentially without
re-fitting a batch model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr


@dataclass
class ProbitADF:
    """Gaussian ADF state for a latent linear predictor.

    ``mean`` and ``cov`` describe the latent vector.  Calling ``update`` with
    a design vector ``x`` and binary outcome ``y`` updates the state for the
    factor ``P(y | z) = Phi((2y-1) x'z)``.  The covariance update is rank one.
    """

    mean: np.ndarray
    cov: np.ndarray

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=float).reshape(-1)
        self.cov = np.asarray(self.cov, dtype=float)
        n = self.mean.size
        if self.cov.shape != (n, n):
            raise ValueError("cov must be square with one row per mean element")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.cov).all():
            raise ValueError("mean and cov must be finite")
        self.cov = _symmetrize_psd(self.cov)

    @classmethod
    def from_diagonal(cls, mean: np.ndarray, variance: np.ndarray) -> "ProbitADF":
        variance = np.asarray(variance, dtype=float).reshape(-1)
        if (variance <= 0).any() or not np.isfinite(variance).all():
            raise ValueError("variance must be finite and positive")
        return cls(np.asarray(mean, dtype=float), np.diag(variance))

    def update(self, x: np.ndarray, y: int | float) -> float:
        """Absorb one label and return its pre-update predictive probability."""
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.mean.size or not np.isfinite(x).all():
            raise ValueError("x must be finite and match the state dimension")
        if y not in (0, 1, 0.0, 1.0):
            raise ValueError("y must be binary")

        sign = 1.0 if float(y) else -1.0
        projected_mean = float(x @ self.mean)
        projected_variance = float(x @ self.cov @ x)
        denominator = np.sqrt(1.0 + max(projected_variance, 0.0))
        z = sign * projected_mean / denominator
        cdf = max(float(ndtr(z)), np.finfo(float).tiny)
        ratio = float(np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi) / cdf)

        cov_x = self.cov @ x
        self.mean = self.mean + cov_x * (sign * ratio / denominator)
        reduction = ratio * (ratio + z) / (1.0 + projected_variance)
        self.cov = _symmetrize_psd(self.cov - reduction * np.outer(cov_x, cov_x))
        return float(ndtr(projected_mean / denominator))

    def predictive_probit(self, x: np.ndarray) -> float:
        """Return ``P(y=1)`` under the probit posterior predictive."""
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.mean.size:
            raise ValueError("x must match the state dimension")
        variance = max(float(x @ self.cov @ x), 0.0)
        return float(ndtr(float(x @ self.mean) / np.sqrt(1.0 + variance)))


def _symmetrize_psd(matrix: np.ndarray) -> np.ndarray:
    matrix = (matrix + matrix.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, 1e-12)
    return (eigenvectors * eigenvalues) @ eigenvectors.T
