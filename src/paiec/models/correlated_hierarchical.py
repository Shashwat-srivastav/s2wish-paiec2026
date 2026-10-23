"""Research-only crossed-effects Bayesian logistic model.

The model uses Gaussian priors on benchmark, system, system-by-benchmark, and
benchmark-by-item intercepts. Its MAP fit is combined with a diagonal Laplace
approximation for posterior-predictive draws. Shared latent effects induce
correlated outcomes; the diagonal approximation does not preserve posterior
covariance between fitted effects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.sparse import csr_matrix


Factor = Hashable
GH_NODES, GH_WEIGHTS = np.polynomial.hermite.hermgauss(20)
GH_NODES = np.sqrt(2.0) * GH_NODES
GH_WEIGHTS = GH_WEIGHTS / np.sqrt(np.pi)


@dataclass(frozen=True)
class EffectScales:
    benchmark: float = 0.8
    system: float = 1.1
    system_benchmark: float = 0.6
    item: float = 1.5

    def __post_init__(self) -> None:
        if any(not np.isfinite(value) or value <= 0 for value in (
            self.benchmark, self.system, self.system_benchmark, self.item
        )):
            raise ValueError("All random-effect scales must be finite and positive.")


class CorrelatedHierarchicalLogistic:
    """MAP + diagonal-Laplace approximation for crossed logistic random effects.

    Rows are Bernoulli outcomes. ``item`` identifiers are nested in benchmarks,
    and the system-by-benchmark effect is shared by all observations from that
    system on that benchmark.
    """

    def __init__(
        self,
        scales: EffectScales = EffectScales(),
        intercept_mean: float = 0.0,
        intercept_sd: float = 2.5,
        max_iter: int = 500,
    ) -> None:
        if not np.isfinite(intercept_mean):
            raise ValueError("intercept_mean must be finite.")
        if not np.isfinite(intercept_sd) or intercept_sd <= 0:
            raise ValueError("intercept_sd must be finite and positive.")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        self.scales = scales
        self.intercept_mean = float(intercept_mean)
        self.intercept_sd = float(intercept_sd)
        self.max_iter = int(max_iter)
        self._is_fitted = False

    def fit(
        self,
        benchmark: Sequence[Factor],
        system: Sequence[Factor],
        item: Sequence[Factor],
        outcomes: Sequence[int | float],
        trials: Sequence[int | float] | None = None,
    ) -> "CorrelatedHierarchicalLogistic":
        b = _as_factor_list(benchmark, "benchmark")
        s = _as_factor_list(system, "system")
        i = _as_factor_list(item, "item")
        y = np.asarray(outcomes, dtype=float)
        if not (len(b) == len(s) == len(i) == len(y)) or len(y) == 0:
            raise ValueError("benchmark, system, item, and outcomes must have equal nonzero lengths.")
        n_trials = np.ones_like(y) if trials is None else np.asarray(trials, dtype=float)
        if len(n_trials) != len(y) or not np.isfinite(n_trials).all():
            raise ValueError("trials must have the same length as outcomes and be finite.")
        if (n_trials <= 0).any() or (n_trials != np.floor(n_trials)).any():
            raise ValueError("trials must contain positive integer counts.")
        if not np.isfinite(y).all() or (y < 0).any() or (y > n_trials).any():
            raise ValueError("outcomes must contain finite success counts between 0 and trials.")
        if (y != np.floor(y)).any():
            raise ValueError("outcomes must contain integer success counts.")

        b_codes, self.benchmark_levels_ = _factor_codes(b)
        s_codes, self.system_levels_ = _factor_codes(s)
        sb_codes, self.system_benchmark_levels_ = _factor_codes(list(zip(b, s)))
        bi_codes, self.benchmark_item_levels_ = _factor_codes(list(zip(b, i)))
        n = len(y)
        offsets = np.cumsum((0, 1, len(self.benchmark_levels_), len(self.system_levels_),
                             len(self.system_benchmark_levels_)))
        columns = np.column_stack((
            np.zeros(n, dtype=np.int64),
            b_codes + offsets[1],
            s_codes + offsets[2],
            sb_codes + offsets[3],
            bi_codes + offsets[4],
        )).ravel()
        rows = np.repeat(np.arange(n), 5)
        design = csr_matrix((np.ones(5 * n), (rows, columns)),
                            shape=(n, int(offsets[-1] + len(self.benchmark_item_levels_))))

        prior_mean = np.zeros(design.shape[1])
        prior_mean[0] = self.intercept_mean
        prior_sd = np.concatenate((
            [self.intercept_sd],
            np.full(len(self.benchmark_levels_), self.scales.benchmark),
            np.full(len(self.system_levels_), self.scales.system),
            np.full(len(self.system_benchmark_levels_), self.scales.system_benchmark),
            np.full(len(self.benchmark_item_levels_), self.scales.item),
        ))
        prior_precision = 1.0 / np.square(prior_sd)
        initial = prior_mean.copy()
        initial[0] = np.log((y.sum() + 0.01) / (n_trials.sum() + 0.01 - y.sum()))

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            eta = np.asarray(design @ parameters).ravel()
            residual = n_trials * expit(eta) - y
            centered = parameters - prior_mean
            value = float(np.dot(n_trials, np.logaddexp(0.0, eta)) - np.dot(y, eta)
                          + 0.5 * np.dot(prior_precision, centered * centered))
            gradient = np.asarray(design.T @ residual).ravel() + prior_precision * centered
            return value, gradient

        result = minimize(
            objective, initial, method="L-BFGS-B", jac=True,
            options={"maxiter": self.max_iter, "ftol": 1e-10, "gtol": 1e-7},
        )
        if not result.success:
            raise RuntimeError(f"Hierarchical logistic MAP fit failed: {result.message}")

        self._design = design
        self._prior_mean = prior_mean
        self._prior_sd = prior_sd
        self._parameters = np.asarray(result.x, dtype=float)
        eta = np.asarray(design @ self._parameters).ravel()
        curvature = n_trials * expit(eta) * (1.0 - expit(eta))
        hessian_diagonal = prior_precision + np.asarray(
            design.power(2).T @ curvature
        ).ravel()
        self._posterior_sd = 1.0 / np.sqrt(hessian_diagonal)
        self._benchmark_index = _index(self.benchmark_levels_)
        self._system_index = _index(self.system_levels_)
        self._system_benchmark_index = _index(self.system_benchmark_levels_)
        self._benchmark_item_index = _index(self.benchmark_item_levels_)
        self._offsets = offsets
        self._is_fitted = True
        self.optimization_result_ = result
        return self

    def predict_marginal_mean(
        self,
        benchmark: Sequence[Factor],
        system: Sequence[Factor],
        item: Sequence[Factor],
    ) -> np.ndarray:
        """Integrate each row's diagonal-Gaussian Laplace predictive distribution."""
        self._check_fitted()
        b = _as_factor_list(benchmark, "benchmark")
        s = _as_factor_list(system, "system")
        i = _as_factor_list(item, "item")
        if not (len(b) == len(s) == len(i)) or not b:
            raise ValueError("benchmark, system, and item must have equal nonzero lengths.")

        linear_mean = np.full(len(b), self._parameters[0], dtype=float)
        linear_variance = np.full(len(b), self._posterior_sd[0] ** 2, dtype=float)
        keys_by_effect = (
            b,
            s,
            list(zip(b, s)),
            list(zip(b, i)),
        )
        effect_specs = (
            (self._benchmark_index, self._offsets[1], self.scales.benchmark),
            (self._system_index, self._offsets[2], self.scales.system),
            (self._system_benchmark_index, self._offsets[3], self.scales.system_benchmark),
            (self._benchmark_item_index, self._offsets[4], self.scales.item),
        )
        for keys, (fitted_index, offset, prior_scale) in zip(keys_by_effect, effect_specs):
            for row, key in enumerate(keys):
                fitted = fitted_index.get(key)
                if fitted is None:
                    variance = prior_scale**2
                else:
                    parameter_index = offset + fitted
                    linear_mean[row] += self._parameters[parameter_index]
                    variance = self._posterior_sd[parameter_index] ** 2
                linear_variance[row] += variance
        eta = linear_mean[:, None] + np.sqrt(linear_variance)[:, None] * GH_NODES[None, :]
        return expit(eta) @ GH_WEIGHTS

    def predictive_draws(
        self,
        benchmark: Sequence[Factor],
        system: Sequence[Factor],
        item: Sequence[Factor],
        draws: int = 1000,
        rng: np.random.Generator | None = None,
        sample_outcomes: bool = False,
    ) -> np.ndarray:
        """Draw joint predictions, reusing each latent effect across matching rows.

        Unknown benchmark/system/item effects are drawn from their Gaussian
        priors. ``sample_outcomes=True`` additionally draws Bernoulli outcomes.
        """
        self._check_fitted()
        b = _as_factor_list(benchmark, "benchmark")
        s = _as_factor_list(system, "system")
        i = _as_factor_list(item, "item")
        if not (len(b) == len(s) == len(i)) or not b:
            raise ValueError("benchmark, system, and item must have equal nonzero lengths.")
        if draws <= 0:
            raise ValueError("draws must be positive.")
        rng = np.random.default_rng() if rng is None else rng

        # Each unique latent group is sampled once per draw so shared effects
        # induce dependence between all corresponding outcome rows.
        keys_by_effect = (
            b,
            s,
            list(zip(b, s)),
            list(zip(b, i)),
        )
        effect_specs = (
            ("benchmark", self._benchmark_index, self._offsets[1], self.scales.benchmark),
            ("system", self._system_index, self._offsets[2], self.scales.system),
            ("system_benchmark", self._system_benchmark_index, self._offsets[3],
             self.scales.system_benchmark),
            ("benchmark_item", self._benchmark_item_index, self._offsets[4], self.scales.item),
        )
        linear = np.full((draws, len(b)), self.intercept_mean, dtype=float)
        intercept_draws = rng.normal(
            self._parameters[0], self._posterior_sd[0], size=draws
        )
        linear += intercept_draws[:, None]
        for keys, (name, fitted_index, offset, prior_scale) in zip(keys_by_effect, effect_specs):
            row_codes, levels = _factor_codes(keys)
            sampled = np.empty((draws, len(levels)), dtype=float)
            for code, level in enumerate(levels):
                fitted = fitted_index.get(level)
                if fitted is None:
                    mean, sd = 0.0, prior_scale
                else:
                    parameter_index = offset + fitted
                    mean = self._parameters[parameter_index]
                    sd = self._posterior_sd[parameter_index]
                sampled[:, code] = rng.normal(mean, sd, size=draws)
            linear += sampled[:, row_codes]
        probabilities = expit(linear)
        if sample_outcomes:
            return rng.binomial(1, probabilities).astype(np.int8)
        return probabilities

    def predict_mean(
        self,
        benchmark: Sequence[Factor],
        system: Sequence[Factor],
        item: Sequence[Factor],
        draws: int = 1000,
        rng: np.random.Generator | None = None,
        batch_size: int = 10_000,
    ) -> np.ndarray:
        """Estimate posterior predictive probabilities in batches."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        total = np.zeros(len(benchmark), dtype=float)
        completed = 0
        while completed < draws:
            count = min(batch_size, draws - completed)
            total += self.predictive_draws(
                benchmark, system, item, count, rng=rng
            ).sum(axis=0)
            completed += count
        return total / draws

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Call fit before requesting predictions.")


def _as_factor_list(values: Sequence[Factor], name: str) -> list[Factor]:
    result = list(values)
    if not result:
        return result
    try:
        for value in result:
            hash(value)
    except TypeError as exc:
        raise ValueError(f"{name} identifiers must be hashable.") from exc
    return result


def _factor_codes(values: Sequence[Factor]) -> tuple[np.ndarray, list[Factor]]:
    levels: list[Factor] = []
    index: dict[Factor, int] = {}
    codes = np.empty(len(values), dtype=np.int64)
    for row, value in enumerate(values):
        code = index.get(value)
        if code is None:
            code = len(levels)
            index[value] = code
            levels.append(value)
        codes[row] = code
    return codes, levels


def _index(levels: Sequence[Factor]) -> dict[Factor, int]:
    return {level: index for index, level in enumerate(levels)}
