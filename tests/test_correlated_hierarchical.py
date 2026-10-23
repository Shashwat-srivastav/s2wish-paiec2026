import numpy as np
import pytest

from paiec.models.correlated_hierarchical import (
    CorrelatedHierarchicalLogistic,
    EffectScales,
)
from paiec.analysis.evaluate_correlated_hierarchical import _blend_predictions


def test_shared_latent_effect_induces_correlated_outcomes():
    model = CorrelatedHierarchicalLogistic()
    model.fit(
        benchmark=["train"] * 8,
        system=["s1", "s1", "s1", "s1", "s2", "s2", "s2", "s2"],
        item=["i1", "i1", "i2", "i2"] * 2,
        outcomes=[1, 1, 0, 1, 0, 0, 1, 0],
    )
    draws = model.predictive_draws(
        benchmark=["new"] * 2,
        system=["same", "same"],
        item=["item-a", "item-b"],
        draws=30_000,
        rng=np.random.default_rng(12),
        sample_outcomes=True,
    )
    assert np.corrcoef(draws.T)[0, 1] > 0.05


def test_repeated_rows_share_item_random_effect():
    model = CorrelatedHierarchicalLogistic(
        scales=EffectScales(benchmark=0.01, system=0.01,
                            system_benchmark=0.01, item=2.0)
    )
    model.fit(
        benchmark=["train"] * 8,
        system=["s1", "s1", "s1", "s1", "s2", "s2", "s2", "s2"],
        item=["i1", "i1", "i2", "i2"] * 2,
        outcomes=[1, 1, 0, 0, 0, 0, 1, 1],
    )
    draws = model.predictive_draws(
        benchmark=["new"] * 2,
        system=["s", "s"],
        item=["same", "same"],
        draws=20_000,
        rng=np.random.default_rng(4),
        sample_outcomes=True,
    )
    assert np.corrcoef(draws.T)[0, 1] > 0.10


def test_fit_rejects_non_binary_outcomes():
    with pytest.raises(ValueError, match="integer"):
        CorrelatedHierarchicalLogistic().fit(
            ["b"], ["s"], ["i"], [0.5]
        )


def test_binomial_cell_counts_match_expanded_binary_rows():
    expanded = CorrelatedHierarchicalLogistic().fit(
        ["b", "b", "b", "b"], ["s"] * 4, ["i"] * 4, [1, 1, 0, 1]
    )
    aggregated = CorrelatedHierarchicalLogistic().fit(
        ["b"], ["s"], ["i"], [3], trials=[4]
    )
    query = (["new"], ["s"], ["new-item"])
    p_expanded = expanded.predict_marginal_mean(*query)
    p_aggregated = aggregated.predict_marginal_mean(*query)
    assert np.allclose(p_expanded, p_aggregated, atol=1e-8, rtol=0)


def test_prediction_rejects_unfitted_model():
    with pytest.raises(RuntimeError, match="fit"):
        CorrelatedHierarchicalLogistic().predict_mean(["b"], ["s"], ["i"])


def test_blend_predictions_shrinks_candidate_toward_baseline():
    baseline = {0: np.array([0.2, 0.8])}
    candidate = {0: np.array([0.6, 0.4])}

    assert np.array_equal(_blend_predictions(baseline, candidate, 0.0)[0], baseline[0])
    assert np.array_equal(_blend_predictions(baseline, candidate, 1.0)[0], candidate[0])
    assert np.allclose(
        _blend_predictions(baseline, candidate, 0.25)[0],
        np.array([0.3, 0.7]),
    )


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan")])
def test_blend_predictions_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        _blend_predictions({0: np.array([0.5])}, {0: np.array([0.5])}, alpha)
