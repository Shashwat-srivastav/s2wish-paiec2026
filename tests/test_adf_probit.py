import numpy as np
import pytest

from paiec.models.adf_probit import ProbitADF


def test_adf_update_moves_toward_observed_label_and_reduces_uncertainty():
    state = ProbitADF.from_diagonal(np.zeros(2), np.ones(2))
    prior_cov = state.cov.copy()
    prior_probability = state.predictive_probit(np.array([1.0, 0.0]))

    state.update(np.array([1.0, 0.0]), 1)

    assert prior_probability == pytest.approx(0.5)
    assert state.mean[0] > 0
    assert state.cov[0, 0] < prior_cov[0, 0]
    assert state.predictive_probit(np.array([1.0, 0.0])) > 0.5


def test_adf_rank_one_update_preserves_unobserved_direction():
    state = ProbitADF.from_diagonal(np.zeros(2), np.ones(2))
    state.update(np.array([1.0, 0.0]), 0)

    assert state.mean[1] == pytest.approx(0.0)
    assert state.cov[1, 1] == pytest.approx(1.0)
    assert state.predictive_probit(np.array([0.0, 1.0])) == pytest.approx(0.5)


def test_adf_rejects_invalid_labels():
    state = ProbitADF.from_diagonal(np.zeros(1), np.ones(1))
    with pytest.raises(ValueError, match="binary"):
        state.update(np.ones(1), 2)
