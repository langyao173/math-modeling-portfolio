import numpy as np
import pytest

from worldcup_forecasting.uncertainty.decomposition import (
    hierarchical_variance_decomposition,
)


def test_hierarchical_variance_decomposition_is_exact_and_labeled() -> None:
    parameter = np.array([-1.0, 1.0])[:, None, None]
    regime = np.array([-2.0, 0.0, 2.0])[None, :, None]
    tournament = np.array([-3.0, -1.0, 1.0, 3.0])[None, None, :]
    draws = 10.0 + parameter + regime + tournament

    result = hierarchical_variance_decomposition(draws)

    assert result.components["parameter"] == pytest.approx(1.0)
    assert result.components["regime_given_parameter"] == pytest.approx(8.0 / 3.0)
    assert result.components["tournament_path_and_match"] == pytest.approx(5.0)
    assert result.components["total"] == pytest.approx(26.0 / 3.0)
    assert result.components["reconstruction_error"] <= 1e-12
    assert sum(result.shares.values()) == pytest.approx(1.0)
    assert result.metadata["interpretation"] == (
        "law_of_total_variance_sensitivity_not_future_coverage_guarantee"
    )


def test_hierarchical_variance_decomposition_handles_zero_total_variance() -> None:
    result = hierarchical_variance_decomposition(np.ones((2, 3, 5)))

    assert result.components["total"] == 0.0
    assert result.shares == {
        "parameter": 0.0,
        "regime_given_parameter": 0.0,
        "tournament_path_and_match": 0.0,
    }


@pytest.mark.parametrize(
    "draws, message",
    [
        (np.ones((2, 3)), "three-dimensional"),
        (np.ones((1, 3, 4)), "at least two"),
        (np.array([[[1.0, np.nan], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]]), "finite"),
    ],
)
def test_hierarchical_variance_decomposition_rejects_invalid_draws(
    draws: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        hierarchical_variance_decomposition(draws)
