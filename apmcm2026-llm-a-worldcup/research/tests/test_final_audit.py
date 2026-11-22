import numpy as np
import pandas as pd
import pytest

from worldcup_forecasting.experiments.final_audit import (
    apply_frozen_2014_audit,
    frozen_audit_protocol_hash,
)
from worldcup_forecasting.models.count import JointGoalBatch, score_matrix


class _FrozenPredictor:
    def __init__(self, matrices: np.ndarray) -> None:
        self._matrices = matrices
        self.metadata = {
            "candidate_id": "d" * 64,
            "final_fit_hash": "f" * 64,
            "selected_config_hash": "c" * 64,
            "fit_end_year": 2010,
        }

    def predict_joint(self, frame: pd.DataFrame) -> JointGoalBatch:
        probability = self._matrices[: len(frame)]
        support = probability.shape[1] - 1
        axis = np.arange(support + 1)
        return JointGoalBatch(
            probability=probability,
            support_max=support,
            retained_mass=np.ones(len(frame)),
            expected_home_goals=(probability.sum(axis=2) * axis).sum(axis=1),
            expected_away_goals=(probability.sum(axis=1) * axis).sum(axis=1),
            model_metadata=self.metadata,
        )


def _inner() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    years = np.repeat([1994, 1998, 2002, 2006, 2010], 15)
    labels = np.tile([0, 1, 2], 25)
    rates = ((1.4, 0.8), (1.1, 1.1), (0.8, 1.4))
    matrices = np.stack([score_matrix(*rates[label], max_score=7) for label in labels])
    ledger = pd.DataFrame(
        {
            "outer_fold_id": [f"inner_{year}" for year in years],
            "match_id": np.arange(1_000, 1_000 + len(years)),
            "prediction_year": years,
            "train_end_year": years - 4,
            "is_oof": True,
            "observed_home_goals": [
                2 if label == 0 else 1 if label == 1 else 0 for label in labels
            ],
            "observed_away_goals": [
                0 if label == 0 else 1 if label == 1 else 2 for label in labels
            ],
            "candidate_id": "d" * 64,
            "config_hash": "c" * 64,
            "feature_schema_hash": "a" * 64,
            "fit_index_hash": "1" * 64,
        }
    )
    rows = []
    for position, row in enumerate(ledger.itertuples(index=False)):
        for home in range(matrices.shape[1]):
            for away in range(matrices.shape[2]):
                rows.append(
                    {
                        "outer_fold_id": row.outer_fold_id,
                        "match_id": row.match_id,
                        "home_goals": home,
                        "away_goals": away,
                        "probability": matrices[position, home, away],
                        "candidate_id": "d" * 64,
                        "fit_index_hash": "1" * 64,
                    }
                )
    return ledger, pd.DataFrame(rows), matrices


def _audit() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Year": [2014] * 6,
            "MatchID": np.arange(2_000, 2_006),
            "target_outcome": ["home_win", "draw", "away_win"] * 2,
            "target_home_goals": [2, 1, 0, 1, 2, 0],
            "target_away_goals": [0, 1, 2, 0, 2, 1],
        }
    )


def test_frozen_audit_states_and_predictions_ignore_2014_targets() -> None:
    inner, grid, matrices = _inner()
    predictor = _FrozenPredictor(matrices)
    identity = {
        "final_fit_hash": "f" * 64,
        "selected_config_hash": "c" * 64,
        "feature_schema_hash": "a" * 64,
        "protocol_hash": frozen_audit_protocol_hash(),
    }
    first = apply_frozen_2014_audit(inner, grid, predictor, _audit(), identity=identity)
    mutated = _audit()
    mutated["target_outcome"] = ["away_win", "home_win", "draw"] * 2
    mutated[["target_home_goals", "target_away_goals"]] = [
        [0, 2],
        [2, 0],
        [1, 1],
    ] * 2
    second = apply_frozen_2014_audit(inner, grid, predictor, mutated, identity=identity)

    assert first.calibration_state == second.calibration_state
    assert first.decision_state == second.decision_state
    pd.testing.assert_frame_equal(
        first.predictions.drop(columns="target_outcome"),
        second.predictions.drop(columns="target_outcome"),
    )
    assert set(first.metrics["evaluation_role"]) == {"audit_only_not_for_selection"}
    assert set(first.predictions["prediction_year"]) == {2014}
    assert first.metadata["selection_maximum_year"] == 2010
    assert first.metadata["audit_target_used_for_model_choice"] is False
    assert len(first.metadata["layer_state_hash"]) == 64
    assert len(first.metadata["frozen_audit_identity_hash"]) == 64
    assert first.metadata["layer_state_hash"] == second.metadata["layer_state_hash"]
    assert (
        first.metadata["frozen_audit_identity_hash"]
        == second.metadata["frozen_audit_identity_hash"]
    )
    mass = first.calibrated_joint_grid.groupby("match_id")["probability"].sum()
    np.testing.assert_allclose(mass, 1.0, atol=1e-12)


def test_frozen_audit_fails_closed_on_identity_mismatch() -> None:
    inner, grid, matrices = _inner()
    predictor = _FrozenPredictor(matrices)

    try:
        apply_frozen_2014_audit(
            inner,
            grid,
            predictor,
            _audit(),
            identity={
                "final_fit_hash": "e" * 64,
                "selected_config_hash": "c" * 64,
                "feature_schema_hash": "a" * 64,
                "protocol_hash": frozen_audit_protocol_hash(),
            },
        )
    except ValueError as error:
        assert "identity" in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Mismatched final identity was accepted")


def test_frozen_audit_rejects_inconsistent_targets_and_malformed_provenance() -> None:
    inner, grid, matrices = _inner()
    predictor = _FrozenPredictor(matrices)
    identity = {
        "final_fit_hash": "f" * 64,
        "selected_config_hash": "c" * 64,
        "feature_schema_hash": "a" * 64,
        "protocol_hash": frozen_audit_protocol_hash(),
    }

    inconsistent = _audit()
    inconsistent.loc[0, "target_outcome"] = "draw"
    with pytest.raises(ValueError, match="outcome.*goal|disagree|consistent"):
        apply_frozen_2014_audit(
            inner, grid, predictor, inconsistent, identity=identity
        )

    fractional_inner = inner.astype(
        {"observed_home_goals": float, "observed_away_goals": float}
    )
    fractional_inner.loc[0, "observed_home_goals"] = 1.5
    with pytest.raises(ValueError, match="inner.*goal|integer"):
        apply_frozen_2014_audit(
            fractional_inner, grid, predictor, _audit(), identity=identity
        )

    noninteger_year = _audit().astype({"Year": float})
    noninteger_year["Year"] = 2014.5
    with pytest.raises(ValueError, match="2014|integer|edition"):
        apply_frozen_2014_audit(
            inner, grid, predictor, noninteger_year, identity=identity
        )

    bad_identity = {**identity, "protocol_hash": "z" * 64}
    with pytest.raises(ValueError, match="hash|identity"):
        apply_frozen_2014_audit(
            inner, grid, predictor, _audit(), identity=bad_identity
        )

    foreign_inner = inner.assign(candidate_id="e" * 64)
    with pytest.raises(ValueError, match="candidate|lineage|identity"):
        apply_frozen_2014_audit(
            foreign_inner, grid, predictor, _audit(), identity=identity
        )

    foreign_grid = grid.assign(fit_index_hash="2" * 64)
    with pytest.raises(ValueError, match="fit.*hash|lineage|grid"):
        apply_frozen_2014_audit(
            inner, foreign_grid, predictor, _audit(), identity=identity
        )
