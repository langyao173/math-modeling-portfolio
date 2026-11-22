"""Behavioral contract tests for structural probability models and registry routing."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from worldcup_forecasting.models import registry
from worldcup_forecasting.models.calibration import TemperatureScaler
from worldcup_forecasting.models.count import (
    JointGoalBatch,
    JointPoissonConfig,
    JointPoissonModel,
    outcome_probabilities,
    score_matrix,
)
from worldcup_forecasting.models.ensemble import ConvexOutcomeEnsemble
from worldcup_forecasting.models.outcome import OutcomeLogit, OutcomeLogitConfig

OUTCOME_FEATURES = (
    "historical_appearances_diff",
    "historical_best_finish_diff",
    "historical_mean_goals_for_diff",
    "historical_mean_goals_conceded_diff",
)


def _model_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Year": [2006] * 3 + [2010] * 3 + [2014] * 3,
            OUTCOME_FEATURES[0]: [-3, 0, 3, -2, 0, 2, -1, 0, 1],
            OUTCOME_FEATURES[1]: [-2, 0, 2, -1, 0, 1, -3, 0, 3],
            OUTCOME_FEATURES[2]: [-1.0, 0.0, 1.0, -0.5, 0.0, 0.5, -1.5, 0.0, 1.5],
            OUTCOME_FEATURES[3]: [1.0, 0.0, -1.0, 0.5, 0.0, -0.5, 1.5, 0.0, -1.5],
            "target_home_goals": [0, 1, 2, 0, 1, 3, 1, 0, 2],
            "target_away_goals": [2, 1, 0, 1, 1, 0, 2, 0, 1],
            "target_outcome": ["away_win", "draw", "home_win"] * 3,
        }
    )


@pytest.mark.parametrize(
    "matrix,error",
    [
        (np.ones(3), "square"),
        (np.ones((2, 3)), "square"),
        (np.empty((0, 0)), "nonempty"),
        (np.array([[np.nan]]), "finite"),
        (np.array([[-0.1]]), "nonnegative"),
        (np.zeros((2, 2)), "positive mass"),
    ],
)
def test_outcome_probabilities_reject_invalid_score_matrices(
    matrix: np.ndarray,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        outcome_probabilities(matrix)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"lambda_home": -1.0, "lambda_away": 1.0}, "intensities"),
        ({"lambda_home": np.inf, "lambda_away": 1.0}, "intensities"),
        ({"lambda_home": 1.0, "lambda_away": 1.0, "rho": np.nan}, "rho"),
        (
            {"lambda_home": 1.0, "lambda_away": 1.0, "min_retained_mass": 0.0},
            "min_retained_mass",
        ),
        (
            {"lambda_home": 1.0, "lambda_away": 1.0, "max_support": 0},
            "max_support",
        ),
        (
            {"lambda_home": 1.0, "lambda_away": 1.0, "max_score": 0},
            "max_score",
        ),
        (
            {"lambda_home": 1.0, "lambda_away": 1.0, "rho": 2.0},
            "nonnegative low-score",
        ),
    ],
)
def test_score_matrix_fails_closed_on_invalid_kernel_parameters(
    kwargs: dict[str, float],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        score_matrix(**kwargs)


def test_score_matrix_support_modes_are_explicit() -> None:
    automatic = score_matrix(3.0, 2.0, max_score=None)
    fixed = score_matrix(3.0, 2.0, max_score=3, adaptive=False)

    assert automatic.shape[0] >= 9
    assert fixed.shape == (4, 4)
    assert np.isclose(automatic.sum(), 1.0)
    assert np.isclose(fixed.sum(), 1.0)

    with pytest.raises(RuntimeError, match="retained only"):
        score_matrix(
            30.0,
            30.0,
            max_score=1,
            min_retained_mass=0.999999,
            max_support=2,
        )
    with pytest.raises(ValueError, match="zero mass"):
        score_matrix(1_000.0, 1_000.0, max_score=1, adaptive=False)


def _valid_joint_batch() -> JointGoalBatch:
    return JointGoalBatch(
        probability=np.array([[[0.2, 0.3], [0.1, 0.4]]]),
        support_max=1,
        retained_mass=np.array([0.95]),
        expected_home_goals=np.array([0.5]),
        expected_away_goals=np.array([0.7]),
        model_metadata={"name": "test"},
    )


def test_joint_goal_batch_derives_all_views_from_one_distribution() -> None:
    batch = _valid_joint_batch()

    np.testing.assert_allclose(batch.total_goal_distribution(), [[0.2, 0.4, 0.4]])
    np.testing.assert_allclose(batch.outcome_probabilities(), [[0.1, 0.6, 0.3]])


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"probability": np.ones((2, 2))}, "shape"),
        ({"support_max": 2}, "support_max"),
        (
            {"probability": np.array([[[np.nan, 0.3], [0.1, 0.6]]])},
            "finite",
        ),
        (
            {"probability": np.array([[[0.2, -0.1], [0.3, 0.6]]])},
            "nonnegative",
        ),
        (
            {"probability": np.array([[[0.2, 0.2], [0.2, 0.2]]])},
            "sum to one",
        ),
        ({"retained_mass": np.array([0.9, 0.8])}, "retained_mass"),
        ({"expected_home_goals": np.array([np.inf])}, "expected_home_goals"),
        ({"expected_away_goals": np.array([])}, "expected_away_goals"),
    ],
)
def test_joint_goal_batch_rejects_inconsistent_views(
    overrides: dict[str, object],
    error: str,
) -> None:
    values: dict[str, object] = {
        "probability": np.array([[[0.2, 0.3], [0.1, 0.4]]]),
        "support_max": 1,
        "retained_mass": np.array([0.95]),
        "expected_home_goals": np.array([0.5]),
        "expected_away_goals": np.array([0.7]),
        "model_metadata": {},
    }
    values.update(overrides)
    with pytest.raises(ValueError, match=error):
        JointGoalBatch(**values)  # type: ignore[arg-type]


def test_joint_poisson_prediction_requires_fit_and_finite_positive_rates() -> None:
    frame = _model_frame()
    model = JointPoissonModel()
    with pytest.raises(RuntimeError, match="fitted before prediction"):
        model.predict_rates(frame)

    class ZeroRatePipeline:
        def predict(self, values: np.ndarray) -> np.ndarray:
            return np.zeros(values.shape[0])

    model.feature_columns_ = OUTCOME_FEATURES
    model.pipeline_ = ZeroRatePipeline()  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="invalid intensities"):
        model.predict_rates(frame)


@pytest.mark.parametrize(
    "mutator,error",
    [
        (lambda frame: pd.DataFrame(), "nonempty DataFrame"),
        (
            lambda frame: frame.assign(target_home_goals=-1),
            "nonnegative",
        ),
        (
            lambda frame: frame.assign(target_away_goals=np.nan),
            "complete and finite",
        ),
        (
            lambda frame: frame.assign(Year=np.nan),
            "Year values must be finite",
        ),
    ],
)
def test_joint_poisson_fit_rejects_invalid_training_data(mutator, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        JointPoissonModel().fit(mutator(_model_frame()))


def test_joint_poisson_configuration_controls_time_and_rho_contracts() -> None:
    frame = _model_frame()
    no_decay = JointPoissonModel(
        replace(JointPoissonConfig(), time_decay_half_life_years=None, rho=0.0)
    ).fit(frame.drop(columns="Year"))
    assert no_decay.predict_joint(frame.drop(columns="Year")).probability.shape[0] == len(frame)

    missing_year_fallback = JointPoissonModel(
        replace(JointPoissonConfig(), rho=0.0)
    ).fit(frame.drop(columns="Year"))
    assert np.isfinite(missing_year_fallback.predict_rates(frame.drop(columns="Year"))).all()

    with pytest.raises(ValueError, match="time_decay_half_life_years"):
        JointPoissonModel(
            replace(JointPoissonConfig(), time_decay_half_life_years=0.0)
        ).fit(frame)
    with pytest.raises(ValueError, match="rho is invalid"):
        JointPoissonModel(replace(JointPoissonConfig(), rho=2.0)).fit(frame)


def test_joint_poisson_requires_at_least_one_signed_feature() -> None:
    config = JointPoissonConfig(
        mandatory_diff_features=(),
        optional_diff_features=(),
        time_decay_half_life_years=None,
    )
    with pytest.raises(ValueError, match="at least one signed difference feature"):
        JointPoissonModel(config).fit(_model_frame())


def test_outcome_logit_validates_configuration_and_training_contracts() -> None:
    with pytest.raises(ValueError, match="exactly four"):
        OutcomeLogit(OutcomeLogitConfig(feature_columns=("only_one",)))

    model = OutcomeLogit()
    with pytest.raises(RuntimeError, match="fitted before prediction"):
        model.predict_proba(np.zeros((1, 4)))
    with pytest.raises(ValueError, match="shape"):
        model.fit(np.zeros((3, 3)), np.array(["home_win", "draw", "away_win"]))
    with pytest.raises(ValueError, match="y is required"):
        model.fit(np.zeros((3, 4)))
    with pytest.raises(ValueError, match="one-dimensional"):
        model.fit(np.zeros((3, 4)), np.array([[0], [1], [2]]))
    with pytest.raises(ValueError, match="Unknown outcome code"):
        model.fit(np.zeros((3, 4)), np.array([0, 1, 3]))
    with pytest.raises(ValueError, match="all three"):
        model.fit(np.zeros((3, 4)), np.array(["draw", "draw", "home_win"]))
    with pytest.raises(ValueError, match="finite"):
        model.fit(
            np.array([[np.nan, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]]),
            np.array(["home_win", "draw", "away_win"]),
        )


def test_outcome_logit_supports_unsymmetrized_mode_and_temporal_report() -> None:
    frame = _model_frame()
    config = OutcomeLogitConfig(swap_augmentation=False, symmetrize_prediction=False)
    model = OutcomeLogit(config).fit(frame)

    probabilities = model.predict_proba(frame)
    assert probabilities.shape == (len(frame), 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert set(model.predict(frame)) <= {"home_win", "draw", "away_win"}
    assert model.coefficients().shape == (3, 4)

    report = model.fit_temporal(frame, train_end_year=2010, test_year=2014)
    assert report["train"]["n"] == 6
    assert report["test"]["n"] == 3
    assert model.temporal_metrics_ == report

    with pytest.raises(ValueError, match="nonempty train and test"):
        model.fit_temporal(frame, test_year=2018)


def test_model_registry_routes_aliases_and_rejects_ambiguous_requests(monkeypatch) -> None:
    monkeypatch.setattr(registry, "_MODEL_REGISTRY", {})

    @registry.register_model("Test-Model", "outcome_classification", ("probability",))
    class TestModel:
        def __init__(self, config: object | None = None) -> None:
            self.config = config

    assert registry.list_models() == ("test_model",)
    assert registry.list_models("goal_distribution") == ()
    assert registry.get_model_spec(" TEST-MODEL ").capabilities == frozenset({"probability"})
    assert isinstance(registry.create_model("test-model"), TestModel)
    assert isinstance(registry.create_model({"name": "test_model"}), TestModel)
    assert isinstance(registry.create_model({"model_name": "test_model"}), TestModel)
    assert isinstance(registry.create_model({"name": " ", "type": "test_model"}), TestModel)

    with pytest.raises(ValueError, match="task kind"):
        registry.create_model("test_model", expected_task_kind="goal_distribution")
    with pytest.raises(KeyError, match="Available models: test_model"):
        registry.get_model_spec("unknown")
    with pytest.raises(ValueError, match="must contain"):
        registry.create_model({"alpha": 1.0})
    with pytest.raises(ValueError, match="nonempty strings"):
        registry.get_model_spec(" ")
    with pytest.raises(ValueError, match="already registered"):

        @registry.register_model("test_model", "outcome_classification")
        class DuplicateModel:
            pass


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_temperature": 0.0},
        {"min_temperature": 2.0, "max_temperature": 1.0},
        {"temperature": 0.0},
        {"temperature": np.nan},
    ],
)
def test_temperature_scaler_rejects_invalid_initialization(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="Temperature|temperature"):
        TemperatureScaler(**kwargs)


@pytest.mark.parametrize(
    "values,error",
    [
        (np.ones(3), "two-dimensional"),
        (np.empty((0, 3)), "nonempty"),
        (np.ones((2, 1)), "two-dimensional"),
        (np.array([[np.nan, 0.0]]), "finite"),
    ],
)
def test_temperature_scaler_rejects_invalid_inputs(values: np.ndarray, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        TemperatureScaler().transform(values)


def test_temperature_scaler_supports_logits_probabilities_and_estimator_aliases() -> None:
    probabilities = np.array([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])
    labels = np.array([0, 1, 2])
    scaler = TemperatureScaler()

    calibrated = scaler.fit_transform(probabilities, labels)
    assert scaler.is_fitted_
    np.testing.assert_allclose(calibrated.sum(axis=1), 1.0)
    np.testing.assert_allclose(scaler.predict_proba(probabilities), calibrated)

    logits = np.log(probabilities)
    np.testing.assert_allclose(scaler.transform(logits).sum(axis=1), 1.0)

    with pytest.raises(ValueError, match="one value per"):
        scaler.fit(probabilities, np.array([0, 1]))
    with pytest.raises(ValueError, match="outside the class range"):
        scaler.fit(probabilities, np.array([0, 1, 3]))


def test_temperature_scaler_reports_optimizer_failure(monkeypatch) -> None:
    import worldcup_forecasting.models.calibration as calibration_module

    monkeypatch.setattr(
        calibration_module,
        "minimize_scalar",
        lambda *args, **kwargs: SimpleNamespace(success=False, fun=np.inf),
    )
    with pytest.raises(RuntimeError, match="optimization failed"):
        TemperatureScaler().fit(np.array([[0.6, 0.4], [0.4, 0.6]]), np.array([0, 1]))


@pytest.mark.parametrize(
    "values,error",
    [
        (np.ones((2, 3)), "shape"),
        (np.empty((0, 2, 3)), "shape"),
        (np.ones((1, 2, 1)), "shape"),
        (np.array([[[np.nan, 0.0], [0.5, 0.5]]]), "finite"),
        (np.array([[[-0.1, 1.1], [0.5, 0.5]]]), "nonnegative"),
        (np.array([[[0.2, 0.2], [0.4, 0.4]]]), "sum to one"),
    ],
)
def test_convex_ensemble_rejects_invalid_probability_stacks(
    values: np.ndarray,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        ConvexOutcomeEnsemble().fit(values, np.array([0, 1]))


def test_convex_ensemble_enforces_label_and_prediction_contracts() -> None:
    components = np.array(
        [
            [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]],
            [[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]],
        ]
    )
    with pytest.raises(ValueError, match="nonnegative"):
        ConvexOutcomeEnsemble(l2_shrinkage=-1.0)
    with pytest.raises(RuntimeError, match="fitted before prediction"):
        ConvexOutcomeEnsemble().predict_proba(components)
    with pytest.raises(ValueError, match="one value per"):
        ConvexOutcomeEnsemble().fit(components, np.array([0, 1]))
    with pytest.raises(ValueError, match="outside the class range"):
        ConvexOutcomeEnsemble().fit(components, np.array([0, 1, 3]))

    ensemble = ConvexOutcomeEnsemble().fit(components, np.array([0, 1, 2]))
    assert ensemble.weights_ is not None
    np.testing.assert_allclose(ensemble.weights_.sum(), 1.0)
    np.testing.assert_allclose(ensemble.predict_proba(components).sum(axis=1), 1.0)
    with pytest.raises(ValueError, match="component count differs"):
        ensemble.predict_proba(components[:1])


def test_convex_ensemble_reports_optimizer_failure(monkeypatch) -> None:
    import worldcup_forecasting.models.ensemble as ensemble_module

    monkeypatch.setattr(
        ensemble_module,
        "minimize",
        lambda *args, **kwargs: SimpleNamespace(success=False, fun=np.inf, message="forced"),
    )
    components = np.array([[[0.7, 0.3], [0.3, 0.7]]])
    with pytest.raises(RuntimeError, match="forced"):
        ConvexOutcomeEnsemble().fit(components, np.array([0, 1]))
