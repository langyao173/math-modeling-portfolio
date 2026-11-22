"""RED contracts for the research-complete forecasting upgrade.

These tests intentionally describe public scientific behaviour that the current
engineering baseline does not yet implement.  Imports happen inside tests so the
RED run reports the missing behaviour directly instead of failing collection.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, recall_score

from worldcup_forecasting.models.count import outcome_probabilities, score_matrix


def _required_symbol(module_name: str, symbol_name: str) -> Any:
    """Load one future public API and turn absence into an intentional RED failure."""
    try:
        module = importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.fail(
            f"RED advanced-research contract is not implemented: import {module_name!r} "
            f"failed ({exc}).",
            pytrace=False,
        )
    if not hasattr(module, symbol_name):
        pytest.fail(
            f"RED advanced-research contract is not implemented: "
            f"{module_name}.{symbol_name} is missing.",
            pytrace=False,
        )
    return getattr(module, symbol_name)


def _advanced_training_frame(seed: int = 418) -> pd.DataFrame:
    """Create a deterministic paired-count panel with identifiable components."""
    rng = np.random.default_rng(seed)
    teams = ("A", "B", "C", "D", "E")
    attack = {"A": 0.65, "B": 0.30, "C": 0.00, "D": -0.25, "E": -0.45}
    vulnerability = {"A": -0.35, "B": 0.40, "C": 0.05, "D": -0.10, "E": 0.15}
    years = (1986, 1990, 1994, 1998, 2002, 2006, 2010)
    rows: list[dict[str, object]] = []
    for index in range(280):
        home = teams[index % len(teams)]
        away = teams[(2 * index + 1) % len(teams)]
        if home == away:
            away = teams[(teams.index(away) + 1) % len(teams)]
        year = years[index % len(years)]
        stage = "final" if index % 5 == 0 else "group_stage"
        host_side = ("home", "away", "neutral")[index % 3]
        pace_signal = -1.0 if index % 4 < 2 else 1.0
        era_effect = 0.18 if year >= 2002 else -0.10
        stage_effect = 0.22 if stage == "final" else -0.04
        pace_effect = 0.38 * pace_signal
        host_home = 0.20 if host_side == "home" else 0.0
        host_away = 0.20 if host_side == "away" else 0.0
        common = 0.12 + era_effect + stage_effect + pace_effect
        home_rate = np.exp(common + attack[home] + vulnerability[away] + host_home)
        away_rate = np.exp(common + attack[away] + vulnerability[home] + host_away)
        rows.append(
            {
                "match_id": index + 1,
                "home_team": home,
                "away_team": away,
                "Year": year,
                "canonical_stage": stage,
                "host_side": host_side,
                "common_pace_signal": pace_signal,
                "target_home_goals": int(rng.poisson(home_rate)),
                "target_away_goals": int(rng.poisson(away_rate)),
            }
        )
    # One deliberately rare team makes partial-pooling strength observable.
    rows.append(
        {
            "match_id": 10_001,
            "home_team": "RARE",
            "away_team": "C",
            "Year": 2010,
            "canonical_stage": "group_stage",
            "host_side": "neutral",
            "common_pace_signal": 0.0,
            "target_home_goals": 7,
            "target_away_goals": 0,
        }
    )
    return pd.DataFrame(rows)


def _advanced_model() -> Any:
    config_type = _required_symbol(
        "worldcup_forecasting.models.advanced_count", "HierarchicalPaceConfig"
    )
    model_type = _required_symbol(
        "worldcup_forecasting.models.advanced_count", "HierarchicalPaceScoreModel"
    )
    config = config_type(
        home_team_column="home_team",
        away_team_column="away_team",
        home_goals_column="target_home_goals",
        away_goals_column="target_away_goals",
        year_column="Year",
        stage_column="canonical_stage",
        host_side_column="host_side",
        common_pace_features=("common_pace_signal",),
        team_shrinkage=4.0,
        max_iter=2_000,
    )
    return model_type(config).fit(_advanced_training_frame())


def _swap_match(frame: pd.DataFrame) -> pd.DataFrame:
    swapped = frame.copy()
    swapped[["home_team", "away_team"]] = frame[["away_team", "home_team"]].to_numpy()
    swapped["host_side"] = frame["host_side"].map(
        {"home": "away", "away": "home", "neutral": "neutral"}
    )
    return swapped


def test_hierarchical_score_model_is_swap_equivariant_and_has_variable_common_pace() -> None:
    """The shared pace term must break the constant-rate-product defect."""
    model = _advanced_model()
    probes = pd.DataFrame(
        [
            {
                "home_team": "A",
                "away_team": "C",
                "Year": 1986,
                "canonical_stage": "group_stage",
                "host_side": "neutral",
                "common_pace_signal": -1.0,
            },
            {
                "home_team": "A",
                "away_team": "C",
                "Year": 2010,
                "canonical_stage": "final",
                "host_side": "neutral",
                "common_pace_signal": 1.0,
            },
            {
                "home_team": "B",
                "away_team": "D",
                "Year": 2006,
                "canonical_stage": "final",
                "host_side": "home",
                "common_pace_signal": 0.7,
            },
        ]
    )
    direct = np.asarray(model.predict_rates(probes), dtype=float)
    reverse = np.asarray(model.predict_rates(_swap_match(probes)), dtype=float)

    assert direct.shape == (3, 2)
    assert np.isfinite(direct).all() and np.all(direct > 0)
    np.testing.assert_allclose(direct[:, 0], reverse[:, 1], atol=1e-10, rtol=0)
    np.testing.assert_allclose(direct[:, 1], reverse[:, 0], atol=1e-10, rtol=0)

    rate_products = direct[:, 0] * direct[:, 1]
    assert np.ptp(rate_products) > 1e-3, (
        "lambda_home * lambda_away is effectively constant; the model lacks a "
        "match-specific symmetric pace component"
    )
    components = model.predict_components(probes)
    required = {"common_pace", "era_effect", "stage_effect", "host_effect"}
    assert required <= set(components.columns)
    assert components["common_pace"].nunique() > 1

    isolated_contexts = pd.DataFrame(
        [
            {
                "home_team": "A",
                "away_team": "C",
                "Year": 1986,
                "canonical_stage": "group_stage",
                "host_side": "neutral",
                "common_pace_signal": 0.0,
            },
            {
                "home_team": "A",
                "away_team": "C",
                "Year": 2010,
                "canonical_stage": "group_stage",
                "host_side": "neutral",
                "common_pace_signal": 0.0,
            },
            {
                "home_team": "A",
                "away_team": "C",
                "Year": 1986,
                "canonical_stage": "final",
                "host_side": "neutral",
                "common_pace_signal": 0.0,
            },
            {
                "home_team": "A",
                "away_team": "C",
                "Year": 1986,
                "canonical_stage": "group_stage",
                "host_side": "home",
                "common_pace_signal": 0.0,
            },
        ]
    )
    isolated_rates = np.asarray(model.predict_rates(isolated_contexts), dtype=float)
    isolated_components = model.predict_components(isolated_contexts)
    assert isolated_components.loc[0, "era_effect"] != pytest.approx(
        isolated_components.loc[1, "era_effect"]
    )
    assert isolated_components.loc[0, "stage_effect"] != pytest.approx(
        isolated_components.loc[2, "stage_effect"]
    )
    assert isolated_components.loc[0, "host_effect"] != pytest.approx(
        isolated_components.loc[3, "host_effect"]
    )
    assert not np.isclose(np.prod(isolated_rates[0]), np.prod(isolated_rates[1]))
    assert not np.isclose(np.prod(isolated_rates[0]), np.prod(isolated_rates[2]))
    assert not np.allclose(isolated_rates[0], isolated_rates[3])


def test_hierarchical_score_model_separates_attack_defence_and_shrinks_sparse_teams() -> None:
    model = _advanced_model()
    effects = model.team_effects_.copy()
    assert {"team", "attack", "defence", "sample_size", "shrinkage_weight"} <= set(
        effects.columns
    )
    assert effects["team"].is_unique
    assert not np.allclose(effects["attack"], effects["defence"])

    frequent = effects.loc[effects["team"].eq("A")].iloc[0]
    rare = effects.loc[effects["team"].eq("RARE")].iloc[0]
    assert rare["sample_size"] < frequent["sample_size"]
    assert 0 <= rare["shrinkage_weight"] < frequent["shrinkage_weight"] <= 1
    # A single 7-0 must not produce an unregularized seven-goal team effect.
    assert abs(float(rare["attack"])) < 2.0

    opponent_probe = pd.DataFrame(
        [
            {
                "home_team": "A",
                "away_team": opponent,
                "Year": 2010,
                "canonical_stage": "group_stage",
                "host_side": "neutral",
                "common_pace_signal": 0.0,
            }
            for opponent in ("B", "C")
        ]
    )
    opponent_components = model.predict_components(opponent_probe)
    assert opponent_components.loc[0, "home_attack_effect"] == pytest.approx(
        opponent_components.loc[1, "home_attack_effect"]
    )
    assert opponent_components.loc[0, "away_defence_effect"] != pytest.approx(
        opponent_components.loc[1, "away_defence_effect"]
    )

    unknown = pd.DataFrame(
        [
            {
                "home_team": "NEVER_SEEN",
                "away_team": "C",
                "Year": 2010,
                "canonical_stage": "group_stage",
                "host_side": "neutral",
                "common_pace_signal": 0.0,
            }
        ]
    )
    rates = np.asarray(model.predict_rates(unknown), dtype=float)
    components = model.predict_components(unknown).iloc[0]
    assert np.isfinite(rates).all() and np.all(rates > 0)
    assert bool(components["home_unknown_team"]) is True
    assert float(components["home_attack_effect"]) == pytest.approx(0.0, abs=1e-12)
    assert float(components["home_defence_effect"]) == pytest.approx(0.0, abs=1e-12)


def _calibration_provenance(count: int) -> pd.DataFrame:
    years = np.resize(np.array([1998, 2002, 2006, 2010]), count)
    return pd.DataFrame(
        {
            "match_id": np.arange(1, count + 1),
            "prediction_year": years,
            "train_end_year": years - 4,
            "outer_fold_id": [f"outer_{year}" for year in years],
            "is_oof": np.ones(count, dtype=bool),
        }
    )


def test_minimum_kl_projection_matches_outcome_margins_without_destroying_score_shape() -> None:
    project = _required_symbol(
        "worldcup_forecasting.models.score_calibration",
        "minimum_kl_outcome_projection",
    )
    base = score_matrix(1.65, 0.95, rho=-0.04, max_score=8, adaptive=False)
    target = {"home_win": 0.41, "draw": 0.31, "away_win": 0.28}
    projected = np.asarray(project(base, target), dtype=float)

    assert projected.shape == base.shape
    assert np.isfinite(projected).all() and np.all(projected >= 0)
    assert projected.sum() == pytest.approx(1.0, abs=1e-12)
    actual = outcome_probabilities(projected)
    for outcome, probability in target.items():
        assert actual[outcome] == pytest.approx(probability, abs=1e-12)

    masks = (
        np.tril(np.ones_like(base, dtype=bool), k=-1),
        np.eye(base.shape[0], dtype=bool),
        np.triu(np.ones_like(base, dtype=bool), k=1),
    )
    for mask in masks:
        positive = mask & (base > 0)
        # The I-projection scales every cell inside one W/D/L partition by one
        # constant, hence all within-partition likelihood ratios are preserved.
        scaling = projected[positive] / base[positive]
        np.testing.assert_allclose(scaling, np.full_like(scaling, scaling[0]), atol=1e-12)

    swapped_target = {
        "home_win": target["away_win"],
        "draw": target["draw"],
        "away_win": target["home_win"],
    }
    swapped = np.asarray(project(base.T, swapped_target), dtype=float)
    np.testing.assert_allclose(swapped, projected.T, atol=1e-12, rtol=0)

    no_home_win_mass = np.diag([0.6, 0.4])
    with pytest.raises(ValueError, match="zero.*mass|support|partition"):
        project(
            no_home_win_mass,
            {"home_win": 0.2, "draw": 0.8, "away_win": 0.0},
        )


def test_diagonal_calibration_reweights_joint_scores_and_preserves_coherence() -> None:
    calibrator_type = _required_symbol(
        "worldcup_forecasting.models.score_calibration", "DiagonalScoreCalibrator"
    )
    base = score_matrix(1.75, 0.85, rho=0.0, max_score=8, adaptive=False)
    reversed_base = base.T.copy()
    joint = np.stack([base, reversed_base] * 8)
    observed = np.asarray([[0, 0], [0, 0], [1, 1], [1, 1]] * 4, dtype=int)
    calibrator = calibrator_type().fit_oof(
        joint,
        observed,
        provenance=_calibration_provenance(len(joint)),
    )
    calibrated = np.asarray(calibrator.transform(joint), dtype=float)

    assert calibrated.shape == joint.shape
    assert np.isfinite(calibrated).all() and np.all(calibrated >= 0)
    np.testing.assert_allclose(calibrated.sum(axis=(1, 2)), 1.0, atol=1e-12)
    np.testing.assert_allclose(calibrated[1], calibrated[0].T, atol=1e-12, rtol=0)
    assert not np.allclose(calibrated[0], joint[0]), (
        "A selected diagonal calibrator must reweight score cells, not merely replace "
        "the derived home/draw/away vector"
    )
    derived = outcome_probabilities(calibrated[0])
    assert derived["draw"] == pytest.approx(float(np.trace(calibrated[0])), abs=1e-12)

    report = calibrator.selection_report_
    assert report["fit_scope"] == "training_oof_only"
    assert report["selected"] is True
    assert report["candidate_log_loss"] <= report["baseline_log_loss"] + 1e-12
    assert report["candidate_brier"] <= report["baseline_brier"] + 1e-12


def test_diagonal_calibration_rejects_non_oof_or_temporally_contaminated_fit_rows() -> None:
    calibrator_type = _required_symbol(
        "worldcup_forecasting.models.score_calibration", "DiagonalScoreCalibrator"
    )
    joint = np.stack([score_matrix(1.2, 1.0, max_score=6, adaptive=False)] * 4)
    observed = np.asarray([[1, 0], [0, 0], [0, 1], [1, 1]], dtype=int)

    in_sample = _calibration_provenance(4)
    in_sample.loc[0, "is_oof"] = False
    with pytest.raises(ValueError, match="OOF|out-of-fold"):
        calibrator_type().fit_oof(joint, observed, provenance=in_sample)

    contaminated = _calibration_provenance(4)
    contaminated.loc[0, "train_end_year"] = contaminated.loc[0, "prediction_year"]
    with pytest.raises(ValueError, match="earlier|chronolog|train_end"):
        calibrator_type().fit_oof(joint, observed, provenance=contaminated)


def test_cost_sensitive_draw_decision_is_not_probability_calibration() -> None:
    decision_type = _required_symbol(
        "worldcup_forecasting.models.decision", "OOFCostSensitiveDecisionRule"
    )
    probabilities = np.asarray(
        [[0.36, 0.34, 0.30]] * 6
        + [[0.52, 0.28, 0.20]] * 6
        + [[0.24, 0.30, 0.46]] * 6,
        dtype=float,
    )
    labels = np.asarray(["draw"] * 6 + ["home_win"] * 6 + ["away_win"] * 6)
    provenance = _calibration_provenance(len(labels))
    baseline = np.asarray([("home_win", "draw", "away_win")[i] for i in probabilities.argmax(1)])

    rule = decision_type().fit_oof(probabilities, labels, provenance=provenance)
    decisions = np.asarray(rule.predict(probabilities))
    untouched = np.asarray(rule.transform_probabilities(probabilities), dtype=float)

    np.testing.assert_array_equal(untouched, probabilities)
    assert recall_score(labels, decisions, labels=["draw"], average="macro") > recall_score(
        labels, baseline, labels=["draw"], average="macro", zero_division=0
    )
    assert f1_score(labels, decisions, average="macro") > f1_score(
        labels, baseline, average="macro"
    )
    assert rule.metadata_["layer_type"] == "decision_only"
    assert rule.metadata_["probability_calibration"] is False
    assert rule.metadata_["fit_scope"] == "training_oof_only"
    assert "log_loss" not in rule.metadata_["valid_evaluation_metrics"]


def test_nested_expanding_origin_contract_and_candidate_budget() -> None:
    folds_fn = _required_symbol(
        "worldcup_forecasting.validation.nested", "nested_expanding_origin_folds"
    )
    budget_type = _required_symbol(
        "worldcup_forecasting.validation.nested", "CandidateBudget"
    )
    years = np.repeat(np.arange(1986, 2015, 4), 3)
    frame = pd.DataFrame({"Year": years, "match_id": np.arange(len(years))})
    folds = folds_fn(
        frame,
        outer_test_years=(1998, 2002, 2006, 2010),
        year_column="Year",
        minimum_inner_folds=2,
    )

    assert len(folds) == 4
    for outer in folds:
        outer_train_years = years[np.asarray(outer.train_index, dtype=int)]
        outer_valid_years = years[np.asarray(outer.validation_index, dtype=int)]
        assert outer_valid_years.min() > outer_train_years.max()
        assert outer_valid_years.max() <= 2010
        assert len(outer.inner_folds) >= 2
        for inner in outer.inner_folds:
            inner_train = years[np.asarray(inner.train_index, dtype=int)]
            inner_valid = years[np.asarray(inner.validation_index, dtype=int)]
            assert inner_train.max() < inner_valid.min() < outer_valid_years.min()

    budget = budget_type(
        max_configurations=80,
        maximum_selection_year=2010,
        forbidden_audit_years=(2014, 2018),
    )
    first = budget.register({"model": "joint", "alpha": 0.1, "seed": 1})
    duplicate = budget.register({"seed": 1, "alpha": 0.1, "model": "joint"})
    assert duplicate == first
    assert budget.count == 1
    with pytest.raises(ValueError, match="2014|audit|selection"):
        budget.register({"model": "contaminated", "data_years": [2010, 2014]})
    for index in range(79):
        budget.register({"model": "candidate", "index": index})
    assert budget.count == 80
    with pytest.raises(ValueError, match="80|budget"):
        budget.register({"model": "candidate", "index": 80})
    budget.freeze()
    with pytest.raises(RuntimeError, match="frozen"):
        budget.register({"model": "late_candidate"})


def _valid_oof_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": [101, 102, 201, 202],
            "prediction_year": [2006, 2006, 2010, 2010],
            "prediction_time": pd.to_datetime(
                ["2006-06-01", "2006-06-02", "2010-06-01", "2010-06-02"]
            ),
            "train_end_year": [2002, 2002, 2006, 2006],
            "outer_fold_id": ["outer_2006", "outer_2006", "outer_2010", "outer_2010"],
            "inner_fold_id": ["deploy", "deploy", "deploy", "deploy"],
            "candidate_id": ["candidate_sha"] * 4,
            "model_id": ["hierarchical_pace"] * 4,
            "seed": [31415] * 4,
            "information_set": ["pre_match"] * 4,
            "fit_index_hash": ["fit_2002", "fit_2002", "fit_2006", "fit_2006"],
            "feature_schema_hash": ["schema_sha"] * 4,
            "home_win": [0.60, 0.25, 0.20, 0.40],
            "draw": [0.25, 0.35, 0.30, 0.30],
            "away_win": [0.15, 0.40, 0.50, 0.30],
            "target_outcome": ["home_win", "away_win", "away_win", "draw"],
            "observed_home_goals": [2, 0, 0, 1],
            "observed_away_goals": [0, 1, 2, 1],
            "is_oof": [True] * 4,
        }
    )


def test_per_match_oof_predictions_have_auditable_persisted_schema(tmp_path: Path) -> None:
    validate = _required_symbol(
        "worldcup_forecasting.validation.research_contracts", "validate_oof_frame"
    )
    write = _required_symbol(
        "worldcup_forecasting.validation.research_contracts", "write_oof_predictions"
    )
    frame = _valid_oof_frame()
    report = validate(frame)
    assert report["row_count"] == len(frame)
    assert report["maximum_prediction_year"] == 2010
    assert report["all_rows_out_of_fold"] is True
    assert isinstance(report["schema_hash"], str) and len(report["schema_hash"]) == 64

    destination = tmp_path / "nested" / "per_match_oof.csv"
    written = Path(write(frame, destination))
    assert written == destination and written.is_file()
    restored = pd.read_csv(written)
    assert "Unnamed: 0" not in restored
    assert restored["match_id"].tolist() == frame["match_id"].tolist()
    assert set(frame.columns) == set(restored.columns)
    validate(restored)

    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate|unique"):
        validate(duplicated)
    contaminated = frame.copy()
    contaminated.loc[0, "train_end_year"] = contaminated.loc[0, "prediction_year"]
    with pytest.raises(ValueError, match="earlier|chronolog|train_end"):
        validate(contaminated)


def _valid_ablation_matrix() -> pd.DataFrame:
    removal = {
        "A0": "none_full_model",
        "A1": "dynamic_ratings",
        "A2": "player_features",
        "A3": "era_effects",
        "A4": "correlated_count_model",
        "A5": "machine_learning_residual",
        "A6": "probability_calibration",
        "A7": "hierarchical_shrinkage",
        "A8": "symmetry_constraint",
    }
    rows = []
    for fold_id, year in (("outer_2006", 2006), ("outer_2010", 2010)):
        for seed in (17, 29):
            for position, (variant, component) in enumerate(removal.items()):
                rows.append(
                    {
                        "variant_id": variant,
                        "removed_component": component,
                        "outer_fold_id": fold_id,
                        "test_year": year,
                        "seed": seed,
                        "evaluation_index_hash": f"eval_{year}",
                        "log_loss": 1.0 + 0.002 * position,
                        "multiclass_brier": 0.60 + 0.002 * position,
                        "runtime_seconds": 0.1,
                        "structural_tests_passed": True,
                    }
                )
    return pd.DataFrame(rows)


def test_a0_a8_ablation_matrix_is_complete_and_pairwise_comparable() -> None:
    validate = _required_symbol(
        "worldcup_forecasting.validation.research_contracts", "validate_ablation_matrix"
    )
    frame = _valid_ablation_matrix()
    report = validate(frame)
    assert report["variants"] == [f"A{index}" for index in range(9)]
    assert report["complete_cartesian_matrix"] is True
    assert report["paired_evaluation_indices"] is True

    missing = frame.loc[~frame["variant_id"].eq("A8")]
    with pytest.raises(ValueError, match="A8|complete"):
        validate(missing)
    unpaired = frame.copy()
    unpaired.loc[unpaired["variant_id"].eq("A3"), "evaluation_index_hash"] = "wrong"
    with pytest.raises(ValueError, match="paired|evaluation_index_hash|identical"):
        validate(unpaired)


def _valid_stress_matrix() -> pd.DataFrame:
    scenarios = (
        "new_or_rare_team",
        "unknown_team_name",
        "player_absence",
        "extreme_scoreline",
        "high_halftime_score",
        "host_to_neutral",
        "group_to_knockout",
        "team_swap",
        "drop_10pct_history",
        "time_decay_perturbation",
        "random_seed_perturbation",
        "format_2026_out_of_range",
    )
    return pd.DataFrame(
        [
            {
                "scenario_id": scenario,
                "model_id": "hierarchical_pace",
                "seed": seed,
                "probability_simplex_error": 1e-15,
                "symmetry_error": 1e-14,
                "halftime_violation": 0.0,
                "finite_outputs": True,
                "structural_tests_passed": True,
                "metric_delta": 0.01,
            }
            for scenario in scenarios
            for seed in (17, 29)
        ]
    )


def test_full_stress_matrix_covers_every_declared_failure_mode() -> None:
    validate = _required_symbol(
        "worldcup_forecasting.validation.research_contracts", "validate_stress_matrix"
    )
    frame = _valid_stress_matrix()
    report = validate(frame)
    assert report["scenario_count"] == 12
    assert report["complete_seed_matrix"] is True
    assert report["all_structural_invariants_passed"] is True

    incomplete = frame.loc[~frame["scenario_id"].eq("format_2026_out_of_range")]
    with pytest.raises(ValueError, match="format_2026_out_of_range|scenario|complete"):
        validate(incomplete)
    broken = frame.copy()
    broken.loc[0, "structural_tests_passed"] = False
    with pytest.raises(ValueError, match="structural|invariant"):
        validate(broken)


def _information_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matches = pd.DataFrame(
        {
            "MatchID": [1, 2],
            "Year": [2002, 2006],
            "Datetime": pd.to_datetime(["2002-06-01 12:00", "2006-06-01 12:00"]),
            "Home Team Initials": ["AAA", "AAA"],
            "Away Team Initials": ["BBB", "BBB"],
            "Home Team Goals": [1, 0],
            "Away Team Goals": [0, 0],
        }
    )
    players = pd.DataFrame(
        {
            "MatchID": [1, 1, 1, 1, 2, 2, 2, 2],
            "Team Initials": ["AAA", "AAA", "BBB", "BBB"] * 2,
            "Player Name": ["A One", "A Two", "B One", "B Two"] * 2,
            "Line-up": ["S", "N", "S", "N", "S", "N", "S", "N"],
            "Event": ["G20", "", "Y40", "", "", "", "", ""],
        }
    )
    coaches = pd.DataFrame(
        {
            "MatchID": [1, 1, 2, 2],
            "Team Initials": ["AAA", "BBB", "AAA", "BBB"],
            "Coach Name": ["Coach A", "Coach B", "Coach A", "Coach C"],
        }
    )
    return matches, players, coaches


def _bundle_frame(bundle: Any, name: str) -> pd.DataFrame:
    value = bundle[name] if isinstance(bundle, dict) else getattr(bundle, name)
    assert isinstance(value, pd.DataFrame)
    return value


def test_player_coach_information_sets_separate_pre_match_kickoff_and_events() -> None:
    build = _required_symbol(
        "worldcup_forecasting.features.information_sets",
        "build_player_coach_information_sets",
    )
    matches, players, coaches = _information_fixture()
    baseline = build(players=players, matches=matches, coaches=coaches)
    pre_match = _bundle_frame(baseline, "pre_match")
    kickoff = _bundle_frame(baseline, "kickoff")
    lineage = _bundle_frame(baseline, "lineage")

    assert {"feature_name", "prediction_information_set", "available_time"} <= set(
        lineage.columns
    )
    prior = lineage["feature_name"].str.contains("player_prior|coach_prior", regex=True)
    current_lineup = lineage["feature_name"].str.contains("current_lineup|current_roster")
    assert prior.any() and current_lineup.any()
    assert set(lineage.loc[prior, "prediction_information_set"]) == {"pre_match"}
    assert set(lineage.loc[current_lineup, "prediction_information_set"]) == {"kickoff"}
    assert not lineage["feature_name"].str.contains("current.*(?:goal|card)", regex=True).any()

    pre_columns = lineage.loc[
        lineage["prediction_information_set"].eq("pre_match"), "feature_name"
    ].tolist()
    kickoff_columns = lineage.loc[
        lineage["prediction_information_set"].eq("kickoff"), "feature_name"
    ].tolist()
    match_key = "MatchID" if "MatchID" in pre_match else "match_id"
    base_pre_row = pre_match.loc[pre_match[match_key].eq(2), pre_columns].reset_index(drop=True)
    base_kickoff_row = kickoff.loc[
        kickoff[match_key].eq(2), kickoff_columns
    ].reset_index(drop=True)

    event_mutation = players.copy()
    event_mutation.loc[event_mutation["MatchID"].eq(2), "Event"] = "G01 Y02"
    event_result = build(players=event_mutation, matches=matches, coaches=coaches)
    pd.testing.assert_frame_equal(
        base_pre_row,
        _bundle_frame(event_result, "pre_match")
        .loc[lambda value: value[match_key].eq(2), pre_columns]
        .reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        base_kickoff_row,
        _bundle_frame(event_result, "kickoff")
        .loc[lambda value: value[match_key].eq(2), kickoff_columns]
        .reset_index(drop=True),
    )

    lineup_mutation = players.copy()
    lineup_mutation.loc[
        lineup_mutation["MatchID"].eq(2) & lineup_mutation["Player Name"].eq("A Two"),
        "Line-up",
    ] = "S"
    lineup_result = build(players=lineup_mutation, matches=matches, coaches=coaches)
    pd.testing.assert_frame_equal(
        base_pre_row,
        _bundle_frame(lineup_result, "pre_match")
        .loc[lambda value: value[match_key].eq(2), pre_columns]
        .reset_index(drop=True),
    )
    changed_kickoff = _bundle_frame(lineup_result, "kickoff").loc[
        lambda value: value[match_key].eq(2), kickoff_columns
    ]
    assert not base_kickoff_row.equals(changed_kickoff.reset_index(drop=True))
