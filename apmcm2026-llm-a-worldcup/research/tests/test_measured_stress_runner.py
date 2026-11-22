"""TDD contracts for executable, evidence-bearing structural stress tests."""

from __future__ import annotations

import importlib
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from worldcup_forecasting.models.advanced_count import (
    EquivariantCountConfig,
    EquivariantCountModel,
)
from worldcup_forecasting.validation.research_contracts import (
    STRESS_SCENARIOS,
    validate_stress_matrix,
)


def _runner() -> Any:
    module_name = "worldcup_forecasting.experiments.stress"
    try:
        module = importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.fail(f"RED measured-stress runner is absent: {module_name} ({exc})", pytrace=False)
    return module.run_measured_stress_tests


def _training_frame(seed: int = 711) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = ("AAA", "BBB", "CCC", "DDD", "EEE")
    rows: list[dict[str, object]] = []
    for index in range(180):
        home = teams[index % len(teams)]
        away = teams[(2 * index + 1) % len(teams)]
        if home == away:
            away = teams[(teams.index(away) + 1) % len(teams)]
        year = (1986, 1990, 1994, 1998, 2002, 2006, 2010)[index % 7]
        stage = "quarter_final" if index % 5 == 0 else "group_stage"
        host_side = ("home", "away", "neutral")[index % 3]
        strength = float((teams.index(home) - teams.index(away)) / 2)
        host_diff = 1.0 if host_side == "home" else -1.0 if host_side == "away" else 0.0
        personnel = float((index % 4) - 1.5)
        history = strength + 0.25 * np.sin(index)
        pace = -0.6 if index % 4 < 2 else 0.7
        era = (year - 1998) / 12
        common = 0.15 + 0.24 * pace + 0.08 * era
        home_rate = np.exp(common + 0.28 * strength + 0.10 * host_diff + 0.04 * personnel)
        away_rate = np.exp(common - 0.28 * strength - 0.10 * host_diff - 0.04 * personnel)
        rows.append(
            {
                "match_id": index + 1,
                "home_team": home,
                "away_team": away,
                "Year": year,
                "canonical_stage": stage,
                "host_side": host_side,
                "pace": pace,
                "era": era,
                "strength_diff": strength,
                "host_indicator_diff": host_diff,
                "player_availability_diff": personnel,
                "home_player_availability": 11.0 + max(personnel, 0.0),
                "away_player_availability": 11.0 + max(-personnel, 0.0),
                "history_goal_diff": history,
                "tournament_teams": 32.0,
                "tournament_matches": 64.0,
                "matches_per_team": 4.5,
                "format_regime": "legacy_32_team",
                "target_home_goals": int(rng.poisson(home_rate)),
                "target_away_goals": int(rng.poisson(away_rate)),
            }
        )
    # A sparse but already-known team is distinct from the unknown-team stress.
    rows.append(
        {
            **rows[-1],
            "match_id": 10_001,
            "home_team": "RARE",
            "away_team": "CCC",
            "Year": 2010,
            "target_home_goals": 1,
            "target_away_goals": 1,
        }
    )
    return pd.DataFrame(rows)


def _config() -> EquivariantCountConfig:
    return EquivariantCountConfig(
        odd_numeric_features=(
            "strength_diff",
            "host_indicator_diff",
            "player_availability_diff",
            "history_goal_diff",
        ),
        even_numeric_features=(
            "pace",
            "era",
            "tournament_teams",
            "tournament_matches",
            "matches_per_team",
        ),
        even_categorical_features=("format_regime",),
        paired_numeric_features=(("home_player_availability", "away_player_availability"),),
        home_team_column="home_team",
        away_team_column="away_team",
        home_goals_column="target_home_goals",
        away_goals_column="target_away_goals",
        stage_column="canonical_stage",
        host_side_column="host_side",
        time_decay_half_life_years=16.0,
        alpha=1.5,
    )


def _probe_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "home_team": ["AAA", "CCC"],
            "away_team": ["BBB", "DDD"],
            "Year": [2010, 2010],
            "canonical_stage": ["group_stage", "group_stage"],
            "host_side": ["home", "away"],
            "pace": [0.2, -0.1],
            "era": [1.0, 1.0],
            "strength_diff": [1.2, -0.6],
            "host_indicator_diff": [1.0, -1.0],
            "player_availability_diff": [0.8, -0.4],
            "home_player_availability": [11.0, 10.0],
            "away_player_availability": [9.0, 11.0],
            "history_goal_diff": [1.0, -0.5],
            "tournament_teams": [32.0, 32.0],
            "tournament_matches": [64.0, 64.0],
            "matches_per_team": [4.5, 4.5],
            "format_regime": ["legacy_32_team", "legacy_32_team"],
            "half_time_home_goals": [1, 0],
            "half_time_away_goals": [0, 1],
        }
    )


class _RefitRecorder:
    def __init__(self, config: EquivariantCountConfig) -> None:
        self.config = config
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, training_frame: pd.DataFrame, overrides: dict[str, Any], seed: int
    ) -> EquivariantCountModel:
        self.calls.append(
            {
                "rows": len(training_frame),
                "years": tuple(sorted(training_frame["Year"].unique())),
                "overrides": dict(overrides),
                "seed": seed,
            }
        )
        supported = {
            key: value
            for key, value in overrides.items()
            if key in {"time_decay_half_life_years"}
        }
        return EquivariantCountModel(replace(self.config, **supported)).fit(training_frame)


def _run() -> tuple[pd.DataFrame, _RefitRecorder, list[int]]:
    training = _training_frame()
    config = _config()
    model = EquivariantCountModel(config).fit(training)
    recorder = _RefitRecorder(config)
    halftime_calls: list[int] = []

    def halftime_checker(frame: pd.DataFrame, probabilities: np.ndarray, seed: int) -> np.ndarray:
        assert frame["half_time_home_goals"].min() >= 6
        assert frame["half_time_away_goals"].min() >= 5
        assert np.isfinite(probabilities).all()
        halftime_calls.append(seed)
        return _valid_halftime_support(frame)

    table = _invoke(model, training, recorder, halftime_checker)
    return table, recorder, halftime_calls


def _valid_halftime_support(frame: pd.DataFrame) -> np.ndarray:
    evidence = np.zeros((len(frame), 12, 12), dtype=float)
    for row, values in enumerate(frame.itertuples(index=False)):
        home = int(values.half_time_home_goals)
        away = int(values.half_time_away_goals)
        evidence[row, home, away] = 0.7
        evidence[row, home + 1, away + 1] = 0.3
    return evidence


def _invoke(
    model: Any, training: pd.DataFrame, refit: Any, checker: Any,
    *, seeds: tuple[int, ...] = (17, 29), cutoff: object = 2010,
) -> pd.DataFrame:
    return _runner()(
        fitted_model=model, training_frame=training, probe_frame=_probe_frame(),
        model_id="s2_equivariant_poisson_glm", seeds=seeds, refit_callback=refit,
        half_time_checker=checker, maximum_training_year=cutoff,
    )


def test_runner_emits_complete_validator_clean_measured_matrix() -> None:
    table, _, _ = _run()
    report = validate_stress_matrix(table)

    assert len(table) == len(STRESS_SCENARIOS) * 2
    assert report["scenario_count"] == 12
    assert report["seed_count"] == 2
    evidence_columns = {
        "baseline_home_win",
        "baseline_draw",
        "baseline_away_win",
        "stressed_home_win",
        "stressed_draw",
        "stressed_away_win",
        "probability_delta",
        "ood_flag",
        "unknown_team_flag",
        "refit_performed",
        "expected_behavior",
        "perturbation",
    }
    assert evidence_columns <= set(table.columns)
    assert table["expected_behavior"].str.len().gt(10).all()
    assert table["perturbation"].str.len().gt(5).all()
    assert table["probability_delta"].ge(0).all()
    sensitive = set(STRESS_SCENARIOS) - {"high_halftime_score", "random_seed_perturbation"}
    for scenario in sensitive:
        assert table.loc[table["scenario_id"].eq(scenario), "probability_delta"].gt(1e-8).all()
    np.testing.assert_allclose(
        table[["stressed_home_win", "stressed_draw", "stressed_away_win"]].sum(1),
        1.0,
        atol=1e-12,
    )


def test_refit_stresses_use_real_training_subsets_and_frozen_overrides() -> None:
    table, recorder, _ = _run()
    refit_scenarios = {
        "drop_10pct_history",
        "time_decay_perturbation",
        "random_seed_perturbation",
    }
    assert set(table.loc[table["refit_performed"], "scenario_id"]) == refit_scenarios
    assert len(recorder.calls) == 9  # baseline verification + 2 + 2 + paired 4
    assert all(max(call["years"]) <= 2010 for call in recorder.calls)

    deletion_calls = [
        call
        for call in recorder.calls
        if call["overrides"]["stress_scenario"] == "drop_10pct_history"
    ]
    assert all(call["rows"] < len(_training_frame()) for call in deletion_calls)
    assert all(call["overrides"]["dropped_years"] for call in deletion_calls)
    decay_calls = [
        call
        for call in recorder.calls
        if call["overrides"]["stress_scenario"] == "time_decay_perturbation"
    ]
    assert {call["overrides"]["time_decay_factor"] for call in decay_calls} == {0.5, 2.0}
    assert {
        call["overrides"]["time_decay_half_life_years"] for call in decay_calls
    } == {8.0, 32.0}


def test_no_decay_selected_model_receives_two_finite_decay_sensitivities() -> None:
    training = _training_frame()
    config = replace(_config(), time_decay_half_life_years=None)
    model = EquivariantCountModel(config).fit(training)
    recorder = _RefitRecorder(config)

    table = _invoke(
        model,
        training,
        recorder,
        lambda frame, probability, seed: _valid_halftime_support(frame),
    )

    decay_calls = [
        call
        for call in recorder.calls
        if call["overrides"]["stress_scenario"] == "time_decay_perturbation"
    ]
    assert {
        call["overrides"]["time_decay_half_life_years"] for call in decay_calls
    } == {8.0, 16.0}
    assert {call["overrides"]["time_decay_factor"] for call in decay_calls} == {None}
    perturbations = table.loc[
        table["scenario_id"].eq("time_decay_perturbation"), "perturbation"
    ]
    assert perturbations.str.contains("no-decay baseline").all()


def test_team_swap_and_high_halftime_are_executed_not_metadata_placeholders() -> None:
    table, _, halftime_calls = _run()
    swapped = table.loc[table["scenario_id"].eq("team_swap")]
    np.testing.assert_allclose(
        swapped["stressed_home_win"], swapped["baseline_away_win"], atol=1e-10
    )
    np.testing.assert_allclose(
        swapped["stressed_away_win"], swapped["baseline_home_win"], atol=1e-10
    )
    np.testing.assert_allclose(swapped["stressed_draw"], swapped["baseline_draw"], atol=1e-10)
    assert swapped["symmetry_error"].max() <= 1e-10
    assert halftime_calls == [17, 29]
    halftime = table.loc[table["scenario_id"].eq("high_halftime_score")]
    assert halftime["halftime_violation"].eq(0.0).all()
    assert halftime["perturbation"].str.contains("6-5").all()


def test_unknown_rare_personnel_and_2026_format_flags_are_measured() -> None:
    table, _, _ = _run()
    rare = table.loc[table["scenario_id"].eq("new_or_rare_team")]
    unknown = table.loc[table["scenario_id"].eq("unknown_team_name")]
    personnel = table.loc[table["scenario_id"].eq("player_absence")]
    format_shift = table.loc[table["scenario_id"].eq("format_2026_out_of_range")]

    assert rare["ood_flag"].all() and not rare["unknown_team_flag"].any()
    assert unknown["ood_flag"].all() and unknown["unknown_team_flag"].all()
    assert personnel["ood_flag"].all()
    assert personnel["personnel_absolute_max"].eq(0.0).all()
    assert format_shift["ood_flag"].all()
    assert format_shift["perturbation"].str.contains("48.*104", regex=True).all()
    assert unknown["probability_delta"].gt(0).all()
    assert format_shift["probability_delta"].gt(0).all()


def test_runner_is_deterministic_and_fails_closed_on_unsafe_protocols() -> None:
    first, _, _ = _run()
    second, _, _ = _run()
    pd.testing.assert_frame_equal(first, second)

    training = _training_frame()
    config = _config()
    model = EquivariantCountModel(config).fit(training)
    recorder = _RefitRecorder(config)
    contaminated = pd.concat(
        [training, training.iloc[[0]].assign(Year=2014, match_id=99_999)],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="2014|maximum|frozen"):
        _invoke(model, contaminated, recorder,
                lambda frame, probability, seed: _valid_halftime_support(frame))
    with pytest.raises(ValueError, match="two|seed"):
        _invoke(model, training, recorder, _valid_halftime_support, seeds=(17,))
    with pytest.raises(ValueError, match="half-time|checker"):
        _invoke(model, training, recorder, None)


@pytest.mark.parametrize("unsafe_cutoff", [2014, 2022, np.nan, 2010.5, "2010", True])
def test_frozen_2010_refit_boundary_cannot_be_relaxed_by_caller(
    unsafe_cutoff: object,
) -> None:
    training = _training_frame()
    config = _config()
    model = EquivariantCountModel(config).fit(training)
    recorder = _RefitRecorder(config)
    with pytest.raises((TypeError, ValueError), match="2010|cutoff|frozen|maximum"):
        _invoke(model, training, recorder, lambda frame, probability, seed: 0.0,
                cutoff=unsafe_cutoff)


def test_invalid_halftime_support_and_unstable_repeat_fit_are_rejected() -> None:
    training = _training_frame()
    config = _config()
    model = EquivariantCountModel(config).fit(training)
    recorder = _RefitRecorder(config)

    def invalid_support(frame: pd.DataFrame, probability: np.ndarray, seed: int) -> np.ndarray:
        evidence = np.zeros((len(frame), 12, 12))
        evidence[:, 0, 0] = 1.0
        return evidence

    with pytest.raises(ValueError, match="half-time|support|structural"):
        _invoke(model, training, recorder, invalid_support)

    class DivergentBaseline(_RefitRecorder):
        def __call__(self, frame: pd.DataFrame, overrides: dict[str, Any], seed: int) -> Any:
            changed = dict(overrides)
            if changed.get("stress_scenario") == "baseline_provenance_check":
                changed["time_decay_half_life_years"] = 4.0
            return super().__call__(frame, changed, seed)

    with pytest.raises(ValueError, match="baseline|provenance"):
        _invoke(model, training, DivergentBaseline(config),
                lambda frame, probability, seed: _valid_halftime_support(frame))

    class UnstableRefit(_RefitRecorder):
        random_calls = 0

        def __call__(self, frame: pd.DataFrame, overrides: dict[str, Any], seed: int) -> Any:
            changed = dict(overrides)
            if changed.get("stress_scenario") == "random_seed_perturbation":
                self.random_calls += 1
                changed["time_decay_half_life_years"] = 8.0 if self.random_calls % 2 else 32.0
            return super().__call__(frame, changed, seed)

    with pytest.raises(ValueError, match="reproduc|determin|structural"):
        _invoke(model, training, UnstableRefit(config), lambda f, p, s: _valid_halftime_support(f))
