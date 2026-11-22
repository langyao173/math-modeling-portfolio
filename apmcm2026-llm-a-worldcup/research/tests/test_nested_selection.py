"""Contracts for frozen, chronological advanced-model selection."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import worldcup_forecasting.experiments.nested_selection as nested_module
from worldcup_forecasting.experiments._nested_selection_residual import residual_design
from worldcup_forecasting.experiments.nested_selection import (
    NestedSelectionConfig,
    fit_final_selected_model,
    frozen_candidate_specs,
    run_nested_selection,
    select_final_development_spec,
)
from worldcup_forecasting.experiments.stress import run_measured_stress_tests
from worldcup_forecasting.models._selective_poisson import OptimizationTimeoutError
from worldcup_forecasting.models.advanced_count import (
    EquivariantCountConfig,
    EquivariantCountModel,
    joint_batch_from_rates,
)
from worldcup_forecasting.simulation.scenario import build_format_scenario_provider
from worldcup_forecasting.validation.research_contracts import validate_oof_frame


class _StepClock:
    def __init__(self, step: float) -> None:
        self.value = 0.0
        self.step = step

    def __call__(self) -> float:
        self.value += self.step
        return self.value


def _count_config(
    *, alpha: float, half_life_years: float | None, shared_score: bool
) -> EquivariantCountConfig:
    return EquivariantCountConfig(
        odd_numeric_features=(
            "strength_diff",
            "host_indicator_diff",
            "player_availability_diff",
            "history_goal_diff",
        ),
        even_numeric_features=(
            "pace",
            "tournament_teams",
            "tournament_matches",
            "matches_per_team",
            "is_knockout",
        ),
        even_categorical_features=("format_regime",),
        paired_numeric_features=(
            ("home_player_availability", "away_player_availability"),
        ),
        optional_odd_numeric_features=(),
        optional_even_numeric_features=(),
        optional_even_categorical_features=(),
        home_team_column="home_team",
        away_team_column="away_team",
        target_home_column="target_home_goals",
        target_away_column="target_away_goals",
        year_column="Year",
        stage_column="canonical_stage",
        host_side_column="host_side",
        alpha=alpha,
        time_decay_half_life_years=half_life_years,
        shared_score=shared_score,
        max_iter=300,
        max_score=5,
        max_support=20,
    )


def _frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    teams = ("A", "B", "C", "D")
    match_id = 0
    for edition_position, year in enumerate(range(1974, 2011, 4)):
        for position in range(3 + edition_position % 4):
            home = teams[position % 4]
            away = teams[(position + edition_position + 1) % 4]
            if home == away:
                away = teams[(teams.index(away) + 1) % 4]
            match_id += 1
            strength = float(teams.index(away) - teams.index(home)) / 3.0
            host_side = ("home", "away", "neutral")[position % 3]
            host_indicator = (
                1.0 if host_side == "home" else -1.0 if host_side == "away" else 0.0
            )
            personnel = 0.5 * strength
            stage = "quarter_final" if position % 3 == 0 else "group_stage"
            rows.append(
                {
                    "match_id": match_id,
                    "Datetime": pd.Timestamp(year=year, month=6, day=position + 1),
                    "Year": year,
                    "home_team": home,
                    "away_team": away,
                    "pace": -0.5 + 0.2 * position + 0.03 * edition_position,
                    "strength_diff": strength,
                    "host_indicator_diff": host_indicator,
                    "player_availability_diff": personnel,
                    "home_player_availability": 11.0 + max(personnel, 0.0),
                    "away_player_availability": 11.0 + max(-personnel, 0.0),
                    "history_goal_diff": strength + 0.05 * edition_position,
                    "canonical_stage": stage,
                    "host_side": host_side,
                    "tournament_teams": 32.0,
                    "tournament_matches": 64.0,
                    "matches_per_team": 4.5,
                    "is_knockout": float(stage != "group_stage"),
                    "format_regime": "legacy_32_team",
                    "target_home_goals": int((position + edition_position) % 3),
                    "target_away_goals": int((2 * position + edition_position) % 3),
                }
            )
    return pd.DataFrame(rows)


def test_frozen_candidate_family_is_exact_hash_stable_and_audit_safe() -> None:
    first, first_manifest = frozen_candidate_specs()
    second, second_manifest = frozen_candidate_specs()
    assert len(first) == 36
    assert [spec.candidate_id for spec in first] == [spec.candidate_id for spec in second]
    assert len(set(spec.candidate_id for spec in first)) == 36
    assert first_manifest == second_manifest
    assert first_manifest == (
        "a95ce1b0fb315e599023bb5822ec56f2224eba867cf4bb4f3c24b8d868b51e43"
    )
    assert {spec.alpha for spec in first} == {0.1, 1.0, 10.0}
    assert {spec.half_life_years for spec in first} == {8.0, 16.0, None}
    assert {spec.shared_score for spec in first} == {False, True}
    assert {spec.residual for spec in first} == {False, True}

    with pytest.raises(ValueError, match="2014|selection|maximum"):
        run_nested_selection(
            _frame(),
            config=NestedSelectionConfig(outer_test_years=(2014,)),
            count_config_factory=_count_config,
        )
    with pytest.raises(ValueError, match="selection|maximum"):
        run_nested_selection(
            _frame(),
            config=NestedSelectionConfig(maximum_selection_year=float("nan")),  # type: ignore[arg-type]
            count_config_factory=_count_config,
        )
    with pytest.raises(ValueError, match="explicit match_id"):
        run_nested_selection(
            _frame().drop(columns="match_id"),
            config=NestedSelectionConfig(outer_test_years=(1998,)),
            count_config_factory=_count_config,
        )
    audit_row = _frame().iloc[[-1]].copy()
    audit_row["match_id"] = 99_999
    audit_row["Year"] = 2014
    with pytest.raises(ValueError, match="cannot load 2014/2018|post-2010"):
        run_nested_selection(
            pd.concat([_frame(), audit_row], ignore_index=True),
            config=NestedSelectionConfig(outer_test_years=(1998,)),
            count_config_factory=_count_config,
        )


def test_nested_selection_is_chronological_complete_and_deterministic() -> None:
    frame = _frame()
    config = NestedSelectionConfig(
        outer_test_years=(1998,), latest_inner_folds=3, seed=29
    )
    first = run_nested_selection(
        frame.sample(frac=1.0, random_state=71),
        config=config,
        feature_schema_hash="f" * 64,
        count_config_factory=_count_config,
        runtime_clock=_StepClock(0.001),
    )
    second = run_nested_selection(
        frame,
        config=replace(config),
        feature_schema_hash="f" * 64,
        count_config_factory=_count_config,
        runtime_clock=_StepClock(1.2),
    )

    assert len(first.candidate_metrics) == 36
    assert len(first.selected_specs) == 1
    assert first.inner_fold_metrics.groupby("candidate_id").size().eq(3).all()
    residual_rows = first.inner_fold_metrics.query("residual and residual_training_rows > 0")
    assert (residual_rows["residual_fit_max_year"] < residual_rows["inner_test_year"]).all()
    assert first.candidate_metrics["selection_objective"].notna().all()
    assert first.candidate_metrics["candidate_id"].is_unique
    assert first.candidate_metrics["config_hash"].is_unique
    assert first.candidate_metrics["candidate_id"].tolist() == second.candidate_metrics[
        "candidate_id"
    ].tolist()
    volatile_runtime = ["measured_runtime_seconds", "outer_fit_runtime_seconds"]
    pd.testing.assert_frame_equal(
        first.selected_specs.drop(columns=volatile_runtime),
        second.selected_specs.drop(columns=volatile_runtime),
    )
    np.testing.assert_allclose(
        first.candidate_metrics["selection_objective"],
        second.candidate_metrics["selection_objective"],
        atol=0.0,
    )
    assert not np.allclose(
        first.candidate_metrics["measured_runtime_seconds"],
        second.candidate_metrics["measured_runtime_seconds"],
    )
    pd.testing.assert_frame_equal(first.oof_ledger, second.oof_ledger)

    metrics = first.candidate_metrics
    recomputed = (
        metrics["mean_log_loss"]
        + 0.25 * metrics["mean_multiclass_brier"]
        + 0.05 * metrics["mean_ece"]
        + 0.05 * metrics["fold_sd_log_loss"]
        + 0.001 * np.log1p(metrics["runtime_seconds"])
        + 0.0005 * metrics["complexity_units"]
    )
    np.testing.assert_allclose(
        recomputed.to_numpy(), metrics["selection_objective"].to_numpy(), atol=1e-14
    )
    assert metrics["runtime_seconds"].between(0.3, 0.5).all()
    assert first.inner_fold_metrics["match_count"].nunique() > 1
    equal_edition_means = first.inner_fold_metrics.groupby("candidate_id")[
        "log_loss"
    ].mean()
    reported_means = metrics.set_index("candidate_id")["mean_log_loss"]
    np.testing.assert_allclose(
        equal_edition_means.loc[reported_means.index].to_numpy(),
        reported_means.to_numpy(),
        atol=1e-14,
    )

    report = validate_oof_frame(first.oof_ledger)
    assert report["maximum_prediction_year"] == 1998
    assert first.oof_ledger["candidate_id"].str.len().eq(64).all()
    assert first.oof_ledger["config_hash"].str.len().eq(64).all()
    assert first.oof_ledger["fit_index_hash"].str.len().eq(64).all()
    assert len(first.rate_ledger) == int(frame["Year"].eq(1998).sum())
    validate_oof_frame(first.selected_inner_oof_ledger)
    assert first.selected_inner_oof_ledger["prediction_year"].nunique() == 3
    assert (
        first.selected_inner_oof_ledger["train_end_year"]
        < first.selected_inner_oof_ledger["prediction_year"]
    ).all()
    applied = first.selected_inner_oof_ledger["residual_applied"]
    if applied.any():
        assert (
            first.selected_inner_oof_ledger.loc[applied, "residual_fit_max_year"]
            < first.selected_inner_oof_ledger.loc[applied, "prediction_year"]
        ).all()
    inner_grid_mass = first.selected_inner_joint_grid.groupby(
        ["selection_outer_fold_id", "outer_fold_id", "match_id"]
    )["probability"].sum()
    np.testing.assert_allclose(inner_grid_mass.to_numpy(), 1.0, atol=1e-12)
    grid_mass = first.joint_grid.groupby(["outer_fold_id", "match_id"])[
        "probability"
    ].sum()
    np.testing.assert_allclose(grid_mass.to_numpy(), 1.0, atol=1e-12)
    assert first.joint_grid.columns.tolist() == [
        "outer_fold_id",
        "match_id",
        "home_goals",
        "away_goals",
        "probability",
    ]
    assert first.rate_ledger.columns[:4].tolist() == [
        "outer_fold_id",
        "match_id",
        "expected_home_goals",
        "expected_away_goals",
    ]
    assert first.metadata["candidate_count"] == 36
    assert first.metadata["maximum_selection_year"] == 2010
    assert first.metadata["runtime_reference_hash"] == (
        "1c86c59e07515948470845f6fbcbfc44657ff58bf221774f197ec712a10158c4"
    )
    assert first.metadata["live_runtime_role"].startswith("diagnostic_only")


def test_residual_selection_deploys_only_from_earlier_structural_oof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = nested_module.evaluate_spec

    def favor_residual(*args: object, **kwargs: object):
        rows, summary = original(*args, **kwargs)
        spec = args[2]
        if (
            spec.residual
            and spec.half_life_years == 16.0
            and not spec.shared_score
            and summary["eligible"]
        ):
            summary = {
                **summary,
                "selection_objective": summary["selection_objective"] - 100.0,
            }
        return rows, summary

    monkeypatch.setattr(nested_module, "evaluate_spec", favor_residual)
    result = run_nested_selection(
        _frame(),
        config=NestedSelectionConfig(outer_test_years=(1998,), latest_inner_folds=3),
        feature_schema_hash="e" * 64,
        count_config_factory=_count_config,
        runtime_clock=lambda: 0.0,
    )
    assert bool(result.selected_specs.iloc[0]["residual"])
    assert result.oof_ledger["residual_applied"].all()
    assert result.rate_ledger["residual_applied"].all()
    applied_by_year = result.selected_inner_oof_ledger.groupby("prediction_year")[
        "residual_applied"
    ].all()
    assert applied_by_year.tolist() == [False, True, True]
    applied = result.selected_inner_oof_ledger["residual_applied"]
    assert (
        result.selected_inner_oof_ledger.loc[applied, "residual_fit_max_year"]
        < result.selected_inner_oof_ledger.loc[applied, "prediction_year"]
    ).all()
    selected = result.selected_specs.iloc[0]
    half_life = selected["half_life_years"]
    selected_config = _count_config(
        alpha=float(selected["alpha"]),
        half_life_years=(None if pd.isna(half_life) else float(half_life)),
        shared_score=bool(selected["shared_score"]),
    )
    expected_joint = joint_batch_from_rates(
        result.rate_ledger[["expected_home_goals", "expected_away_goals"]].to_numpy(),
        shared_fraction=float(result.rate_ledger["shared_fraction"].iloc[0]),
        max_score=selected_config.max_score,
        min_retained_mass=selected_config.min_retained_mass,
        max_support=selected_config.max_support,
    )
    for position, match_id in enumerate(result.rate_ledger["match_id"]):
        observed = result.joint_grid.loc[
            result.joint_grid["match_id"].eq(match_id), "probability"
        ].to_numpy()
        np.testing.assert_allclose(
            observed, expected_joint.probability[position].ravel(), atol=1e-14
        )


def test_candidate_factory_failure_is_contained_and_early_folds_are_never_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_structural = nested_module.structural_predictions
    requested_windows: list[tuple[int, ...]] = []

    def guarded_structural(*args: object, **kwargs: object):
        folds = args[1]
        years = tuple(fold.validation_year for fold in folds)
        requested_windows.append(years)
        assert years == (1986, 1990, 1994)
        return original_structural(*args, **kwargs)

    monkeypatch.setattr(nested_module, "structural_predictions", guarded_structural)

    def partly_failing_factory(
        *, alpha: float, half_life_years: float | None, shared_score: bool
    ) -> EquivariantCountConfig:
        if alpha == 10.0:
            raise RuntimeError("deliberate candidate-local failure")
        returned_alpha = 1.0 if alpha == 0.1 else alpha
        return _count_config(
            alpha=returned_alpha,
            half_life_years=half_life_years,
            shared_score=shared_score,
        )

    result = run_nested_selection(
        _frame(),
        config=NestedSelectionConfig(outer_test_years=(1998,), latest_inner_folds=3),
        feature_schema_hash="d" * 64,
        count_config_factory=partly_failing_factory,
        runtime_clock=lambda: 0.0,
    )
    failed = result.candidate_metrics.query("alpha == 10.0")
    mislabeled = result.candidate_metrics.query("alpha == 0.1")
    assert len(failed) == 12
    assert len(mislabeled) == 12
    assert (~failed["eligible"]).all()
    assert (~mislabeled["eligible"]).all()
    assert failed["failure_reason"].str.contains("candidate-local failure").all()
    assert mislabeled["failure_reason"].str.contains(
        "does not match the frozen candidate spec"
    ).all()
    assert bool(result.selected_specs.iloc[0]["eligible"])
    assert float(result.selected_specs.iloc[0]["alpha"]) == 1.0
    assert requested_windows


def test_timeout_candidate_is_recorded_ineligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs, budget_hash = frozen_candidate_specs()
    good = next(spec for spec in specs if spec.alpha == 1.0 and not spec.residual)
    timed_out = next(spec for spec in specs if spec.alpha == 10.0 and not spec.residual)
    monkeypatch.setattr(
        nested_module,
        "frozen_candidate_specs",
        lambda: ((good, timed_out), budget_hash),
    )
    original_structural = nested_module.structural_predictions

    def bounded_structural(*args: object, **kwargs: object):
        model_config = args[2]
        if model_config.alpha == 10.0:
            raise OptimizationTimeoutError("fixed candidate wall-clock budget exceeded")
        return original_structural(*args, **kwargs)

    monkeypatch.setattr(nested_module, "structural_predictions", bounded_structural)
    result = run_nested_selection(
        _frame(),
        config=NestedSelectionConfig(outer_test_years=(1998,), latest_inner_folds=1),
        feature_schema_hash="e" * 64,
        count_config_factory=_count_config,
        runtime_clock=lambda: 0.0,
    )

    failed = result.candidate_metrics.loc[
        result.candidate_metrics["candidate_id"].eq(timed_out.candidate_id)
    ]
    assert len(failed) == 1
    assert not bool(failed.iloc[0]["eligible"])
    assert "OptimizationTimeoutError" in str(failed.iloc[0]["failure_reason"])
    assert result.selected_specs.iloc[0]["candidate_id"] == good.candidate_id


def test_final_development_selection_and_fit_never_load_audit_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame()
    original = nested_module.evaluate_spec

    def favor_residual(*args: object, **kwargs: object):
        rows, summary = original(*args, **kwargs)
        spec = args[2]
        if (
            spec.residual
            and spec.half_life_years == 16.0
            and not spec.shared_score
            and summary["eligible"]
        ):
            summary = {
                **summary,
                "selection_objective": summary["selection_objective"] - 100.0,
            }
        return rows, summary

    monkeypatch.setattr(nested_module, "evaluate_spec", favor_residual)
    result = select_final_development_spec(
        frame,
        latest_inner_folds=3,
        feature_schema_hash="c" * 64,
        count_config_factory=_count_config,
        runtime_clock=lambda: 0.0,
    )

    assert result.metadata["selection_inner_test_years"] == [2002, 2006, 2010]
    assert result.metadata["development_cutoff_year"] == 2010
    assert result.metadata["audit_rows_loaded"] is False
    assert len(result.candidate_metrics) == 36
    assert result.inner_fold_metrics.groupby("candidate_id").size().eq(3).all()
    assert result.selected_candidate_hash == result.selected_spec.candidate_id
    assert result.selected_spec.residual
    assert result.selected_config_hash == result.selected_summary.iloc[0]["config_hash"]
    assert set(result.selected_inner_oof_ledger["prediction_year"]) == {
        2002,
        2006,
        2010,
    }
    assert result.selected_inner_oof_ledger["prediction_year"].max() <= 2010
    assert (
        result.selected_inner_oof_ledger["train_end_year"]
        < result.selected_inner_oof_ledger["prediction_year"]
    ).all()
    validate_oof_frame(result.selected_inner_oof_ledger)
    mass = result.selected_inner_joint_grid.groupby(
        ["selection_outer_fold_id", "outer_fold_id", "match_id"]
    )["probability"].sum()
    np.testing.assert_allclose(mass.to_numpy(), 1.0, atol=1e-12)

    fitted = fit_final_selected_model(frame, result, runtime_clock=lambda: 0.0)
    assert fitted.residual_applied
    assert fitted.residual_oof_max_year == 2010
    assert len(fitted.final_fit_hash) == 64
    assert fitted.metadata["candidate_id"] == result.selected_candidate_hash
    assert fitted.config == result.selected_model_config
    assert fitted.home_team_column_ == "home_team"
    assert fitted.away_team_column_ == "away_team"
    assert fitted.year_column_ == "Year"
    assert fitted.stage_column_ == "canonical_stage"
    assert fitted.host_side_column_ == "host_side"
    assert fitted.paired_columns_ == (
        ("home_player_availability", "away_player_availability"),
    )
    assert "strength_diff" in fitted.odd_columns_
    assert "tournament_teams" in fitted.even_columns_
    assert "format_regime" in fitted.context_columns_
    assert fitted.shared_fraction_ == fitted.metadata["shared_fraction"]
    audit_features = frame.tail(2).copy()
    audit_features["match_id"] = [10_001, 10_002]
    audit_features["Year"] = 2014
    audit_features["Datetime"] = pd.to_datetime(["2014-06-12", "2014-06-13"])
    audit_features = audit_features.drop(
        columns=["target_home_goals", "target_away_goals"]
    )
    aliases = audit_features.copy()
    aliases["host_side"] = ["home_team", -1]
    alias_design, _, _ = residual_design(aliases, result.selected_model_config)
    np.testing.assert_array_equal(alias_design["host_side_signed"], [1.0, -1.0])
    unknown_host = aliases.copy()
    unknown_host.loc[0, "host_side"] = "mystery_host"
    with pytest.raises(ValueError, match="Unknown host-side value"):
        residual_design(unknown_host, result.selected_model_config)
    rates = fitted.predict_rates(audit_features)
    joint = fitted.predict_joint(audit_features)
    probabilities = fitted.predict_proba(audit_features)
    direct_joint = joint_batch_from_rates(
        rates,
        shared_fraction=float(fitted.metadata["shared_fraction"]),
        max_score=result.selected_model_config.max_score,
        min_retained_mass=result.selected_model_config.min_retained_mass,
        max_support=result.selected_model_config.max_support,
    )
    assert rates.shape == (2, 2) and np.isfinite(rates).all() and (rates > 0).all()
    assert joint.probability.shape[0] == 2
    np.testing.assert_allclose(joint.probability, direct_joint.probability, atol=1e-14)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)
    assert joint.model_metadata["final_fit_hash"] == fitted.final_fit_hash

    scenario_teams = tuple(f"T{index:02d}" for index in range(48))
    profile_rows: list[pd.DataFrame] = []
    template = frame.loc[frame["Year"].eq(2010)].iloc[[0]]
    for position in range(24):
        profile_rows.append(
            template.assign(
                match_id=20_000 + position,
                home_team=scenario_teams[2 * position],
                away_team=scenario_teams[2 * position + 1],
            )
        )
    scenario = build_format_scenario_provider(
        fitted,
        pd.concat(profile_rows, ignore_index=True),
        teams=scenario_teams,
    )
    scenario_pmf = scenario.score_provider(
        scenario.teams[0], scenario.teams[1], "group_stage"
    )
    assert np.isclose(scenario_pmf.sum(), 1.0, atol=1e-12)

    stress_probe = audit_features.assign(
        half_time_home_goals=[1, 0], half_time_away_goals=[0, 1]
    )

    def refit_callback(
        training: pd.DataFrame, overrides: dict[str, object], seed: int
    ) -> object:
        del seed
        if overrides.get("stress_scenario") == "baseline_provenance_check":
            return fitted
        replacement = {}
        if "time_decay_half_life_years" in overrides:
            replacement["time_decay_half_life_years"] = float(
                overrides["time_decay_half_life_years"]
            )
        return EquivariantCountModel(
            replace(result.selected_model_config, **replacement)
        ).fit(training)

    def halftime_checker(
        probe: pd.DataFrame, probabilities: np.ndarray, seed: int
    ) -> np.ndarray:
        del probabilities, seed
        evidence = np.zeros((len(probe), 8, 8), dtype=float)
        for row, values in enumerate(probe.itertuples(index=False)):
            evidence[
                row,
                int(values.half_time_home_goals),
                int(values.half_time_away_goals),
            ] = 1.0
        return evidence

    stress = run_measured_stress_tests(
        fitted_model=fitted,
        training_frame=frame,
        probe_frame=stress_probe,
        model_id="final_selected_equivariant_count",
        seeds=(17, 29),
        refit_callback=refit_callback,
        half_time_checker=halftime_checker,
        maximum_training_year=2010,
    )
    assert len(stress) == 24
    assert stress["structural_tests_passed"].all()
    with pytest.raises(ValueError, match="after its fit cutoff"):
        fitted.predict_rates(frame.tail(1))
    tampered = frame.copy()
    tampered.loc[0, "pace"] = float(tampered.loc[0, "pace"]) + 0.01
    with pytest.raises(ValueError, match="differs from the frozen selection frame"):
        fit_final_selected_model(tampered, result, runtime_clock=lambda: 0.0)

    audit_row = frame.iloc[[-1]].copy()
    audit_row["match_id"] = int(frame["match_id"].max()) + 1
    audit_row["Year"] = 2014
    audit_row["target_home_goals"] = "not-even-read"
    with pytest.raises(ValueError, match="cannot load 2014/2018|post-2010"):
        select_final_development_spec(
            pd.concat([frame, audit_row], ignore_index=True),
            count_config_factory=_count_config,
        )
