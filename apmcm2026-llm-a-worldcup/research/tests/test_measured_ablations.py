"""Measured, paired A0--A8 ablation contracts."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from worldcup_forecasting.experiments._ablation_helpers import (
    residual_design,
    swap_frame,
)
from worldcup_forecasting.experiments.ablations import (
    AblationConfig,
    run_measured_ablations,
)
from worldcup_forecasting.models.advanced_count import (
    EquivariantCountConfig,
    EquivariantCountModel,
)
from worldcup_forecasting.validation.research_contracts import validate_ablation_matrix


def _count_config(
    *,
    alpha: float,
    half_life_years: float | None,
    shared_score: bool,
    include_dynamic: bool = True,
    include_personnel: bool = True,
    include_era_format: bool = True,
    enforce_shrinkage: bool = True,
) -> EquivariantCountConfig:
    odd = ["strength_diff"]
    if include_dynamic:
        odd.append("dynamic_diff")
    pairs: list[tuple[str, str]] = []
    if include_personnel:
        pairs.append(("home_player", "away_player"))
    even = ["pace"]
    if include_era_format:
        even.append("era_signal")
    return EquivariantCountConfig(
        odd_numeric_features=tuple(odd),
        even_numeric_features=tuple(even),
        optional_odd_numeric_features=(),
        optional_even_numeric_features=(),
        optional_even_categorical_features=(),
        paired_numeric_features=tuple(pairs),
        home_team_column="home_team",
        away_team_column="away_team",
        target_home_column="target_home_goals",
        target_away_column="target_away_goals",
        year_column="Year",
        alpha=alpha,
        team_shrinkage=alpha if enforce_shrinkage else 0.0,
        time_decay_half_life_years=half_life_years,
        shared_score=shared_score,
        max_iter=500,
        max_score=6,
        max_support=24,
    )


def _frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    teams = ("A", "B", "C", "D")
    match_id = 0
    for edition, year in enumerate(range(1970, 2003, 4)):
        for position in range(7):
            home = teams[(position + edition) % len(teams)]
            away = teams[(2 * position + edition + 1) % len(teams)]
            if home == away:
                away = teams[(teams.index(away) + 1) % len(teams)]
            home_rank = teams.index(home)
            away_rank = teams.index(away)
            match_id += 1
            # The ordered home process is intentionally not a mirror image of
            # the away process, so the A8 diagnostic has measurable work to do.
            home_goal = int((2 * position + edition + home_rank) % 4)
            away_goal = int((position + 2 * edition + away_rank) % 3)
            rows.append(
                {
                    "MatchID": match_id,
                    "Datetime": pd.Timestamp(year=year, month=6, day=position + 1),
                    "Year": year,
                    "home_team": home,
                    "away_team": away,
                    "pace": 0.15 * position + 0.04 * edition,
                    "era_signal": float(edition >= 4),
                    "strength_diff": float(away_rank - home_rank),
                    "dynamic_diff": float((home_rank + edition) - away_rank) / 3.0,
                    "home_player": float(2 + home_rank + edition),
                    "away_player": float(1 + away_rank + edition / 2),
                    "target_home_goals": home_goal,
                    "target_away_goals": away_goal,
                }
            )
    return pd.DataFrame(rows)


def _selected(*, residual: bool, shared: bool) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "outer_fold_id": "outer_1998",
                "outer_test_year": 1998,
                "candidate_id": "c" * 64,
                "alpha": 1.0,
                "half_life_years": 16.0,
                "shared_score": shared,
                "residual": residual,
            }
        ]
    )


def test_ablation_engine_emits_complete_paired_measured_matrix() -> None:
    frame = _frame().sample(frac=1.0, random_state=71)
    config = AblationConfig(outer_test_years=(1998,), latest_inner_folds=3)
    result = run_measured_ablations(
        frame,
        _selected(residual=True, shared=True),
        config=config,
        count_config_factory=_count_config,
        feature_schema_hash="a" * 64,
        runtime_clock=lambda: 0.0,
    )

    report = validate_ablation_matrix(result.matrix)
    assert report["complete_cartesian_matrix"]
    assert len(result.matrix) == 18
    assert set(result.matrix["seed"]) == {17, 29}
    assert result.matrix.groupby(["outer_fold_id", "seed"])["variant_id"].nunique().eq(9).all()
    assert result.matrix["evaluation_index_hash"].nunique() == 1
    assert result.matrix["probability_simplex_passed"].all()
    assert result.matrix["expected_structural_status_observed"].all()

    eligible = result.matrix.query("variant_id != 'A8'")
    assert eligible["structurally_eligible"].all()
    assert eligible["symmetry_constraint_passed"].all()
    assert eligible["symmetry_error"].max() <= config.symmetry_tolerance
    diagnostic = result.matrix.query("variant_id == 'A8'")
    assert not diagnostic["structurally_eligible"].any()
    assert not diagnostic["symmetry_constraint_passed"].any()
    assert diagnostic["symmetry_error"].min() > config.symmetry_tolerance
    assert diagnostic["structural_tests_passed"].all()

    matches = int(frame["Year"].eq(1998).sum())
    assert len(result.predictions) == matches * 9 * 2
    probability = result.predictions[["home_win", "draw", "away_win"]].to_numpy()
    np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=1e-12)
    assert result.predictions.groupby(["variant_id", "seed"]).size().eq(matches).all()
    a7 = result.predictions.query("variant_id == 'A7'")
    assert a7.groupby("seed")["home_win"].std().gt(1e-3).all()
    assert (result.predictions["train_end_year"] < result.predictions["test_year"]).all()
    residual = result.predictions.loc[result.predictions["residual_applied"]]
    assert not residual.empty
    assert (residual["residual_fit_max_year"] < residual["test_year"]).all()
    calibrated = result.predictions.loc[result.predictions["calibration_applied"]]
    assert (calibrated["calibration_fit_max_year"] < calibrated["test_year"]).all()
    assert result.metadata["audit_year_used"] is False
    assert result.metadata["maximum_outer_test_year"] == 1998
    assert result.metadata["seed_protocol"] == [17, 29]


def test_absent_components_are_zero_effect_and_run_is_order_deterministic() -> None:
    frame = _frame()
    config = AblationConfig(
        outer_test_years=(1998,), latest_inner_folds=3, information_set="kickoff"
    )
    kwargs = {
        "config": config,
        "count_config_factory": _count_config,
        "feature_schema_hash": "b" * 64,
        "runtime_clock": lambda: 0.0,
    }
    first = run_measured_ablations(
        frame.sample(frac=1.0, random_state=11),
        _selected(residual=False, shared=False),
        **kwargs,
    )
    second = run_measured_ablations(frame, _selected(residual=False, shared=False), **kwargs)

    inactive = first.matrix.query("variant_id in ['A4', 'A5']")
    assert not inactive["component_active_in_a0"].any()
    assert inactive["zero_effect_by_design"].all()
    for variant in ("A4", "A5"):
        variant_rows = first.predictions.loc[
            first.predictions["variant_id"].eq(variant)
        ].reset_index(drop=True)
        full_rows = first.predictions.query("variant_id == 'A0'").reset_index(drop=True)
        np.testing.assert_allclose(
            variant_rows[["home_win", "draw", "away_win"]],
            full_rows[["home_win", "draw", "away_win"]],
            atol=0,
            rtol=0,
        )
    pd.testing.assert_frame_equal(first.matrix, second.matrix)
    pd.testing.assert_frame_equal(first.predictions, second.predictions)
    assert first.predictions["information_set"].eq("kickoff").all()


def test_ablation_engine_fails_closed_on_audit_year_or_bad_selection_contract() -> None:
    frame = _frame()
    audit_row = frame.iloc[[-1]].copy()
    audit_row["MatchID"] = 99_999
    audit_row["Year"] = 2014
    audit_row["target_home_goals"] = 99
    audit_row["target_away_goals"] = 99
    with pytest.raises(ValueError, match="2014|audit|2010|development"):
        run_measured_ablations(
            pd.concat([frame, audit_row], ignore_index=True),
            _selected(residual=False, shared=False),
            config=AblationConfig(outer_test_years=(1998,)),
            count_config_factory=_count_config,
        )
    with pytest.raises(ValueError, match="2014|audit|2010|development"):
        run_measured_ablations(
            frame,
            _selected(residual=False, shared=False).assign(
                outer_fold_id="outer_2014", outer_test_year=2014
            ),
            config=AblationConfig(
                outer_test_years=(2014,), maximum_development_year=2014
            ),
            count_config_factory=_count_config,
        )
    fractional = frame.astype(
        {"target_home_goals": float, "target_away_goals": float}
    )
    fractional.loc[fractional["Year"].eq(1998).idxmax(), "target_home_goals"] = 1.9
    with pytest.raises(ValueError, match="integer|count|goal"):
        run_measured_ablations(
            fractional,
            _selected(residual=False, shared=False),
            config=AblationConfig(outer_test_years=(1998,)),
            count_config_factory=_count_config,
        )
    duplicate_ids = frame.copy()
    duplicate_ids.loc[duplicate_ids.index[-1], "MatchID"] = duplicate_ids.loc[
        duplicate_ids.index[0], "MatchID"
    ]
    with pytest.raises(ValueError, match="match.*unique|duplicate|identif"):
        run_measured_ablations(
            duplicate_ids,
            _selected(residual=False, shared=False),
            config=AblationConfig(outer_test_years=(1998,)),
            count_config_factory=_count_config,
        )
    with pytest.raises(ValueError, match="2014|audit|2010"):
        run_measured_ablations(
            frame,
            _selected(residual=False, shared=False).assign(
                outer_fold_id="outer_2014", outer_test_year=2014
            ),
            config=AblationConfig(outer_test_years=(2014,)),
            count_config_factory=_count_config,
        )
    duplicated = pd.concat(
        [_selected(residual=False, shared=False)] * 2, ignore_index=True
    )
    with pytest.raises(ValueError, match="exactly one|duplicate|selected"):
        run_measured_ablations(
            frame,
            duplicated,
            config=replace(AblationConfig(), outer_test_years=(1998,)),
            count_config_factory=_count_config,
        )


def test_a6_activity_is_defined_by_a0_calibration_not_its_forced_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _SelectedTemperature:
        def __init__(self, *_: object, **__: object) -> None:
            self.chosen_method_ = "temperature"

        def fit_oof(self, *_: object, **__: object) -> _SelectedTemperature:
            return self

        def transform(self, joint: np.ndarray) -> np.ndarray:
            return np.asarray(joint, dtype=float).copy()

    monkeypatch.setattr(
        "worldcup_forecasting.experiments.ablations.ChronologicalCalibrationSelector",
        _SelectedTemperature,
    )
    result = run_measured_ablations(
        _frame(),
        _selected(residual=False, shared=False),
        config=AblationConfig(outer_test_years=(1998,), latest_inner_folds=3),
        count_config_factory=_count_config,
        feature_schema_hash="c" * 64,
        runtime_clock=lambda: 0.0,
    )
    removed = result.matrix.query("variant_id == 'A6'")
    assert removed["component_active_in_a0"].all()
    assert not removed["zero_effect_by_design"].any()


def test_ablation_residual_host_parsing_matches_structural_contract() -> None:
    config = replace(
        _count_config(alpha=1.0, half_life_years=16.0, shared_score=False),
        host_side_column="host_side",
    )
    probe = _frame().iloc[:5].copy()
    probe["host_side"] = ["home_team", "away_team", 1, -1, "neutral"]
    design, _, _ = residual_design(probe, config)
    np.testing.assert_array_equal(
        design["host_side_signed"].to_numpy(), [1.0, -1.0, 1.0, -1.0, 0.0]
    )
    swapped = swap_frame(probe, config)
    assert swapped["host_side"].tolist() == [
        "away",
        "home",
        "away",
        "home",
        "neutral",
    ]
    with pytest.raises(ValueError, match="Unknown host-side"):
        residual_design(probe.assign(host_side="unexpected"), config)


def test_a8_records_empirical_symmetry_without_becoming_eligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "worldcup_forecasting.experiments._ablation_helpers.AsymmetricSidePoisson",
        EquivariantCountModel,
    )
    result = run_measured_ablations(
        _frame(),
        _selected(residual=False, shared=False),
        config=AblationConfig(outer_test_years=(1998,), latest_inner_folds=3),
        count_config_factory=_count_config,
        feature_schema_hash="d" * 64,
        runtime_clock=lambda: 0.0,
    )
    diagnostic = result.matrix.query("variant_id == 'A8'")
    assert diagnostic["symmetry_constraint_passed"].all()
    assert not diagnostic["structurally_eligible"].any()
    predictions = result.predictions.query("variant_id == 'A8'")
    assert predictions["symmetry_constraint_passed"].all()
