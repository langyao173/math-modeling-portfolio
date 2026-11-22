"""Contracts for the preregistered A7 selective-shrinkage ablation."""

from __future__ import annotations

from dataclasses import replace
from multiprocessing import active_children
from time import perf_counter

import numpy as np
import pandas as pd
import pytest

from worldcup_forecasting.models._selective_poisson import (
    DiagonalPenaltyPoissonRegressor,
    OptimizationTimeoutError,
    build_selective_penalty_profile,
    original_scale_coefficients,
)
from worldcup_forecasting.models.advanced_count import (
    EquivariantCountConfig,
    EquivariantCountModel,
)


def _frame(seed: int = 9047) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = ("A", "B", "C", "D")
    attack = np.asarray([0.45, 0.15, -0.10, -0.35])
    defence = np.asarray([-0.25, 0.20, 0.05, 0.30])
    rows: list[dict[str, object]] = []
    for index in range(160):
        home_index = index % len(teams)
        away_index = (index * 3 + 1) % len(teams)
        if away_index == home_index:
            away_index = (away_index + 1) % len(teams)
        pace = float(np.sin(index / 9.0))
        strength = float(attack[home_index] - attack[away_index])
        host_side = ("home", "away", "neutral")[index % 3]
        host = 0.18 if host_side == "home" else -0.18 if host_side == "away" else 0.0
        common = 0.12 + 0.20 * pace
        home_rate = np.exp(
            common + attack[home_index] + defence[away_index] + 0.13 * strength + host
        )
        away_rate = np.exp(
            common + attack[away_index] + defence[home_index] - 0.13 * strength - host
        )
        rows.append(
            {
                "home_team": teams[home_index],
                "away_team": teams[away_index],
                "Year": 1986 + 4 * (index % 7),
                "stage": "knockout" if index % 5 == 0 else "group",
                "host_side": host_side,
                "pace": pace,
                "strength_diff": strength,
                "target_home_goals": int(rng.poisson(home_rate)),
                "target_away_goals": int(rng.poisson(away_rate)),
            }
        )
    return pd.DataFrame(rows)


def _config(*, alpha: float = 1.0, team_penalty: float | None = None) -> EquivariantCountConfig:
    return EquivariantCountConfig(
        odd_numeric_features=("strength_diff",),
        even_numeric_features=("pace",),
        even_categorical_features=("stage",),
        optional_odd_numeric_features=(),
        optional_even_numeric_features=(),
        optional_even_categorical_features=(),
        home_team_column="home_team",
        away_team_column="away_team",
        target_home_column="target_home_goals",
        target_away_column="target_away_goals",
        host_side_column="host_side",
        alpha=alpha,
        team_shrinkage=team_penalty,
        time_decay_half_life_years=None,
        max_iter=5_000,
        tolerance=1e-10,
    )


def _swap(frame: pd.DataFrame) -> pd.DataFrame:
    swapped = frame.copy()
    swapped[["home_team", "away_team"]] = frame[["away_team", "home_team"]].to_numpy()
    swapped["strength_diff"] = -frame["strength_diff"]
    swapped["host_side"] = frame["host_side"].map(
        {"home": "away", "away": "home", "neutral": "neutral"}
    )
    return swapped


def test_default_penalty_remains_the_uniform_poisson_glm() -> None:
    frame = _frame()
    implicit = EquivariantCountModel(_config(alpha=1.7)).fit(frame)
    explicit = EquivariantCountModel(_config(alpha=1.7, team_penalty=1.7)).fit(frame)

    np.testing.assert_array_equal(implicit.predict_rates(frame), explicit.predict_rates(frame))
    pd.testing.assert_frame_equal(
        implicit.coefficient_penalty_table(), explicit.coefficient_penalty_table()
    )
    table = implicit.coefficient_penalty_table()
    np.testing.assert_array_equal(table["effective_l2_penalty"], 1.7)
    assert not isinstance(
        implicit.pipeline_.named_steps["poisson"], DiagonalPenaltyPoissonRegressor
    )


def test_legacy_uniform_artifact_without_new_audit_attributes_remains_usable() -> None:
    frame = _frame()
    model = EquivariantCountModel(_config(alpha=1.0)).fit(frame)
    del model.penalty_optimizer_
    del model.penalty_profile_

    probe = frame.drop(columns=["target_home_goals", "target_away_goals"]).iloc[:3]
    joint = model.predict_joint(probe)
    assert joint.model_metadata["penalty_parameterization"] == (
        "sklearn_uniform_poisson_regressor"
    )
    table = model.coefficient_penalty_table()
    np.testing.assert_array_equal(table["effective_l2_penalty"], 1.0)


def test_selective_profile_changes_only_attack_and_defence_penalties() -> None:
    frame = _frame()
    baseline = EquivariantCountModel(_config(alpha=1.0, team_penalty=1.0)).fit(frame)
    ablated = EquivariantCountModel(_config(alpha=1.0, team_penalty=0.0)).fit(frame)
    base_table = baseline.coefficient_penalty_table().set_index("feature_name")
    a7_table = ablated.coefficient_penalty_table().set_index("feature_name")

    assert base_table.index.equals(a7_table.index)
    team = a7_table["coefficient_group"].isin(
        ["attacker_team", "opponent_defender_team"]
    )
    assert team.any() and (~team).any()
    np.testing.assert_array_equal(base_table["effective_l2_penalty"], 1.0)
    np.testing.assert_array_equal(a7_table.loc[~team, "effective_l2_penalty"], 1.0)
    np.testing.assert_array_equal(a7_table.loc[team, "effective_l2_penalty"], 0.0)

    # Coefficients exposed for interpretation multiply the original, unscaled
    # design, rather than the internal solver reparameterization.
    assert ablated.pipeline_ is not None
    recovered = original_scale_coefficients(ablated.pipeline_)
    encoded = ablated.pipeline_.named_steps["design"].transform(ablated._perspectives(frame))
    eta = np.asarray(encoded) @ recovered + ablated.pipeline_.named_steps["poisson"].intercept_
    expected = ablated.pipeline_.predict(ablated._perspectives(frame))
    np.testing.assert_allclose(np.exp(eta), expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("alpha", [0.1, 1.0, 10.0])
def test_exact_unshrunk_fit_improves_objective_and_satisfies_constrained_kkt(
    alpha: float,
) -> None:
    frame = _frame()
    model = EquivariantCountModel(_config(alpha=alpha, team_penalty=0.0)).fit(frame)
    assert model.pipeline_ is not None
    optimizer = model.pipeline_.named_steps["poisson"]
    assert isinstance(optimizer, DiagonalPenaltyPoissonRegressor)

    design = np.asarray(
        model.pipeline_.named_steps["design"].transform(model._perspectives(frame)),
        dtype=float,
    )
    target = np.concatenate(
        [frame["target_home_goals"].to_numpy(float), frame["target_away_goals"].to_numpy(float)]
    )
    rate = np.exp(optimizer.intercept_ + design @ optimizer.coef_)
    residual = (rate - target) / target.size
    full_gradient = design.T @ residual + optimizer.effective_penalties_ * optimizer.coef_
    intercept_gradient = float(residual.sum())
    projected_gradient = optimizer.identifiable_parameter_basis_.T @ np.concatenate(
        ([intercept_gradient], full_gradient)
    )
    independent_kkt = float(np.max(np.abs(projected_gradient), initial=0.0))
    constraint_residual = optimizer.constraint_matrix_ @ optimizer.coef_
    augmented_design = np.column_stack([np.ones(design.shape[0]), design])

    assert optimizer.objective_ < optimizer.intercept_only_objective_ - 1e-4
    assert independent_kkt <= 1e-7
    assert optimizer.kkt_residual_ == pytest.approx(independent_kkt, abs=1e-12)
    np.testing.assert_allclose(constraint_residual, 0.0, atol=1e-12)
    assert optimizer.constraint_rank_ == 2
    assert np.linalg.matrix_rank(optimizer.team_contrast_basis_) == design.shape[1] - 2
    assert optimizer.parameter_nullspace_.shape[1] > 0
    np.testing.assert_allclose(
        augmented_design @ optimizer.parameter_nullspace_, 0.0, atol=1e-10
    )
    np.testing.assert_allclose(
        optimizer.effective_penalties_[:, None] * optimizer.parameter_nullspace_[1:],
        0.0,
        atol=1e-12,
    )
    assert optimizer.hessian_rank_ == optimizer.identifiable_rank_
    assert np.linalg.matrix_rank(optimizer.identifiable_parameter_basis_) == (
        optimizer.identifiable_rank_
    )
    assert float(np.std(model.predict_rates(frame))) > 1e-3


def test_disconnected_team_graph_has_unique_order_invariant_gauge() -> None:
    frame = _frame()
    first = EquivariantCountModel(_config(alpha=1.0, team_penalty=0.0)).fit(frame)
    second = EquivariantCountModel(_config(alpha=1.0, team_penalty=0.0)).fit(
        frame.sample(frac=1.0, random_state=19)
    )
    assert first.pipeline_ is not None and second.pipeline_ is not None
    first_poisson = first.pipeline_.named_steps["poisson"]
    second_poisson = second.pipeline_.named_steps["poisson"]

    np.testing.assert_allclose(first_poisson.coef_, second_poisson.coef_, atol=1e-8)
    assert first_poisson.intercept_ == pytest.approx(second_poisson.intercept_, abs=1e-8)
    np.testing.assert_allclose(first.predict_rates(frame), second.predict_rates(frame), atol=1e-8)


def test_exact_unshrunk_team_blocks_preserve_swap_and_probability_invariants() -> None:
    model = EquivariantCountModel(_config(alpha=1.0, team_penalty=0.0)).fit(_frame())
    probe = _frame(seed=91).drop(columns=["target_home_goals", "target_away_goals"]).iloc[:16]
    direct_rate = model.predict_rates(probe)
    swapped_rate = model.predict_rates(_swap(probe))
    np.testing.assert_allclose(direct_rate[:, 0], swapped_rate[:, 1], atol=1e-12, rtol=0)
    np.testing.assert_allclose(direct_rate[:, 1], swapped_rate[:, 0], atol=1e-12, rtol=0)

    direct_joint = model.predict_joint(probe).probability
    swapped_joint = model.predict_joint(_swap(probe)).probability
    np.testing.assert_allclose(direct_joint, swapped_joint.transpose(0, 2, 1), atol=1e-12)
    np.testing.assert_allclose(direct_joint.sum(axis=(1, 2)), 1.0, atol=1e-12)
    direct_probability = model.predict_proba(probe)
    swapped_probability = model.predict_proba(_swap(probe))
    np.testing.assert_allclose(direct_probability.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(direct_probability[:, 0], swapped_probability[:, 2], atol=1e-12)
    np.testing.assert_allclose(direct_probability[:, 1], swapped_probability[:, 1], atol=1e-12)


def test_exact_unshrunk_cold_start_uses_population_team_effects() -> None:
    frame = _frame()
    model = EquivariantCountModel(_config(alpha=1.0, team_penalty=0.0)).fit(frame)
    probe = frame.drop(columns=["target_home_goals", "target_away_goals"]).iloc[[0]].copy()
    probe.loc[:, "home_team"] = "UNSEEN_HOME"
    probe.loc[:, "away_team"] = "UNSEEN_AWAY"

    components = model.predict_components(probe)
    assert components.loc[components.index[0], "home_unknown_team"]
    assert components.loc[components.index[0], "away_unknown_team"]
    np.testing.assert_allclose(
        components[
            [
                "home_attack_effect",
                "away_defence_effect",
                "away_attack_effect",
                "home_defence_effect",
            ]
        ],
        0.0,
        atol=0.0,
    )
    assert np.isfinite(model.predict_rates(probe)).all()


def test_invalid_or_incomplete_penalty_groups_fail_closed() -> None:
    names = (
        "numeric__even::pace",
        "categorical__attacker_team_A",
        "categorical__opponent_defender_team_A",
    )
    with pytest.raises(ValueError, match="exactly attacker_team"):
        build_selective_penalty_profile(
            names,
            base_penalty=1.0,
            team_penalty=1e-6,
            team_groups=("attacker_team", "context::stage"),
        )
    with pytest.raises(ValueError, match="missing required penalty group"):
        build_selective_penalty_profile(
            names[:-1], base_penalty=1.0, team_penalty=1e-6
        )
    with pytest.raises(ValueError, match="positive team penalty with zero base"):
        build_selective_penalty_profile(names, base_penalty=0.0, team_penalty=1.0)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        EquivariantCountModel(replace(_config(), team_shrinkage=-1.0)).fit(_frame())


def test_selective_optimizer_enforces_fixed_wall_clock_budget() -> None:
    design = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [-1.0, 1.0, 0.0],
            [0.5, 0.0, 1.0],
        ]
    )
    target = np.asarray([1.0, 2.0, 0.0, 1.0])
    optimizer = DiagonalPenaltyPoissonRegressor(
        (1.0, 0.0, 0.0),
        ("non_team_context", "attacker_team", "opponent_defender_team"),
        wall_clock_limit_seconds=0.01,
    )
    started = perf_counter()
    with pytest.raises(OptimizationTimeoutError, match="wall-clock budget"):
        optimizer.fit(design, target)
    elapsed = perf_counter() - started

    assert elapsed < 5.0
    assert not any(
        child.name == "diagonal-penalty-poisson-fit" for child in active_children()
    )
