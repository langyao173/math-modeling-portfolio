"""Measured, deterministic stress experiments for equivariant score models."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from worldcup_forecasting.validation.research_contracts import (
    STRESS_SCENARIOS,
    validate_stress_matrix,
)

RefitCallback = Callable[[pd.DataFrame, dict[str, Any], int], Any]
HalfTimeChecker = Callable[[pd.DataFrame, np.ndarray, int], np.ndarray]
_FROZEN_MAXIMUM_TRAINING_YEAR, _INVARIANT_TOLERANCE = 2010, 1e-10
_PERSONNEL_FRAGMENTS = ("player", "lineup", "roster", "coach")

_EXPECTED_BEHAVIOR = dict(zip(STRESS_SCENARIOS, (
    "Sparse known teams remain finite under shrinkage.", "Unknown teams use population effects.",
    "Absolute personnel removal remains finite.", "Extreme history preserves invariants.",
    "Full-time support respects the 6-5 lower bound.", "Neutral status changes context only.",
    "Knockout stage uses fitted context effects.", "Team exchange swaps WDL exactly.",
    "Whole-edition deletion measures sensitivity.", "Half-life factors bound recency effects.",
    "Same-seed refits reproduce probabilities.", "The modeled 48/104 context is OOD.",
), strict=True))


def run_measured_stress_tests(
    *, fitted_model: Any,
    training_frame: pd.DataFrame, probe_frame: pd.DataFrame,
    model_id: str, seeds: Sequence[int],
    refit_callback: RefitCallback, half_time_checker: HalfTimeChecker | None,
    maximum_training_year: int = 2010,
) -> pd.DataFrame:
    """Execute the preregistered 12-scenario matrix using pre-audit data only."""
    frozen_seeds = _validate_inputs(
        fitted_model, training_frame, probe_frame, model_id, seeds,
        refit_callback, half_time_checker, maximum_training_year,
    )
    resolved_maximum = int(maximum_training_year)
    provenance_model = refit_callback(
        training_frame.copy(deep=True),
        {"stress_scenario": "baseline_provenance_check", "maximum_training_year": resolved_maximum},
        frozen_seeds[0],
    )
    if not callable(getattr(provenance_model, "predict_proba", None)):
        raise TypeError("refit_callback must return a fitted probability model")
    baseline = _predict_probabilities(fitted_model, probe_frame)
    provenance_probability = _predict_probabilities(provenance_model, probe_frame)
    baseline_provenance_error = float(np.max(np.abs(baseline - provenance_probability)))
    if baseline_provenance_error > 1e-12:
        raise ValueError("The supplied fitted-model baseline failed pre-2014 provenance refitting")
    baseline_mean = baseline.mean(axis=0)
    known_teams = _training_teams(fitted_model, training_frame)
    rows: list[dict[str, Any]] = []
    for seed_position, seed in enumerate(frozen_seeds):
        for scenario in STRESS_SCENARIOS:
            stressed, fit_frame, overrides, flags, perturbation = _prepare_scenario(
                scenario, seed=seed, seed_position=seed_position, fitted_model=fitted_model,
                training_frame=training_frame, probe_frame=probe_frame, known_teams=known_teams,
            )
            refit_performed = fit_frame is not None
            fit_overrides = {**overrides, "maximum_training_year": resolved_maximum}
            scenario_model = (
                refit_callback(fit_frame, fit_overrides, seed)
                if fit_frame is not None
                else fitted_model
            )
            if not callable(getattr(scenario_model, "predict_proba", None)):
                raise TypeError("refit_callback must return a fitted probability model")
            stressed_probability = _predict_probabilities(scenario_model, stressed)
            reverse_frame = _swap_probe(stressed, scenario_model)
            reverse_probability = _predict_probabilities(scenario_model, reverse_frame)
            reproducibility_error = 0.0
            if scenario == "random_seed_perturbation":
                repeated_model = refit_callback(fit_frame.copy(deep=True), fit_overrides, seed)
                if not callable(getattr(repeated_model, "predict_proba", None)):
                    raise TypeError("refit_callback must return a fitted probability model")
                repeated = _predict_probabilities(repeated_model, stressed)
                repeated_reverse = _predict_probabilities(repeated_model, reverse_frame)
                reproducibility_error = float(
                    max(np.max(np.abs(stressed_probability - repeated)),
                        np.max(np.abs(reverse_probability - repeated_reverse)))
                )
            symmetry_error = float(np.max(
                np.abs(stressed_probability[:, [2, 1, 0]] - reverse_probability)
            ))
            simplex_error = float(np.max(np.abs(stressed_probability.sum(axis=1) - 1.0)))
            finite_outputs = bool(np.isfinite(stressed_probability).all()
                                  and np.isfinite(reverse_probability).all())
            halftime_violation = 0.0
            if scenario == "high_halftime_score":
                assert half_time_checker is not None
                evidence = half_time_checker(stressed.copy(), stressed_probability.copy(), seed)
                halftime_violation = _halftime_support_violation(stressed, evidence)
            stressed_mean = stressed_probability.mean(axis=0)
            probability_delta = float(np.max(np.abs(stressed_probability - baseline)))
            structural_passed = bool(
                finite_outputs and simplex_error <= 1e-10 and symmetry_error <= 1e-10
                and halftime_violation <= 1e-12 and reproducibility_error <= 1e-12
                and baseline_provenance_error <= 1e-12
            )
            rows.append({
                    "scenario_id": scenario, "model_id": model_id, "seed": seed,
                    "probability_simplex_error": simplex_error, "simplex_error": simplex_error,
                    "symmetry_error": symmetry_error, "halftime_violation": halftime_violation,
                    "reproducibility_error": reproducibility_error,
                    "baseline_provenance_error": baseline_provenance_error,
                    "finite_outputs": finite_outputs, "structural_tests_passed": structural_passed,
                    "metric_delta": probability_delta, "probability_delta": probability_delta,
                    "baseline_home_win": float(baseline_mean[0]),
                    "baseline_draw": float(baseline_mean[1]),
                    "baseline_away_win": float(baseline_mean[2]),
                    "stressed_home_win": float(stressed_mean[0]),
                    "stressed_draw": float(stressed_mean[1]),
                    "stressed_away_win": float(stressed_mean[2]),
                    "personnel_absolute_max": _personnel_absolute_max(stressed, scenario_model),
                    "ood_flag": bool(flags["ood"]),
                    "unknown_team_flag": bool(flags["unknown_team"]),
                    "refit_performed": refit_performed, "refit_indicator": refit_performed,
                    "expected_behavior": _EXPECTED_BEHAVIOR[scenario], "perturbation": perturbation,
                })
    table = pd.DataFrame(rows)
    validate_stress_matrix(table)
    return table


def _validate_inputs(
    fitted_model: Any, training_frame: pd.DataFrame, probe_frame: pd.DataFrame,
    model_id: str, seeds: Sequence[int],
    refit_callback: RefitCallback,
    half_time_checker: HalfTimeChecker | None,
    maximum_training_year: int,
) -> tuple[int, ...]:
    for name, frame in (("training_frame", training_frame), ("probe_frame", probe_frame)):
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError(f"{name} must be a nonempty DataFrame")
    if not callable(getattr(fitted_model, "predict_proba", None)):
        raise TypeError("fitted_model must expose predict_proba")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be nonempty")
    if not callable(refit_callback):
        raise ValueError("A refit callback is required for measured refit scenarios")
    if not callable(half_time_checker):
        raise ValueError("A declared half-time checker is required")
    frozen = tuple(seeds)
    if len(frozen) < 2 or len(set(frozen)) != len(frozen):
        raise ValueError("At least two unique frozen seeds are required")
    if any(isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) for seed in frozen):
        raise ValueError("Stress seeds must be integers")
    valid_cutoff_type = isinstance(maximum_training_year, (int, np.integer))
    if isinstance(maximum_training_year, bool) or not valid_cutoff_type:
        raise TypeError("The frozen maximum training-year cutoff must be an integer <= 2010")
    resolved_maximum = int(maximum_training_year)
    if not 0 < resolved_maximum <= _FROZEN_MAXIMUM_TRAINING_YEAR:
        raise ValueError("The caller cannot relax the frozen 2010 maximum training-year cutoff")
    year_column = _year_column(fitted_model, training_frame)
    years = pd.to_numeric(training_frame[year_column], errors="coerce")
    if years.isna().any() or years.max() > resolved_maximum:
        raise ValueError(f"Stress refitting is frozen through {resolved_maximum}; 2014+ forbidden")
    return tuple(int(seed) for seed in frozen)


def _prepare_scenario(
    scenario: str, *, seed: int, seed_position: int, fitted_model: Any,
    training_frame: pd.DataFrame,
    probe_frame: pd.DataFrame,
    known_teams: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, Any], dict[str, bool], str]:
    stressed = probe_frame.copy(deep=True)
    flags = {"ood": False, "unknown_team": False}
    home, _ = _team_columns(fitted_model)
    if scenario == "new_or_rare_team":
        rare = _rarest_team(fitted_model, training_frame)
        stressed[home] = rare
        flags["ood"] = True
        return stressed, None, {}, flags, f"home team replaced by sparse known team {rare}"
    if scenario == "unknown_team_name":
        unknown = "ZZZ_STRESS_UNKNOWN"
        if unknown in known_teams:
            raise RuntimeError("Synthetic unknown team collides with training data")
        stressed[home] = unknown
        flags.update(ood=True, unknown_team=True)
        return stressed, None, {}, flags, f"home team replaced by valid unseen team {unknown}"
    if scenario == "player_absence":
        pairs = _personnel_pairs(fitted_model)
        if not pairs:
            raise ValueError("player_absence requires paired absolute personnel features")
        columns = [column for pair in pairs for column in pair]
        columns.extend(_matching_model_features(fitted_model, _PERSONNEL_FRAGMENTS))
        columns = list(dict.fromkeys(columns))
        stressed.loc[:, columns] = 0.0
        flags["ood"] = True
        return stressed, None, {}, flags, f"absolute personnel removal in {columns}"
    if scenario == "extreme_scoreline":
        columns = _matching_model_features(
            fitted_model, ("history", "goal", "elo", "attack", "defen")
        )
        if not columns:
            raise ValueError("extreme_scoreline stress requires a numeric history feature")
        for position, column in enumerate(columns):
            stressed[column] = _extreme_value(training_frame[column]) * (-1) ** position
        flags["ood"] = True
        return stressed, None, {}, flags, f"history features moved beyond training range {columns}"
    if scenario == "high_halftime_score":
        stressed[["half_time_home_goals", "half_time_away_goals"]] = (6, 5)
        flags["ood"] = True
        return stressed, None, {}, flags, "declared 6-5 halftime lower-bound check"
    if scenario == "host_to_neutral":
        _neutralize_host(stressed, fitted_model)
        return stressed, None, {}, flags, "host status neutralized with signed inputs zeroed"
    if scenario == "group_to_knockout":
        stage = getattr(fitted_model, "stage_column_", None)
        if not stage or stage not in stressed:
            raise ValueError("group_to_knockout stress requires a fitted stage column")
        stressed[stage] = "quarter_final"
        return stressed, None, {}, flags, "canonical stage changed group_stage to quarter_final"
    if scenario == "team_swap":
        return _swap_probe(stressed, fitted_model), None, {}, flags, "exact S2 team exchange"
    if scenario == "drop_10pct_history":
        reduced, dropped = _drop_tournament_blocks(training_frame, fitted_model, seed)
        overrides = {"stress_scenario": scenario, "dropped_years": dropped}
        return stressed, reduced, overrides, flags, f"refit after dropping editions {dropped}"
    if scenario == "time_decay_perturbation":
        configured = fitted_model.config.time_decay_half_life_years
        if configured is None:
            half_life = 8.0 if seed_position % 2 == 0 else 16.0
            factor: float | None = None
            perturbation = f"no-decay baseline replaced by half-life {half_life:g} years"
        else:
            factor = 0.5 if seed_position % 2 == 0 else 2.0
            half_life = float(configured) * factor
            perturbation = f"half-life x{factor}"
        overrides = {
            "stress_scenario": scenario,
            "time_decay_factor": factor,
            "time_decay_half_life_years": half_life,
        }
        return stressed, training_frame.copy(), overrides, flags, perturbation
    if scenario == "random_seed_perturbation":
        overrides = {"stress_scenario": scenario, "random_seed": seed}
        return stressed, training_frame.copy(), overrides, flags, f"repeat refit seed {seed}"
    if scenario == "format_2026_out_of_range":
        _apply_2026_format(stressed, fitted_model)
        flags["ood"] = True
        return stressed, None, {}, flags, "context changed to 48 teams and 104 matches"
    raise ValueError(f"Unknown stress scenario: {scenario}")


def _predict_probabilities(model: Any, frame: pd.DataFrame) -> np.ndarray:
    probability = np.asarray(model.predict_proba(frame), dtype=float)
    if probability.shape != (len(frame), 3):
        raise ValueError("Stress model must return one three-class WDL row per probe")
    if not np.isfinite(probability).all() or np.any(probability < 0):
        raise ValueError("Stress model returned invalid probabilities")
    return probability


def _halftime_support_violation(frame: pd.DataFrame, evidence: np.ndarray) -> float:
    probability = np.asarray(evidence, dtype=float)
    if probability.ndim != 3 or probability.shape[0] != len(frame):
        raise ValueError("half-time checker must return one joint score-support grid per row")
    if min(probability.shape[1:]) < 1 or not np.isfinite(probability).all():
        raise ValueError("half-time score-support evidence must be finite and nonempty")
    if np.any(probability < 0):
        raise ValueError("half-time score-support probabilities must be nonnegative")
    total = probability.sum(axis=(1, 2))
    if float(np.max(np.abs(total - 1.0))) > _INVARIANT_TOLERANCE:
        raise ValueError("half-time score-support rows must each sum to one")
    home = pd.to_numeric(frame["half_time_home_goals"], errors="coerce").to_numpy(float)
    away = pd.to_numeric(frame["half_time_away_goals"], errors="coerce").to_numpy(float)
    finite = np.isfinite(home).all() and np.isfinite(away).all()
    nonnegative = bool(np.all(home >= 0) and np.all(away >= 0))
    integer = np.allclose(home, np.round(home)) and np.allclose(away, np.round(away))
    supported = bool(np.all(home < probability.shape[1]) and np.all(away < probability.shape[2]))
    if not (finite and nonnegative and integer and supported):
        raise ValueError("half-time goals must be supported nonnegative integer states")
    home_index = np.arange(probability.shape[1])[None, :, None]
    away_index = np.arange(probability.shape[2])[None, None, :]
    invalid = (home_index < home[:, None, None]) | (away_index < away[:, None, None])
    return float(np.max(np.sum(probability * invalid, axis=(1, 2))))


def _team_columns(model: Any) -> tuple[str, str]:
    home = getattr(model, "home_team_column_", None)
    away = getattr(model, "away_team_column_", None)
    if not home or not away:
        raise ValueError("Fitted model does not expose frozen home/away team columns")
    return str(home), str(away)


def _swap_probe(frame: pd.DataFrame, model: Any) -> pd.DataFrame:
    swapped = frame.copy(deep=True)
    home, away = _team_columns(model)
    swapped[[home, away]] = frame[[away, home]].to_numpy()
    for column in getattr(model, "odd_columns_", ()):
        swapped[column] = -pd.to_numeric(frame[column], errors="raise")
    for home_column, away_column in getattr(model, "paired_columns_", ()):
        swapped[[home_column, away_column]] = frame[[away_column, home_column]].to_numpy()
    host = getattr(model, "host_side_column_", None)
    if host and host in frame:
        swapped[host] = frame[host].map(
            {"home": "away", "away": "home", "neutral": "neutral"}
        )
        if swapped[host].isna().any():
            raise ValueError("Host-side stress encountered an unknown host label")
    return swapped


def _training_teams(model: Any, frame: pd.DataFrame) -> set[str]:
    home, away = _team_columns(model)
    return set(frame[home].astype(str)) | set(frame[away].astype(str))


def _rarest_team(model: Any, frame: pd.DataFrame) -> str:
    home, away = _team_columns(model)
    appearances = pd.concat([frame[home], frame[away]], ignore_index=True).astype(str)
    counts = appearances.value_counts(sort=False)
    return next(team for team in appearances if counts[team] == int(counts.min()))


def _matching_model_features(model: Any, fragments: Sequence[str]) -> list[str]:
    columns = [*getattr(model, "odd_columns_", ()), *getattr(model, "even_columns_", ())]
    return [name for name in columns if any(part in name.casefold() for part in fragments)]


def _personnel_pairs(model: Any) -> tuple[tuple[str, str], ...]:
    return tuple(
        (str(home), str(away))
        for home, away in getattr(model, "paired_columns_", ())
        if any(part in f"{home} {away}".casefold() for part in _PERSONNEL_FRAGMENTS)
    )


def _personnel_absolute_max(frame: pd.DataFrame, model: Any) -> float:
    columns = [column for pair in _personnel_pairs(model) for column in pair]
    if not columns:
        return 0.0
    values = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Personnel stress features must be finite")
    return float(np.max(np.abs(values)))


def _extreme_value(series: pd.Series) -> float:
    """Return a severe supported shift five full training spans above the maximum.

    The separate boundary-refusal audit deliberately probes ten spans.  Keeping
    this invariant scenario at five spans lets the declared finite score support
    test normalization and symmetry instead of conflating them with safe refusal.
    """
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError("Cannot stress an entirely missing history feature")
    span = max(float(np.ptp(values)), float(np.std(values)), 1.0)
    return float(np.max(values) + 5.0 * span)


def _neutralize_host(frame: pd.DataFrame, model: Any) -> None:
    host = getattr(model, "host_side_column_", None)
    host_features = _matching_model_features(model, ("host",))
    if not host and not host_features:
        raise ValueError("host_to_neutral stress requires a fitted host feature")
    if host:
        frame[host] = "neutral"
    for column in host_features:
        frame[column] = 0.0


def _drop_tournament_blocks(
    frame: pd.DataFrame, model: Any, seed: int,
) -> tuple[pd.DataFrame, tuple[int, ...]]:
    year = _year_column(model, frame)
    years = pd.to_numeric(frame[year], errors="raise").astype(int)
    target = max(1, int(np.ceil(0.10 * len(frame))))
    removed = 0
    dropped: list[int] = []
    for candidate in np.random.default_rng(seed).permutation(np.sort(years.unique())):
        dropped.append(int(candidate))
        removed += int(years.eq(candidate).sum())
        if removed >= target:
            break
    reduced = frame.loc[~years.isin(dropped)].copy()
    if reduced.empty:
        raise ValueError("Tournament-block deletion removed all training data")
    return reduced, tuple(sorted(dropped))


def _year_column(model: Any, frame: pd.DataFrame) -> str:
    candidate = getattr(model, "year_column_", None) or getattr(model.config, "year_column", None)
    if not candidate or candidate not in frame:
        raise ValueError("Stress protocol requires a fitted tournament year column")
    return str(candidate)


def _apply_2026_format(frame: pd.DataFrame, model: Any) -> None:
    replacements: Mapping[str, Any] = {"tournament_teams": 48.0, "tournament_matches": 104.0,
                                       "matches_per_team": 5.0,
                                       "format_regime": "expanded_48_team"}
    required = {"tournament_teams", "tournament_matches"}
    model_features = {*getattr(model, "odd_columns_", ()), *getattr(model, "even_columns_", ()),
                      *getattr(model, "context_columns_", ())}
    if not required.issubset(frame.columns) or not required.issubset(model_features):
        raise ValueError("2026 format stress requires modeled 48-team and 104-match fields")
    for column, value in replacements.items():
        if column in frame:
            frame[column] = value
    frame[_year_column(model, frame)] = 2026


__all__ = ["run_measured_stress_tests"]
