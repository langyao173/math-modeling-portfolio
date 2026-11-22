"""Strictly pre-match dynamic Elo and scoring-form features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._schema import base_match_frame, match_times, numeric_series, team_series


@dataclass
class _TeamState:
    elo: float
    attack: float
    defense: float
    last_time: pd.Timestamp | None = None
    matches: int = 0


@dataclass(frozen=True)
class _Snapshot:
    elo: float
    attack: float
    defense: float
    cutoff: pd.Timestamp | None
    matches: int


def build_dynamic_features(
    matches: pd.DataFrame,
    *,
    initial_elo: float = 1500.0,
    k_factor: float = 24.0,
    state_baseline: float = 1.0,
    state_update_weight: float = 0.25,
    state_half_life_days: float = 730.0,
    state_update_before_year: int | None = None,
) -> pd.DataFrame:
    """Build pre-match Elo and exponentially decayed attack/defence states.

    Rows are evaluated in timestamp batches.  All matches with the same timestamp
    see the state that existed *before* that timestamp, independent of input row
    order.  Only after the complete batch is predicted are outcomes incorporated.
    This predict-then-update rule is the central leakage guard.  When
    ``state_update_before_year`` is set, rows in that year and later receive a
    snapshot of the previously frozen state but can never update it.  That
    stricter mode supports a batch-blind external audit.
    """

    _validate_parameters(
        initial_elo,
        k_factor,
        state_baseline,
        state_update_weight,
        state_half_life_days,
    )
    base = base_match_frame(matches, include_targets=True)
    times = match_times(matches)
    home = team_series(matches, "home_team").to_numpy()
    away = team_series(matches, "away_team").to_numpy()
    home_goals = numeric_series(matches, "home_goals").to_numpy()
    away_goals = numeric_series(matches, "away_goals").to_numpy()
    years = pd.to_numeric(base["Year"], errors="raise").astype(int).to_numpy()

    n_rows = len(matches)
    values = {
        column: np.zeros(n_rows, dtype=float)
        for column in (
            "home_pre_match_elo",
            "away_pre_match_elo",
            "home_attack_state",
            "away_attack_state",
            "home_defense_state",
            "away_defense_state",
            "home_dynamic_history_matches",
            "away_dynamic_history_matches",
        )
    }
    cutoff_values: dict[str, list[pd.Timestamp | pd.NaT]] = {
        "home_dynamic_cutoff_time": [pd.NaT] * n_rows,
        "away_dynamic_cutoff_time": [pd.NaT] * n_rows,
    }
    states: dict[str, _TeamState] = {}

    order = pd.DataFrame(
        {
            "position": np.arange(n_rows),
            "time": times.to_numpy(),
            "match_id": base["MatchID"].astype(str).to_numpy(),
        }
    ).sort_values(["time", "match_id", "position"], kind="mergesort")
    _reject_same_team_timestamp(times, home, away)

    for current_time, batch in order.groupby("time", sort=True):
        timestamp = pd.Timestamp(current_time)
        positions = batch["position"].to_numpy(dtype=int)
        snapshots: dict[tuple[int, str], _Snapshot] = {}

        # Snapshot every side before any result at this timestamp is observed.
        for position in positions:
            for side, team in (("home", home[position]), ("away", away[position])):
                snapshot = _snapshot(
                    states.get(str(team)),
                    timestamp,
                    initial_elo=initial_elo,
                    baseline=state_baseline,
                    half_life_days=state_half_life_days,
                )
                snapshots[(position, side)] = snapshot
                values[f"{side}_pre_match_elo"][position] = snapshot.elo
                values[f"{side}_attack_state"][position] = snapshot.attack
                values[f"{side}_defense_state"][position] = snapshot.defense
                values[f"{side}_dynamic_history_matches"][position] = snapshot.matches
                cutoff_values[f"{side}_dynamic_cutoff_time"][position] = (
                    snapshot.cutoff if snapshot.cutoff is not None else pd.NaT
                )

        update_positions = [
            int(position)
            for position in positions
            if state_update_before_year is None
            or years[int(position)] < int(state_update_before_year)
        ]
        if not update_positions:
            continue

        elo_deltas: dict[str, float] = {}
        observations: dict[str, list[tuple[float, float]]] = {}
        for position in update_positions:
            home_team = str(home[position])
            away_team = str(away[position])
            home_snapshot = snapshots[(position, "home")]
            away_snapshot = snapshots[(position, "away")]
            expected_home = _elo_expectation(home_snapshot.elo, away_snapshot.elo)
            actual_home = float(
                1.0
                if home_goals[position] > away_goals[position]
                else 0.0
                if home_goals[position] < away_goals[position]
                else 0.5
            )
            delta = k_factor * (actual_home - expected_home)
            elo_deltas[home_team] = elo_deltas.get(home_team, 0.0) + delta
            elo_deltas[away_team] = elo_deltas.get(away_team, 0.0) - delta
            observations.setdefault(home_team, []).append(
                (float(home_goals[position]), float(away_goals[position]))
            )
            observations.setdefault(away_team, []).append(
                (float(away_goals[position]), float(home_goals[position]))
            )

        # A team normally appears once per timestamp.  Averaging repeated synthetic
        # observations keeps tied batches deterministic and independent of row order.
        for team, team_observations in observations.items():
            prior = _snapshot(
                states.get(team),
                timestamp,
                initial_elo=initial_elo,
                baseline=state_baseline,
                half_life_days=state_half_life_days,
            )
            goals_for = float(np.mean([item[0] for item in team_observations]))
            goals_against = float(np.mean([item[1] for item in team_observations]))
            states[team] = _TeamState(
                elo=prior.elo + elo_deltas.get(team, 0.0),
                attack=(1.0 - state_update_weight) * prior.attack + state_update_weight * goals_for,
                defense=(1.0 - state_update_weight) * prior.defense
                + state_update_weight * goals_against,
                last_time=timestamp,
                matches=prior.matches + len(team_observations),
            )

    result = base.copy()
    for column, column_values in values.items():
        result[column] = column_values
    for column, column_values in cutoff_values.items():
        result[column] = pd.to_datetime(column_values)
    result["pre_match_elo_diff"] = result["home_pre_match_elo"] - result["away_pre_match_elo"]
    result["attack_state_diff"] = result["home_attack_state"] - result["away_attack_state"]
    result["defense_state_diff"] = result["home_defense_state"] - result["away_defense_state"]
    result["home_dynamic_history_missing"] = (
        result["home_dynamic_history_matches"].eq(0).astype("int8")
    )
    result["away_dynamic_history_missing"] = (
        result["away_dynamic_history_matches"].eq(0).astype("int8")
    )
    return result.reset_index(drop=True)


def _snapshot(
    state: _TeamState | None,
    current_time: pd.Timestamp,
    *,
    initial_elo: float,
    baseline: float,
    half_life_days: float,
) -> _Snapshot:
    if state is None:
        return _Snapshot(initial_elo, baseline, baseline, None, 0)
    if state.last_time is None or state.last_time >= current_time:
        # Equality is possible only while evaluating a tied timestamp batch; no
        # already-updated state is intentionally exposed to that same batch.
        elapsed_days = 0.0
    else:
        elapsed_days = (current_time - state.last_time).total_seconds() / 86_400.0
    decay = float(0.5 ** (elapsed_days / half_life_days))
    return _Snapshot(
        elo=state.elo,
        attack=baseline + decay * (state.attack - baseline),
        defense=baseline + decay * (state.defense - baseline),
        cutoff=state.last_time,
        matches=state.matches,
    )


def _elo_expectation(home_elo: float, away_elo: float) -> float:
    difference = float(np.clip(away_elo - home_elo, -1600.0, 1600.0))
    return 1.0 / (1.0 + 10.0 ** (difference / 400.0))


def _reject_same_team_timestamp(
    times: pd.Series,
    home: np.ndarray,
    away: np.ndarray,
) -> None:
    appearances = pd.concat(
        [
            pd.DataFrame({"time": times.to_numpy(), "team": home}),
            pd.DataFrame({"time": times.to_numpy(), "team": away}),
        ],
        ignore_index=True,
    )
    duplicate = appearances.duplicated(["time", "team"], keep=False)
    if duplicate.any():
        examples = appearances.loc[duplicate, ["time", "team"]].drop_duplicates().head(5)
        raise ValueError(
            "A team cannot play multiple matches at the same timestamp; offending "
            f"(time, team) pairs include {examples.to_dict('records')}."
        )


def _validate_parameters(
    initial_elo: float,
    k_factor: float,
    state_baseline: float,
    state_update_weight: float,
    state_half_life_days: float,
) -> None:
    parameters = (initial_elo, k_factor, state_baseline, state_half_life_days)
    if not all(np.isfinite(value) for value in parameters):
        raise ValueError("Dynamic-feature parameters must be finite.")
    if k_factor <= 0 or state_half_life_days <= 0 or state_baseline < 0:
        raise ValueError("k_factor/half-life must be positive and baseline nonnegative.")
    if not 0.0 < state_update_weight <= 1.0:
        raise ValueError("state_update_weight must lie in (0, 1].")
