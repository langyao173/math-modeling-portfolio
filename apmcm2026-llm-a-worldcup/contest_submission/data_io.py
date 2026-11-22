"""CSV input and labeled/query-row separation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from features import STAGE_ORDER, clean_team_name, normalize_stage, score_outcome
except ImportError:
    from .features import STAGE_ORDER, clean_team_name, normalize_stage, score_outcome


def read_csv_flexible(path: str | Path) -> pd.DataFrame:
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error or ValueError(f"无法读取 CSV：{path}")


def derive_match_fields(
    raw: pd.DataFrame,
    entity_policy: str = "conservative",
) -> pd.DataFrame:
    required = [
        "Year",
        "Home Team Name",
        "Away Team Name",
        "Home Team Goals",
        "Away Team Goals",
    ]
    missing = [name for name in required if name not in raw.columns]
    if missing:
        raise ValueError(f"WorldCupMatches.csv 缺少必要列：{', '.join(missing)}")
    data = raw.copy()
    numeric_columns = [
        "Year",
        "Home Team Goals",
        "Away Team Goals",
        "Half-time Home Goals",
        "Half-time Away Goals",
        "Attendance",
    ]
    for column in numeric_columns:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data["Home Team Name"] = data["Home Team Name"].map(
        lambda value: clean_team_name(value, policy=entity_policy)
    )
    data["Away Team Name"] = data["Away Team Name"].map(
        lambda value: clean_team_name(value, policy=entity_policy)
    )
    data["TotalGoals"] = data["Home Team Goals"] + data["Away Team Goals"]
    if {"Half-time Home Goals", "Half-time Away Goals"}.issubset(data.columns):
        data["HalfTimeTotal"] = (
            data["Half-time Home Goals"] + data["Half-time Away Goals"]
        )
    else:
        data["HalfTimeTotal"] = np.nan
    if "Stage" not in data:
        data["Stage"] = "Group Stage"
    data["StageNormalized"] = data["Stage"].map(normalize_stage)
    data["StageCode"] = data["StageNormalized"].map(STAGE_ORDER).astype(float)
    data["ScoreOutcome"] = [
        score_outcome(home, away)
        for home, away in zip(data["Home Team Goals"], data["Away Team Goals"])
    ]
    keys = [name for name in ("Year", "Datetime", "MatchID") if name in data]
    return data.sort_values(keys, na_position="last", kind="mergesort").reset_index(drop=True)


def load_competition_data(
    training_dir: Path,
    entity_policy: str = "conservative",
):
    cups = read_csv_flexible(training_dir / "WorldCups_preview.csv")
    raw_matches = read_csv_flexible(training_dir / "WorldCupMatches.csv")
    players = read_csv_flexible(training_dir / "WorldCupPlayers.csv")
    required_targets = ["Home Team Goals", "Away Team Goals"]
    missing = [name for name in required_targets if name not in raw_matches]
    if missing:
        raise ValueError(f"WorldCupMatches.csv 缺少必要列：{', '.join(missing)}")

    numeric_targets = raw_matches[required_targets].apply(pd.to_numeric, errors="coerce")
    invalid_text = raw_matches[required_targets].notna() & numeric_targets.isna()
    if invalid_text.any().any():
        column = invalid_text.any()[invalid_text.any()].index[0]
        raise ValueError(f"目标列“{column}”含非数值内容。")

    labeled = numeric_targets.notna().all(axis=1)
    training_raw = raw_matches.loc[labeled].drop_duplicates().copy()
    embedded_query = raw_matches.loc[~labeled].copy()
    embedded_query["__query_order__"] = np.arange(len(embedded_query))
    matches = derive_match_fields(training_raw, entity_policy=entity_policy)
    return (
        cups,
        matches,
        players.drop_duplicates().reset_index(drop=True),
        embedded_query.reset_index(drop=True),
    )


def load_prediction_data(
    training_dir: Path,
    entity_policy: str = "conservative",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load only the match table needed by the formal prediction path."""
    raw_matches = read_csv_flexible(training_dir / "WorldCupMatches.csv")
    required = ["Home Team Goals", "Away Team Goals"]
    missing = [name for name in required if name not in raw_matches]
    if missing:
        raise ValueError(f"WorldCupMatches.csv 缺少必要列：{', '.join(missing)}")
    numeric = raw_matches[required].apply(pd.to_numeric, errors="coerce")
    invalid_text = raw_matches[required].notna() & numeric.isna()
    if invalid_text.any().any():
        column = invalid_text.any()[invalid_text.any()].index[0]
        raise ValueError(f"目标列“{column}”含非数值内容。")
    labeled = numeric.notna().all(axis=1)
    training_raw = raw_matches.loc[labeled].drop_duplicates().copy()
    embedded_query = raw_matches.loc[~labeled].copy()
    embedded_query["__query_order__"] = np.arange(len(embedded_query))
    return (
        derive_match_fields(training_raw, entity_policy=entity_policy),
        embedded_query.reset_index(drop=True),
    )
