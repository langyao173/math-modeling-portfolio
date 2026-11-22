"""Prediction input contract.

Formal mode accepts exactly one source: ``query.csv`` or unlabeled rows in the
official match table. Demo rows are available only through ``--demo``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

try:
    from features import clean_team_name, normalize_stage
except ImportError:
    from ..features import clean_team_name, normalize_stage


HOME_COLUMNS = ["Home Team Name", "HomeTeam", "home_team", "home", "主队"]
AWAY_COLUMNS = ["Away Team Name", "AwayTeam", "away_team", "away", "客队"]
YEAR_COLUMNS = ["Year", "year", "比赛年份", "年份"]
STAGE_COLUMNS = ["Stage", "StageNormalized", "stage", "比赛阶段", "阶段"]
ATTENDANCE_COLUMNS = ["Attendance", "attendance", "观众数"]
HT_HOME_COLUMNS = [
    "Half-time Home Goals",
    "HalfTimeHomeGoals",
    "ht_home_goals",
    "半场主队进球",
]
HT_AWAY_COLUMNS = [
    "Half-time Away Goals",
    "HalfTimeAwayGoals",
    "ht_away_goals",
    "半场客队进球",
]
ID_COLUMNS = ["id", "ID", "sample_id", "SampleID", "match_id", "MatchID", "编号"]
TARGET_HOME_COLUMNS = ["Home Team Goals", "HomeGoals", "home_goals", "主队进球"]
TARGET_AWAY_COLUMNS = ["Away Team Goals", "AwayGoals", "away_goals", "客队进球"]


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    available = set(columns)
    return next((name for name in candidates if name in available), None)


def _validate_optional_target(frame: pd.DataFrame, candidates: list[str]) -> None:
    column = first_existing(frame.columns, candidates)
    if not column:
        return
    nonempty = frame[column].notna()
    invalid = nonempty & pd.to_numeric(frame[column], errors="coerce").isna()
    if invalid.any():
        rows = ", ".join(str(int(i) + 2) for i in frame.index[invalid][:5])
        raise ValueError(f"目标列“{column}”含非数值内容，出错行：{rows}。")


def normalize_query_frame(
    frame: pd.DataFrame,
    source: str,
    entity_policy: str = "conservative",
) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise ValueError(f"待预测文件“{source}”为空，无法生成预测。")
    data = frame.copy().reset_index(drop=True)
    home_col = first_existing(data.columns, HOME_COLUMNS)
    away_col = first_existing(data.columns, AWAY_COLUMNS)
    if not home_col or not away_col:
        raise ValueError(
            f"待预测文件“{source}”必须同时包含主队列和客队列；支持中英文列名。"
        )
    _validate_optional_target(data, TARGET_HOME_COLUMNS)
    _validate_optional_target(data, TARGET_AWAY_COLUMNS)

    id_col = first_existing(data.columns, ID_COLUMNS)
    if id_col and data[id_col].notna().any():
        if data[id_col].isna().any():
            raise ValueError(f"编号列“{id_col}”部分为空；请全部填写或全部留空。")
        if data[id_col].duplicated().any():
            duplicate = data.loc[data[id_col].duplicated(keep=False), id_col].iloc[0]
            raise ValueError(f"编号列“{id_col}”存在重复值“{duplicate}”。")
    elif id_col:
        data = data.drop(columns=[id_col])

    data["__OriginalHomeTeam__"] = data[home_col].astype("string").fillna("")
    data["__OriginalAwayTeam__"] = data[away_col].astype("string").fillna("")
    data["__HomeTeam__"] = data[home_col].map(
        lambda value: clean_team_name(value, policy=entity_policy)
    )
    data["__AwayTeam__"] = data[away_col].map(
        lambda value: clean_team_name(value, policy=entity_policy)
    )
    if (data["__HomeTeam__"] == "").any() or (data["__AwayTeam__"] == "").any():
        raise ValueError("主队或客队名称存在空值。")

    year_col = first_existing(data.columns, YEAR_COLUMNS)
    if year_col:
        parsed_year = pd.to_numeric(data[year_col], errors="coerce")
        invalid_year = data[year_col].notna() & parsed_year.isna()
        if invalid_year.any():
            raise ValueError(f"年份列“{year_col}”存在无法解析的值。")
        data["__Year__"] = parsed_year.fillna(2026).astype(int)
    else:
        data["__Year__"] = 2026
    stage_col = first_existing(data.columns, STAGE_COLUMNS)
    data["__Stage__"] = (
        data[stage_col].map(normalize_stage) if stage_col else "Group Stage"
    )
    data["__QueryOrder__"] = np.arange(len(data))
    return data


def discover_query_frame(
    testdata_dir: Path | None,
    embedded_query: pd.DataFrame | None,
    csv_reader: Callable[[Path], pd.DataFrame],
    entity_policy: str = "conservative",
) -> tuple[str, pd.DataFrame]:
    explicit: list[Path] = []
    if testdata_dir and testdata_dir.is_dir():
        explicit = [
            path
            for path in testdata_dir.iterdir()
            if path.is_file() and path.name.lower() == "query.csv"
        ]
    if len(explicit) > 1:
        raise ValueError("测试目录存在多个大小写不同的 query.csv，输入来源冲突。")
    has_embedded = embedded_query is not None and len(embedded_query) > 0
    if explicit and has_embedded:
        raise ValueError(
            "同时发现 query.csv 和 WorldCupMatches.csv 无标签行；请只保留一种查询来源。"
        )
    if explicit:
        source = explicit[0].name
        return source, normalize_query_frame(
            csv_reader(explicit[0]), source, entity_policy=entity_policy
        )
    if has_embedded:
        source = "WorldCupMatches.csv 中的无标签行"
        return source, normalize_query_frame(
            embedded_query.drop(columns=["__query_order__"], errors="ignore"),
            source,
            entity_policy=entity_policy,
        )
    raise ValueError(
        "未发现待预测数据：请在 testdata 放置 query.csv，或在 "
        "WorldCupMatches.csv 中提供无标签比赛行。若仅需演示，请运行 "
        "python main.py --demo。"
    )
