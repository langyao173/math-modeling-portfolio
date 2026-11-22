"""Formal prediction interfaces for match outcome and full-time goals."""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

try:
    from contracts.input_schema import (
        ATTENDANCE_COLUMNS,
        HT_AWAY_COLUMNS,
        HT_HOME_COLUMNS,
        ID_COLUMNS,
        discover_query_frame,
        first_existing,
        normalize_query_frame,
    )
    from contracts.output_schema import build_minimal_submission, write_atomic_submission
    from data_io import read_csv_flexible
    from features import (
        CLASS_CHINESE_BY_INT,
        TeamStateEngine,
    )
    from models import MulticlassLogisticModel, P3_FINAL_CONFIG
    from problem2_analysis import (
        P2_FEATURES,
        empirical_interval_radius,
        make_problem2_estimator,
        prepare_problem2_model_frame,
    )
except ImportError:
    from .contracts.input_schema import (
        ATTENDANCE_COLUMNS,
        HT_AWAY_COLUMNS,
        HT_HOME_COLUMNS,
        ID_COLUMNS,
        discover_query_frame,
        first_existing,
        normalize_query_frame,
    )
    from .contracts.output_schema import build_minimal_submission, write_atomic_submission
    from .data_io import read_csv_flexible
    from .features import (
        CLASS_CHINESE_BY_INT,
        TeamStateEngine,
    )
    from .models import MulticlassLogisticModel, P3_FINAL_CONFIG
    from .problem2_analysis import (
        P2_FEATURES,
        empirical_interval_radius,
        make_problem2_estimator,
        prepare_problem2_model_frame,
    )


def _safe_number(value, default=np.nan) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if pd.notna(parsed) else float(default)


class MatchResultPredictor:
    """Official four-feature multiclass logistic-regression predictor."""

    def __init__(
        self,
        matches: pd.DataFrame,
        cutoff_year: int,
        config: dict | None = None,
    ):
        self.cutoff_year = int(cutoff_year)
        self.config = dict(P3_FINAL_CONFIG if config is None else config)
        self.entity_policy = str(self.config.get("entity_policy", "conservative"))
        self.history = matches[matches["Year"] < self.cutoff_year].copy()
        if self.history.empty:
            raise ValueError(f"查询年份 {self.cutoff_year} 之前没有可用历史比赛。")
        self.training_max_year = int(self.history["Year"].max())
        self.known_teams = set(self.history["Home Team Name"]) | set(
            self.history["Away Team Name"]
        )
        self.engine = TeamStateEngine(
            form_decay=float(self.config.get("form_decay", 0.8)),
            entity_policy=self.entity_policy,
        )
        self.engine.fit(self.history)
        self.training_features = TeamStateEngine(
            form_decay=float(self.config.get("form_decay", 0.8)),
            entity_policy=self.entity_policy,
            era_tournaments=5,
        ).transform_history_frozen(self.history)
        self.model = MulticlassLogisticModel.fit(
            self.training_features,
            self.config,
        )
        self.feature_names = list(self.config["feature_names"])

    def _matrix_and_warnings(
        self,
        query: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[str]]:
        rows = []
        warning_messages: list[str] = []
        for _, source in query.iterrows():
            home = source["__HomeTeam__"]
            away = source["__AwayTeam__"]
            unknown = [name for name in (home, away) if name not in self.known_teams]
            if unknown:
                message = (
                    "未知球队：" + "、".join(dict.fromkeys(unknown))
                    + "。程序使用中性先验；请检查拼写或补充球队别名。"
                )
                warnings.warn(message, UserWarning, stacklevel=3)
            else:
                message = ""
            warning_messages.append(message)
            values = self.engine.feature_row(
                home,
                away,
                source["__Stage__"],
                source["__Year__"],
            )
            for name in self.feature_names:
                if name in source.index and pd.notna(source[name]):
                    values[name] = _safe_number(source[name], values.get(name, 0.0))
            rows.append({name: values.get(name, 0.0) for name in self.feature_names})
        matrix = pd.DataFrame(rows, columns=self.feature_names, index=query.index)
        if not np.isfinite(matrix.to_numpy(float)).all():
            raise ValueError("构造出的赛前特征存在非有限值。")
        return matrix, warning_messages

    def predict_dataframe(
        self,
        frame: pd.DataFrame,
        source: str = "内存查询数据",
    ) -> pd.DataFrame:
        query = (
            frame
            if "__HomeTeam__" in frame.columns
            else normalize_query_frame(
                frame,
                source,
                entity_policy=self.entity_policy,
            )
        )
        matrix, entity_warnings = self._matrix_and_warnings(query)
        probabilities = self.model.predict_proba(matrix)
        prediction = probabilities.argmax(axis=1).astype(int)
        return pd.DataFrame(
            {
                "主队": query["__HomeTeam__"].to_numpy(),
                "客队": query["__AwayTeam__"].to_numpy(),
                "prediction": prediction,
                "预测结果": [
                    CLASS_CHINESE_BY_INT[int(value)] for value in prediction
                ],
                "主队胜概率": probabilities[:, 0],
                "平局概率": probabilities[:, 1],
                "客队胜概率": probabilities[:, 2],
                "概率最大类别": probabilities.argmax(axis=1),
                "硬分类与概率一致": prediction == probabilities.argmax(axis=1),
                "模型": self.config["model_name"],
                "球队实体策略": self.entity_policy,
                "球队实体警告": entity_warnings,
                "训练数据实际最大年份": self.training_max_year,
            },
            index=query.index,
        )


class MatchGoalPredictor:
    """Predict second-half increments and add the known half-time score."""

    def __init__(self, matches: pd.DataFrame, cutoff_year: int):
        history = matches[matches["Year"] < int(cutoff_year)].copy()
        if history.empty:
            raise ValueError("没有可用于单场进球模型训练的历史比赛。")
        self.training_max_year = int(history["Year"].max())
        self.model_frame = prepare_problem2_model_frame(history)
        self.model = make_problem2_estimator("poisson").fit(
            self.model_frame[P2_FEATURES],
            self.model_frame["SecondHalfGoals"],
        )
        (
            self.empirical_interval_radius,
            self.oof_residuals,
        ) = empirical_interval_radius(self.model_frame, kind="increment", coverage=0.8)
        self.engine = TeamStateEngine(
            form_decay=float(P3_FINAL_CONFIG["form_decay"]),
            entity_policy=str(P3_FINAL_CONFIG["entity_policy"]),
        ).fit(history)

    def predict_full_time_goals_after_halftime(
        self,
        frame: pd.DataFrame,
        source: str = "内存查询数据",
    ) -> pd.DataFrame:
        query = (
            frame
            if "__HomeTeam__" in frame.columns
            else normalize_query_frame(frame, source)
        )
        attendance_col = first_existing(query.columns, ATTENDANCE_COLUMNS)
        half_home_col = first_existing(query.columns, HT_HOME_COLUMNS)
        half_away_col = first_existing(query.columns, HT_AWAY_COLUMNS)
        rows = []
        half_totals = []
        for _, item in query.iterrows():
            values = self.engine.feature_row(
                item["__HomeTeam__"],
                item["__AwayTeam__"],
                item["__Stage__"],
                item["__Year__"],
            )
            half_home = (
                _safe_number(item[half_home_col], 0.0) if half_home_col else 0.0
            )
            half_away = (
                _safe_number(item[half_away_col], 0.0) if half_away_col else 0.0
            )
            attendance = (
                _safe_number(item[attendance_col], np.nan)
                if attendance_col
                else np.nan
            )
            half_total = half_home + half_away
            half_totals.append(half_total)
            rows.append(
                {
                    "HalfTimeTotal": half_total,
                    "AttendanceLog": (
                        np.log1p(attendance)
                        if np.isfinite(attendance) and attendance >= 0
                        else np.nan
                    ),
                    "HomeStrength": values["HomeHistAvgGF"],
                    "AwayStrength": values["AwayHistAvgGF"],
                    "StageNormalized": item["__Stage__"],
                }
            )
        matrix = pd.DataFrame(rows, columns=P2_FEATURES, index=query.index)
        increment = np.clip(self.model.predict(matrix), 0.0, None)
        point = np.asarray(half_totals) + increment
        lower = np.maximum(
            np.asarray(half_totals),
            point - self.empirical_interval_radius,
        )
        upper = point + self.empirical_interval_radius
        return pd.DataFrame(
            {
                "预测全场总进球": point,
                "80%区间下限": lower,
                "80%区间上限": upper,
                "最可能整数进球": np.rint(point).astype(int),
                "区间口径": "基于历届滚动折外绝对残差的80%经验预测区间",
                "状态协议": "整个查询批次冻结；查询内部不更新球队状态",
                "训练数据实际最大年份": self.training_max_year,
            },
            index=query.index,
        )

    def predict_dataframe(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.predict_full_time_goals_after_halftime(frame)


def default_2026_matchups() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": 1,
                "Year": 2026,
                "Stage": "Group Stage",
                "Home Team Name": "Brazil",
                "Away Team Name": "Argentina",
                "Half-time Home Goals": 0,
                "Half-time Away Goals": 0,
                "Attendance": 70_000,
            },
            {
                "id": 2,
                "Year": 2026,
                "Stage": "Group Stage",
                "Home Team Name": "Germany",
                "Away Team Name": "France",
                "Half-time Home Goals": 1,
                "Half-time Away Goals": 0,
                "Attendance": 65_000,
            },
            {
                "id": 3,
                "Year": 2026,
                "Stage": "Round of 16",
                "Home Team Name": "Spain",
                "Away Team Name": "Netherlands",
                "Half-time Home Goals": 0,
                "Half-time Away Goals": 1,
                "Attendance": 60_000,
            },
            {
                "id": 4,
                "Year": 2026,
                "Stage": "Quarter-finals",
                "Home Team Name": "England",
                "Away Team Name": "Portugal",
                "Half-time Home Goals": 1,
                "Half-time Away Goals": 1,
                "Attendance": 72_000,
            },
        ]
    )


def _query_cutoff_year(query: pd.DataFrame) -> int:
    values = pd.to_numeric(query["__Year__"], errors="coerce").dropna()
    return int(values.min()) if not values.empty else 2026


def generate_evaluator_predictions(
    matches: pd.DataFrame,
    embedded_query: pd.DataFrame | None,
    testdata_dir: Path | None,
    output_root: Path,
    output_dirs: dict[str, Path] | None = None,
    model_config: dict | None = None,
    demo: bool = False,
    write_details: bool = False,
) -> dict:
    selected = dict(P3_FINAL_CONFIG if model_config is None else model_config)
    if demo:
        source_name = "显式 --demo 内置2026演示输入"
        query = normalize_query_frame(
            default_2026_matchups(),
            source_name,
            entity_policy=str(selected.get("entity_policy", "conservative")),
        )
    else:
        source_name, query = discover_query_frame(
            testdata_dir,
            embedded_query,
            read_csv_flexible,
            entity_policy=str(selected.get("entity_policy", "conservative")),
        )

    cutoff_year = _query_cutoff_year(query)
    predictor = MatchResultPredictor(
        matches,
        cutoff_year=cutoff_year,
        config=selected,
    )
    rich = predictor.predict_dataframe(query, source_name)
    minimal = build_minimal_submission(query, rich)
    cla_path = write_atomic_submission(minimal, Path(output_root) / "cla_pre.csv")

    rich_path = None
    goal_path = None
    goal_output = pd.DataFrame()
    if write_details:
        if not output_dirs:
            raise ValueError("写入详细结果时必须提供输出目录。")
        rich_path = output_dirs["tables"] / "3_4_待预测比赛统一概率.csv"
        rich_output = rich.copy()
        id_col = first_existing(query.columns, ID_COLUMNS)
        if id_col:
            rich_output.insert(0, "id", query[id_col].to_numpy())
        rich_output.to_csv(rich_path, index=False, encoding="utf-8-sig")

        goal_output = MatchGoalPredictor(
            matches,
            cutoff_year,
        ).predict_full_time_goals_after_halftime(query, source_name)
        if id_col:
            goal_output.insert(0, "id", query[id_col].to_numpy())
        goal_path = output_dirs["tables"] / "2_3_待预测比赛全场进球区间.csv"
        goal_output.to_csv(goal_path, index=False, encoding="utf-8-sig")
    return {
        "source": source_name,
        "query": query,
        "classification": rich,
        "minimal": minimal,
        "goal_prediction": goal_output,
        "cla_path": cla_path,
        "rich_path": rich_path,
        "goal_path": goal_path,
        "cutoff_year": cutoff_year,
        "training_max_year": predictor.training_max_year,
    }
