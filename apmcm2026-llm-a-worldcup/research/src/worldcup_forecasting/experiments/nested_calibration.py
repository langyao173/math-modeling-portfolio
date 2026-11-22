"""Outer-fold-safe calibration and coherent score-grid reconciliation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from worldcup_forecasting.experiments.calibration_selection import (
    CalibrationSelectionConfig,
    ChronologicalCalibrationSelector,
)
from worldcup_forecasting.experiments.evidence import reconstruct_joint_batch


@dataclass(frozen=True)
class NestedCalibrationResult:
    """Calibrated outer OOF evidence and the fitted inner-only states."""

    oof_ledger: pd.DataFrame
    joint_grid: pd.DataFrame
    selection_table: pd.DataFrame
    decision_training_ledger: pd.DataFrame
    states: dict[str, dict[str, Any]]
    metadata: dict[str, Any]


def apply_nested_calibration(
    selected_inner_oof_ledger: pd.DataFrame,
    selected_inner_joint_grid: pd.DataFrame,
    outer_oof_ledger: pd.DataFrame,
    outer_joint_grid: pd.DataFrame,
    *,
    config: CalibrationSelectionConfig | None = None,
) -> NestedCalibrationResult:
    """Fit each calibrator on its outer training window's selected inner OOF only."""
    inner = selected_inner_oof_ledger.copy()
    outer = outer_oof_ledger.copy()
    _validate_ledgers(inner, outer)
    parent_column = (
        "selection_outer_fold_id" if "selection_outer_fold_id" in inner else "outer_fold_id"
    )
    calibrated_rows: list[pd.DataFrame] = []
    grid_rows: list[pd.DataFrame] = []
    selection_rows: list[dict[str, Any]] = []
    decision_parts: list[pd.DataFrame] = []
    states: dict[str, dict[str, Any]] = {}
    for outer_fold_id, outer_block in outer.groupby("outer_fold_id", sort=True):
        outer_block = outer_block.sort_values("match_id").copy()
        inner_block = inner.loc[inner[parent_column].eq(outer_fold_id)].sort_values(
            ["prediction_year", "match_id"]
        )
        if inner_block.empty:
            raise ValueError(f"No selected inner OOF rows exist for outer fold {outer_fold_id!r}")
        outer_year = _single_year(outer_block, "outer")
        inner_max = int(inner_block["prediction_year"].max())
        if inner_max >= outer_year:
            raise ValueError("Every selected inner prediction must be strictly earlier than outer")
        inner_joint = reconstruct_joint_batch(
            selected_inner_joint_grid,
            inner_block[["outer_fold_id", "match_id"]],
        )
        observed_inner = inner_block[
            ["observed_home_goals", "observed_away_goals"]
        ].to_numpy(int)
        selector = ChronologicalCalibrationSelector(config).fit_oof(
            inner_joint,
            observed_inner,
            provenance=inner_block[["is_oof", "prediction_year", "train_end_year"]],
        )
        outer_joint = reconstruct_joint_batch(
            outer_joint_grid,
            outer_block[["outer_fold_id", "match_id"]],
        )
        calibrated_joint = selector.transform(outer_joint)
        calibrated_outcome = _outcome_margins(calibrated_joint)
        outer_block.loc[:, ["home_win", "draw", "away_win"]] = calibrated_outcome
        if "model_id" in outer_block:
            chosen_method = str(selector.chosen_method_)
            outer_block["model_id"] = outer_block["model_id"].astype(str).map(
                lambda value, method=chosen_method: f"{value}__calibrated_{method}"
            )
        calibrated_rows.append(outer_block)
        grid_rows.append(
            _long_joint_grid(
                outer_block[["outer_fold_id", "match_id"]], calibrated_joint
            )
        )
        report = selector.selection_report_
        selection_rows.append(
            {
                "outer_fold_id": str(outer_fold_id),
                "outer_prediction_year": outer_year,
                "inner_min_prediction_year": int(inner_block["prediction_year"].min()),
                "inner_max_prediction_year": inner_max,
                "inner_oof_rows": int(len(inner_block)),
                "inner_edition_count": int(inner_block["prediction_year"].nunique()),
                "inner_oof_only": bool(inner_block["is_oof"].astype(bool).all()),
                "chosen_method": str(selector.chosen_method_),
                "comparison_years": json.dumps(report["comparison_years"]),
                "fallback_reason": report["fallback_reason"],
                "selection_report_json": json.dumps(report, sort_keys=True),
            }
        )
        decision_parts.append(
            _decision_training_rows(
                selector,
                inner_block.reset_index(drop=True),
                selection_outer_fold_id=str(outer_fold_id),
            )
        )
        states[str(outer_fold_id)] = selector.state_dict()
    calibrated = pd.concat(calibrated_rows, ignore_index=True).sort_values(
        ["prediction_year", "match_id"]
    ).reset_index(drop=True)
    grids = pd.concat(grid_rows, ignore_index=True).sort_values(
        ["outer_fold_id", "match_id", "home_goals", "away_goals"]
    ).reset_index(drop=True)
    selection = pd.DataFrame(selection_rows).sort_values("outer_prediction_year").reset_index(
        drop=True
    )
    decision_training = pd.concat(decision_parts, ignore_index=True).sort_values(
        ["selection_outer_fold_id", "prediction_year", "match_id"]
    ).reset_index(drop=True)
    metadata = {
        "outer_fold_count": int(calibrated["outer_fold_id"].nunique()),
        "maximum_outer_prediction_year": int(calibrated["prediction_year"].max()),
        "maximum_inner_prediction_year": int(inner["prediction_year"].max()),
        "calibration_fit_scope": "selected_inner_oof_only_within_each_outer_training_window",
        "coherent_reconciliation": "minimum_kl_outcome_partition_projection",
        "audit_year_used": False,
    }
    return NestedCalibrationResult(
        calibrated, grids, selection, decision_training, states, metadata
    )


def _validate_ledgers(inner: pd.DataFrame, outer: pd.DataFrame) -> None:
    required_inner = {
        "outer_fold_id",
        "match_id",
        "prediction_year",
        "train_end_year",
        "is_oof",
        "observed_home_goals",
        "observed_away_goals",
    }
    required_outer = {
        "outer_fold_id",
        "match_id",
        "prediction_year",
        "train_end_year",
        "is_oof",
        "home_win",
        "draw",
        "away_win",
    }
    for label, frame, required in (
        ("inner", inner, required_inner),
        ("outer", outer, required_outer),
    ):
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError(f"{label} calibration ledger must be a nonempty DataFrame")
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{label} calibration ledger is missing columns: {missing}")
        if not frame["is_oof"].astype(bool).all():
            raise ValueError(f"{label} calibration ledger must contain OOF rows only")
        train = pd.to_numeric(frame["train_end_year"], errors="coerce")
        predict = pd.to_numeric(frame["prediction_year"], errors="coerce")
        if train.isna().any() or predict.isna().any() or not train.lt(predict).all():
            raise ValueError(f"{label} calibration chronology is invalid")
    if int(outer["prediction_year"].max()) > 2010:
        raise ValueError("2014/audit predictions are forbidden during nested calibration selection")


def _single_year(frame: pd.DataFrame, label: str) -> int:
    years = pd.to_numeric(frame["prediction_year"], errors="raise").unique()
    if len(years) != 1:
        raise ValueError(f"Each {label} fold must contain exactly one prediction year")
    return int(years[0])


def _outcome_margins(joint: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            [np.tril(matrix, -1).sum(), np.trace(matrix), np.triu(matrix, 1).sum()]
            for matrix in joint
        ],
        dtype=float,
    )


def _long_joint_grid(order: pd.DataFrame, joint: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = order.itertuples(index=False, name=None)
    for matrix, (outer_fold_id, match_id) in zip(joint, keys, strict=True):
        for home in range(matrix.shape[0]):
            for away in range(matrix.shape[1]):
                rows.append(
                    {
                        "outer_fold_id": outer_fold_id,
                        "match_id": match_id,
                        "home_goals": home,
                        "away_goals": away,
                        "probability": float(matrix[home, away]),
                    }
                )
    return pd.DataFrame(rows)


def _decision_training_rows(
    selector: ChronologicalCalibrationSelector,
    inner_block: pd.DataFrame,
    *,
    selection_outer_fold_id: str,
) -> pd.DataFrame:
    chosen = selector.crossfit_predictions_.loc[
        selector.crossfit_predictions_["method"].eq(selector.chosen_method_)
    ].copy()
    if chosen.empty:
        raise RuntimeError("Chosen calibration method has no cross-fitted decision rows")
    positions = chosen["row_position"].to_numpy(int)
    source = inner_block.iloc[positions].reset_index(drop=True)
    train_max = pd.to_numeric(chosen["train_max_year"], errors="coerce").to_numpy(float)
    fallback = pd.to_numeric(source["train_end_year"], errors="raise").to_numpy(float)
    resolved_train_end = np.where(np.isfinite(train_max), train_max, fallback)
    if (
        not np.isfinite(resolved_train_end).all()
        or not np.equal(resolved_train_end, np.floor(resolved_train_end)).all()
    ):
        raise RuntimeError("Decision-training fit years must be finite integers")
    resolved_train_end = resolved_train_end.astype(int)
    result = pd.DataFrame(
        {
            "selection_outer_fold_id": selection_outer_fold_id,
            "match_id": source["match_id"].to_numpy(),
            "prediction_year": chosen["prediction_year"].to_numpy(int),
            "train_end_year": resolved_train_end,
            "is_oof": True,
            "calibration_method": str(selector.chosen_method_),
            "home_win": chosen["home_win"].to_numpy(float),
            "draw": chosen["draw"].to_numpy(float),
            "away_win": chosen["away_win"].to_numpy(float),
            "target_label": chosen["target_label"].to_numpy(int),
        }
    )
    if not (result["train_end_year"] < result["prediction_year"]).all():
        raise RuntimeError("Decision training rows are not strictly chronological OOF")
    return result


__all__ = ["NestedCalibrationResult", "apply_nested_calibration"]
