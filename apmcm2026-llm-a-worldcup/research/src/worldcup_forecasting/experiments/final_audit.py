"""One-time 2014 audit applied only after model/calibration/decision freezing."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

from worldcup_forecasting.models.decision import OOFCostSensitiveDecisionRule
from worldcup_forecasting.models.score_calibration import outcome_margins
from worldcup_forecasting.validation.joint_metrics import joint_forecast_metrics
from worldcup_forecasting.validation.metrics import classification_metrics

from .calibration_selection import (
    CalibrationSelectionConfig,
    ChronologicalCalibrationSelector,
)
from .evidence import reconstruct_joint_batch

_OUTCOMES = np.asarray(("home_win", "draw", "away_win"), dtype=object)
_TARGET_COLUMNS = {
    "target_outcome",
    "target_home_goals",
    "target_away_goals",
}
_AUDIT_PROTOCOL_VERSION = "frozen_2014_audit_v1"


@dataclass(frozen=True)
class FrozenAuditResult:
    """Predictions, score grids, metrics, fitted states, and audit provenance."""

    predictions: pd.DataFrame
    raw_joint_grid: pd.DataFrame
    calibrated_joint_grid: pd.DataFrame
    metrics: pd.DataFrame
    calibration_state: dict[str, Any]
    decision_state: dict[str, Any]
    metadata: dict[str, Any]


def apply_frozen_2014_audit(
    selected_inner_oof_ledger: pd.DataFrame,
    selected_inner_joint_grid: pd.DataFrame,
    fitted_predictor: Any,
    audit_frame: pd.DataFrame,
    *,
    identity: dict[str, str],
    calibration_config: CalibrationSelectionConfig | None = None,
    pit_seed: int = 2014,
) -> FrozenAuditResult:
    """Freeze all fitted states, then score the isolated 2014 audit exactly once."""
    resolved_calibration = calibration_config or CalibrationSelectionConfig()
    expected_protocol_hash = frozen_audit_protocol_hash(
        resolved_calibration, pit_seed=pit_seed
    )
    inner = _validate_inner(selected_inner_oof_ledger)
    audit = _validate_audit_structure(audit_frame)
    predictor_metadata = _validate_identity(
        fitted_predictor, identity, expected_protocol_hash
    )
    inner_lineage_hash = _validate_inner_lineage(
        inner, selected_inner_joint_grid, predictor_metadata, identity
    )
    selector, rule = _freeze_layers(
        inner,
        selected_inner_joint_grid,
        resolved_calibration,
    )
    calibration_state = selector.state_dict()
    decision_state = dict(rule.metadata_)
    layer_state_hash = _hash_payload(
        {
            "protocol_hash": expected_protocol_hash,
            "inner_lineage_hash": inner_lineage_hash,
            "calibration_state": calibration_state,
            "decision_state": decision_state,
        }
    )
    frozen_identity_hash = _hash_payload(
        {"frozen_identity": identity, "layer_state_hash": layer_state_hash}
    )

    # Targets are physically absent from the frame exposed to serving code.
    feature_frame = audit.drop(columns=list(_TARGET_COLUMNS), errors="ignore")
    raw_batch = fitted_predictor.predict_joint(feature_frame)
    raw_joint = np.asarray(raw_batch.probability, dtype=float)
    calibrated_joint = selector.transform(raw_joint)
    raw_outcome = outcome_margins(raw_joint)
    calibrated_outcome = outcome_margins(calibrated_joint)
    decision_label = rule.predict(calibrated_outcome)
    match_ids = audit["MatchID"].to_numpy()

    # Only this block opens outcome/score targets; no fitted object changes below.
    labels, goals = _audit_targets(audit)
    prediction = pd.DataFrame(
        {
            "match_id": match_ids,
            "prediction_year": 2014,
            "raw_home_win": raw_outcome[:, 0],
            "raw_draw": raw_outcome[:, 1],
            "raw_away_win": raw_outcome[:, 2],
            "home_win": calibrated_outcome[:, 0],
            "draw": calibrated_outcome[:, 1],
            "away_win": calibrated_outcome[:, 2],
            "argmax_label": _OUTCOMES[calibrated_outcome.argmax(axis=1)],
            "decision_label": decision_label,
            "target_outcome": labels,
        }
    )
    metrics = _audit_metrics(
        labels,
        goals,
        raw_outcome,
        calibrated_outcome,
        raw_joint,
        calibrated_joint,
        decision_label,
        pit_seed=pit_seed,
    )
    metadata: dict[str, Any] = {
        "evaluation_year": 2014,
        "evaluation_role": "audit_only_not_for_selection",
        "selection_maximum_year": int(inner["prediction_year"].max()),
        "audit_target_used_for_model_choice": False,
        "target_columns_exposed_to_predictor": False,
        "calibration_fit_scope": "development_selected_inner_oof_only",
        "decision_fit_scope": "development_calibrator_crossfit_oof_only",
        "identity": dict(identity),
        "predictor_metadata": predictor_metadata,
        "inner_lineage_hash": inner_lineage_hash,
        "layer_state_hash": layer_state_hash,
        "frozen_audit_identity_hash": frozen_identity_hash,
    }
    return FrozenAuditResult(
        predictions=prediction,
        raw_joint_grid=_long_grid(match_ids, raw_joint),
        calibrated_joint_grid=_long_grid(match_ids, calibrated_joint),
        metrics=metrics,
        calibration_state=calibration_state,
        decision_state=decision_state,
        metadata=metadata,
    )


def frozen_audit_protocol_hash(
    calibration_config: CalibrationSelectionConfig | None = None,
    *,
    pit_seed: int = 2014,
) -> str:
    """Hash every tunable calibration, decision, and metric control pre-audit."""
    config = calibration_config or CalibrationSelectionConfig()
    if config.maximum_development_year > 2010:
        raise ValueError("Frozen audit calibration cannot exceed the 2010 boundary")
    if isinstance(pit_seed, (bool, np.bool_)) or not isinstance(
        pit_seed, (int, np.integer)
    ):
        raise ValueError("pit_seed must be an integer")
    decision = OOFCostSensitiveDecisionRule()
    return _hash_payload(
        {
            "protocol_version": _AUDIT_PROTOCOL_VERSION,
            "calibration_config": asdict(config),
            "decision_config": asdict(decision),
            "pit_seed": int(pit_seed),
            "target_isolation": "drop_targets_before_predict_then_open_after_states_frozen",
        }
    )


def _validate_inner(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "outer_fold_id",
        "match_id",
        "prediction_year",
        "train_end_year",
        "is_oof",
        "observed_home_goals",
        "observed_away_goals",
    }
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Selected inner OOF ledger must be nonempty")
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Selected inner OOF ledger is missing columns: {missing}")
    years = pd.to_numeric(frame["prediction_year"], errors="coerce")
    train = pd.to_numeric(frame["train_end_year"], errors="coerce")
    chronology = np.column_stack((years.to_numpy(float), train.to_numpy(float)))
    integer_chronology = np.isfinite(chronology).all() and np.equal(
        chronology, np.floor(chronology)
    ).all()
    oof = frame["is_oof"]
    strict_oof = oof.map(lambda value: isinstance(value, (bool, np.bool_)))
    if (
        not integer_chronology
        or years.max() > 2010
        or not train.lt(years).all()
        or not strict_oof.all()
        or not oof.astype(bool).all()
    ):
        raise ValueError("Final calibration requires strict pre-2014 chronological OOF rows")
    goals = frame[["observed_home_goals", "observed_away_goals"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(float)
    valid_goals = (
        np.isfinite(goals).all()
        and np.all(goals >= 0)
        and np.equal(goals, np.floor(goals)).all()
    )
    if not valid_goals:
        raise ValueError("Selected inner OOF goal targets must be nonnegative integers")
    keys = ["outer_fold_id", "match_id"]
    if frame["match_id"].isna().any() or frame.duplicated(keys).any():
        raise ValueError("Selected inner OOF match keys must be complete and unique")
    return frame.sort_values(["prediction_year", "match_id"]).reset_index(drop=True)


def _validate_audit_structure(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"Year", "MatchID", *_TARGET_COLUMNS}
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("2014 audit frame must be nonempty")
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"2014 audit frame is missing columns: {missing}")
    years = pd.to_numeric(frame["Year"], errors="coerce").to_numpy(float)
    valid_year = (
        np.isfinite(years).all()
        and np.equal(years, np.floor(years)).all()
        and set(years.astype(int)) == {2014}
    )
    if not valid_year:
        raise ValueError("Frozen audit accepts exactly the 2014 edition")
    if frame["MatchID"].isna().any() or frame["MatchID"].duplicated().any():
        raise ValueError("Frozen audit match identifiers must be complete and unique")
    return frame.sort_values("MatchID").reset_index(drop=True)


def _validate_identity(
    predictor: Any,
    identity: dict[str, str],
    expected_protocol_hash: str,
) -> dict[str, Any]:
    required = {
        "final_fit_hash",
        "selected_config_hash",
        "feature_schema_hash",
        "protocol_hash",
    }
    valid_hashes = all(
        isinstance(identity.get(key), str)
        and len(identity[key]) == 64
        and all(character in "0123456789abcdef" for character in identity[key])
        for key in required
    )
    if set(identity) != required or not valid_hashes:
        raise ValueError("Frozen identity must contain the four exact 64-character hashes")
    if identity["protocol_hash"] != expected_protocol_hash:
        raise ValueError("Frozen protocol hash disagrees with calibration/decision controls")
    metadata = getattr(predictor, "metadata", None)
    if not isinstance(metadata, dict):
        raise ValueError("Fitted predictor does not expose immutable metadata")
    if (
        metadata.get("final_fit_hash") != identity["final_fit_hash"]
        or metadata.get("selected_config_hash") != identity["selected_config_hash"]
        or int(metadata.get("fit_end_year", -1)) != 2010
    ):
        raise ValueError("Fitted predictor identity disagrees with the frozen audit identity")
    if not callable(getattr(predictor, "predict_joint", None)):
        raise TypeError("Fitted predictor must expose predict_joint")
    return dict(metadata)


def _validate_inner_lineage(
    inner: pd.DataFrame,
    grid: pd.DataFrame,
    predictor_metadata: dict[str, Any],
    identity: dict[str, str],
) -> str:
    """Bind calibration/decision evidence to the exact selected fitted model."""
    ledger_required = {
        "candidate_id",
        "config_hash",
        "feature_schema_hash",
        "fit_index_hash",
    }
    missing = sorted(ledger_required.difference(inner.columns))
    if missing:
        raise ValueError(f"Selected inner OOF lineage is missing columns: {missing}")
    candidate_id = predictor_metadata.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("Fitted predictor metadata is missing its selected candidate identity")
    expected = {
        "candidate_id": candidate_id,
        "config_hash": identity["selected_config_hash"],
        "feature_schema_hash": identity["feature_schema_hash"],
    }
    for column, value in expected.items():
        if not inner[column].astype(str).eq(value).all():
            raise ValueError(f"Selected inner OOF {column} disagrees with frozen identity")
    fit_hash = inner["fit_index_hash"].astype(str)
    if not fit_hash.str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("Selected inner OOF fit_index_hash values must be SHA-256 digests")

    grid_required = {
        "outer_fold_id",
        "match_id",
        "candidate_id",
        "fit_index_hash",
    }
    if not isinstance(grid, pd.DataFrame) or grid.empty:
        raise ValueError("Selected inner joint grid must be a nonempty DataFrame")
    missing_grid = sorted(grid_required.difference(grid.columns))
    if missing_grid:
        raise ValueError(f"Selected inner joint-grid lineage is missing: {missing_grid}")
    keys = ["outer_fold_id", "match_id"]
    grouped = grid.groupby(keys, dropna=False)[["candidate_id", "fit_index_hash"]].nunique(
        dropna=False
    )
    if not grouped.eq(1).all().all():
        raise ValueError("Selected inner joint grid has inconsistent per-match lineage")
    grid_lineage = grid[
        [*keys, "candidate_id", "fit_index_hash"]
    ].drop_duplicates()
    ledger_lineage = inner[[*keys, "candidate_id", "fit_index_hash"]]
    compared = ledger_lineage.merge(
        grid_lineage,
        on=keys,
        how="outer",
        suffixes=("_ledger", "_grid"),
        indicator=True,
        validate="one_to_one",
    )
    valid = (
        compared["_merge"].eq("both").all()
        and compared["candidate_id_ledger"].eq(compared["candidate_id_grid"]).all()
        and compared["fit_index_hash_ledger"].eq(compared["fit_index_hash_grid"]).all()
    )
    if not valid:
        raise ValueError("Selected inner joint-grid fit lineage disagrees with OOF ledger")
    lineage = inner[
        [
            "outer_fold_id",
            "match_id",
            "candidate_id",
            "config_hash",
            "feature_schema_hash",
            "fit_index_hash",
        ]
    ].astype(str)
    return _hash_payload(lineage.to_dict(orient="records"))


def _freeze_layers(
    inner: pd.DataFrame,
    grid: pd.DataFrame,
    config: CalibrationSelectionConfig | None,
) -> tuple[ChronologicalCalibrationSelector, OOFCostSensitiveDecisionRule]:
    joint = reconstruct_joint_batch(grid, inner[["outer_fold_id", "match_id"]])
    goals = inner[["observed_home_goals", "observed_away_goals"]].to_numpy(int)
    selector = ChronologicalCalibrationSelector(config).fit_oof(
        joint,
        goals,
        provenance=inner[["is_oof", "prediction_year", "train_end_year"]],
    )
    chosen = selector.crossfit_predictions_.loc[
        selector.crossfit_predictions_["method"].eq(selector.chosen_method_)
    ].copy()
    positions = chosen["row_position"].to_numpy(int)
    source = inner.iloc[positions].reset_index(drop=True)
    train_max = pd.to_numeric(chosen["train_max_year"], errors="coerce").to_numpy(float)
    fallback = pd.to_numeric(source["train_end_year"], errors="raise").to_numpy(float)
    train_end = np.where(np.isfinite(train_max), train_max, fallback)
    if not np.isfinite(train_end).all() or not np.equal(train_end, np.floor(train_end)).all():
        raise RuntimeError("Frozen decision-training fit years must be finite integers")
    provenance = pd.DataFrame(
        {
            "is_oof": True,
            "prediction_year": chosen["prediction_year"].to_numpy(int),
            "train_end_year": train_end.astype(int),
        }
    )
    rule = OOFCostSensitiveDecisionRule().fit_oof(
        chosen[["home_win", "draw", "away_win"]].to_numpy(float),
        chosen["target_label"].to_numpy(int),
        provenance=provenance,
    )
    return selector, rule


def _audit_targets(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    labels = frame["target_outcome"].astype(str).to_numpy()
    if not set(labels).issubset(set(_OUTCOMES)):
        raise ValueError("2014 audit contains an unknown outcome label")
    goals = frame[["target_home_goals", "target_away_goals"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(float)
    if not np.isfinite(goals).all() or np.any(goals < 0) or not np.equal(
        goals, np.floor(goals)
    ).all():
        raise ValueError("2014 audit goals must be finite nonnegative integers")
    derived = np.where(
        goals[:, 0] > goals[:, 1],
        "home_win",
        np.where(goals[:, 0] == goals[:, 1], "draw", "away_win"),
    )
    if not np.array_equal(labels, derived):
        raise ValueError("2014 audit outcome labels disagree with observed goal counts")
    return labels, goals.astype(int)


def _audit_metrics(
    labels: np.ndarray,
    goals: np.ndarray,
    raw_outcome: np.ndarray,
    calibrated_outcome: np.ndarray,
    raw_joint: np.ndarray,
    calibrated_joint: np.ndarray,
    decision_label: np.ndarray,
    *,
    pit_seed: int,
) -> pd.DataFrame:
    rows = []
    for name, outcome, joint in (
        ("advanced_raw", raw_outcome, raw_joint),
        ("advanced_calibrated", calibrated_outcome, calibrated_joint),
    ):
        classification, _ = classification_metrics(labels, outcome)
        rows.append(
            {
                "model": name,
                "evaluation_role": "audit_only_not_for_selection",
                **classification,
                **joint_forecast_metrics(goals, joint, pit_seed=pit_seed),
            }
        )
    truth = np.asarray([int(np.flatnonzero(_OUTCOMES == label)[0]) for label in labels])
    decision = np.asarray(
        [int(np.flatnonzero(_OUTCOMES == label)[0]) for label in decision_label]
    )
    rows.append(
        {
            "model": "advanced_cost_sensitive_decision",
            "evaluation_role": "audit_only_not_for_selection",
            "accuracy": float(accuracy_score(truth, decision)),
            "macro_f1": float(f1_score(truth, decision, average="macro")),
            "recall_draw": float(
                recall_score(truth, decision, labels=[1], average="macro", zero_division=0)
            ),
            "decision_draw_predictions": int(np.sum(decision == 1)),
            "proper_score_applicable": False,
        }
    )
    return pd.DataFrame(rows)


def _long_grid(match_ids: np.ndarray, joint: np.ndarray) -> pd.DataFrame:
    rows = []
    for position, match_id in enumerate(match_ids):
        for home in range(joint.shape[1]):
            for away in range(joint.shape[2]):
                rows.append(
                    {
                        "match_id": match_id,
                        "home_goals": home,
                        "away_goals": away,
                        "probability": float(joint[position, home, away]),
                    }
                )
    return pd.DataFrame(rows)


def _hash_payload(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


__all__ = [
    "FrozenAuditResult",
    "apply_frozen_2014_audit",
    "frozen_audit_protocol_hash",
]
