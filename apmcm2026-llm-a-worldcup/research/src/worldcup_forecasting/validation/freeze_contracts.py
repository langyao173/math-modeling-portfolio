"""Pure validation and serialization contracts for a pre-2014 freeze."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifact_integrity import file_sha256, payload_sha256

OUTER_YEARS = (1986, 1990, 1994, 1998, 2002, 2006, 2010)
REQUIRED_OUTPUTS = (
    "DATA_HASHES.json",
    "ENVIRONMENT.txt",
    "FEATURE_SCHEMA.json",
    "MODEL_REGISTRY.csv",
    "PRE2014_PREDICTIONS.parquet",
    "SELECTION_PROTOCOL.md",
)
PACKAGE_NAMES = (
    "hydra-core",
    "joblib",
    "matplotlib",
    "numpy",
    "omegaconf",
    "pandas",
    "pyarrow",
    "pyyaml",
    "scikit-learn",
    "scipy",
    "seaborn",
    "statsmodels",
)


def json_object(path: Path) -> dict[str, Any]:
    """Read one required JSON object."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def jsonable(value: Any) -> Any:
    """Normalize fitted scientific state into canonical JSON-compatible values."""
    if is_dataclass(value) and not isinstance(value, type):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        normalized = [jsonable(item) for item in value]
        return sorted(normalized) if isinstance(value, (set, frozenset)) else normalized
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Path):
        return value.as_posix()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Cannot serialize freeze value of type {type(value).__name__}")


def validate_snapshot(root: Path) -> dict[str, Any]:
    """Validate the immutable 1930-2010 development snapshot and its hashes."""
    snapshot_dir = root / "data" / "development_snapshot"
    manifest = json_object(snapshot_dir / "snapshot_manifest.json")
    if (
        manifest.get("cutoff_year") != 2010
        or manifest.get("audit_outcomes_used") is not False
        or manifest.get("development_rows") != 772
    ):
        raise ValueError("Development snapshot does not satisfy the frozen 2010 boundary")
    for record in manifest.get("artifacts", []):
        path = snapshot_dir / str(record["path"])
        if path.stat().st_size != record["bytes"] or file_sha256(path) != record["sha256"]:
            raise ValueError(f"Development snapshot artifact changed: {path.name}")
    return manifest


def validate_oof(frame: pd.DataFrame) -> pd.DataFrame:
    """Require a coherent, unique, seven-edition temporal OOF ledger."""
    required = {
        "match_id",
        "prediction_year",
        "train_end_year",
        "outer_fold_id",
        "candidate_id",
        "feature_schema_hash",
        "home_win",
        "draw",
        "away_win",
        "is_oof",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Outer OOF ledger is missing columns: {missing}")
    result = frame.sort_values(["prediction_year", "match_id"]).reset_index(drop=True)
    years = pd.to_numeric(result["prediction_year"], errors="raise").astype(int)
    train_end = pd.to_numeric(result["train_end_year"], errors="raise").astype(int)
    probabilities = result[["home_win", "draw", "away_win"]].to_numpy(float)
    oof = result["is_oof"].astype(str).str.casefold().eq("true")
    if tuple(sorted(years.unique())) != OUTER_YEARS:
        raise ValueError("Outer OOF years differ from the frozen seven-fold protocol")
    if int(years.max()) > 2010 or not bool((train_end < years).all()) or not bool(oof.all()):
        raise ValueError("Outer predictions violate the pre-2014 temporal boundary")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("Outer OOF probabilities are invalid")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12):
        raise ValueError("Outer OOF probabilities do not sum to one")
    if result["match_id"].duplicated().any():
        raise ValueError("Outer OOF ledger contains duplicate match IDs")
    return result


def candidate_registry(candidate: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    """Reduce seven-fold candidate rows to the exact 36-candidate registry."""
    if candidate["candidate_id"].nunique() != 36:
        raise ValueError("The frozen candidate registry must contain exactly 36 candidates")
    candidate = candidate.copy()
    candidate["eligible"] = candidate["eligible"].astype(str).str.casefold().eq("true")
    selected_counts = selected.groupby("candidate_id").size().rename("selected_outer_folds")
    runtime_column = (
        "measured_runtime_seconds"
        if "measured_runtime_seconds" in candidate
        else "runtime_seconds"
    )
    if runtime_column not in candidate:
        raise ValueError("Candidate registry requires a frozen runtime reference")

    def failure_reasons(values: pd.Series) -> str:
        return "|".join(sorted(set(values.dropna())))

    registry = (
        candidate.groupby("candidate_id", as_index=False)
        .agg(
            candidate_order=("candidate_order", "first"),
            config_hash=("config_hash", "first"),
            alpha=("alpha", "first"),
            half_life_years=("half_life_years", "first"),
            shared_score=("shared_score", "first"),
            residual=("residual", "first"),
            eligible_all_folds=("eligible", "all"),
            eligible_fold_count=("eligible", "sum"),
            maximum_runtime_reference_seconds=(runtime_column, "max"),
            failure_reasons=("failure_reason", failure_reasons),
        )
        .merge(selected_counts, how="left", on="candidate_id")
        .fillna({"selected_outer_folds": 0})
        .sort_values("candidate_order")
        .reset_index(drop=True)
    )
    registry["selected_outer_folds"] = registry["selected_outer_folds"].astype(int)
    return registry


def model_state(model: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract exact feature order, preprocessing state, and fitted count weights."""
    fitted = model._model
    design = fitted.pipeline_.named_steps["design"]
    numeric = design.named_transformers_["numeric"]
    categorical = design.named_transformers_["categorical"]
    imputer = numeric.named_steps["imputer"]
    scaler = numeric.named_steps["scaler"]
    poisson = fitted.pipeline_.named_steps["poisson"]
    schema = {
        "schema_version": 1,
        "information_set": "pre_match",
        "ordered_input_features": [str(value) for value in design.feature_names_in_],
        "ordered_transformed_features": [str(value) for value in design.get_feature_names_out()],
        "odd_features": list(fitted.odd_columns_),
        "even_features": list(fitted.even_columns_),
        "context_features": list(fitted.context_columns_),
        "paired_features": [list(pair) for pair in fitted.paired_columns_],
        "team_categories": sorted(str(value) for value in fitted.team_categories_),
    }
    state = {
        "numeric_imputer_statistics": jsonable(imputer.statistics_),
        "numeric_scaler_mean": jsonable(scaler.mean_),
        "numeric_scaler_scale": jsonable(scaler.scale_),
        "categorical_categories": jsonable(categorical.categories_),
        "poisson_intercept": float(poisson.intercept_),
        "poisson_coefficients": jsonable(poisson.coef_),
        "poisson_iterations": int(poisson.n_iter_),
        "team_effects": jsonable(fitted.team_effects_.to_dict(orient="records")),
    }
    schema["preprocessing_state_sha256"] = payload_sha256(state)
    return schema, state


def hash_data(root: Path, snapshot: dict[str, Any]) -> dict[str, Any]:
    """Hash raw, processed, and formal development data without parsing raw targets."""
    records: list[dict[str, Any]] = []
    for directory, role in (("raw", "supplied_raw"), ("processed", "canonical_processed")):
        for path in sorted((root / "data" / directory).glob("*")):
            if path.is_file():
                records.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "role": role,
                        "bytes": path.stat().st_size,
                        "sha256": file_sha256(path),
                    }
                )
    for record in snapshot["artifacts"]:
        records.append(
            {
                "path": f"data/development_snapshot/{record['path']}",
                "role": "formal_selection_input_through_2010",
                "bytes": record["bytes"],
                "sha256": record["sha256"],
            }
        )
    return {
        "schema_version": 1,
        "records": records,
        "selection_data_role": "development_snapshot_only",
        "selection_cutoff_year": 2010,
        "audit_year": 2014,
        "raw_hashing_does_not_parse_or_expose_targets": True,
        "physical_raw_no_read_claim": False,
    }


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Write a compressed Parquet ledger through an atomic same-directory replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as stream:
            temporary = stream.name
        frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=False)
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        raise


__all__ = [
    "OUTER_YEARS",
    "PACKAGE_NAMES",
    "REQUIRED_OUTPUTS",
    "candidate_registry",
    "hash_data",
    "json_object",
    "jsonable",
    "model_state",
    "validate_oof",
    "validate_snapshot",
    "write_parquet_atomic",
]
