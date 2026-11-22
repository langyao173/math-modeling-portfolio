"""Orchestrate the immutable, hash-bound pre-2014 research freeze."""

from __future__ import annotations

import io
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .artifact_integrity import (
    build_stage_manifest,
    file_sha256,
    payload_sha256,
    read_verified_artifact_bytes,
    verify_stage_manifest,
    write_json_atomic,
)
from .freeze_contracts import (
    OUTER_YEARS,
    PACKAGE_NAMES,
    REQUIRED_OUTPUTS,
    candidate_registry,
    hash_data,
    jsonable,
    model_state,
    validate_oof,
    validate_snapshot,
    write_parquet_atomic,
)


def git_output(root: Path, *args: str) -> str:
    """Run one read-only Git query and return normalized stdout."""
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def _seal_nested_stage(nested: Path) -> dict[str, Any]:
    required = {
        "candidate_metrics.csv",
        "final_fitted_model.joblib",
        "final_selection_metadata.json",
        "frozen_model_identity.json",
        "outer_oof_ledger.csv",
        "selected_specs.csv",
    }
    relative = sorted(
        path.name
        for path in nested.iterdir()
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    missing = sorted(required.difference(relative))
    if missing:
        raise FileNotFoundError(f"Nested freeze inputs are incomplete: {missing}")
    manifest = build_stage_manifest(nested, relative)
    manifest_path = write_json_atomic(manifest, nested / "artifact_manifest.json")
    return verify_stage_manifest(manifest_path, required_paths=required)


def _verified_csv(nested: Path, name: str, binding: dict[str, Any]) -> pd.DataFrame:
    payload = read_verified_artifact_bytes(nested, name, binding)
    return pd.read_csv(io.BytesIO(payload))


def _verified_json(nested: Path, name: str, binding: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(read_verified_artifact_bytes(nested, name, binding))
    if not isinstance(payload, dict):
        raise ValueError(f"Verified nested JSON is not an object: {name}")
    return payload


def _verified_model(nested: Path, binding: dict[str, Any]) -> Any:
    payload = read_verified_artifact_bytes(nested, "final_fitted_model.joblib", binding)
    return joblib.load(io.BytesIO(payload))


def _environment(root: Path, source_commit: str, seeds: list[int]) -> str:
    versions = {name: metadata.version(name) for name in PACKAGE_NAMES}
    frozen = subprocess.run(
        ["uv", "pip", "freeze"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    lines = [
        "PRE-2014 FREEZE ENVIRONMENT",
        f"source_commit={source_commit}",
        f"python={sys.version.replace(chr(10), ' ')}",
        f"platform={platform.platform()}",
        f"processor={platform.processor()}",
        f"seeds={','.join(str(seed) for seed in seeds)}",
        "packages=" + json.dumps(versions, sort_keys=True),
        "",
        "uv pip freeze:",
        frozen,
        "",
    ]
    return "\n".join(lines)


def _selection_protocol(root: Path, source_commit: str, identity: dict[str, Any]) -> str:
    sources = (
        root / "plan" / "PREREGISTRATION.md",
        root / "plan" / "protocol_lock.json",
        root / "plan" / "preaudit_amendment_lock.json",
    )
    protocol_hashes = {
        path.relative_to(root).as_posix(): file_sha256(path) for path in sources
    }
    disclosure = (
        "Existing historical 2014 visibility is disclosed; this is software isolation, "
        "not researcher blinding."
    )
    return "\n".join(
        [
            "# Frozen Pre-2014 Selection Protocol",
            "",
            f"- Source commit: `{source_commit}`",
            "- Formal selection input: `data/development_snapshot` through 2010 only.",
            "- Outer folds: 1986, 1990, 1994, 1998, 2002, 2006, 2010.",
            "- Candidate budget: exactly 36 configurations.",
            f"- Final candidate ID: `{identity['selected_candidate_hash']}`",
            f"- Final config hash: `{identity['selected_config_hash']}`",
            f"- Final fit hash: `{identity['final_fit_hash']}`",
            "- 2014 role: one-shot external audit only; never model selection.",
            f"- {disclosure}",
            "- Timeout/non-convergence rule: ineligible for production promotion.",
            "",
            "## Bound protocol sources",
            "",
            *[f"- `{path}`: `{digest}`" for path, digest in protocol_hashes.items()],
            "",
        ]
    )


def _freeze_manifest(
    root: Path,
    source_commit: str,
    identity: dict[str, Any],
    model: Any,
    schema: dict[str, Any],
    binding: dict[str, Any],
    records: list[dict[str, Any]],
    seeds: list[int],
) -> dict[str, Any]:
    model_hash = binding["artifact_records"]["final_fitted_model.joblib"]["sha256"]
    return {
        "schema_version": 1,
        "freeze_id": "rc0-pre2014",
        "status": "pre2014_model_selection_frozen",
        "generated_at": datetime.now(UTC).astimezone().isoformat(),
        "source_commit": source_commit,
        "source_commit_parent": git_output(root, "rev-parse", f"{source_commit}^"),
        "training_boundary": {"minimum_year": 1930, "maximum_year": 2010},
        "outer_validation_years": list(OUTER_YEARS),
        "candidate_count": 36,
        "final_candidate_id": identity["selected_candidate_hash"],
        "selected_config_hash": identity["selected_config_hash"],
        "final_fit_hash": identity["final_fit_hash"],
        "feature_schema_hash": identity["feature_schema_hash"],
        "hyperparameters": jsonable(model.model_config),
        "random_seeds": seeds,
        "preprocessing_state_sha256": schema["preprocessing_state_sha256"],
        "model_file_sha256": model_hash,
        "calibration_and_decision": {
            "outer_fold_states_path": "temp/advanced_evidence/evaluation",
            "final_audit_layers_frozen_only_after_positive_promotion": True,
            "ensemble_weights": None,
            "ensemble_status": "not_used_by_selected_model",
        },
        "software_versions": {name: metadata.version(name) for name in PACKAGE_NAMES},
        "predeclared_2014_metrics": [
            "accuracy",
            "macro_f1",
            "confusion_matrix",
            "log_loss",
            "multiclass_brier",
            "ranked_probability_score",
            "calibration_error",
            "scoreline_nll",
            "total_goal_crps",
            "dixon_coles_comparison",
            "official_logit_comparison",
            "simple_elo_comparison",
        ],
        "human_blinding_disclosure": identity["human_blinding_disclosure"],
        "formal_preaudit_reads_combined_raw": False,
        "advanced_2014_audit_opened": False,
        "artifact_records": records,
        "nested_input_manifest_sha256": binding["manifest_sha256"],
        "nested_input_artifact_count": binding["artifact_count"],
    }


def build_freeze(
    root: Path,
    evidence_root: Path,
    output: Path,
    *,
    source_commit: str,
) -> dict[str, Any]:
    """Build and verify every required pre-2014 freeze artifact."""
    root = root.resolve()
    evidence_root = evidence_root.resolve()
    output = output.resolve()
    if source_commit != git_output(root, "rev-parse", "HEAD"):
        raise RuntimeError("Freeze source commit is not HEAD")
    if git_output(root, "status", "--porcelain"):
        raise RuntimeError("Freeze requires the exact clean source commit at HEAD")

    nested = evidence_root / "nested"
    binding = _seal_nested_stage(nested)
    snapshot = validate_snapshot(root)
    identity = _verified_json(nested, "frozen_model_identity.json", binding)
    selection = _verified_json(nested, "final_selection_metadata.json", binding)
    model = _verified_model(nested, binding)
    if (
        identity.get("fit_end_year") != 2010
        or identity.get("audit_outcomes_used_for_selection") is not False
        or selection.get("maximum_selection_year") != 2010
        or selection.get("audit_rows_loaded") is not False
    ):
        raise ValueError("Final model identity is not a valid pre-2014 freeze")

    oof = validate_oof(_verified_csv(nested, "outer_oof_ledger.csv", binding))
    candidate = _verified_csv(nested, "candidate_metrics.csv", binding)
    selected = _verified_csv(nested, "selected_specs.csv", binding)
    registry = candidate_registry(candidate, selected)
    schema, preprocessing = model_state(model)
    if schema["preprocessing_state_sha256"] != payload_sha256(preprocessing):
        raise AssertionError("Preprocessing state fingerprint is unstable")
    if identity["feature_schema_hash"] != snapshot["feature_schema_hash"]:
        raise ValueError("Model and development snapshot feature schemas disagree")

    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(hash_data(root, snapshot), output / "DATA_HASHES.json")
    write_json_atomic(
        {**schema, "preprocessing_state": preprocessing}, output / "FEATURE_SCHEMA.json"
    )
    registry.to_csv(output / "MODEL_REGISTRY.csv", index=False, lineterminator="\n")
    write_parquet_atomic(oof, output / "PRE2014_PREDICTIONS.parquet")
    (output / "SELECTION_PROTOCOL.md").write_text(
        _selection_protocol(root, source_commit, identity), encoding="utf-8", newline="\n"
    )
    seeds = [1729, 2718, 31415, 57721, 65537]
    (output / "ENVIRONMENT.txt").write_text(
        _environment(root, source_commit, seeds), encoding="utf-8", newline="\n"
    )
    records = build_stage_manifest(output, REQUIRED_OUTPUTS)
    manifest = _freeze_manifest(
        root, source_commit, identity, model, schema, binding, records, seeds
    )
    write_json_atomic(manifest, output / "FREEZE_MANIFEST.json")
    for record in records:
        path = output / record["path"]
        if path.stat().st_size != record["bytes"] or file_sha256(path) != record["sha256"]:
            message = f"Freeze artifact changed during final verification: {record['path']}"
            raise RuntimeError(message)
    return manifest


__all__ = ["build_freeze", "git_output"]
