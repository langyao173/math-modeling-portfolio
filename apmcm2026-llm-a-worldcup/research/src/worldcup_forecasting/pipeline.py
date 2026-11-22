"""One-click, configuration-driven World Cup forecasting pipeline."""

from __future__ import annotations

import json
import logging
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from worldcup_forecasting.application import (
    REQUIRED_ARTIFACTS,
    RESEARCH_REQUIRED_ARTIFACTS,
    ResearchRunResult,
    ensure_advanced_research,
    publish_advanced_research,
    run_official_analysis,
    write_csv,
    write_json,
)
from worldcup_forecasting.application.runtime import staged_output_directory
from worldcup_forecasting.config import load_config, resolved_config_dict
from worldcup_forecasting.data import build_canonical_data, write_audit_artifacts
from worldcup_forecasting.experiments.matlab_bridge import verify_matlab_parity
from worldcup_forecasting.reporting.documents import write_analysis_bundle, write_model_card
from worldcup_forecasting.reporting.manifest import write_manifest
from worldcup_forecasting.reporting.model_state import stable_hash


def run_pipeline(
    project_root: str | Path | None = None,
    config_name: str = "final",
    output_override: str | Path | None = None,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Run audit, official baselines, research evaluation, and reporting.

    The manifest is deliberately the final write. A successful return therefore
    certifies that every mandatory artifact existed and was nonempty when hashed.
    """
    started = time.perf_counter()
    root = _resolve_project_root(project_root)
    config = load_config(root, config_name, overrides)
    _validate_profile_isolation(config)
    _validate_deployment_contract(config)
    final_output_dir = _resolve_output(root, config, output_override)
    default_output_dir = (root / "outputs").resolve()
    raw_dir = _resolve_from_root(root, str(config.paths.raw_dir))
    registry_dir = _resolve_from_root(root, str(config.paths.registry_dir))
    processed_dir = _resolve_from_root(root, str(config.paths.processed_dir))
    configured_data_paths = [raw_dir, registry_dir, processed_dir]
    if "interim_dir" in config.paths:
        configured_data_paths.append(
            _resolve_from_root(root, str(config.paths.interim_dir))
        )

    with staged_output_directory(
        root,
        final_output_dir,
        default_output_dir,
        clean=bool(config.runtime.clean_outputs),
        preserve_matlab="research" in config,
        protected_paths=configured_data_paths,
    ) as workspace:
        output_dir = workspace.staging_dir
        logger = _configure_logging(output_dir)
        try:
            logger.info("Pipeline start config=%s project_root=%s", config_name, root)
            advanced_preparation = None
            if "research" in config:
                advanced_preparation = ensure_advanced_research(root, config, logger)
            logger.info("Building SHA-256-verified canonical data")
            bundle = build_canonical_data(raw_dir)
            write_audit_artifacts(bundle, raw_dir, output_dir / "audit", registry_dir)
            processed_dir.mkdir(parents=True, exist_ok=True)
            write_csv(bundle.tournaments, processed_dir / "tournaments.csv")
            write_csv(bundle.matches, processed_dir / "matches.csv")
            write_csv(bundle.players, processed_dir / "players.csv")
            write_json(resolved_config_dict(config), output_dir / "audit" / "resolved_config.json")
            if "research" in config:
                _validate_optional_matlab(bundle, output_dir, logger)

            if "research" in config:
                if advanced_preparation is None:  # pragma: no cover - guarded by config branch
                    raise RuntimeError("Research preflight did not return a preparation")
                research = publish_advanced_research(
                    advanced_preparation,
                    output_dir,
                    config,
                    logger,
                )
                official = run_official_analysis(bundle, output_dir, config, logger)
            else:
                official = run_official_analysis(bundle, output_dir, config, logger)
                selected_model = (
                    str(config.deployment.selected_model)
                    if "deployment" in config
                    else "outcome_logit"
                )
                research = ResearchRunResult(
                    cv_metrics=pd.DataFrame(),
                    comparisons=[],
                    selected_model=selected_model,
                    promotion_decision=(
                        "frozen_deployment_configuration_no_search"
                        if "deployment" in config
                        else "official_mode_baseline_only"
                    ),
                    figure_catalog=[],
                )

            figure_catalog = [*official.figure_catalog, *research.figure_catalog]
            write_analysis_bundle(
                output_dir / "analysis",
                research.cv_metrics,
                official.official_metrics,
                figure_catalog,
                research.comparisons,
            )
            write_model_card(
                output_dir / "reports" / "MODEL_CARD.md",
                research.selected_model,
                "2014-07-13",
                official.official_metrics,
                research.promotion_decision,
            )
            runtime_seconds = time.perf_counter() - started
            run_summary = {
                "status": "ready_for_manifest",
                "config_name": config_name,
                "track": str(config.project.track),
                "data_cutoff": 2014,
                "raw_data_hashes": bundle.raw_hashes,
                "canonical_rows": {
                    "tournaments": len(bundle.tournaments),
                    "matches": len(bundle.matches),
                    "players": len(bundle.players),
                },
                "selected_model": research.selected_model,
                "promotion_decision": research.promotion_decision,
                "runtime_seconds": runtime_seconds,
            }
            write_json(run_summary, output_dir / "reports" / "run_summary.json")
            logger.info(
                "All analyses complete in %.3f seconds; closing logs before manifest",
                runtime_seconds,
            )
        finally:
            _close_logger(logger)

        artifact_profile = "research" if "research" in config else "official"
        required_artifacts = set(REQUIRED_ARTIFACTS)
        if artifact_profile == "research":
            required_artifacts.update(RESEARCH_REQUIRED_ARTIFACTS)
        _assert_required_nonempty(output_dir, required_artifacts)
        config_payload = resolved_config_dict(config)
        write_manifest(
            output_dir,
            run_metadata={
                "run": run_summary,
                "environment": {
                    "python": sys.version.split()[0],
                    "platform": platform.platform(),
                },
                "resolved_config_sha256": stable_hash(config_payload),
                "scenario_warning": (
                    "2026 outputs are scenario-only; all supplied observations end in 2014."
                ),
                "artifact_profile": artifact_profile,
            },
            required_paths=required_artifacts,
        )
        workspace.promote()

    manifest_path = final_output_dir / "manifest.json"
    return {
        "status": "complete",
        "manifest_path": str(manifest_path),
        "output_dir": str(final_output_dir),
        "config_name": config_name,
        "runtime_seconds": runtime_seconds,
    }


def _resolve_project_root(project_root: str | Path | None) -> Path:
    if project_root is None:
        candidate = Path(__file__).resolve().parents[2]
    else:
        candidate = Path(project_root)
    resolved = candidate.expanduser().resolve()
    if not (resolved / "configs").is_dir() or not (resolved / "data" / "raw").is_dir():
        raise FileNotFoundError(f"Not a World Cup forecasting project root: {resolved}")
    return resolved


def _validate_deployment_contract(config: Any) -> None:
    """Fail closed if deployment metadata disagrees with executable production code."""
    if "deployment" not in config:
        return
    expected = {
        "selected_model": "joint_poisson_static_history",
        "model_artifact": "models/problem3_joint_poisson_static_2014.json",
    }
    mismatches = {}
    for field, value in expected.items():
        configured = (
            str(config.deployment[field])
            if field in config.deployment
            else "<missing>"
        )
        if configured != value:
            mismatches[field] = {"expected": value, "configured": configured}
    if mismatches:
        raise ValueError(
            "Deployment configuration is not executable by the frozen production path: "
            f"{mismatches}"
        )


def _validate_profile_isolation(config: Any) -> None:
    """Reject research configuration in official or frozen-final execution."""
    track = str(config.project.track)
    if track in {"closed", "final_frozen_closed"} and "research" in config:
        raise ValueError(
            f"Execution profile {track!r} must not reference research configuration"
        )


def _resolve_output(root: Path, config: Any, override: str | Path | None) -> Path:
    if override is not None:
        return Path(override).expanduser().resolve()
    return _resolve_from_root(root, str(config.paths.output_dir))


def _resolve_from_root(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


class _JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "level": record.levelname,
                "message": record.getMessage(),
            },
            ensure_ascii=False,
        )


def _configure_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"worldcup_forecasting.pipeline.{id(output_dir)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    plain = logging.FileHandler(output_dir / "logs" / "pipeline.log", encoding="utf-8")
    plain.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    structured = logging.FileHandler(output_dir / "logs" / "run.jsonl", encoding="utf-8")
    structured.setFormatter(_JsonLineFormatter())
    logger.addHandler(plain)
    logger.addHandler(structured)
    return logger


def _close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def _assert_required_nonempty(output_dir: Path, required: set[str]) -> None:
    missing = []
    empty = []
    for relative in sorted(required):
        path = output_dir / relative
        if not path.is_file():
            missing.append(relative)
        elif path.stat().st_size == 0:
            empty.append(relative)
    if missing or empty:
        raise RuntimeError(f"Artifact gate failed; missing={missing}, empty={empty}")


def _validate_optional_matlab(bundle: Any, output_dir: Path, logger: logging.Logger) -> None:
    matlab_dir = output_dir / "matlab"
    retained_files = (
        sorted(path for path in matlab_dir.rglob("*") if path.is_file())
        if matlab_dir.is_dir()
        else []
    )
    if not retained_files:
        return
    required = {
        "experiment_card.json",
        "parity_descriptive_statistics.csv",
        "parity_stage_statistics.csv",
        "parity_summary.json",
    }
    present = {path.relative_to(matlab_dir).as_posix() for path in retained_files}
    missing = sorted(required.difference(present))
    if missing:
        raise RuntimeError(
            "Retained MATLAB evidence is incomplete and cannot enter the manifest; "
            f"missing={missing}"
        )
    logger.info("Validating preserved MATLAB R2026a evidence against Python")
    report = verify_matlab_parity(
        bundle,
        matlab_dir,
        report_path=output_dir / "audit" / "matlab_parity_verification.json",
    )
    if report["status"] != "passed":
        raise RuntimeError("Preserved MATLAB evidence failed parity validation")
