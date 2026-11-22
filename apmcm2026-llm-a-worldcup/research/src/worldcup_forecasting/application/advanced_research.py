"""Production adapter for the sealed advanced research stage machine.

This module deliberately does not build canonical data.  Its preflight is called
before the production pipeline is allowed to read the combined raw source, so a
research run can only proceed after the development stages, promotion decision,
and cached one-time audit have all passed their independent integrity checks.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from worldcup_forecasting.validation._promotion_artifacts import checkpoint_status
from worldcup_forecasting.validation.audit_orchestration import verify_cached_audit
from worldcup_forecasting.validation.promotion import verify_promotion_certificate
from worldcup_forecasting.validation.stage_orchestration import PREAUDIT_STAGE_ORDER

from .research import ResearchRunResult
from .runtime import assert_output_tree_has_no_links, write_csv, write_json


@dataclass(frozen=True, slots=True)
class AdvancedResearchPreparation:
    """Integrity-verified evidence available before canonical 2014 is opened."""

    evidence_dir: Path
    command: tuple[str, ...]
    stage_bindings: dict[str, dict[str, Any]]
    promotion_certificate: dict[str, Any]
    audit_summary: dict[str, Any]


def ensure_advanced_research(
    project_root: Path,
    config: Any,
    logger: Any,
) -> AdvancedResearchPreparation:
    """Run/resume the bundled ``all`` stage and verify its complete sealed result."""
    root = project_root.expanduser().resolve()
    evidence, script, timeout = _advanced_paths(root, config)
    command = (
        sys.executable,
        str(script),
        "--stage",
        "all",
        "--output",
        str(evidence),
    )
    cached_audit_state = evidence / "audit" / "audit_state.json"
    if cached_audit_state.is_file():
        logger.info("Research preflight: verifying sealed cached audit without orchestration rerun")
        verified = _verify_advanced_evidence(evidence)
        return replace(verified, command=command)

    logger.info("Research preflight: completing advanced stages before canonical raw read")
    subprocess.run(
        list(command),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    verified = _verify_advanced_evidence(evidence)
    logger.info(
        "Research preflight verified seven stages, promotion certificate, and cached audit"
    )
    return replace(verified, command=command)


def publish_advanced_research(
    preparation: AdvancedResearchPreparation,
    output_dir: Path,
    config: Any,
    logger: Any,
) -> ResearchRunResult:
    """Copy the verified evidence byte-for-byte and publish stable report tables."""
    del config
    source = preparation.evidence_dir.resolve()
    current = _verify_advanced_evidence(source)
    if _preparation_identity(current) != _preparation_identity(preparation):
        raise RuntimeError("Advanced evidence changed after orchestration verification")

    assert_output_tree_has_no_links(source)
    source_inventory, source_digest = _tree_inventory(source)
    destination = output_dir.resolve() / "evidence" / "advanced"
    if destination.exists():
        raise RuntimeError(f"Advanced evidence destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, symlinks=False, copy_function=shutil.copy2)
    assert_output_tree_has_no_links(destination)
    copied_inventory, copied_digest = _tree_inventory(destination)
    if source_inventory != copied_inventory or source_digest != copied_digest:
        raise RuntimeError("Advanced evidence copy failed byte-for-byte integrity verification")

    tables = output_dir.resolve() / "tables"
    required_sources = {
        "cv": destination / "evaluation" / "combined_fold_metrics.csv",
        "comparisons": destination / "evaluation" / "paired_tournament_intervals.csv",
        "calibration": destination / "evaluation" / "calibration_metadata.json",
        "ablation": destination / "ablation" / "ablation_matrix.csv",
        "audit": destination / "audit" / "audit_metrics.csv",
        "stress": destination / "stress_stage_summary.json",
    }
    missing = sorted(str(path) for path in required_sources.values() if not path.is_file())
    if missing:
        raise RuntimeError(f"Verified advanced evidence lacks publication inputs: {missing}")

    cv_metrics = _read_csv(required_sources["cv"])
    comparisons_frame = _read_csv(required_sources["comparisons"])
    audit_metrics = _read_csv(required_sources["audit"])
    write_csv(cv_metrics, tables / "research_cv_metrics.csv")
    write_csv(comparisons_frame, tables / "research_paired_comparisons.csv")
    write_csv(_read_csv(required_sources["ablation"]), tables / "research_ablation_matrix.csv")
    write_csv(audit_metrics, tables / "research_frozen_2014.csv")

    calibration = _read_json_object(required_sources["calibration"])
    stress = _read_json_object(required_sources["stress"])
    gate = preparation.promotion_certificate.get("gate")
    if not isinstance(gate, dict) or gate.get("system_promoted") is not True:
        raise RuntimeError("Advanced promotion certificate lacks a positive frozen gate")
    write_json(calibration, tables / "research_calibration.json")
    write_json(stress, tables / "research_stress_tests.json")
    write_json(gate, tables / "research_promotion_gate.json")
    write_csv(_gate_table(gate), tables / "research_promotion_gate.csv")

    selected_model = str(gate.get("candidate", "advanced_nested_calibrated"))
    summary = {
        "status": "advanced_research_evidence_published",
        "stage_order": list(PREAUDIT_STAGE_ORDER),
        "all_stage_checkpoints_verified": True,
        "promotion": {
            "decision": preparation.promotion_certificate.get("decision"),
            "system_promoted": True,
            "evidence_bundle_sha256": preparation.promotion_certificate.get(
                "evidence_bundle_sha256"
            ),
            "final_fit_hash": preparation.promotion_certificate.get("model_identity", {}).get(
                "final_fit_hash"
            ),
        },
        "audit": preparation.audit_summary,
        "source_file_count": len(source_inventory),
        "source_tree_sha256": source_digest,
        "copied_tree_sha256": copied_digest,
        "copy_verified_byte_for_byte": True,
        "selected_model": selected_model,
        "claim_boundary": (
            "single_frozen_2014_audit_not_external_validation_or_2026_performance_claim"
        ),
    }
    write_json(summary, tables / "research_advanced_summary.json")
    logger.info(
        "Published immutable advanced evidence (%d files, tree_sha256=%s)",
        len(source_inventory),
        source_digest,
    )
    return ResearchRunResult(
        cv_metrics=cv_metrics,
        comparisons=comparisons_frame.to_dict(orient="records"),
        selected_model=selected_model,
        promotion_decision="promoted_then_one_time_2014_audit_complete",
        figure_catalog=[],
    )


def _verify_advanced_evidence(evidence_dir: Path) -> AdvancedResearchPreparation:
    """Fail closed unless the entire advanced state machine is already complete."""
    evidence = evidence_dir.expanduser().resolve()
    if not evidence.is_dir():
        raise FileNotFoundError(f"Advanced evidence directory does not exist: {evidence}")
    stage_bindings: dict[str, dict[str, Any]] = {}
    for stage in PREAUDIT_STAGE_ORDER:
        binding = checkpoint_status(evidence, stage)
        if binding.get("status") != "complete":
            raise RuntimeError(f"Advanced stage {stage!r} is not complete")
        stage_bindings[stage] = binding

    certificate = verify_promotion_certificate(
        evidence / "promotion" / "promotion_certificate.json",
        revalidate_artifacts=True,
    )
    if (
        certificate.get("certificate_type") != "preaudit_promotion"
        or certificate.get("decision") != "promote_to_one_time_audit"
        or not isinstance(certificate.get("gate"), dict)
        or certificate["gate"].get("system_promoted") is not True
    ):
        raise RuntimeError("Advanced evidence has no valid positive promotion certificate")
    audit = verify_cached_audit(evidence)
    if (
        audit.get("status") != "one_time_2014_audit_complete"
        or audit.get("target_reads") != 1
        or audit.get("cached_repeats_only") is not True
    ):
        raise RuntimeError("Advanced evidence has no valid cached one-time audit")
    return AdvancedResearchPreparation(
        evidence_dir=evidence,
        command=(),
        stage_bindings=stage_bindings,
        promotion_certificate=certificate,
        audit_summary=audit,
    )


def _advanced_paths(project_root: Path, config: Any) -> tuple[Path, Path, float]:
    research = config.research
    evidence_value = str(research.advanced_evidence_dir)
    script_value = str(research.advanced_orchestration_script)
    evidence = _resolve_inside(project_root, evidence_value, "advanced evidence")
    script = _resolve_inside(project_root, script_value, "advanced orchestration script")
    bundled = (project_root / "scripts" / "run_advanced_experiments.py").resolve()
    if script != bundled or not script.is_file():
        raise ValueError("Research orchestration must use the bundled advanced script")
    forbidden = tuple(
        (project_root / value).resolve()
        for value in ("data", "outputs", "src", "tests", "configs", "plan")
    )
    if any(evidence == path or evidence.is_relative_to(path) for path in forbidden):
        raise ValueError(f"Advanced evidence path overlaps a protected project path: {evidence}")
    timeout = float(research.advanced_timeout_seconds)
    if not (timeout > 0):
        raise ValueError("advanced_timeout_seconds must be positive")
    return evidence, script, timeout


def _resolve_inside(root: Path, value: str, label: str) -> Path:
    candidate = Path(value)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError(f"{label.title()} must stay inside the project root: {resolved}")
    return resolved


def _tree_inventory(root: Path) -> tuple[list[dict[str, Any]], str]:
    records: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest = sha256(path.read_bytes()).hexdigest()
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    if not records:
        raise RuntimeError("Advanced evidence tree is empty")
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return records, sha256(encoded).hexdigest()


def _preparation_identity(value: AdvancedResearchPreparation) -> dict[str, Any]:
    return {
        "stage_bindings": value.stage_bindings,
        "promotion_certificate": value.promotion_certificate,
        "audit_summary": value.audit_summary,
    }


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise RuntimeError(f"Advanced publication table is empty: {path}")
    return frame


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Advanced publication JSON is unreadable: {path}") from error
    if not isinstance(value, dict) or not value:
        raise RuntimeError(f"Advanced publication JSON is not a nonempty object: {path}")
    return value


def _gate_table(gate: dict[str, Any]) -> pd.DataFrame:
    checks = gate.get("checks")
    if not isinstance(checks, dict) or not checks:
        raise RuntimeError("Advanced promotion gate has no executable checks")
    evidence = gate.get("metric_evidence", {})
    return pd.DataFrame(
        [
            {
                "check": name,
                "passed": bool(passed),
                "evidence_json": json.dumps(evidence, sort_keys=True, separators=(",", ":")),
            }
            for name, passed in sorted(checks.items())
        ]
    )


__all__ = [
    "AdvancedResearchPreparation",
    "ensure_advanced_research",
    "publish_advanced_research",
]
