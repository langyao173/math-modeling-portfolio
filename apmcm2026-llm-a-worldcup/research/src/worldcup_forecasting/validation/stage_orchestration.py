"""Strict, resumable ordering for pre-audit and one-time-audit stages."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ._promotion_artifacts import checkpoint_status
from .promotion import verify_promotion_certificate

PREAUDIT_STAGE_ORDER = (
    "nested",
    "evaluate",
    "stress",
    "ablation",
    "parameter",
    "simulate",
    "decompose",
)


def run_preaudit_pipeline(
    output_root: Path,
    *,
    stage_runners: Mapping[str, Callable[[Path], dict[str, Any]]],
    promote: Callable[[Path], dict[str, Any]],
    checkpoint: Callable[[Path, str], dict[str, Any]] = checkpoint_status,
    verify_certificate: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resume only verified checkpoints and stop on any partial/corrupt stage."""
    root = output_root.resolve()
    if set(stage_runners) != set(PREAUDIT_STAGE_ORDER):
        raise ValueError("Pre-audit runner map must contain the seven exact ordered stages")
    stage_results: dict[str, Any] = {}
    for stage in PREAUDIT_STAGE_ORDER:
        before = checkpoint(root, stage)
        if before.get("status") == "complete":
            stage_results[stage] = {"status": "verified_checkpoint_reused"}
            continue
        if before.get("status") != "missing":
            raise RuntimeError(f"Unexpected checkpoint state for {stage}: {before!r}")
        stage_results[stage] = stage_runners[stage](root)
        after = checkpoint(root, stage)
        if after.get("status") != "complete":
            raise RuntimeError(f"Stage {stage!r} did not produce a verifiable checkpoint")

    certificate_path = root / "promotion" / "promotion_certificate.json"
    if certificate_path.is_file():
        verifier = verify_certificate or (
            lambda path: verify_promotion_certificate(path, revalidate_artifacts=True)
        )
        certificate = verifier(certificate_path)
        promotion_status = "verified_certificate_reused"
    else:
        if (root / "promotion" / "promotion_rejection.json").exists():
            raise RuntimeError(
                "A prior signed/hash-bound promotion rejection exists; evidence must not be rerun"
            )
        certificate = promote(root)
        promotion_status = "certificate_issued"
    return {
        "status": "preaudit_complete_audit_authorized",
        "stage_order": list(PREAUDIT_STAGE_ORDER),
        "stage_results": stage_results,
        "promotion_status": promotion_status,
        "promotion_certificate_type": _certificate_type(certificate),
        "audit_executed": False,
    }


def run_all_pipeline(
    output_root: Path,
    *,
    preaudit: Callable[[Path], dict[str, Any]],
    audit: Callable[[Path], dict[str, Any]],
) -> dict[str, Any]:
    """Run verified pre-audit first, then exactly one audit (or cached verification)."""
    root = output_root.resolve()
    preaudit_result = preaudit(root)
    if preaudit_result.get("status") != "preaudit_complete_audit_authorized":
        raise RuntimeError("Audit cannot start before a complete authorized pre-audit")
    audit_result = audit(root)
    return {
        "status": "all_stages_complete",
        "preaudit": preaudit_result,
        "audit": audit_result,
    }


def _certificate_type(certificate: dict[str, Any]) -> str | None:
    if "payload" in certificate and isinstance(certificate["payload"], dict):
        return certificate["payload"].get("certificate_type")
    return certificate.get("certificate_type")


__all__ = ["PREAUDIT_STAGE_ORDER", "run_all_pipeline", "run_preaudit_pipeline"]
