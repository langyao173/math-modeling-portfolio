from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from omegaconf import OmegaConf

from worldcup_forecasting.application import advanced_research
from worldcup_forecasting.application.advanced_research import (
    AdvancedResearchPreparation,
    ensure_advanced_research,
    publish_advanced_research,
)
from worldcup_forecasting.validation.stage_orchestration import PREAUDIT_STAGE_ORDER


def _config() -> object:
    return OmegaConf.create(
        {
            "research": {
                "advanced_evidence_dir": "temp/advanced_evidence",
                "advanced_orchestration_script": "scripts/run_advanced_experiments.py",
                "advanced_timeout_seconds": 7200,
            }
        }
    )


def _minimal_evidence(root: Path) -> None:
    (root / "evaluation").mkdir(parents=True)
    (root / "ablation").mkdir()
    (root / "audit").mkdir()
    (root / "promotion").mkdir()
    pd.DataFrame(
        {
            "test_year": [1986, 1986],
            "model": ["advanced_nested_calibrated", "legacy_mirrored_dixon_coles"],
            "log_loss": [0.91, 0.98],
            "multiclass_brier": [0.55, 0.58],
        }
    ).to_csv(root / "evaluation" / "combined_fold_metrics.csv", index=False)
    pd.DataFrame(
        {
            "metric": ["log_loss"],
            "mean_difference": [-0.07],
        }
    ).to_csv(root / "evaluation" / "paired_tournament_intervals.csv", index=False)
    pd.DataFrame(
        {"variant_id": ["A0", "A1"], "log_loss": [0.91, 0.94]}
    ).to_csv(root / "ablation" / "ablation_matrix.csv", index=False)
    pd.DataFrame(
        {"model": ["advanced_nested_calibrated"], "log_loss": [0.89]}
    ).to_csv(root / "audit" / "audit_metrics.csv", index=False)
    (root / "evaluation" / "calibration_metadata.json").write_text(
        json.dumps({"calibration_fit_scope": "nested_inner_oof_only"}), encoding="utf-8"
    )
    (root / "stress_stage_summary.json").write_text(
        json.dumps({"status": "stress_stage_complete", "all_structural_checks_passed": True}),
        encoding="utf-8",
    )
    (root / "promotion" / "promotion_certificate.json").write_text("{}", encoding="utf-8")
    (root / "audit" / "audit_state.json").write_text("{}", encoding="utf-8")
    (root / "audit" / "audit_result_certificate.json").write_text("{}", encoding="utf-8")


def _verified_preparation(evidence: Path) -> AdvancedResearchPreparation:
    return AdvancedResearchPreparation(
        evidence_dir=evidence,
        command=("python", "run_advanced_experiments.py", "--stage", "all"),
        stage_bindings={stage: {"status": "complete"} for stage in PREAUDIT_STAGE_ORDER},
        promotion_certificate={
            "certificate_type": "preaudit_promotion",
            "decision": "promote_to_one_time_audit",
            "gate": {
                "system_promoted": True,
                "candidate": "advanced_nested_calibrated",
                "checks": {"empirical_probability_gate": True},
                "metric_evidence": {"log_loss": {"mean_candidate_minus_baseline": -0.07}},
            },
            "model_identity": {"final_fit_hash": "a" * 64},
            "evidence_bundle_sha256": "b" * 64,
        },
        audit_summary={
            "status": "one_time_2014_audit_complete",
            "target_reads": 1,
            "cached_repeats_only": True,
        },
    )


def test_ensure_advanced_research_uses_cached_audit_without_orchestration_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    script = project / "scripts" / "run_advanced_experiments.py"
    script.parent.mkdir(parents=True)
    script.write_text("# bundled", encoding="utf-8")
    evidence = project / "temp" / "advanced_evidence"
    (evidence / "audit").mkdir(parents=True)
    (evidence / "audit" / "audit_state.json").write_text("{}", encoding="utf-8")
    calls: list[object] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((tuple(command), kwargs))
        return SimpleNamespace(returncode=0, stdout='{"status":"all_stages_complete"}', stderr="")

    verified = _verified_preparation(evidence)

    def fake_verify(path: Path) -> AdvancedResearchPreparation:
        calls.append(("verify", path))
        return verified

    monkeypatch.setattr(advanced_research.subprocess, "run", fake_run)
    monkeypatch.setattr(advanced_research, "_verify_advanced_evidence", fake_verify)

    result = ensure_advanced_research(project, _config(), SimpleNamespace(info=lambda *a: None))

    assert calls == [("verify", evidence.resolve())]
    assert result.evidence_dir == evidence.resolve()
    assert result.command[1:] == (
        str(script.resolve()),
        "--stage",
        "all",
        "--output",
        str(evidence.resolve()),
    )


def test_ensure_advanced_research_runs_bundled_all_when_audit_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    script = project / "scripts" / "run_advanced_experiments.py"
    script.parent.mkdir(parents=True)
    script.write_text("# bundled", encoding="utf-8")
    evidence = project / "temp" / "advanced_evidence"
    evidence.mkdir(parents=True)
    calls: list[object] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((tuple(command), kwargs))
        return SimpleNamespace(returncode=0, stdout='{"status":"all_stages_complete"}', stderr="")

    verified = _verified_preparation(evidence)

    def fake_verify(path: Path) -> AdvancedResearchPreparation:
        calls.append(("verify", path))
        return verified

    monkeypatch.setattr(advanced_research.subprocess, "run", fake_run)
    monkeypatch.setattr(advanced_research, "_verify_advanced_evidence", fake_verify)

    result = ensure_advanced_research(project, _config(), SimpleNamespace(info=lambda *a: None))

    command, kwargs = calls[0]
    assert command[1:] == (
        str(script.resolve()),
        "--stage",
        "all",
        "--output",
        str(evidence.resolve()),
    )
    assert kwargs["cwd"] == project.resolve()
    assert kwargs["check"] is True
    assert calls[1] == ("verify", evidence.resolve())
    assert result.evidence_dir == evidence.resolve()


def test_verify_advanced_evidence_requires_every_stage_certificate_and_cached_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    stage_calls: list[str] = []

    def checkpoint(path: Path, stage: str) -> dict[str, object]:
        assert path == evidence.resolve()
        stage_calls.append(stage)
        return {"stage": stage, "status": "complete", "manifest": {"manifest_sha256": stage}}

    monkeypatch.setattr(advanced_research, "checkpoint_status", checkpoint)
    monkeypatch.setattr(
        advanced_research,
        "verify_promotion_certificate",
        lambda path, revalidate_artifacts: {
            "certificate_type": "preaudit_promotion",
            "decision": "promote_to_one_time_audit",
            "gate": {"system_promoted": True},
            "model_identity": {"final_fit_hash": "a" * 64},
            "evidence_bundle_sha256": "b" * 64,
        },
    )
    monkeypatch.setattr(
        advanced_research,
        "verify_cached_audit",
        lambda path: {
            "status": "one_time_2014_audit_complete",
            "target_reads": 1,
            "cached_repeats_only": True,
        },
    )

    result = advanced_research._verify_advanced_evidence(evidence)

    assert stage_calls == list(PREAUDIT_STAGE_ORDER)
    assert set(result.stage_bindings) == set(PREAUDIT_STAGE_ORDER)


def test_verify_advanced_evidence_fails_closed_on_incomplete_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    monkeypatch.setattr(
        advanced_research,
        "checkpoint_status",
        lambda path, stage: {"stage": stage, "status": "missing"},
    )

    with pytest.raises(RuntimeError, match="not complete"):
        advanced_research._verify_advanced_evidence(evidence)


def test_publish_advanced_research_copies_exact_evidence_and_writes_stable_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = tmp_path / "advanced_evidence"
    output = tmp_path / "staging"
    _minimal_evidence(evidence)
    prepared = _verified_preparation(evidence)
    monkeypatch.setattr(
        advanced_research,
        "_verify_advanced_evidence",
        lambda path: prepared,
    )

    result = publish_advanced_research(
        prepared, output, _config(), SimpleNamespace(info=lambda *a: None)
    )

    copied = output / "evidence" / "advanced"
    assert (copied / "audit" / "audit_metrics.csv").read_bytes() == (
        evidence / "audit" / "audit_metrics.csv"
    ).read_bytes()
    assert pd.read_csv(output / "tables" / "research_cv_metrics.csv").shape == (2, 4)
    assert pd.read_csv(output / "tables" / "research_frozen_2014.csv").shape == (1, 2)
    summary = json.loads(
        (output / "tables" / "research_advanced_summary.json").read_text(encoding="utf-8")
    )
    assert summary["stage_order"] == list(PREAUDIT_STAGE_ORDER)
    assert summary["audit"]["target_reads"] == 1
    assert summary["source_tree_sha256"] == summary["copied_tree_sha256"]
    assert result.selected_model == "advanced_nested_calibrated"
    assert result.promotion_decision == "promoted_then_one_time_2014_audit_complete"


def test_publish_advanced_research_refuses_source_changed_after_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = tmp_path / "advanced_evidence"
    _minimal_evidence(evidence)
    prepared = _verified_preparation(evidence)
    changed = _verified_preparation(evidence)
    changed = replace(
        changed,
        promotion_certificate={
            **changed.promotion_certificate,
            "evidence_bundle_sha256": "c" * 64,
        },
    )
    monkeypatch.setattr(advanced_research, "_verify_advanced_evidence", lambda path: changed)

    with pytest.raises(RuntimeError, match="changed after orchestration"):
        publish_advanced_research(
            prepared,
            tmp_path / "staging",
            _config(),
            SimpleNamespace(info=lambda *a: None),
        )
