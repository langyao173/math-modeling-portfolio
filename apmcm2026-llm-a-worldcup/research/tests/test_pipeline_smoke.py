import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from omegaconf import OmegaConf

import worldcup_forecasting.pipeline as pipeline_module
from worldcup_forecasting.application.research import ResearchRunResult
from worldcup_forecasting.application.runtime import (
    REQUIRED_ARTIFACTS,
    RESEARCH_REQUIRED_ARTIFACTS,
)
from worldcup_forecasting.pipeline import run_pipeline


def test_research_pipeline_completes_advanced_state_machine_before_raw_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "project"
    for relative in ("configs", "data/raw", "data/registry", "data/processed"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    config = OmegaConf.create(
        {
            "project": {"track": "closed_research"},
            "paths": {
                "raw_dir": "data/raw",
                "registry_dir": "data/registry",
                "processed_dir": "data/processed",
                "output_dir": "outputs",
            },
            "runtime": {"clean_outputs": True},
            "research": {
                "advanced_evidence_dir": "temp/advanced_evidence",
                "advanced_orchestration_script": "scripts/run_advanced_experiments.py",
                "advanced_timeout_seconds": 7200,
            },
        }
    )
    events: list[str] = []
    preparation = object()

    def preflight(*args: object, **kwargs: object) -> object:
        events.append("advanced_all_verified")
        return preparation

    bundle = SimpleNamespace(
        tournaments=pd.DataFrame(),
        matches=pd.DataFrame(),
        players=pd.DataFrame(),
        raw_hashes={},
    )

    def raw_read(path: Path) -> object:
        assert events == ["advanced_all_verified"]
        events.append("combined_raw_read")
        return bundle

    research = ResearchRunResult(
        cv_metrics=pd.DataFrame(),
        comparisons=[],
        selected_model="advanced_nested_calibrated",
        promotion_decision="promoted_then_one_time_2014_audit_complete",
        figure_catalog=[],
    )

    def publish(value: object, *args: object, **kwargs: object) -> ResearchRunResult:
        assert value is preparation
        events.append("advanced_evidence_published")
        return research

    def official(*args: object, **kwargs: object) -> SimpleNamespace:
        events.append("official_q1_q3")
        return SimpleNamespace(figure_catalog=[], official_metrics={})

    monkeypatch.setattr(pipeline_module, "load_config", lambda *a, **k: config)
    monkeypatch.setattr(pipeline_module, "ensure_advanced_research", preflight)
    monkeypatch.setattr(pipeline_module, "build_canonical_data", raw_read)
    monkeypatch.setattr(pipeline_module, "publish_advanced_research", publish)
    monkeypatch.setattr(pipeline_module, "run_official_analysis", official)
    monkeypatch.setattr(pipeline_module, "write_audit_artifacts", lambda *a, **k: None)
    monkeypatch.setattr(pipeline_module, "write_analysis_bundle", lambda *a, **k: [])
    monkeypatch.setattr(pipeline_module, "write_model_card", lambda *a, **k: None)
    monkeypatch.setattr(pipeline_module, "REQUIRED_ARTIFACTS", set())
    monkeypatch.setattr(pipeline_module, "RESEARCH_REQUIRED_ARTIFACTS", set())

    result = run_pipeline(
        project_root=root,
        config_name="research",
        output_override=tmp_path / "research-output",
    )

    assert result["status"] == "complete"
    assert events == [
        "advanced_all_verified",
        "combined_raw_read",
        "advanced_evidence_published",
        "official_q1_q3",
    ]


@pytest.mark.slow
def test_one_click_pipeline_writes_complete_manifest(project_root: Path, tmp_path: Path) -> None:
    result = run_pipeline(
        project_root=project_root,
        config_name="official",
        output_override=tmp_path / "outputs",
    )
    manifest_path = Path(result["manifest_path"])
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "tables/problem1_descriptive_statistics.csv",
        "tables/problem1_2018_prediction.csv",
        "tables/problem2_stage_statistics.csv",
        "tables/problem2_metrics.csv",
        "tables/problem3_metrics.csv",
        "figures/problem1_total_goals_trend.png",
        "figures/problem1_goals_per_match_trend.png",
        "figures/problem2_halftime_fulltime_scatter.png",
        "figures/problem3_feature_correlation_heatmap.png",
        "figures/problem3_confusion_matrix.png",
        "predictions/2026_half_time_demo.json",
        "predictions/2026_outcome_demo.json",
        "audit/schema_report.json",
        "audit/data_lineage.json",
        "logs/pipeline.log",
    }
    assert required.issubset(set(manifest["relative_paths"]))
    assert set(manifest["required_paths"]) == REQUIRED_ARTIFACTS
    assert manifest["artifact_profile"] == "official"
    assert manifest["status"] == "complete"


@pytest.mark.slow
def test_research_pipeline_writes_temporal_evidence(
    project_root: Path,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "research_outputs"

    result = run_pipeline(
        project_root=project_root,
        config_name="research",
        output_override=output_dir,
    )

    assert result["status"] == "complete"
    metrics = pd.read_csv(output_dir / "tables" / "research_cv_metrics.csv")
    assert set(metrics["test_year"]) == {1986, 1990, 1994, 1998, 2002, 2006, 2010}
    assert set(metrics["model"]) == {
        "official_outcome_logit",
        "independent_mirrored_poisson",
        "legacy_mirrored_dixon_coles",
        "advanced_nested_raw",
        "advanced_nested_calibrated",
    }
    assert metrics.groupby("model").size().eq(7).all()
    assert metrics[["log_loss", "multiclass_brier"]].notna().all().all()
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["required_paths"]) == (
        REQUIRED_ARTIFACTS | RESEARCH_REQUIRED_ARTIFACTS
    )
    assert manifest["artifact_profile"] == "research"
    stress = json.loads(
        (output_dir / "tables" / "research_stress_tests.json").read_text(encoding="utf-8")
    )
    assert stress["all_structural_checks_passed"] is True
    summary = json.loads(
        (output_dir / "tables" / "research_advanced_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["all_stage_checkpoints_verified"] is True
    assert summary["audit"]["target_reads"] == 1
    assert summary["copy_verified_byte_for_byte"] is True
