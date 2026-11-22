"""Tests for the evidence-bound manuscript artifact builder."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_paper_artifacts import (  # noqa: E402
    assert_citations_resolve,
    build_all,
    load_evidence,
    summarize_ablation,
    summarize_audit,
    summarize_evaluation,
)


def test_load_evidence_enforces_frozen_development_boundary() -> None:
    """Development, one-shot audit, and scenario evidence must remain separate."""
    evidence = load_evidence(PROJECT_ROOT)

    assert evidence["nested_stage"]["outer_oof_rows"] == 412
    assert evidence["nested_metadata"]["maximum_selection_year"] == 2010
    assert evidence["final_selection_metadata"]["audit_rows_loaded"] is False
    assert evidence["evaluation_stage"]["audit_year_loaded"] is False
    assert evidence["ablation_stage"]["audit_outcomes_used"] is False
    assert evidence["simulation_metadata"]["factual_2026_entrants_used"] is False
    assert evidence["simulation_stage"]["scenario_only_not_factual_forecast"] is True
    assert evidence["statistical_closure"]["passed"] is True
    assert evidence["variance_decomposition"]["metadata"]["replacement_simulation_manifest_sha256"]
    assert evidence["verified_audit"]["target_reads"] == 1
    assert evidence["verified_audit"]["selection_reopened"] is False


def test_evaluation_summary_preserves_mixed_joint_score_evidence() -> None:
    """WDL gains must not be generalized to every joint-distribution metric."""
    summary = summarize_evaluation(load_evidence(PROJECT_ROOT))
    paired = summary["paired"].set_index("metric")

    assert paired.loc["log_loss", "mean_difference"] < 0
    assert paired.loc["multiclass_brier", "mean_difference"] < 0
    assert (
        paired.loc["scoreline_log_loss", "ci_low"] < 0 < paired.loc["scoreline_log_loss", "ci_high"]
    )
    assert paired.loc["total_goal_crps", "mean_difference"] > 0
    assert paired.loc["total_goal_crps", "ci_low"] < 0 < paired.loc["total_goal_crps", "ci_high"]


def test_ablation_summary_preserves_zero_and_negative_results() -> None:
    """Inactive and structurally invalid components require explicit labels."""
    ablation = summarize_ablation(load_evidence(PROJECT_ROOT)).set_index("variant_id")

    assert ablation.loc["A4", "mean_delta_log_loss_vs_a0"] == pytest.approx(0.0)
    assert ablation.loc["A5", "mean_delta_log_loss_vs_a0"] == pytest.approx(0.0)
    assert bool(ablation.loc["A4", "component_active_in_a0"]) is False
    assert bool(ablation.loc["A5", "component_active_in_a0"]) is False
    assert ablation.loc["A7", "mean_delta_log_loss_vs_a0"] > 0.14
    assert "shrinkage is essential" in ablation.loc["A7", "interpretation"]
    assert bool(ablation.loc["A8", "structurally_eligible"]) is False


def test_one_shot_audit_preserves_adverse_external_result() -> None:
    audit = summarize_audit(load_evidence(PROJECT_ROOT)).set_index("model")

    assert audit.loc["advanced_calibrated", "log_loss"] > audit.loc[
        "legacy_mirrored_dixon_coles", "log_loss"
    ]
    assert audit.loc["advanced_calibrated", "multiclass_brier"] > audit.loc[
        "legacy_mirrored_dixon_coles", "multiclass_brier"
    ]
    assert audit.loc["advanced_cost_sensitive_decision", "recall_draw"] > 0


def test_citation_audit_detects_missing_bibliography_keys() -> None:
    """Every LaTeX citation key must resolve in the bibliography."""
    tex = r"Established count models \citep{maher1982,dixoncoles1997}."
    bib = "@article{maher1982, title={A}}\n@article{dixoncoles1997, title={B}}"
    assert_citations_resolve(tex, bib)

    with pytest.raises(ValueError, match="missing_key"):
        assert_citations_resolve(tex + r" \citet{missing_key}", bib)


@pytest.mark.integration
def test_build_all_generates_reproducible_tables_figures_and_manifest(tmp_path: Path) -> None:
    """The builder produces all manuscript-facing artifacts from frozen evidence."""
    manifest = build_all(PROJECT_ROOT, tmp_path)
    required = {
        "tables/model_benchmark.csv",
        "tables/model_benchmark.tex",
        "tables/paired_metrics.csv",
        "tables/paired_metrics.tex",
        "tables/ablation_summary.csv",
        "tables/ablation_summary.tex",
        "tables/robustness_summary.tex",
        "tables/scenario_summary.tex",
        "tables/audit_2014.csv",
        "tables/audit_2014.tex",
        "tables/paper_evidence.json",
        "figures/paired_metrics.pdf",
        "figures/paired_metrics.png",
        "figures/ablation_effects.pdf",
        "figures/ablation_effects.png",
        "figures/format_transport.pdf",
        "figures/format_transport.png",
        "figures/scenario_advancement.pdf",
        "figures/scenario_advancement.png",
        "figures/uncertainty_decomposition.pdf",
        "figures/uncertainty_decomposition.png",
        "figures/fold_improvement.pdf",
        "figures/fold_improvement.png",
        "figures/calibration_curve.pdf",
        "figures/calibration_curve.png",
        "figures/joint_score_heatmap.pdf",
        "figures/joint_score_heatmap.png",
        "figures/runtime_pareto.pdf",
        "figures/runtime_pareto.png",
        "figures/selection_stability.pdf",
        "figures/selection_stability.png",
        "figures/audit_2014_comparison.pdf",
        "figures/audit_2014_comparison.png",
    }

    artifact_paths = {entry["path"] for entry in manifest["artifacts"]}
    assert required <= artifact_paths
    assert "tables/pending_evidence.tex" not in artifact_paths
    assert manifest["evidence_boundary"]["maximum_selection_year"] == 2010
    assert manifest["evidence_boundary"]["advanced_audit_opened"] is True
    assert manifest["evidence_boundary"]["audit_target_reads"] == 1
    assert manifest["evidence_boundary"]["simulation_2026_status"] == (
        "promoted_replacement_format_scenario"
    )
    assert (
        manifest["superseded_evidence"]["former_simulation_2026_format"]["numerical_results_usable"]
        is False
    )
    assert all(entry["sha256"] for entry in manifest["artifacts"])

    persisted = json.loads((tmp_path / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert persisted == manifest

    snapshot = json.loads((tmp_path / "tables" / "paper_evidence.json").read_text(encoding="utf-8"))
    assert snapshot["scenario"]["stage"]["scenario_only_not_factual_forecast"] is True
    assert snapshot["boundary"]["advanced_2014_audit_opened"] is True
    assert snapshot["audit_2014"]["stage"]["target_reads"] == 1
    assert snapshot["audit_2014"]["stage"]["selection_reopened"] is False
    assert snapshot["estimands"] == [
        "nested_validation_1986_2010",
        "one_shot_2014_external_audit",
        "illustrative_2026_format_scenario",
    ]
    assert snapshot["variance_decomposition"]["metadata"]["replacement_simulation_manifest_sha256"]
    assert snapshot["superseded_evidence"] == manifest["superseded_evidence"]
