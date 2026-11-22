"""Build manuscript tables and figures from hash-verified research evidence.

The builder is read-only with respect to model outputs.  It keeps frozen development validation,
the one-shot 2014 external audit, and the promoted replacement 2026 format scenario as three
separate estimands.  Superseded tournament outputs can never enter this evidence layer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from worldcup_forecasting.validation.audit_orchestration import verify_cached_audit
from worldcup_forecasting.validation.promotion import verify_promotion_certificate

plt.switch_backend("Agg")


EVIDENCE_RELATIVE_PATHS = {
    "nested_stage": "nested_stage_summary.json",
    "nested_metadata": "nested/nested_metadata.json",
    "final_selection_metadata": "nested/final_selection_metadata.json",
    "frozen_model_identity": "nested/frozen_model_identity.json",
    "candidate_metrics": "nested/candidate_metrics.csv",
    "selected_specs": "nested/selected_specs.csv",
    "raw_oof_ledger": "nested/outer_oof_ledger.csv",
    "evaluation_stage": "evaluation_stage_summary.json",
    "evaluation_lineage": "evaluation/evaluation_lineage.json",
    "combined_metrics": "evaluation/combined_fold_metrics.csv",
    "paired_metrics": "evaluation/paired_tournament_intervals.csv",
    "decision_metrics": "evaluation/decision_fold_metrics.csv",
    "robust_sensitivity": "evaluation/kl_edition_robust_sensitivity.csv",
    "ablation_stage": "ablation_stage_summary.json",
    "ablation_metadata": "ablation/ablation_metadata.json",
    "ablation_summary": "ablation/ablation_summary.csv",
    "ablation_paired": "ablation/ablation_paired_deltas.csv",
    "stress_stage": "stress_stage_summary.json",
    "format_transport_summary": "stress/format_transport_summary.json",
    "format_transport_audits": "stress/format_transport_audits.csv",
    "extreme_support_boundary": "stress/extreme_support_boundary.json",
    "parameter_stage": "parameter_uncertainty_stage_summary.json",
    "parameter_metadata": "uncertainty/parameter_metadata.json",
    "parameter_probability_summary": "uncertainty/parameter_probability_summary.csv",
    "statistical_closure": "evaluation/statistical_closure.json",
    "statistical_fold_comparison": "evaluation/statistical_fold_comparison.csv",
    "statistical_model_ranking": "evaluation/statistical_model_ranking.csv",
    "statistical_classwise_calibration": ("evaluation/statistical_classwise_calibration.csv"),
    "calibrated_oof_ledger": "evaluation/calibrated_oof_ledger.csv",
    "calibrated_joint_grid": "evaluation/calibrated_joint_grid.csv",
    "simulation_stage": "simulation_stage_summary.json",
    "simulation_metadata": "simulation/scenario_metadata.json",
    "simulation_team_summary": "simulation/scenario_team_summary.csv",
    "simulation_stage_counts": "simulation/scenario_stage_summary.csv",
    "adaptive_stopping_ledger": "simulation/adaptive_stopping_ledger.csv",
    "variance_decomposition": "uncertainty/variance_decomposition.json",
    "promotion_certificate": "promotion/promotion_certificate.json",
    "audit_stage": "audit/audit_stage_summary.json",
    "audit_metadata": "audit/audit_metadata.json",
    "audit_metrics": "audit/audit_metrics.csv",
    "audit_external": "audit/2014_external_audit.json",
    "audit_manifest": "audit/audit_artifact_manifest.json",
    "audit_result_certificate": "audit/audit_result_certificate.json",
    "audit_state": "audit/audit_state.json",
    "blind_prediction_manifest": "audit/blind_prediction_manifest.json",
}

SUPERSEDED_EVIDENCE = {
    "former_simulation_2026_format": {
        "status": "superseded_and_excluded",
        "reason": (
            "The superseded engine paired adjacent qualifiers instead of implementing the "
            "official FIFA 2026 Regulations, Article 12 and Annex C match graph."
        ),
        "numerical_results_usable": False,
        "replacement_status": "recomputed_under_promoted_official_annex_c_engine",
    },
    "former_variance_decomposition": {
        "status": "superseded_and_excluded",
        "reason": "The decomposition depended on the withdrawn tournament engine.",
        "numerical_results_usable": False,
        "replacement_status": "recomputed_and_bound_to_replacement_simulation_manifest",
    },
}

DEPRECATED_OUTPUTS = ("tables/pending_evidence.tex",)

MODEL_LABELS = {
    "advanced_nested_calibrated": "Advanced (nested calibrated)",
    "advanced_nested_raw": "Advanced (raw)",
    "legacy_mirrored_dixon_coles": "Legacy mirrored Dixon--Coles",
    "independent_mirrored_poisson": "Independent mirrored Poisson",
    "official_outcome_logit": "Official multinomial logit",
    "advanced_calibrated": "Advanced (frozen calibrated)",
    "advanced_raw": "Advanced (frozen raw)",
    "advanced_cost_sensitive_decision": "Advanced cost-sensitive decision",
}

METRIC_LABELS = {
    "log_loss": "WDL log loss",
    "multiclass_brier": "Multiclass Brier",
    "scoreline_log_loss": "Scoreline NLL",
    "total_goal_crps": "Total-goal CRPS",
}

ABLATION_INTERPRETATIONS = {
    "A0": "Reference configuration.",
    "A1": "Dynamic-state removal is small and unstable across editions.",
    "A2": "Removing player features improves the mean; no retained player-value claim.",
    "A3": "Removing era effects improves the mean; no retained era-benefit claim.",
    "A4": "Exact zero: the selected A0 model had no correlated-count component.",
    "A5": "Exact zero: the selected A0 model had no residual learner.",
    "A6": "Calibration has mixed, near-zero aggregate effects.",
    "A7": "Zero team-effect penalty is much worse; shrinkage is essential here.",
    "A8": "Asymmetric diagnostic degrades scores and violates the structural contract.",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False})


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def load_evidence(project_root: Path) -> dict[str, Any]:
    """Load and validate the frozen evidence used by the manuscript layer."""
    project_root = Path(project_root).resolve()
    evidence_root = project_root / "temp" / "advanced_evidence"
    if not evidence_root.is_dir():
        raise FileNotFoundError(f"advanced evidence directory not found: {evidence_root}")

    evidence: dict[str, Any] = {
        "project_root": project_root,
        "evidence_root": evidence_root,
        "source_paths": {},
    }
    for key, relative in EVIDENCE_RELATIVE_PATHS.items():
        path = evidence_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required paper evidence missing: {path}")
        evidence["source_paths"][key] = path
        evidence[key] = pd.read_csv(path) if path.suffix == ".csv" else _read_json(path)

    evidence["verified_promotion_certificate"] = verify_promotion_certificate(
        evidence["source_paths"]["promotion_certificate"],
        authentication_key=None,
        revalidate_artifacts=True,
    )
    evidence["verified_audit"] = verify_cached_audit(evidence_root)

    _validate_evidence_boundary(evidence)
    return evidence


def _validate_evidence_boundary(evidence: dict[str, Any]) -> None:
    failures: list[str] = []
    if evidence["nested_metadata"].get("maximum_selection_year") != 2010:
        failures.append("nested selection is not frozen at 2010")
    if evidence["final_selection_metadata"].get("audit_rows_loaded") is not False:
        failures.append("final selection reports loaded audit rows")
    if evidence["evaluation_stage"].get("audit_year_loaded") is not False:
        failures.append("evaluation reports loaded audit rows")
    if evidence["ablation_stage"].get("audit_outcomes_used") is not False:
        failures.append("ablation reports used audit outcomes")
    if evidence["parameter_metadata"].get("post_2010_rows_used") is not False:
        failures.append("parameter uncertainty reports post-2010 rows")
    certificate_payload = evidence["verified_promotion_certificate"]
    gate = certificate_payload.get("gate", {})
    if gate.get("system_promoted") is not True:
        failures.append("promotion certificate does not authorize the evidence bundle")
    if evidence["statistical_closure"].get("passed") is not True:
        failures.append("seven-fold statistical closure did not pass")
    if gate.get("simulation_checks", {}).get("passed") is not True:
        failures.append("replacement simulation did not pass all 14 checks")
    if gate.get("uncertainty_checks", {}).get("passed") is not True:
        failures.append("replacement variance decomposition did not pass")
    if evidence["simulation_metadata"].get("factual_2026_entrants_used") is not False:
        failures.append("2026 format scenario is mislabeled as a factual entrant forecast")
    if (
        evidence["variance_decomposition"]
        .get("metadata", {})
        .get("replacement_simulation_manifest_sha256")
        is None
    ):
        failures.append("variance decomposition lacks replacement-simulation lineage")
    audit = evidence["verified_audit"]
    if (
        audit.get("status") != "one_time_2014_audit_complete"
        or audit.get("target_reads") != 1
        or audit.get("selection_reopened") is not False
        or audit.get("cached_repeats_only") is not True
    ):
        failures.append("one-shot 2014 audit state is not final and cached-only")
    audit_metadata = evidence["audit_metadata"]
    if (
        audit_metadata.get("all_predictions_completed_before_target_access") is not True
        or audit_metadata.get("target_columns_exposed_to_predictors") is not False
        or audit_metadata.get("audit_target_used_for_model_choice") is not False
        or audit_metadata.get("advanced_identity", {}).get("final_fit_hash")
        != evidence["frozen_model_identity"].get("final_fit_hash")
    ):
        failures.append("2014 audit is not bound to the frozen blind prediction identity")
    if evidence["audit_external"].get("claim_boundary") != (
        "external_audit_only_not_model_selection"
    ):
        failures.append("2014 audit claim boundary is missing")

    for key, year_column in (
        ("combined_metrics", "test_year"),
        ("ablation_paired", "test_year"),
    ):
        maximum = pd.to_numeric(evidence[key][year_column], errors="raise").max()
        if int(maximum) > 2010:
            failures.append(f"{key} contains a post-2010 scored row")

    if failures:
        raise ValueError("paper evidence boundary failed: " + "; ".join(failures))


def summarize_evaluation(evidence: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Create equal-edition model summaries and paired comparison records."""
    metrics = [
        "accuracy",
        "macro_f1",
        "log_loss",
        "multiclass_brier",
        "scoreline_log_loss",
        "total_goal_crps",
        "calibration_error",
        "recall_draw",
    ]
    combined = _numeric(evidence["combined_metrics"], metrics + ["test_year"])
    rows: list[dict[str, Any]] = []
    for model, group in combined.groupby("model", sort=False):
        record: dict[str, Any] = {
            "model": model,
            "model_label": MODEL_LABELS.get(model, model),
            "folds": int(group["test_year"].nunique()),
        }
        for metric in metrics:
            record[metric] = float(group[metric].mean()) if group[metric].notna().any() else np.nan
        rows.append(record)
    benchmark = pd.DataFrame(rows)

    paired = _numeric(
        evidence["paired_metrics"],
        ["clusters", "replicates", "mean_difference", "median_difference", "ci_low", "ci_high"],
    )
    paired["metric_label"] = paired["metric"].map(METRIC_LABELS)

    candidate = combined.loc[combined["model"] == "advanced_nested_calibrated"]
    baseline = combined.loc[combined["model"] == "legacy_mirrored_dixon_coles"]
    wins: dict[str, int] = {}
    for metric in METRIC_LABELS:
        merged = candidate[["test_year", metric]].merge(
            baseline[["test_year", metric]], on="test_year", suffixes=("_candidate", "_baseline")
        )
        wins[metric] = int((merged[f"{metric}_candidate"] < merged[f"{metric}_baseline"]).sum())
    paired["fold_wins"] = paired["metric"].map(wins).astype(int)
    paired["fold_total"] = 7
    return {"benchmark": benchmark, "paired": paired, "folds": combined}


def summarize_audit(evidence: dict[str, Any]) -> pd.DataFrame:
    """Return the sealed one-shot 2014 metrics without mixing them into development means."""
    metrics = _numeric(
        evidence["audit_metrics"],
        [
            "accuracy",
            "macro_f1",
            "log_loss",
            "multiclass_brier",
            "ranked_probability_score",
            "calibration_error",
            "recall_draw",
            "scoreline_log_loss",
            "total_goal_crps",
            "decision_draw_predictions",
        ],
    )
    if set(metrics["evaluation_role"].astype(str)) != {"audit_only_not_for_selection"}:
        raise ValueError("2014 metrics lost their audit-only role")
    metrics["model_label"] = metrics["model"].map(MODEL_LABELS).fillna(metrics["model"])
    return metrics


def _edition_cluster_interval(
    frame: pd.DataFrame, value_column: str, seed: int = 20260718, replicates: int = 5000
) -> tuple[float, float]:
    edition_means = frame.groupby("test_year", sort=True)[value_column].mean().to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(edition_means, size=(replicates, len(edition_means)), replace=True).mean(
        axis=1
    )
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def summarize_ablation(evidence: dict[str, Any]) -> pd.DataFrame:
    """Return A0--A8 means with edition-cluster intervals and structural eligibility."""
    summary = _numeric(
        evidence["ablation_summary"],
        [
            "paired_evaluations",
            "mean_log_loss",
            "mean_multiclass_brier",
            "mean_delta_log_loss_vs_a0",
            "mean_delta_brier_vs_a0",
            "log_loss_improvement_count_vs_a0",
            "brier_improvement_count_vs_a0",
        ],
    )
    summary["component_active_in_a0"] = _as_bool(summary["component_active_in_a0"])
    summary["structurally_eligible"] = _as_bool(summary["structurally_eligible"])
    summary["zero_effect_by_design"] = _as_bool(summary["zero_effect_by_design"])
    paired = _numeric(
        evidence["ablation_paired"],
        ["test_year", "seed", "delta_log_loss_vs_a0", "delta_brier_vs_a0"],
    )
    paired["structurally_eligible"] = _as_bool(paired["structurally_eligible"])
    paired["zero_effect_by_design"] = _as_bool(paired["zero_effect_by_design"])

    interval_rows: list[dict[str, Any]] = []
    for index, (variant, group) in enumerate(paired.groupby("variant_id", sort=True)):
        ll_low, ll_high = _edition_cluster_interval(
            group, "delta_log_loss_vs_a0", seed=20260718 + index
        )
        br_low, br_high = _edition_cluster_interval(
            group, "delta_brier_vs_a0", seed=20261718 + index
        )
        interval_rows.append(
            {
                "variant_id": variant,
                "delta_log_loss_ci_low": ll_low,
                "delta_log_loss_ci_high": ll_high,
                "delta_brier_ci_low": br_low,
                "delta_brier_ci_high": br_high,
            }
        )
    result = summary.merge(pd.DataFrame(interval_rows), on="variant_id", validate="one_to_one")
    result["interpretation"] = result["variant_id"].map(ABLATION_INTERPRETATIONS)
    return result.sort_values("variant_id").reset_index(drop=True)


def _escape_tex(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _fmt(value: Any, digits: int = 3, signed: bool = False) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):+.{digits}f}" if signed else f"{float(value):.{digits}f}"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_model_table(frame: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabularx}{\linewidth}{@{}Yrrrrr@{}}",
        r"\toprule",
        r"Model & WDL LL & Brier & Score NLL & Goal CRPS & Draw recall \\",
        r"\midrule",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"{_escape_tex(row.model_label)} & {_fmt(row.log_loss)} & "
            f"{_fmt(row.multiclass_brier)} & {_fmt(row.scoreline_log_loss)} & "
            f"{_fmt(row.total_goal_crps)} & {_fmt(row.recall_draw)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    _write_text(path, "\n".join(lines))


def _write_audit_table(frame: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabularx}{\linewidth}{@{}Yrrrrr@{}}",
        r"\toprule",
        r"Model & WDL LL & Brier & Accuracy & Macro-F1 & Draw recall \\",
        r"\midrule",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"{_escape_tex(row.model_label)} & {_fmt(row.log_loss)} & "
            f"{_fmt(row.multiclass_brier)} & {_fmt(row.accuracy)} & "
            f"{_fmt(row.macro_f1)} & {_fmt(row.recall_draw)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    _write_text(path, "\n".join(lines))


def _write_paired_table(frame: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabularx}{\linewidth}{@{}Yrrr@{}}",
        r"\toprule",
        r"Metric & Mean $\Delta$ & 95\% edition-bootstrap CI & Fold wins \\",
        r"\midrule",
    ]
    for row in frame.itertuples(index=False):
        ci = f"[{_fmt(row.ci_low, signed=True)}, {_fmt(row.ci_high, signed=True)}]"
        lines.append(
            f"{_escape_tex(row.metric_label)} & {_fmt(row.mean_difference, signed=True)} & "
            f"{ci} & {row.fold_wins}/{row.fold_total} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    _write_text(path, "\n".join(lines))


def _write_ablation_table(frame: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabularx}{\linewidth}{@{}llrrY@{}}",
        r"\toprule",
        r"ID & Removed component & $\Delta$LL & $\Delta$Brier & Evidence decision \\",
        r"\midrule",
    ]
    for row in frame.itertuples(index=False):
        status = row.interpretation
        if not row.structurally_eligible:
            status = "Structurally ineligible diagnostic. " + status
        lines.append(
            f"{row.variant_id} & {_escape_tex(row.removed_component)} & "
            f"{_fmt(row.mean_delta_log_loss_vs_a0, signed=True)} & "
            f"{_fmt(row.mean_delta_brier_vs_a0, signed=True)} & {_escape_tex(status)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    _write_text(path, "\n".join(lines))


def _write_key_value_table(rows: list[tuple[str, str, str]], path: Path) -> None:
    lines = [
        r"\begin{tabularx}{\linewidth}{@{}lY Y@{}}",
        r"\toprule",
        r"Check & Measured result & Claim boundary \\",
        r"\midrule",
    ]
    for check, result, boundary in rows:
        lines.append(f"{_escape_tex(check)} & {_escape_tex(result)} & {_escape_tex(boundary)} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    _write_text(path, "\n".join(lines))


def _set_figure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save_figure(fig: plt.Figure, base: Path) -> list[Path]:
    paths = [base.with_suffix(".pdf"), base.with_suffix(".png")]
    for path in paths:
        fig.savefig(path, dpi=300)
    plt.close(fig)
    return paths


def _plot_paired_metrics(frame: pd.DataFrame, base: Path) -> list[Path]:
    plot = frame.copy().reset_index(drop=True)
    y = np.arange(len(plot))
    means = plot["mean_difference"].to_numpy(float)
    lower = means - plot["ci_low"].to_numpy(float)
    upper = plot["ci_high"].to_numpy(float) - means
    decisive = plot["ci_high"].to_numpy(float) < 0
    colors = np.where(decisive, "#0072B2", "#666666")

    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    for idx in range(len(plot)):
        ax.errorbar(
            means[idx],
            y[idx],
            xerr=np.array([[lower[idx]], [upper[idx]]]),
            fmt="o" if decisive[idx] else "s",
            color=colors[idx],
            capsize=3,
            linewidth=1.2,
        )
    ax.set_yticks(y, plot["metric_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Advanced calibrated minus legacy baseline (lower is better)")
    ax.set_title("Paired edition-level evidence is metric-dependent")
    ax.grid(axis="x", alpha=0.2)
    return _save_figure(fig, base)


def _plot_ablation(frame: pd.DataFrame, base: Path) -> list[Path]:
    plot = frame.loc[frame["variant_id"] != "A0"].copy().reset_index(drop=True)
    y = np.arange(len(plot))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), sharey=True)
    specifications = [
        (
            "mean_delta_log_loss_vs_a0",
            "delta_log_loss_ci_low",
            "delta_log_loss_ci_high",
            "Log loss",
        ),
        ("mean_delta_brier_vs_a0", "delta_brier_ci_low", "delta_brier_ci_high", "Brier"),
    ]
    for ax, (mean_col, low_col, high_col, title) in zip(axes, specifications, strict=True):
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        for idx, row in plot.iterrows():
            marker = "x" if not bool(row["structurally_eligible"]) else "o"
            color = "#D55E00" if row["variant_id"] in {"A7", "A8"} else "#0072B2"
            mean = float(row[mean_col])
            ax.errorbar(
                mean,
                y[idx],
                xerr=np.array([[mean - float(row[low_col])], [float(row[high_col]) - mean]]),
                fmt=marker,
                color=color,
                capsize=2.5,
                linewidth=1,
            )
        ax.set_title(f"$\\Delta$ {title}")
        ax.set_xlabel("Ablation minus A0 (lower is better)")
        ax.grid(axis="x", alpha=0.2)
    labels = [f"{row.variant_id}: {row.removed_component}" for row in plot.itertuples(index=False)]
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.suptitle("Ablations expose inactive, unsupported, and structurally required components")
    fig.tight_layout()
    return _save_figure(fig, base)


def _plot_format_transport(frame: pd.DataFrame, base: Path) -> list[Path]:
    audits = _numeric(
        frame,
        ["test_year", "direct_total_absolute_error", "offset_absolute_error"],
    ).sort_values("test_year")
    audits["is_team_field_expansion"] = _as_bool(audits["is_team_field_expansion"])
    x = np.arange(len(audits))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.5, 3.1))
    ax.bar(
        x - width / 2,
        audits["direct_total_absolute_error"],
        width,
        label="Direct total extrapolation",
        color="#999999",
    )
    ax.bar(
        x + width / 2,
        audits["offset_absolute_error"],
        width,
        label="Match-volume offset",
        color="#009E73",
    )
    labels = [
        f"{int(year)}{'†' if expansion else ''}"
        for year, expansion in zip(
            audits["test_year"], audits["is_team_field_expansion"], strict=True
        )
    ]
    ax.set_xticks(x, labels)
    ax.set_ylabel("Absolute total-goal error")
    ax.set_xlabel("Historical expansion audit († team field also expanded)")
    ax.set_title("Match-volume separation helps team-field expansions, not every change")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    return _save_figure(fig, base)


def _plot_scenario_advancement(frame: pd.DataFrame, base: Path) -> list[Path]:
    columns = [
        "team",
        "champion_probability",
        "final_probability",
        "semi_final_probability",
        "quarter_final_probability",
    ]
    plot = _numeric(frame[columns], columns[1:]).nlargest(12, "champion_probability")
    plot = plot.sort_values("champion_probability")
    y = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    left = np.zeros(len(plot))
    for column, label, color in (
        ("champion_probability", "Champion", "#D55E00"),
        ("final_probability", "Runner-up", "#E69F00"),
        ("semi_final_probability", "Semi-final exit", "#56B4E9"),
        ("quarter_final_probability", "Quarter-final exit", "#0072B2"),
    ):
        incremental = plot[column].to_numpy(float) - left
        ax.barh(y, incremental, left=left, label=label, color=color)
        left = plot[column].to_numpy(float)
    ax.set_yticks(y, plot["team"].astype(str))
    ax.set_xlabel("Illustrative advancement probability")
    ax.set_title("Replacement official-format scenario: leading profiles")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="x", alpha=0.2)
    return _save_figure(fig, base)


def _plot_variance_decomposition(payload: dict[str, Any], base: Path) -> list[Path]:
    shares = payload["shares"]
    labels = ["Parameter", "Regime | parameter", "Tournament path + match"]
    values = [
        float(shares["parameter"]),
        float(shares["regime_given_parameter"]),
        float(shares["tournament_path_and_match"]),
    ]
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    bars = ax.bar(labels, values, color=["#0072B2", "#E69F00", "#009E73"])
    ax.bar_label(bars, labels=[f"{value:.1%}" for value in values], padding=3)
    ax.set_ylim(0, max(values) * 1.2 + 0.02)
    ax.set_ylabel("Share of total variance")
    ax.set_title("Balanced sensitivity decomposition of tournament total goals")
    ax.grid(axis="y", alpha=0.2)
    return _save_figure(fig, base)


def _plot_fold_improvement(frame: pd.DataFrame, base: Path) -> list[Path]:
    plot = _numeric(
        frame,
        ["test_year", "improvement_log_loss", "improvement_multiclass_brier"],
    ).sort_values("test_year")
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.plot(plot["test_year"], plot["improvement_log_loss"], "o-", label="Log loss")
    ax.plot(
        plot["test_year"],
        plot["improvement_multiclass_brier"],
        "s-",
        label="Brier",
    )
    ax.set_xlabel("Outer test edition")
    ax.set_ylabel("Legacy minus advanced (positive favors advanced)")
    ax.set_title("Seven-fold improvement and temporal heterogeneity")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    return _save_figure(fig, base)


def _calibration_points(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for outcome in ("home_win", "draw", "away_win"):
        probability = pd.to_numeric(frame[outcome], errors="raise")
        target = frame["target_outcome"].astype(str).eq(outcome).astype(float)
        bins = pd.qcut(probability.rank(method="first"), q=10, labels=False)
        for _, index in probability.groupby(bins).groups.items():
            rows.append(
                {
                    "predicted": float(probability.loc[index].mean()),
                    "observed": float(target.loc[index].mean()),
                }
            )
    return pd.DataFrame(rows)


def _plot_calibration(raw: pd.DataFrame, calibrated: pd.DataFrame, base: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    ax.plot([0, 1], [0, 1], color="black", linewidth=0.8, linestyle="--")
    for frame, label, marker in (
        (raw, "Raw", "o"),
        (calibrated, "Nested calibrated", "s"),
    ):
        points = _calibration_points(frame).sort_values("predicted")
        ax.scatter(
            points["predicted"],
            points["observed"],
            s=22,
            alpha=0.7,
            label=label,
            marker=marker,
        )
    ax.set(
        xlabel="Predicted probability",
        ylabel="Observed frequency",
        xlim=(0, 0.85),
        ylim=(0, 0.85),
    )
    ax.set_title("Development OOF reliability (class-pooled deciles)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    return _save_figure(fig, base)


def _plot_joint_heatmap(frame: pd.DataFrame, base: Path) -> list[Path]:
    first = frame.sort_values(["outer_fold_id", "match_id"]).iloc[0]
    block = frame.loc[
        frame["outer_fold_id"].eq(first["outer_fold_id"])
        & frame["match_id"].eq(first["match_id"])
    ]
    matrix = block.pivot(index="home_goals", columns="away_goals", values="probability")
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    image = ax.imshow(matrix.to_numpy(float), origin="lower", cmap="viridis", aspect="auto")
    fig.colorbar(image, ax=ax, label="Probability")
    ax.set_xlabel("Away goals")
    ax.set_ylabel("Home goals")
    ax.set_title(f"Coherent calibrated score grid (match {int(first['match_id'])})")
    return _save_figure(fig, base)


def _plot_runtime_pareto(frame: pd.DataFrame, selected: pd.DataFrame, base: Path) -> list[Path]:
    candidates = _numeric(frame, ["mean_log_loss", "runtime_seconds", "complexity_units"])
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    scatter = ax.scatter(
        candidates["runtime_seconds"],
        candidates["mean_log_loss"],
        c=candidates["complexity_units"],
        cmap="viridis",
        s=24,
        alpha=0.55,
    )
    chosen = _numeric(selected, ["mean_log_loss", "runtime_seconds"])
    ax.scatter(
        chosen["runtime_seconds"],
        chosen["mean_log_loss"],
        marker="x",
        s=70,
        color="#D55E00",
        label="Selected by fold",
    )
    fig.colorbar(scatter, ax=ax, label="Declared complexity units")
    ax.set_xlabel("Frozen runtime reference (s)")
    ax.set_ylabel("Inner mean log loss")
    ax.set_title("Runtime--performance evidence, not live wall-clock timing")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    return _save_figure(fig, base)


def _plot_selection_stability(frame: pd.DataFrame, base: Path) -> list[Path]:
    counts = frame["candidate_id"].astype(str).value_counts().sort_values()
    labels = [f"{value[:8]}..." for value in counts.index]
    fig, ax = plt.subplots(figsize=(5.8, 2.8))
    ax.barh(np.arange(len(counts)), counts.to_numpy(), color="#0072B2")
    ax.set_yticks(np.arange(len(counts)), labels)
    ax.set_xlabel("Outer folds selected")
    ax.set_title("Selection stability across seven development editions")
    ax.set_xlim(0, 7)
    ax.grid(axis="x", alpha=0.2)
    return _save_figure(fig, base)


def _plot_audit(frame: pd.DataFrame, base: Path) -> list[Path]:
    plot = frame.loc[frame["log_loss"].notna()].copy()
    labels = plot["model_label"].astype(str).str.replace("mirrored ", "", regex=False)
    x = np.arange(len(plot))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar(x - width / 2, plot["log_loss"], width, label="Log loss", color="#0072B2")
    ax.bar(x + width / 2, plot["multiclass_brier"], width, label="Brier", color="#E69F00")
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("One-shot 2014 loss (lower is better)")
    ax.set_title("External audit does not reproduce the development superiority")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    return _save_figure(fig, base)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def build_tables(evidence: dict[str, Any], table_dir: Path) -> list[Path]:
    """Write deterministic manuscript tables and a machine-readable evidence snapshot."""
    table_dir.mkdir(parents=True, exist_ok=True)
    evaluation = summarize_evaluation(evidence)
    ablation = summarize_ablation(evidence)
    audit = summarize_audit(evidence)
    benchmark = evaluation["benchmark"]
    paired = evaluation["paired"]

    outputs: list[Path] = []
    for name, frame, writer in (
        ("model_benchmark", benchmark, _write_model_table),
        ("paired_metrics", paired, _write_paired_table),
        ("ablation_summary", ablation, _write_ablation_table),
        ("audit_2014", audit, _write_audit_table),
    ):
        csv_path = table_dir / f"{name}.csv"
        tex_path = table_dir / f"{name}.tex"
        frame.to_csv(csv_path, index=False, lineterminator="\n")
        writer(frame, tex_path)
        outputs.extend([csv_path, tex_path])

    stress = evidence["stress_stage"]
    format_summary = evidence["format_transport_summary"]
    boundary = evidence["extreme_support_boundary"]
    robust_rows = [
        (
            "Structural stress matrix",
            f"{stress['stress_rows']} rows; {stress['scenario_count']} scenarios; all passed",
            "Invariant checks, not predictive accuracy.",
        ),
        (
            "Swap/simplex errors",
            f"max {stress['maximum_symmetry_error']:.2e} / {stress['maximum_simplex_error']:.2e}",
            "Numerical structural certificate.",
        ),
        (
            "Extreme support",
            f"safe refusal at score support {boundary['declared_maximum_score_support']}",
            "Abstention, not a valid extreme forecast.",
        ),
        (
            "All five format changes",
            f"MAE {format_summary['direct_total_mae']:.2f} direct vs "
            f"{format_summary['offset_mae']:.2f} offset",
            "Mixed historical pseudo-transport result.",
        ),
        (
            "Three team-field expansions",
            f"MAE {format_summary['team_field_expansion_direct_mae']:.2f} direct vs "
            f"{format_summary['team_field_expansion_offset_mae']:.2f} offset; 3/3 wins",
            "Historical subgroup; not a 2026 guarantee.",
        ),
    ]
    robust_path = table_dir / "robustness_summary.tex"
    _write_key_value_table(robust_rows, robust_path)
    outputs.append(robust_path)

    simulation = evidence["simulation_stage"]
    teams = _numeric(
        evidence["simulation_team_summary"],
        [
            "champion_probability",
            "champion_mcse",
            "final_probability",
            "expected_group_points",
            "expected_tournament_goals",
        ],
    )
    leader = teams.sort_values("champion_probability", ascending=False).iloc[0]
    scenario_rows = [
        (
            "Replacement simulation",
            f"{simulation['simulations']:,} tournaments; adaptive MCSE thresholds met",
            "Illustrative format scenario, not a factual entrant forecast.",
        ),
        (
            "Monte Carlo precision",
            f"max advancement/champion MCSE "
            f"{simulation['maximum_advancement_mcse']:.4f} / "
            f"{simulation['maximum_champion_mcse']:.4f}",
            "Predeclared adaptive thresholds or maximum-run termination.",
        ),
        (
            "Highest illustrative champion probability",
            f"{leader['team']}: {leader['champion_probability']:.3f}",
            "Scenario-team labels are frozen model-transport profiles.",
        ),
        (
            "Expected tournament goals",
            f"{simulation['mean_tournament_goals']:.2f} "
            f"(MCSE {simulation['mean_tournament_goals_mcse']:.3f})",
            "104-match official-format engine.",
        ),
    ]
    scenario_path = table_dir / "scenario_summary.tex"
    _write_key_value_table(scenario_rows, scenario_path)
    outputs.append(scenario_path)

    decision = _numeric(
        evidence["decision_metrics"],
        [
            "macro_f1_gain",
            "draw_recall_gain",
            "argmax_draw_predictions",
            "decision_draw_predictions",
        ],
    )
    evidence_snapshot = {
        "boundary": {
            "development_rows": int(evidence["nested_stage"]["development_rows"]),
            "outer_oof_rows": int(evidence["nested_stage"]["outer_oof_rows"]),
            "outer_test_years": evidence["nested_metadata"]["outer_test_years"],
            "maximum_selection_year": 2010,
            "advanced_2014_audit_opened": True,
            "simulation_2026_status": "promoted_replacement_format_scenario",
        },
        "estimands": [
            "nested_validation_1986_2010",
            "one_shot_2014_external_audit",
            "illustrative_2026_format_scenario",
        ],
        "model_benchmark": benchmark.to_dict(orient="records"),
        "paired_metrics": paired.to_dict(orient="records"),
        "ablation": ablation.to_dict(orient="records"),
        "decision_layer": {
            "draw_positive_folds": int((decision["decision_draw_predictions"] > 0).sum()),
            "macro_f1_improved_folds": int((decision["macro_f1_gain"] > 0).sum()),
            "mean_macro_f1_gain": float(decision["macro_f1_gain"].mean()),
            "probabilities_modified": False,
        },
        "stress": stress,
        "format_transport": format_summary,
        "parameter_uncertainty": evidence["parameter_stage"],
        "statistical_closure": evidence["statistical_closure"],
        "audit_2014": {
            "stage": evidence["verified_audit"],
            "metadata": evidence["audit_metadata"],
            "metrics": audit.to_dict(orient="records"),
            "claim_boundary": evidence["audit_external"]["claim_boundary"],
        },
        "scenario": {
            "stage": simulation,
            "metadata": evidence["simulation_metadata"],
            "team_summary": teams.to_dict(orient="records"),
            "stopping_ledger": evidence["adaptive_stopping_ledger"].to_dict(orient="records"),
        },
        "variance_decomposition": evidence["variance_decomposition"],
        "superseded_evidence": SUPERSEDED_EVIDENCE,
    }
    evidence_path = table_dir / "paper_evidence.json"
    _write_text(evidence_path, json.dumps(_json_safe(evidence_snapshot), indent=2, sort_keys=True))
    outputs.append(evidence_path)
    return outputs


def build_figures(evidence: dict[str, Any], figure_dir: Path) -> list[Path]:
    """Write vector and raster versions of the claim-bearing figures."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    _set_figure_style()
    evaluation = summarize_evaluation(evidence)
    ablation = summarize_ablation(evidence)
    outputs: list[Path] = []
    outputs.extend(_plot_paired_metrics(evaluation["paired"], figure_dir / "paired_metrics"))
    outputs.extend(
        _plot_fold_improvement(
            evidence["statistical_fold_comparison"], figure_dir / "fold_improvement"
        )
    )
    outputs.extend(
        _plot_calibration(
            evidence["raw_oof_ledger"],
            evidence["calibrated_oof_ledger"],
            figure_dir / "calibration_curve",
        )
    )
    outputs.extend(
        _plot_joint_heatmap(evidence["calibrated_joint_grid"], figure_dir / "joint_score_heatmap")
    )
    outputs.extend(_plot_ablation(ablation, figure_dir / "ablation_effects"))
    outputs.extend(
        _plot_format_transport(evidence["format_transport_audits"], figure_dir / "format_transport")
    )
    outputs.extend(
        _plot_scenario_advancement(
            evidence["simulation_team_summary"], figure_dir / "scenario_advancement"
        )
    )
    outputs.extend(
        _plot_variance_decomposition(
            evidence["variance_decomposition"], figure_dir / "uncertainty_decomposition"
        )
    )
    outputs.extend(
        _plot_runtime_pareto(
            evidence["candidate_metrics"], evidence["selected_specs"], figure_dir / "runtime_pareto"
        )
    )
    outputs.extend(
        _plot_selection_stability(evidence["selected_specs"], figure_dir / "selection_stability")
    )
    outputs.extend(_plot_audit(summarize_audit(evidence), figure_dir / "audit_2014_comparison"))
    return outputs


def citation_keys_from_tex(tex: str) -> set[str]:
    """Extract comma-separated keys from common natbib citation commands."""
    keys: set[str] = set()
    pattern = re.compile(r"\\cite\w*\s*(?:\[[^\]]*\]\s*){0,2}\{([^}]*)\}")
    for match in pattern.finditer(tex):
        keys.update(key.strip() for key in match.group(1).split(",") if key.strip())
    return keys


def bib_keys_from_text(bib: str) -> set[str]:
    """Extract entry keys from a BibTeX database."""
    return set(re.findall(r"@[A-Za-z]+\s*\{\s*([^,\s]+)\s*,", bib))


def assert_citations_resolve(tex: str, bib: str) -> None:
    """Raise when the manuscript contains a citation absent from the bibliography."""
    missing = sorted(citation_keys_from_tex(tex) - bib_keys_from_text(bib))
    if missing:
        raise ValueError("unresolved citation keys: " + ", ".join(missing))


def build_all(project_root: Path, output_root: Path) -> dict[str, Any]:
    """Build every manuscript-facing artifact and a deterministic hash manifest."""
    project_root = Path(project_root).resolve()
    output_root = Path(output_root).resolve()
    for relative in DEPRECATED_OUTPUTS:
        deprecated = output_root / relative
        if deprecated.is_file():
            deprecated.unlink()
    evidence = load_evidence(project_root)
    paths = build_tables(evidence, output_root / "tables")
    paths.extend(build_figures(evidence, output_root / "figures"))
    artifacts = [
        {
            "path": path.relative_to(output_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(paths)
    ]
    source_hashes = {
        key: {
            "path": path.relative_to(project_root).as_posix(),
            "sha256": _sha256(path),
        }
        for key, path in sorted(evidence["source_paths"].items())
    }
    manifest = {
        "schema_version": 1,
        "builder": "scripts/build_paper_artifacts.py",
        "evidence_boundary": {
            "maximum_selection_year": 2010,
            "advanced_audit_opened": True,
            "audit_target_reads": int(evidence["verified_audit"]["target_reads"]),
            "simulation_2026_status": "promoted_replacement_format_scenario",
        },
        "superseded_evidence": SUPERSEDED_EVIDENCE,
        "source_hashes": source_hashes,
        "artifacts": artifacts,
    }
    _write_text(
        output_root / "artifact_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True),
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root containing temp/advanced_evidence.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Artifact root (default: <project-root>/doc).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_root = args.output_root or args.project_root / "doc"
    manifest = build_all(args.project_root, output_root)
    print(
        json.dumps(
            {
                "status": "paper_artifacts_built",
                "artifact_count": len(manifest["artifacts"]),
                "output_root": str(output_root.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
