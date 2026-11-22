"""Generated research, audit, and model-card documents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def write_analysis_bundle(
    analysis_dir: Path,
    cv_metrics: pd.DataFrame,
    official_metrics: dict[str, dict[str, Any]],
    figure_catalog: list[dict[str, str]],
    comparisons: list[dict[str, Any]],
) -> list[Path]:
    """Write the strict analysis bundle required by the selected skill."""
    analysis_dir.mkdir(parents=True, exist_ok=True)
    report_path = analysis_dir / "analysis-report.md"
    stats_path = analysis_dir / "stats-appendix.md"
    catalog_path = analysis_dir / "figure-catalog.md"

    fold_count = int(cv_metrics["test_year"].nunique()) if not cv_metrics.empty else 0
    model_count = int(cv_metrics["model"].nunique()) if not cv_metrics.empty else 0
    metric_summary = (
        cv_metrics.groupby("model", observed=True)
        .mean(numeric_only=True)
        .drop(columns=["test_year"], errors="ignore")
        .round(6)
        if not cv_metrics.empty
        else pd.DataFrame()
    )
    report = [
        "# Strict Analysis Report",
        "",
        "## Analysis question",
        "",
        "Do leakage-safe probabilistic enhancements improve proper scoring rules across multiple "
        "expanding-origin World Cup folds without degrading mandatory baseline behavior?",
        "",
        "## Comparison unit and evidence",
        "",
        f"- Unit of repeated comparison: held-out tournament edition (n={fold_count}).",
        f"- Compared model families: {model_count}.",
        "- The 2014 edition is excluded from candidate promotion and reported only "
        "as the frozen official test.",
        "- The evaluated candidates are deterministic; research mode records an explicit "
        "repeat-fit probability check.",
        "",
        "## Mean outer-fold metrics",
        "",
        _markdown_table(metric_summary.reset_index()),
        "",
        "## Frozen official test metrics",
        "",
    ]
    for task, metrics in official_metrics.items():
        report.extend([f"### {task}", "", _markdown_table(pd.DataFrame([metrics])), ""])
    report.extend(
        [
            "## Interpretation boundary",
            "",
            "Observed fold differences support model-selection decisions only. They do not "
            "establish a causal mechanism, and seven or fewer historical tournament folds "
            "provide limited inferential power. The supplied data stop in 2014, so 2026 "
            "outputs are explicitly stale-cutoff scenarios.",
        ]
    )
    report_path.write_text("\n".join(report), encoding="utf-8")

    stats = [
        "# Statistical Appendix",
        "",
        "## Protocol",
        "",
        "- Expanding-origin outer folds, never random K-fold.",
        "- Paired Wilcoxon signed-rank comparison across tournament folds when n>=3.",
        "- Signed-rank rank-biserial effect size and exact fold win count.",
        "- No p-value is interpreted as evidence from the single 2014 holdout.",
        "",
        "## Paired comparisons",
        "",
        _markdown_table(pd.DataFrame(comparisons)),
        "",
        "## Assumptions and limitations",
        "",
        "World Cup editions are temporally ordered and heterogeneous. Random-fold estimates "
        "are therefore not used. Calibration is fit on pooled development OOF base "
        "predictions; its application to frozen 2014 is audit-only and is not used for "
        "selection. Development-fit diagnostics are not presented as held-out evidence.",
    ]
    stats_path.write_text("\n".join(stats), encoding="utf-8")

    catalog = ["# Figure Catalog", ""]
    for index, item in enumerate(figure_catalog, start=1):
        catalog.extend(
            [
                f"## Figure {index}: {item['filename']}",
                "",
                f"- **Purpose:** {item['purpose']}",
                f"- **Data source:** {item['data_source']}",
                f"- **Observation:** {item['observation']}",
                f"- **Interpretation:** {item['interpretation']}",
                f"- **Caveat:** {item['caveat']}",
                "",
            ]
        )
    catalog_path.write_text("\n".join(catalog), encoding="utf-8")
    return [report_path, stats_path, catalog_path]


def write_model_card(
    path: Path,
    selected_model: str,
    data_cutoff: str,
    official_metrics: dict[str, dict[str, Any]],
    promotion_decision: str,
) -> Path:
    """Write a bounded model card for judges and downstream users."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Model Card",
        "",
        f"- **Selected outcome system:** {selected_model}",
        f"- **Training data cutoff:** {data_cutoff}",
        "- **Intended use:** reproducible educational/competition analysis of supplied "
        "World Cup data.",
        "- **Not intended for:** betting, live 2026 operations, or claims using "
        "post-2014 team strength.",
        "- **Information sets:** pre-match outcome model; half-time update goal model.",
        f"- **Promotion decision:** {promotion_decision}",
        "",
        "## Frozen official metrics",
        "",
    ]
    for task, metrics in official_metrics.items():
        lines.extend([f"### {task}", "", _markdown_table(pd.DataFrame([metrics])), ""])
    lines.extend(
        [
            "## Known limitations",
            "",
            "The source ends in 2014; historical formats, national-team identities, and "
            "player identifiers are heterogeneous; one 2014 attendance value is missing; "
            "no official 2026 matchup/input table is supplied. "
            "Probabilities and intervals are therefore accompanied by cutoff and OOD metadata.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No comparable observations were available."
    headers = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_format_cell(value) for value in row) + " |")
    return "\n".join(lines)


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")
