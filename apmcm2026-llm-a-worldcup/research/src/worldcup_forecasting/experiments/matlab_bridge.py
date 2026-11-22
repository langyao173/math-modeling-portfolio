"""Validate retained MATLAB R2026a evidence against Python data oracles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from worldcup_forecasting.data import CanonicalDataBundle, parse_attendance
from worldcup_forecasting.reporting.tables import descriptive_statistics, stage_statistics

_ADVANCED_OUTER_YEARS = (1986, 1990, 1994, 1998, 2002, 2006, 2010)


def verify_advanced_matlab_artifacts(
    matlab_dir: str | Path,
    report_path: str | Path | None = None,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Validate the independent advanced MATLAB prototype and its bounded claim."""
    directory = Path(matlab_dir)
    metrics_path = directory / "advanced_prototype_metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Required advanced MATLAB artifact is missing: {metrics_path}")
    metrics = pd.read_csv(metrics_path)
    invariants = _read_json(directory / "advanced_theory_invariants.json")
    card = _read_json(directory / "advanced_experiment_card.json")
    required_metrics = {"test_year", "log_loss", "multiclass_brier", "accuracy", "n"}
    missing = sorted(required_metrics.difference(metrics.columns))
    if missing:
        raise ValueError(f"Advanced MATLAB metrics are missing columns: {missing}")
    years = pd.to_numeric(metrics["test_year"], errors="coerce")
    if years.isna().any() or tuple(years.astype(int)) != _ADVANCED_OUTER_YEARS:
        raise ValueError("Advanced MATLAB outer years must be 1986-2010; 2014 audit is forbidden")
    numeric_columns = ["log_loss", "multiclass_brier", "accuracy", "n"]
    numeric = metrics[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Advanced MATLAB metrics must be finite")
    if (
        (numeric[["log_loss", "multiclass_brier", "n"]] < 0).any().any()
        or not numeric["accuracy"].between(0, 1).all()
        or (numeric["n"] <= 0).any()
    ):
        raise ValueError("Advanced MATLAB metrics are outside valid ranges")
    invariant_fields = (
        "swap_error",
        "rate_product_range",
        "kl_margin_error",
        "kl_swap_error",
    )
    try:
        invariant_values = {name: float(invariants[name]) for name in invariant_fields}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Advanced MATLAB invariant fields are incomplete or nonnumeric") from exc
    if not np.isfinite(list(invariant_values.values())).all():
        raise ValueError("Advanced MATLAB invariants must be finite")
    errors = [
        abs(invariant_values["swap_error"]),
        abs(invariant_values["kl_margin_error"]),
        abs(invariant_values["kl_swap_error"]),
    ]
    if max(errors) > tolerance or invariant_values["rate_product_range"] <= tolerance:
        raise ValueError("Advanced MATLAB structural invariants failed")
    if invariants.get("all_passed") is not True:
        raise ValueError("Advanced MATLAB all_passed flag is false")
    claim_scope = invariants.get("claim_scope")
    if claim_scope != "independent prototype; not production evidence":
        raise ValueError("Advanced MATLAB claim scope must remain independent/non-production")
    if card.get("production_dependency") is not False:
        raise ValueError("Advanced MATLAB prototype cannot be a production dependency")
    runtime = card.get("runtime_seconds")
    if not isinstance(runtime, (int, float)) or not np.isfinite(runtime) or runtime < 0:
        raise ValueError("Advanced MATLAB experiment card requires finite nonnegative runtime")
    if not isinstance(card.get("experiment_id"), str) or not card["experiment_id"].strip():
        raise ValueError("Advanced MATLAB experiment card requires an experiment_id")
    report: dict[str, Any] = {
        "status": "passed",
        "outer_years": list(_ADVANCED_OUTER_YEARS),
        "fold_count": len(_ADVANCED_OUTER_YEARS),
        "mean_log_loss": float(numeric["log_loss"].mean()),
        "mean_multiclass_brier": float(numeric["multiclass_brier"].mean()),
        "mean_accuracy": float(numeric["accuracy"].mean()),
        "maximum_invariant_error": float(max(errors)),
        "rate_product_range": invariant_values["rate_product_range"],
        "matlab_release": invariants.get("matlab_release"),
        "runtime_seconds": float(runtime),
        "production_dependency": False,
        "claim_scope": claim_scope,
    }
    if report_path is not None:
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return report


def verify_matlab_parity(
    bundle: CanonicalDataBundle,
    matlab_dir: str | Path,
    report_path: str | Path | None = None,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Compare a retained MATLAB experiment with canonical Python results.

    MATLAB output is evidence only. This function reads it after execution and
    fails the report if any frozen data/statistics contract differs.
    """
    directory = Path(matlab_dir)
    summary = _read_json(directory / "parity_summary.json")
    card = _read_json(directory / "experiment_card.json")
    matlab_descriptive = pd.read_csv(directory / "parity_descriptive_statistics.csv")
    matlab_stages = pd.read_csv(directory / "parity_stage_statistics.csv")
    python_descriptive = descriptive_statistics(bundle.tournaments)
    python_stages = stage_statistics(bundle.matches)

    checks: list[dict[str, Any]] = []
    numeric_errors: list[float] = []

    def exact(name: str, expected: Any, observed: Any) -> None:
        passed = expected == observed
        checks.append({"name": name, "expected": expected, "observed": observed, "passed": passed})
        if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
            numeric_errors.append(abs(float(expected) - float(observed)))

    exact(
        "raw_match_rows",
        int(bundle.data_lineage["inputs"]["matches"]["rows"]),
        int(summary["raw_match_rows"]),
    )
    exact("canonical_match_rows", len(bundle.matches), int(summary["canonical_match_rows"]))
    exact(
        "raw_player_rows",
        int(bundle.data_lineage["inputs"]["players"]["rows"]),
        int(summary["raw_player_rows"]),
    )
    exact("canonical_player_rows", len(bundle.players), int(summary["canonical_player_rows"]))
    matches_2014 = bundle.matches.loc[bundle.matches["Year"] == 2014]
    exact("matches_2014", len(matches_2014), int(summary["matches_2014"]))
    exact(
        "goals_2014",
        int(matches_2014["full_time_total_goals"].sum()),
        int(summary["goals_2014"]),
    )
    exact(
        "attendance_2014",
        int(matches_2014["Attendance"].sum(skipna=True)),
        int(summary["attendance_2014"]),
    )
    expected_examples = [
        parse_attendance("590.549"),
        parse_attendance("1.045.246"),
        parse_attendance("3.386.810"),
    ]
    exact(
        "attendance_parse_examples",
        expected_examples,
        [int(value) for value in summary["attendance_parse_examples"]],
    )
    exact("matlab_not_production_dependency", False, bool(card["production_dependency"]))

    correlation = float(
        bundle.matches[["half_time_total_goals", "full_time_total_goals"]].corr().iloc[0, 1]
    )
    correlation_error = abs(correlation - float(summary["halftime_fulltime_correlation"]))
    numeric_errors.append(correlation_error)
    checks.append(
        {
            "name": "halftime_fulltime_correlation",
            "expected": correlation,
            "observed": float(summary["halftime_fulltime_correlation"]),
            "absolute_error": correlation_error,
            "passed": correlation_error <= tolerance,
        }
    )

    numeric_errors.extend(
        _table_check(
            "descriptive_statistics",
            python_descriptive,
            matlab_descriptive,
            key="indicator",
            numeric_columns=("mean", "median", "minimum", "maximum", "sample_std", "n"),
            tolerance=tolerance,
            checks=checks,
        )
    )
    numeric_errors.extend(
        _table_check(
            "stage_statistics",
            python_stages,
            matlab_stages,
            key="canonical_stage",
            numeric_columns=("matches", "goal_sum", "mean_goals"),
            tolerance=tolerance,
            checks=checks,
        )
    )
    maximum_error = max(numeric_errors, default=0.0)
    report: dict[str, Any] = {
        "status": "passed" if all(check["passed"] for check in checks) else "failed",
        "tolerance": float(tolerance),
        "maximum_absolute_numeric_error": float(maximum_error),
        "checks": checks,
        "matlab_version": card.get("matlab_version"),
        "production_dependency": False,
    }
    if report_path is not None:
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return report


def _table_check(
    name: str,
    expected: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    key: str,
    numeric_columns: tuple[str, ...],
    tolerance: float,
    checks: list[dict[str, Any]],
) -> list[float]:
    left = expected.set_index(key).sort_index()
    right = observed.set_index(key).sort_index()
    same_keys = left.index.tolist() == right.index.tolist()
    errors: list[float] = []
    if same_keys:
        difference = np.abs(
            left.loc[:, numeric_columns].to_numpy(dtype=float)
            - right.loc[:, numeric_columns].to_numpy(dtype=float)
        )
        errors = difference.ravel().tolist()
        maximum = float(difference.max(initial=0.0))
    else:
        maximum = float("inf")
    checks.append(
        {
            "name": name,
            "same_keys": same_keys,
            "maximum_absolute_error": maximum,
            "passed": same_keys and maximum <= tolerance,
        }
    )
    return errors


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required MATLAB artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"MATLAB JSON artifact must contain an object: {path}")
    return value
