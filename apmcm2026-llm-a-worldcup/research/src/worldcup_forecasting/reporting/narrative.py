"""Data-derived summaries with bounded statistical claims."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrendResult:
    """Automated trend and possible structural-break evidence."""

    slope_per_year: float
    correlation: float
    turning_year: int | None
    bic_improvement: float
    conclusion: str


def analyze_trend(frame: pd.DataFrame, value_column: str) -> TrendResult:
    """Fit a global line and a deterministic one-break alternative."""
    years = frame["Year"].to_numpy(dtype=float)
    values = frame[value_column].to_numpy(dtype=float)
    if years.size < 8 or not np.isfinite(values).all():
        raise ValueError("Trend analysis requires at least eight finite observations")
    centered_year = years - years.mean()
    design = np.column_stack([np.ones(years.size), centered_year])
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    fitted = design @ coefficients
    single_sse = float(np.sum((values - fitted) ** 2))
    single_bic = _bic(single_sse, years.size, parameters=2)

    best_year: int | None = None
    best_bic = np.inf
    for split in range(4, years.size - 3):
        hinge = np.maximum(centered_year - centered_year[split], 0.0)
        piecewise = np.column_stack([np.ones(years.size), centered_year, hinge])
        piece_coefficients, *_ = np.linalg.lstsq(piecewise, values, rcond=None)
        residual_sse = float(np.sum((values - piecewise @ piece_coefficients) ** 2))
        candidate_bic = _bic(residual_sse, years.size, parameters=3)
        if candidate_bic < best_bic:
            best_bic = candidate_bic
            best_year = int(years[split])
    improvement = float(single_bic - best_bic)
    accepted_year = best_year if improvement >= 2.0 else None
    correlation = float(np.corrcoef(years, values)[0, 1])
    direction = "increased" if coefficients[1] > 0 else "decreased"
    if accepted_year is None:
        conclusion = (
            f"{value_column} {direction} by {abs(coefficients[1]):.3f} per calendar year "
            f"over the observed editions (Pearson r={correlation:.3f}); no single breakpoint "
            "improved BIC by the pre-specified threshold."
        )
    else:
        conclusion = (
            f"{value_column} {direction} by {abs(coefficients[1]):.3f} per calendar year "
            f"overall (Pearson r={correlation:.3f}). A one-break model selected {accepted_year} "
            f"with BIC improvement {improvement:.2f}; this is descriptive evidence, "
            "not a causal claim."
        )
    return TrendResult(float(coefficients[1]), correlation, accepted_year, improvement, conclusion)


def correlation_strength(correlation: float) -> str:
    """Map absolute Pearson correlation to a declared qualitative scale."""
    magnitude = abs(float(correlation))
    if magnitude < 0.2:
        return "very weak"
    if magnitude < 0.4:
        return "weak"
    if magnitude < 0.6:
        return "moderate"
    if magnitude < 0.8:
        return "strong"
    return "very strong"


def _bic(sse: float, observations: int, parameters: int) -> float:
    safe_sse = max(sse, np.finfo(float).tiny)
    return float(observations * np.log(safe_sse / observations) + parameters * np.log(observations))
