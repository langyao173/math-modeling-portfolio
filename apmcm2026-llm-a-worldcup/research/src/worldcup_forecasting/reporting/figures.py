"""Publication-quality, program-generated competition figures."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]


def configure_style() -> None:
    """Apply a compact, colorblind-safe scientific style."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )


def plot_tournament_trend(
    frame: pd.DataFrame,
    value_column: str,
    ylabel: str,
    stem: Path,
    dpi: int = 300,
) -> list[Path]:
    """Plot one edition-level chronological trend to PNG and PDF."""
    configure_style()
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.plot(
        frame["Year"],
        frame[value_column],
        color=OKABE_ITO[0],
        marker="o",
        markersize=4.5,
        linewidth=1.8,
    )
    axis.set(xlabel="Tournament year", ylabel=ylabel)
    axis.grid(alpha=0.25)
    return _save_both(figure, stem, dpi)


def plot_halftime_fulltime(frame: pd.DataFrame, stem: Path, dpi: int = 300) -> list[Path]:
    """Plot half-time against full-time totals with an OLS guide."""
    configure_style()
    x = frame["half_time_total_goals"].to_numpy(dtype=float)
    y = frame["full_time_total_goals"].to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(6.2, 4.5))
    axis.scatter(x, y, s=18, alpha=0.35, color=OKABE_ITO[0], edgecolors="none")
    slope, intercept = np.polyfit(x, y, 1)
    grid = np.linspace(x.min(), x.max(), 100)
    axis.plot(grid, intercept + slope * grid, color=OKABE_ITO[1], linewidth=2)
    axis.set(xlabel="Half-time total goals", ylabel="Full-time total goals")
    axis.grid(alpha=0.25)
    return _save_both(figure, stem, dpi)


def plot_correlation_heatmap(correlation: pd.DataFrame, stem: Path, dpi: int = 300) -> list[Path]:
    """Plot the feature/outcome correlation matrix."""
    configure_style()
    width = max(6.5, correlation.shape[1] * 0.75)
    height = max(4.5, correlation.shape[0] * 0.58)
    figure, axis = plt.subplots(figsize=(width, height))
    sns.heatmap(
        correlation,
        vmin=-1,
        vmax=1,
        center=0,
        cmap="vlag",
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        ax=axis,
    )
    axis.set_xticklabels(axis.get_xticklabels(), rotation=35, ha="right")
    return _save_both(figure, stem, dpi)


def plot_confusion(
    matrix: np.ndarray,
    labels: Sequence[str],
    stem: Path,
    dpi: int = 300,
) -> list[Path]:
    """Plot a labeled confusion matrix."""
    configure_style()
    figure, axis = plt.subplots(figsize=(5.3, 4.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=axis,
    )
    axis.set(xlabel="Predicted", ylabel="Observed")
    return _save_both(figure, stem, dpi)


def plot_model_comparison(
    table: pd.DataFrame,
    metric: str,
    stem: Path,
    dpi: int = 300,
) -> list[Path]:
    """Plot fold-level model scores with visible observations."""
    configure_style()
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    sns.boxplot(data=table, x="model", y=metric, color=OKABE_ITO[5], ax=axis)
    sns.stripplot(data=table, x="model", y=metric, color="#000000", alpha=0.65, ax=axis)
    axis.set(xlabel="Model", ylabel=metric.replace("_", " ").title())
    axis.tick_params(axis="x", rotation=20)
    return _save_both(figure, stem, dpi)


def plot_reliability(
    labels: np.ndarray,
    probabilities: np.ndarray,
    stem: Path,
    dpi: int = 300,
    bins: int = 8,
) -> list[Path]:
    """Plot top-label reliability with bin counts."""
    configure_style()
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correctness = predicted == labels
    edges = np.linspace(0, 1, bins + 1)
    rows: list[tuple[float, float, int]] = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            rows.append(
                (float(confidence[mask].mean()), float(correctness[mask].mean()), int(mask.sum()))
            )
    figure, axis = plt.subplots(figsize=(5.2, 4.5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="#666666", label="Perfect calibration")
    if rows:
        x, y, counts = zip(*rows, strict=True)
        axis.scatter(x, y, s=np.asarray(counts) * 8 + 18, color=OKABE_ITO[0], alpha=0.8)
        axis.plot(x, y, color=OKABE_ITO[0], linewidth=1.5)
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Empirical accuracy")
    axis.legend(loc="upper left")
    return _save_both(figure, stem, dpi)


def _save_both(figure: plt.Figure, stem: Path, dpi: int) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    figure.savefig(png, dpi=dpi)
    figure.savefig(pdf)
    plt.close(figure)
    return [png, pdf]
