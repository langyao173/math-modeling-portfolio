"""Required figures with deterministic Chinese-font detection and fallback."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from scipy import stats


BLUE = "#2f6f9f"
ORANGE = "#d97732"
PURPLE = "#5b3a8c"
NAVY = "#17324d"
GRAY = "#7b8794"
LIGHT_GRAY = "#e8ebef"
GRID_ALPHA = 0.16
DPI = 240

# Orange–white–purple diverging map (corrgram reference style).
CORR_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "apmcm_corr",
    ["#c45c26", "#f3e6dc", "#ffffff", "#d9cce8", "#5b3a8c"],
)
VIF_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "apmcm_vif",
    [PURPLE, ORANGE],
)

_FEATURE_LABELS_ZH = {
    "TournamentExperienceDiff": "参赛次数差",
    "BestStageDiff": "最佳成绩差",
    "HistAvgGFDiff": "场均进球差",
    "HistAvgGADiff": "场均失球差",
}
_FEATURE_LABELS_EN = {
    "TournamentExperienceDiff": "Experience diff.",
    "BestStageDiff": "Best-stage diff.",
    "HistAvgGFDiff": "Goals-for diff.",
    "HistAvgGADiff": "Goals-against diff.",
}
_OUTCOME_LABELS_ZH = {
    "Home Win": "主胜",
    "Draw": "平局",
    "Away Win": "客胜",
}
_OUTCOME_LABELS_EN = {
    "Home Win": "Home win",
    "Draw": "Draw",
    "Away Win": "Away win",
}

_CHINESE_FONT_CANDIDATES = (
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
)
_AVAILABLE_FONTS = {item.name for item in font_manager.fontManager.ttflist}
_FORCED_LANGUAGE = os.environ.get("APMCM_PLOT_LANGUAGE", "").lower()
CHINESE_FONT = (
    None
    if _FORCED_LANGUAGE == "en"
    else next(
        (
            name
            for name in _CHINESE_FONT_CANDIDATES
            if name in _AVAILABLE_FONTS
        ),
        None,
    )
)
PLOT_LANGUAGE = "zh" if CHINESE_FONT else "en"

plt.rcParams.update(
    {
        "font.sans-serif": [CHINESE_FONT or "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#9aa3ad",
        "axes.linewidth": 0.8,
        "xtick.color": NAVY,
        "ytick.color": NAVY,
        "text.color": NAVY,
        "axes.labelcolor": NAVY,
        "axes.titlecolor": NAVY,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def font_status() -> dict[str, str | bool]:
    return {
        "chinese_font_available": CHINESE_FONT is not None,
        "selected_font": CHINESE_FONT or "DejaVu Sans",
        "plot_language": PLOT_LANGUAGE,
    }


def _t(chinese: str, english: str) -> str:
    return chinese if PLOT_LANGUAGE == "zh" else english


def _feature_label(name: str) -> str:
    mapping = _FEATURE_LABELS_ZH if PLOT_LANGUAGE == "zh" else _FEATURE_LABELS_EN
    return mapping.get(name, name)


def _outcome_label(name: str) -> str:
    mapping = _OUTCOME_LABELS_ZH if PLOT_LANGUAGE == "zh" else _OUTCOME_LABELS_EN
    return mapping.get(name, name)


def _style_axes(ax, grid_axis: str | None = "both") -> None:
    ax.tick_params(length=3, width=0.7, labelsize=9.5)
    if grid_axis:
        ax.grid(axis=grid_axis if grid_axis != "both" else "both", alpha=GRID_ALPHA, linewidth=0.7)
        ax.set_axisbelow(True)


def _finish(fig, destination: Path, dpi: int = DPI) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _significance_stars(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def _gaussian_kde_1d(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    clean = values[np.isfinite(values)]
    if clean.size < 2 or np.nanstd(clean) <= 0:
        return np.zeros_like(grid)
    try:
        density = stats.gaussian_kde(clean)(grid)
    except Exception:
        return np.zeros_like(grid)
    peak = float(np.max(density)) if density.size else 0.0
    if peak <= 0:
        return np.zeros_like(grid)
    return density / peak


def _annotate_extrema(
    ax,
    years: np.ndarray,
    values: np.ndarray,
    color: str,
) -> None:
    if years.size == 0:
        return
    i_max = int(np.nanargmax(values))
    i_min = int(np.nanargmin(values))
    for index, kind in ((i_max, "max"), (i_min, "min")):
        ax.scatter(
            [years[index]],
            [values[index]],
            s=42,
            color=color,
            zorder=4,
            edgecolors="white",
            linewidths=0.8,
        )
        offset = 8 if kind == "max" else -10
        ax.annotate(
            f"{int(years[index])}: {values[index]:.2f}",
            xy=(years[index], values[index]),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if kind == "max" else "top",
            fontsize=8,
            color=NAVY,
        )


def plot_problem1_trends(
    data: pd.DataFrame,
    total_destination: Path,
    average_destination: Path,
    breakpoint_year: int | None = None,
    breakpoint_fit: np.ndarray | None = None,
) -> None:
    years = data["Year"].to_numpy(float)
    totals = data["GoalsScored"].to_numpy(float)
    matches = data["MatchesPlayed"].to_numpy(float)
    slope_total = float(np.polyfit(years, totals, 1)[0])

    # Fig1: dual-axis time series (goals vs tournament scale) — not a side lollipop.
    fig, ax = plt.subplots(figsize=(9.4, 4.35))
    ax_scale = ax.twinx()
    bar_width = max(2.2, float(np.median(np.diff(years))) * 0.55) if len(years) > 1 else 2.5
    ax_scale.bar(
        years,
        matches,
        width=bar_width,
        color="#c5d4e0",
        edgecolor="white",
        linewidth=0.4,
        alpha=0.85,
        zorder=1,
        label=_t("比赛场次（右轴）", "Matches (right)"),
    )
    ax.fill_between(years, totals, alpha=0.10, color=BLUE, linewidth=0, zorder=2)
    ax.plot(
        years,
        totals,
        marker="o",
        markersize=5.4,
        markerfacecolor="white",
        markeredgewidth=1.4,
        color=BLUE,
        linewidth=2.15,
        label=_t("总进球（左轴）", "Total goals (left)"),
        zorder=3,
    )
    trend = np.polyval(np.polyfit(years, totals, 1), years)
    ax.plot(
        years,
        trend,
        color=PURPLE,
        linestyle="--",
        linewidth=1.25,
        alpha=0.9,
        label=_t(f"线性趋势 {slope_total:+.2f}/年", f"Linear trend {slope_total:+.2f}/yr"),
        zorder=3,
    )
    _annotate_extrema(ax, years, totals, BLUE)
    ax.set_xlabel(_t("世界杯年份", "World Cup year"))
    ax.set_ylabel(_t("总进球", "Total goals"), color=BLUE)
    ax_scale.set_ylabel(_t("比赛场次", "Matches played"), color="#5f7385")
    ax.tick_params(axis="y", colors=BLUE)
    ax_scale.tick_params(axis="y", colors="#5f7385")
    ax_scale.spines["right"].set_visible(True)
    ax_scale.spines["right"].set_color("#9aa3ad")
    ax.set_title(
        _t(
            "历届世界杯总进球与赛会规模（双轴）",
            "Total goals vs tournament scale (dual axis)",
        ),
        fontsize=12.5,
        pad=10,
    )
    handles_left, labels_left = ax.get_legend_handles_labels()
    handles_right, labels_right = ax_scale.get_legend_handles_labels()
    ax.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        frameon=False,
        loc="upper left",
        fontsize=8.5,
    )
    _style_axes(ax, grid_axis="y")
    ax_scale.grid(False)
    fig.tight_layout()
    _finish(fig, total_destination, dpi=280)

    averages = data["GoalsPerMatch"].to_numpy(float)
    mean_value = float(np.mean(averages))
    slope_avg = float(np.polyfit(years, averages, 1)[0])
    fig = plt.figure(figsize=(10.8, 4.35))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[2.35, 1.0], wspace=0.28)
    ax = fig.add_subplot(gs[0])
    ax_dist = fig.add_subplot(gs[1])

    ax.fill_between(years, averages, alpha=0.12, color=ORANGE, linewidth=0)
    ax.plot(
        years,
        averages,
        marker="o",
        markersize=5.2,
        markerfacecolor="white",
        markeredgewidth=1.35,
        color=ORANGE,
        linewidth=2.0,
        zorder=3,
        label=_t("场均进球", "Goals per match"),
    )
    ax.axhline(
        mean_value,
        color=GRAY,
        linestyle="--",
        linewidth=1.15,
        label=_t(f"历届均值 {mean_value:.2f}", f"Mean {mean_value:.2f}"),
    )
    if breakpoint_fit is not None and len(breakpoint_fit) == len(years):
        ax.plot(
            years,
            breakpoint_fit,
            color=PURPLE,
            linewidth=1.6,
            alpha=0.9,
            label=_t("分段拟合", "Segmented fit"),
        )
    if breakpoint_year is not None:
        ax.axvline(
            float(breakpoint_year),
            color=PURPLE,
            linestyle=":",
            linewidth=1.2,
            alpha=0.9,
            label=_t(f"候选断点 {breakpoint_year}", f"Breakpoint {breakpoint_year}"),
        )
    ax.plot(
        years,
        np.polyval(np.polyfit(years, averages, 1), years),
        color=BLUE,
        linestyle="--",
        linewidth=1.05,
        alpha=0.75,
        label=_t(f"线性 {slope_avg:+.3f}/年", f"Linear {slope_avg:+.3f}/yr"),
    )
    _annotate_extrema(ax, years, averages, ORANGE)
    ax.set(
        xlabel=_t("世界杯年份", "World Cup year"),
        ylabel=_t("场均进球", "Goals per match"),
    )
    ax.legend(frameon=False, loc="best", fontsize=8)
    _style_axes(ax)

    lo, hi = float(np.nanmin(averages)), float(np.nanmax(averages))
    grid = np.linspace(lo, hi, 100)
    density = _gaussian_kde_1d(averages, grid)
    ax_dist.fill_betweenx(grid, 0, density, color=ORANGE, alpha=0.18, linewidth=0)
    ax_dist.plot(density, grid, color=ORANGE, linewidth=1.4)
    ax_dist.axhline(mean_value, color=GRAY, linestyle="--", linewidth=1.0)
    ax_dist.boxplot(
        averages,
        vert=True,
        positions=[1.15],
        widths=0.18,
        manage_ticks=False,
        patch_artist=True,
        boxprops={"facecolor": "#f3e6dc", "edgecolor": ORANGE, "linewidth": 1.0},
        medianprops={"color": PURPLE, "linewidth": 1.3},
        whiskerprops={"color": GRAY},
        capprops={"color": GRAY},
        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": ORANGE, "alpha": 0.5},
    )
    ax_dist.set_xlim(0, 1.4)
    ax_dist.set_ylim(lo - 0.15, hi + 0.15)
    ax_dist.set_xticks([])
    ax_dist.set_xlabel(_t("密度 / 箱线", "Density / box"), fontsize=9)
    ax_dist.set_title(_t("分布诊断", "Distribution"), fontsize=10.5, pad=6)
    ax_dist.text(
        0.05,
        0.96,
        f"n={len(averages)}\nσ={np.nanstd(averages):.2f}",
        transform=ax_dist.transAxes,
        va="top",
        fontsize=8,
        color=NAVY,
    )
    _style_axes(ax_dist, grid_axis="y")
    fig.suptitle(
        _t("历届世界杯场均进球趋势与分布", "Goals-per-match trend and distribution"),
        fontsize=12.5,
        y=0.98,
    )
    _finish(fig, average_destination, dpi=280)


def plot_problem2_stage(stage_stats: pd.DataFrame, destination: Path) -> None:
    frame = stage_stats.dropna(subset=["AverageTotalGoals"]).copy()
    labels = (
        frame["StageChinese"]
        if PLOT_LANGUAGE == "zh"
        else frame["StageNormalized"]
    ).tolist()
    means = frame["AverageTotalGoals"].to_numpy(float)
    medians = (
        frame["MedianTotalGoals"].to_numpy(float)
        if "MedianTotalGoals" in frame
        else means.copy()
    )
    counts = (
        frame["MatchCount"].to_numpy(float)
        if "MatchCount" in frame
        else np.ones(len(frame), dtype=float)
    )
    stds = (
        frame["StandardDeviation"].to_numpy(float)
        if "StandardDeviation" in frame
        else np.full(len(frame), np.nan)
    )
    se = np.where(
        (counts > 1) & np.isfinite(stds),
        stds / np.sqrt(counts),
        np.nan,
    )
    n = len(means)
    stage_cmap = mcolors.LinearSegmentedColormap.from_list(
        "apmcm_stage",
        ["#c45c26", "#b8896a", "#7a6aa5", "#5b3a8c"],
    )
    palette = [mcolors.to_hex(stage_cmap(i / max(n - 1, 1))) for i in range(n)]
    overall = float(np.average(means, weights=np.maximum(counts, 1)))
    size = 55 + 140 * (counts / max(float(np.nanmax(counts)), 1.0))

    # Fig3: horizontal dumbbell / forest plot — distinct from dual-axis fig1.
    fig, ax = plt.subplots(figsize=(9.0, 4.7))
    y_pos = np.arange(n)
    ax.axvline(
        overall,
        color=GRAY,
        linestyle="--",
        linewidth=1.15,
        label=_t(f"加权均值 {overall:.2f}", f"Weighted mean {overall:.2f}"),
        zorder=1,
    )
    for y, mean, median, err, color, count, marker_size in zip(
        y_pos, means, medians, se, palette, counts, size
    ):
        lo = mean - 1.96 * err if np.isfinite(err) else mean
        hi = mean + 1.96 * err if np.isfinite(err) else mean
        ax.plot([lo, hi], [y, y], color=color, linewidth=2.0, solid_capstyle="round", zorder=2)
        ax.plot(
            [median, mean],
            [y, y],
            color=color,
            linewidth=1.1,
            alpha=0.55,
            zorder=2,
        )
        ax.scatter(
            [median],
            [y],
            s=42,
            marker="D",
            facecolors="white",
            edgecolors=color,
            linewidths=1.3,
            zorder=4,
        )
        ax.scatter(
            [mean],
            [y],
            s=marker_size,
            color=color,
            edgecolors="white",
            linewidths=0.8,
            zorder=5,
            alpha=0.95,
        )
        ax.text(
            hi + 0.04,
            y,
            f"μ={mean:.2f}  M={median:.2f}  n={int(count)}",
            va="center",
            fontsize=8,
            color=NAVY,
        )

    # Legend proxies.
    ax.scatter([], [], s=70, color=PURPLE, edgecolors="white", label=_t("均值（点大小∝n）", "Mean (size ∝ n)"))
    ax.scatter(
        [],
        [],
        s=42,
        marker="D",
        facecolors="white",
        edgecolors=ORANGE,
        linewidths=1.3,
        label=_t("中位数", "Median"),
    )
    ax.plot([], [], color=GRAY, linewidth=2.0, label=_t("均值 95% SE", "Mean 95% SE"))

    ax.set_yticks(y_pos, labels)
    ax.set_xlabel(_t("场均总进球", "Average total goals"))
    ax.set_xlim(
        max(0.0, float(np.nanmin(means - np.nan_to_num(1.96 * se))) - 0.15),
        float(np.nanmax(means + np.nan_to_num(1.96 * se))) + 1.35,
    )
    ax.set_ylim(-0.7, n - 0.3)
    ax.invert_yaxis()
    ax.set_title(
        _t(
            "各阶段进球：均值—中位数哑铃图（点大小∝样本量）",
            "Goals by stage: mean–median dumbbell (size ∝ n)",
        ),
        fontsize=12.5,
        pad=10,
    )
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    _style_axes(ax, grid_axis="x")
    fig.tight_layout()
    _finish(fig, destination, dpi=280)


def plot_problem2_half_full_scatter(
    complete: pd.DataFrame,
    pearson_r: float,
    destination: Path,
    pearson_p: float | None = None,
) -> None:
    x_raw = complete["HalfTimeTotal"].to_numpy(float)
    y_raw = complete["TotalGoals"].to_numpy(float)
    mask = np.isfinite(x_raw) & np.isfinite(y_raw)
    x, y = x_raw[mask], y_raw[mask]
    n = len(x)

    fig = plt.figure(figsize=(8.6, 6.2))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[4.0, 1.15],
        height_ratios=[1.15, 4.0],
        wspace=0.06,
        hspace=0.06,
        left=0.10,
        right=0.96,
        top=0.90,
        bottom=0.10,
    )
    ax_histx = fig.add_subplot(gs[0, 0])
    ax_histy = fig.add_subplot(gs[1, 1])
    ax = fig.add_subplot(gs[1, 0], sharex=ax_histx, sharey=ax_histy)
    ax_stats = fig.add_subplot(gs[0, 1])
    ax_stats.axis("off")

    ax.scatter(x, y, alpha=0.28, s=18, color=BLUE, edgecolors="none", zorder=2)
    slope = intercept = sigma = np.nan
    if len(np.unique(x)) > 1:
        slope, intercept = np.polyfit(x, y, 1)
        grid = np.linspace(float(np.min(x)), float(np.max(x)), 120)
        fitted = slope * grid + intercept
        resid = y - (slope * x + intercept)
        sigma = float(np.std(resid, ddof=1)) if len(resid) > 2 else float(np.std(resid))
        ax.fill_between(
            grid,
            fitted - 1.96 * sigma,
            fitted + 1.96 * sigma,
            color=ORANGE,
            alpha=0.14,
            linewidth=0,
            zorder=1,
            label=_t("约 95% 残差带", "~95% residual band"),
        )
        ax.plot(grid, fitted, color=ORANGE, linewidth=2.1, zorder=3, label=_t("OLS 拟合", "OLS fit"))
        ax.legend(frameon=False, loc="upper left", fontsize=8)

    ax.set_xlabel(_t("半场总进球", "Half-time total goals"))
    ax.set_ylabel(_t("全场总进球", "Full-time total goals"))
    _style_axes(ax)

    x_grid = np.linspace(float(np.min(x)), float(np.max(x)), 80)
    y_grid = np.linspace(float(np.min(y)), float(np.max(y)), 80)
    ax_histx.fill_between(x_grid, _gaussian_kde_1d(x, x_grid), color=BLUE, alpha=0.25, linewidth=0)
    ax_histx.plot(x_grid, _gaussian_kde_1d(x, x_grid), color=BLUE, linewidth=1.2)
    ax_histx.set_ylim(0, None)
    ax_histx.tick_params(labelbottom=False, labelleft=False, length=0)
    for spine in ax_histx.spines.values():
        spine.set_visible(False)

    dens_y = _gaussian_kde_1d(y, y_grid)
    ax_histy.fill_betweenx(y_grid, 0, dens_y, color=ORANGE, alpha=0.25, linewidth=0)
    ax_histy.plot(dens_y, y_grid, color=ORANGE, linewidth=1.2)
    ax_histy.set_xlim(0, None)
    ax_histy.tick_params(labelbottom=False, labelleft=False, length=0)
    for spine in ax_histy.spines.values():
        spine.set_visible(False)

    p_text = f"{pearson_p:.3g}" if pearson_p is not None and np.isfinite(pearson_p) else "—"
    stars = _significance_stars(float(pearson_p)) if pearson_p is not None else ""
    stats_lines = [
        _t("相关诊断", "Correlation diagnostics"),
        f"n = {n}",
        f"Pearson r = {pearson_r:.3f}{stars}",
        f"p = {p_text}",
    ]
    if np.isfinite(slope):
        stats_lines.append(f"slope = {slope:.3f}")
        stats_lines.append(f"σ_resid = {sigma:.3f}")
    ax_stats.text(
        0.02,
        0.95,
        "\n".join(stats_lines),
        transform=ax_stats.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color=NAVY,
        linespacing=1.45,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#f7f4f8",
            "edgecolor": LIGHT_GRAY,
            "linewidth": 0.8,
        },
    )
    fig.suptitle(
        _t("半场与全场总进球相关（含边际分布）", "Half- vs full-time goals with margins"),
        fontsize=12.5,
        y=0.97,
    )
    _finish(fig, destination, dpi=300)


def plot_problem3_feature_correlation(
    feature_frame: pd.DataFrame,
    spearman_rho: pd.DataFrame,
    spearman_p: pd.DataFrame,
    vif: pd.DataFrame,
    outcome_corr: pd.DataFrame,
    destination: Path,
) -> None:
    names = list(feature_frame.columns)
    n = len(names)
    if n == 0:
        raise ValueError("feature_frame must contain at least one feature column")
    labels = [_feature_label(name) for name in names]
    rho = spearman_rho.loc[names, names].to_numpy(float)
    pmat = spearman_p.loc[names, names].to_numpy(float)
    data = feature_frame[names].to_numpy(float)

    vif_map = {
        str(row["Feature"]): float(row["VIF"])
        for _, row in vif.iterrows()
    }
    vif_values = np.array([vif_map.get(name, np.nan) for name in names], float)

    outcome_cols = [col for col in ("Home Win", "Draw", "Away Win") if col in outcome_corr.columns]
    if not outcome_cols:
        outcome_cols = list(outcome_corr.columns)
    outcome_matrix = outcome_corr.loc[names, outcome_cols].to_numpy(float)
    outcome_xlabels = [_outcome_label(col) for col in outcome_cols]

    fig = plt.figure(figsize=(12.8, 6.9))
    outer = GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[1.62, 1.0],
        wspace=0.22,
        left=0.07,
        right=0.98,
        top=0.84,
        bottom=0.10,
    )
    # Dedicated bottom row for Spearman colorbar; x-labels sit on the top row.
    left_block = outer[0].subgridspec(
        2,
        1,
        height_ratios=[20.0, 1.05],
        hspace=0.22,
    )
    left = left_block[0].subgridspec(1, 2, width_ratios=[4.0, 1.15], wspace=0.18)
    corr_grid = left[0].subgridspec(n, n, wspace=0.04, hspace=0.04)

    norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
    scatter_color = ORANGE
    kde_color = PURPLE

    axes = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            ax = fig.add_subplot(corr_grid[i, j])
            axes[i, j] = ax
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
                spine.set_color(LIGHT_GRAY)

            if i == j:
                values = data[:, i]
                lo, hi = np.nanpercentile(values, [1, 99])
                if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
                    if not np.isfinite(lo) or lo == hi:
                        lo, hi = lo - 1.0, hi + 1.0
                grid = np.linspace(lo, hi, 80)
                density = _gaussian_kde_1d(values, grid)
                ax.fill_between(grid, density, color=kde_color, alpha=0.18, linewidth=0)
                ax.plot(grid, density, color=kde_color, linewidth=1.15)
                ax.set_ylim(0, 1.08)
                ax.set_xlim(lo, hi)
            elif i > j:
                ax.scatter(
                    data[:, j],
                    data[:, i],
                    s=7,
                    alpha=0.28,
                    color=scatter_color,
                    edgecolors="none",
                )
            else:
                value = float(rho[i, j])
                ax.set_facecolor(CORR_CMAP(norm(value)))
                stars = _significance_stars(float(pmat[i, j]))
                text_color = "white" if abs(value) >= 0.55 else NAVY
                ax.text(
                    0.5,
                    0.58 if stars else 0.5,
                    f"{value:.2f}",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=text_color,
                    fontweight="medium",
                )
                if stars:
                    ax.text(
                        0.5,
                        0.28,
                        stars,
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=text_color,
                    )
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

            if i == 0:
                ax.set_xlabel(labels[j], fontsize=7.8, rotation=28, ha="left")
                ax.xaxis.set_label_position("top")
                ax.xaxis.set_label_coords(0.5, 1.18)
            if j == 0:
                ax.set_ylabel(labels[i], fontsize=8.0)
                ax.yaxis.set_label_coords(-0.22, 0.5)

    ax_vif = fig.add_subplot(left[1])
    y_pos = np.arange(n)
    vmax = float(np.nanmax(vif_values)) if np.isfinite(vif_values).any() else 1.0
    vmax = max(vmax * 1.12, 1.2)
    colors = [VIF_CMAP(i / max(n - 1, 1)) for i in range(n)]
    for y, value, color in zip(y_pos, vif_values, colors):
        if not np.isfinite(value):
            continue
        ax_vif.hlines(y, 0.0, value, color=color, linewidth=1.6, alpha=0.85)
        ax_vif.plot(value, y, "o", color=color, markersize=7.5, zorder=3)
        ax_vif.text(
            value + 0.03 * vmax,
            y,
            f"{value:.2f}",
            va="center",
            ha="left",
            fontsize=8,
            color=NAVY,
        )
    ax_vif.axvline(1.0, color=GRAY, linestyle="--", linewidth=1.0, alpha=0.85)
    ax_vif.set_yticks(y_pos)
    ax_vif.set_yticklabels(labels, fontsize=8)
    ax_vif.set_xlim(0.0, vmax)
    ax_vif.set_ylim(-0.6, n - 0.4)
    ax_vif.invert_yaxis()  # align row 0 with corrgram top row
    ax_vif.set_xlabel("VIF", fontsize=9)
    ax_vif.set_title(_t("VIF 值", "VIF value"), fontsize=10, pad=6)
    ax_vif.spines["top"].set_visible(False)
    ax_vif.spines["right"].set_visible(False)
    _style_axes(ax_vif, grid_axis="x")

    ax_out = fig.add_subplot(outer[1])
    image = ax_out.imshow(outcome_matrix, cmap=CORR_CMAP, vmin=-1, vmax=1, aspect="auto")
    for row in range(outcome_matrix.shape[0]):
        for column in range(outcome_matrix.shape[1]):
            value = float(outcome_matrix[row, column])
            text_color = "white" if abs(value) >= 0.55 else NAVY
            ax_out.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                color=text_color,
            )
    ax_out.set_xticks(range(len(outcome_xlabels)), outcome_xlabels)
    ax_out.set_yticks(range(n), labels)
    ax_out.set_xlabel(_t("结局指示变量", "Outcome indicator"), fontsize=10)
    ax_out.set_title(
        _t("特征 × 三类结局相关", "Features × three outcomes"),
        fontsize=11,
        pad=8,
    )
    for spine in ax_out.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax_out, fraction=0.046, pad=0.04)
    cbar.set_label(
        _t("Pearson（对指示变量）", "Pearson vs indicators"),
        fontsize=8.5,
    )
    cbar.ax.tick_params(labelsize=8)

    sm = plt.cm.ScalarMappable(cmap=CORR_CMAP, norm=norm)
    sm.set_array([])
    cax = fig.add_subplot(left_block[1])
    cbar_left = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar_left.set_label(_t("Spearman 相关", "Spearman correlation"), fontsize=8.5)
    cbar_left.ax.tick_params(labelsize=8)

    fig.suptitle(
        _t(
            "四特征诊断：Spearman corrgram + VIF，及与三类结局指示变量相关",
            "Four-feature diagnostics: Spearman corrgram + VIF, and outcome indicators",
        ),
        fontsize=12.5,
        y=0.97,
    )
    _finish(fig, destination, dpi=320)


def plot_problem3_confusion(matrix: np.ndarray, destination: Path) -> None:
    labels = (
        ["主胜", "平局", "客胜"]
        if PLOT_LANGUAGE == "zh"
        else ["Home win", "Draw", "Away win"]
    )
    counts = np.asarray(matrix, dtype=float)
    total = float(counts.sum())
    row_sums = counts.sum(axis=1, keepdims=True)
    row_norm = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)
    recall = np.diag(row_norm)
    precision_den = counts.sum(axis=0)
    precision = np.divide(
        np.diag(counts),
        precision_den,
        out=np.zeros(3, dtype=float),
        where=precision_den > 0,
    )
    correct = int(np.trace(counts))
    accuracy = correct / total if total > 0 else 0.0

    fig = plt.figure(figsize=(12.4, 5.7))
    # Row0: heatmaps / metrics; Row1: horizontal colorbars under heatmaps only.
    gs = GridSpec(
        2,
        3,
        figure=fig,
        height_ratios=[12.0, 1.35],
        width_ratios=[1.15, 1.15, 1.05],
        hspace=0.38,
        wspace=0.40,
        left=0.07,
        right=0.98,
        top=0.88,
        bottom=0.14,
    )
    ax_count = fig.add_subplot(gs[0, 0])
    ax_recall = fig.add_subplot(gs[0, 1])
    ax_metrics = fig.add_subplot(gs[0, 2])
    cax1 = fig.add_subplot(gs[1, 0])
    cax2 = fig.add_subplot(gs[1, 1])

    image = ax_count.imshow(counts, cmap="Blues", vmin=0)
    vmax = float(np.nanmax(counts)) if counts.size else 1.0
    for row in range(3):
        for column in range(3):
            count = int(counts[row, column])
            pct = 100.0 * count / total if total > 0 else 0.0
            text_color = "white" if counts[row, column] >= 0.55 * vmax else NAVY
            ax_count.text(
                column,
                row,
                f"{count}\n({pct:.1f}%)",
                ha="center",
                va="center",
                fontsize=10,
                color=text_color,
                linespacing=1.2,
            )
    ax_count.set_xticks([0, 1, 2], labels)
    ax_count.set_yticks([0, 1, 2], labels)
    ax_count.set_title(_t("计数混淆矩阵", "Count confusion matrix"), fontsize=11, pad=8)
    ax_count.set_xlabel(_t("预测类别", "Predicted"), fontsize=9.5)
    ax_count.set_ylabel(_t("真实类别", "Actual"), fontsize=9.5)
    for spine in ax_count.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, cax=cax1, orientation="horizontal")
    cbar.set_label(_t("场次计数", "Count"), fontsize=8)
    cbar.ax.tick_params(labelsize=7.5)

    image2 = ax_recall.imshow(row_norm, cmap="Purples", vmin=0, vmax=1)
    for row in range(3):
        for column in range(3):
            value = float(row_norm[row, column])
            text_color = "white" if value >= 0.55 else NAVY
            ax_recall.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=11,
                color=text_color,
            )
    ax_recall.set_xticks([0, 1, 2], labels)
    ax_recall.set_yticks([0, 1, 2], labels)
    ax_recall.set_title(
        _t("行归一化（召回视角）", "Row-normalized (recall view)"),
        fontsize=11,
        pad=8,
    )
    ax_recall.set_xlabel(_t("预测类别", "Predicted"), fontsize=9.5)
    # Avoid repeating the same y-label next to the previous panel.
    ax_recall.set_ylabel("")
    for spine in ax_recall.spines.values():
        spine.set_visible(False)
    cbar2 = fig.colorbar(image2, cax=cax2, orientation="horizontal")
    cbar2.set_label(_t("行内占比", "Row share"), fontsize=8)
    cbar2.ax.tick_params(labelsize=7.5)

    y_pos = np.arange(3)
    ax_metrics.hlines(y_pos - 0.14, 0, recall, color=PURPLE, linewidth=1.6, label="Recall")
    ax_metrics.plot(recall, y_pos - 0.14, "o", color=PURPLE, markersize=7)
    ax_metrics.hlines(y_pos + 0.14, 0, precision, color=ORANGE, linewidth=1.6, label="Precision")
    ax_metrics.plot(precision, y_pos + 0.14, "o", color=ORANGE, markersize=7)
    for y, rec, prec in zip(y_pos, recall, precision):
        ax_metrics.text(
            min(rec + 0.04, 1.28),
            y - 0.14,
            f"{rec:.2f}",
            va="center",
            fontsize=8,
            color=PURPLE,
        )
        ax_metrics.text(
            min(prec + 0.04, 1.28),
            y + 0.14,
            f"{prec:.2f}",
            va="center",
            fontsize=8,
            color=ORANGE,
        )
    ax_metrics.axvline(
        accuracy,
        color=GRAY,
        linestyle="--",
        linewidth=1.1,
        alpha=0.9,
        label=_t(f"准确率 {accuracy:.3f}", f"Accuracy {accuracy:.3f}"),
    )
    ax_metrics.set_yticks(y_pos, labels)
    ax_metrics.set_xlim(0, 1.35)
    ax_metrics.set_ylim(-0.7, 2.7)
    ax_metrics.invert_yaxis()
    ax_metrics.set_xlabel(_t("比率", "Rate"), fontsize=9.5)
    ax_metrics.set_title(
        _t(
            f"分类表现（正确 {correct}/{int(total)}）",
            f"Class metrics ({correct}/{int(total)} correct)",
        ),
        fontsize=11,
        pad=8,
    )
    ax_metrics.legend(frameon=False, fontsize=8, loc="lower right")
    _style_axes(ax_metrics, grid_axis="x")

    fig.suptitle(
        _t(
            "2014 逻辑回归混淆矩阵与分类诊断",
            "2014 logistic regression confusion diagnostics",
        ),
        fontsize=12.5,
        y=0.96,
    )
    _finish(fig, destination, dpi=360)


__all__ = [
    "font_status",
    "plot_problem1_trends",
    "plot_problem2_half_full_scatter",
    "plot_problem2_stage",
    "plot_problem3_confusion",
    "plot_problem3_feature_correlation",
]
