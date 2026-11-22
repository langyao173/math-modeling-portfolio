# -*- coding: utf-8 -*-
"""Shared publication-style settings for all quantitative manuscript figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager


METRIC_COLORS = {
    "R": "#0F4D92",      # blue: thermal resistance
    "dP": "#B64342",    # red/orange: pressure drop
    "phi": "#2E7D32",   # green: temperature uniformity
}

NEUTRAL = {
    "black": "#272727",
    "dark": "#4D4D4D",
    "mid": "#767676",
    "light": "#CFCECE",
    "pale": "#F2F3F5",
}

DESIGN_COLORS = {
    "recommended": "#C1121F",
    "topsis": "#6A4C93",
    "ideal": "#C1121F",
    "cheby": "#00897B",
    "minmax": "#E76F51",
    "coverage": "#C1121F",
    "single": "#4D4D4D",
}

DESIGN_MARKERS = {
    "recommended": "*",
    "topsis": "P",
    "ideal": "*",
    "cheby": "X",
    "minmax": "D",
    "coverage": "*",
    "single": "o",
}


def apply_pub_style() -> None:
    """Apply a restrained journal-style matplotlib configuration."""
    preferred_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
        "DejaVu Sans",
        "Arial",
        "sans-serif",
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    font_chain = [name for name in preferred_fonts if name == "sans-serif" or name in installed]
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": font_chain,
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "figure.dpi": 120,
        "savefig.dpi": 450,
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "grid.linewidth": 0.35,
        "grid.alpha": 0.22,
    })


def panel_label(ax, label: str) -> None:
    text_kwargs = dict(
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=NEUTRAL["black"],
    )
    if hasattr(ax, "text2D"):
        ax.text2D(-0.12, 1.06, label, **text_kwargs)
    else:
        ax.text(-0.12, 1.06, label, **text_kwargs)


def save_pub(fig, filename: str | Path, *, dpi: int = 450) -> None:
    """Save raster preview plus editable vector versions."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
