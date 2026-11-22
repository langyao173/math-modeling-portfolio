"""Minimal, self-contained Matplotlib style shared by the extension figures.

Serif text (STIX, Times-like), no background grid, colour-blind-safe palette
(Okabe-Ito), vector PDF + 300-dpi PNG export.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
MM = 1 / 25.4
SINGLE, DOUBLE = 89 * MM, 183 * MM


def apply(fontsize: float = 7.0) -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": fontsize,
        "axes.labelsize": fontsize, "axes.titlesize": fontsize, "legend.fontsize": fontsize - 0.5,
        "xtick.labelsize": fontsize - 0.5, "ytick.labelsize": fontsize - 0.5,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.direction": "out", "ytick.direction": "out", "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.prop_cycle": matplotlib.cycler(color=OKABE_ITO),
    })


def save(fig, path_stem) -> None:
    fig.savefig(f"{path_stem}.pdf")
    fig.savefig(f"{path_stem}.png", dpi=300)
    plt.close(fig)
