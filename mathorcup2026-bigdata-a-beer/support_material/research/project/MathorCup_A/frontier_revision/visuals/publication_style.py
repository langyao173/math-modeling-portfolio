"""Shared, restrained Matplotlib style for source-backed research figures."""
from pathlib import Path
import os
import subprocess
from xml.sax.saxutils import escape
os.environ.setdefault("MPLCONFIGDIR",str(Path(__file__).resolve().parent/"qa/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

COLORS = dict(ink="#263B43", text="#53656D", teal="#166B68", light_teal="#4D9993",
              orange="#CB8748", gray="#A9B5BA", line="#D5DDDF", pale="#E8F1EF",
              reject="#A96A75", slate="#718F99", olive="#869060")
FONT_CONFIG=None


def configure(font_path):
    global FONT_CONFIG
    font_manager.fontManager.addfont(str(font_path))
    qa=Path(__file__).resolve().parent/"qa";qa.mkdir(parents=True,exist_ok=True)
    FONT_CONFIG=qa/"quantitative_fonts.conf"
    FONT_CONFIG.write_text('<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "fonts.dtd"><fontconfig>'
        '<include ignore_missing="yes">/etc/fonts/fonts.conf</include>'
        f'<dir>{escape(str(Path(font_path).parent))}</dir><cachedir>{escape(str(qa/"fontcache"))}</cachedir></fontconfig>')
    plt.rcParams.update({
        "font.family":"Noto Sans CJK SC", "font.size":10.5,
        "axes.labelsize":10.5, "axes.titlesize":11.5, "axes.titleweight":"normal",
        "axes.labelcolor":COLORS["ink"], "text.color":COLORS["ink"],
        "xtick.color":COLORS["text"], "ytick.color":COLORS["text"],
        "axes.edgecolor":COLORS["gray"], "axes.linewidth":.7,
        "axes.spines.top":False, "axes.spines.right":False,
        "axes.unicode_minus":False, "grid.color":COLORS["line"],
        "grid.linewidth":.55, "grid.alpha":.75,
        "xtick.major.size":3, "ytick.major.size":3,
        "xtick.major.width":.6, "ytick.major.width":.6,
        "legend.frameon":False, "legend.fontsize":9.5,
        "figure.facecolor":"white", "axes.facecolor":"white",
        "savefig.facecolor":"white", "svg.fonttype":"none",
        "svg.hashsalt":"mathorcup-a-source-backed-figures-v2",
        "pdf.fonttype":3, "ps.fonttype":3,
        "lines.linewidth":1.8, "lines.markersize":5.5,
        "mathtext.fontset":"dejavusans",
    })


def panel_title(ax, letter, title, y=1.06):
    ax.text(0,y,letter,transform=ax.transAxes,color=COLORS["orange"],
            fontsize=13.5,fontweight="bold",va="bottom")
    ax.annotate(title,(0,y),xycoords=ax.transAxes,xytext=(17,0),
                textcoords="offset points",fontsize=11.5,va="bottom")


def clean(ax, grid="y"):
    ax.set_axisbelow(True)
    if grid:ax.grid(axis=grid)
    ax.tick_params(pad=4)


def dot_handle(color, label, marker="o", filled=True):
    return Line2D([],[],color=color,marker=marker,lw=0,
                  markerfacecolor=color if filled else "white",
                  markeredgewidth=1.4,label=label,markersize=6)


def save(fig, path, notes=None):
    """Use one fixed canvas for all three formats; preserve vector text."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    meta={"Creator":"Python / Matplotlib; source-backed MathorCup A figures"}
    fig.savefig(path.with_suffix(".svg"),metadata={"Creator":meta["Creator"],"Date":None})
    fig.savefig(path.with_suffix(".png"),dpi=240)
    # Cairo embeds the supplied OpenType/CFF CJK font correctly. Matplotlib's
    # Type42 embedding produced visually corrupt CID glyph maps in this runtime.
    # Exporting the native-text SVG through Inkscape preserves vector text.
    env=os.environ.copy();env["FONTCONFIG_FILE"]=str(FONT_CONFIG)
    env["INKSCAPE_PROFILE_DIR"]=str(Path(__file__).resolve().parent/"qa/inkscape")
    subprocess.run(["inkscape",str(path.with_suffix(".svg")),"--export-type=pdf",
                    f'--export-filename={path.with_suffix(".pdf")}'],check=True,
                   env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if notes:path.with_suffix(".caption.txt").write_text(notes,encoding="utf-8")
    plt.close(fig)
