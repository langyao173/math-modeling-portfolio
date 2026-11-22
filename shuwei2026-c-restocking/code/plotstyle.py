# -*- coding: utf-8 -*-
"""Common Matplotlib style for all report figures (Matplotlib + NumPy + Pandas only; English text only)."""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import FIGURES

# Okabe-Ito colour-blind-safe palette
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', red='#D55E00', purple='#CC79A7', sky='#56B4E9', yellow='#F0E442',
         black='#222222', grey='#888888', light='#CCCCCC')
PHASE_COL = ['#56B4E9', '#009E73', '#E69F00', '#D55E00']
# one accent colour per problem (dark, light tint) used by the schematic figures
P_COL = {1: ('#1F6FB2', '#E6F0F9'), 2: ('#D4660A', '#FCEEE0'), 3: ('#0B8A66', '#E2F4EE'), 0: ('#4A4F57', '#F1F2F4')}
W_FULL = 6.6          # inches: text width of the report


SANS = ['Arial', 'Helvetica', 'Liberation Sans', 'TeX Gyre Heros', 'DejaVu Sans']      # metric-compatible sans-serif TrueType fonts


def setup():
    from matplotlib import font_manager
    have = set(font_manager.get_font_names())
    sans = [f for f in SANS if f in have] or ['DejaVu Sans']        # only installed families (no font warnings in the logs)
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': sans,
        'mathtext.fontset': 'custom', 'mathtext.rm': 'sans', 'mathtext.it': 'sans:italic', 'mathtext.bf': 'sans:bold',
        'mathtext.cal': 'sans:italic', 'mathtext.sf': 'sans', 'mathtext.tt': 'sans',
        'font.size': 8.0, 'axes.titlesize': 8.0, 'axes.labelsize': 8.0, 'xtick.labelsize': 7.0, 'ytick.labelsize': 7.0,
        'legend.fontsize': 6.8, 'legend.frameon': False, 'legend.handlelength': 1.8, 'legend.borderaxespad': 0.3,
        'axes.linewidth': 0.7, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
        'grid.linewidth': 0.35, 'grid.color': '#E3E3E3', 'grid.linestyle': '-', 'axes.axisbelow': True,
        'lines.linewidth': 1.3, 'lines.markersize': 3.5, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'out', 'ytick.direction': 'out',
        'figure.dpi': 120, 'savefig.dpi': 300, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'figure.constrained_layout.use': True,
    })


def label(ax, s, x=-0.16, y=1.04):
    ax.text(x, y, '(%s)' % s, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='left')


def save(fig, name):
    fig.savefig(os.path.join(FIGURES, name + '.pdf'))
    fig.savefig(os.path.join(FIGURES, name + '.png'), dpi=200)
    plt.close(fig)
    print('saved', name)
